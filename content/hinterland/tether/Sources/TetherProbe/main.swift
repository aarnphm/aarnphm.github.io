import Foundation
import TetherCore

@main
struct TetherProbe {
  @MainActor
  static func main() async {
    do {
      let args = Array(CommandLine.arguments.dropFirst())
      guard let action = args.first else { throw ProbeError.usage }
      switch action {
      case "devices":
        let monitor = DeviceMonitor()
        var snapshot: [GarminDevice] = []
        monitor.onChange = { snapshot = $0 }
        try monitor.start()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(snapshot), as: UTF8.self))
        monitor.stop()
      case "import":
        guard args.count == 3 else { throw ProbeError.usage }
        let archive = ActivityArchive(root: URL(fileURLWithPath: args[2]))
        let result = try await archive.ingest(
          file: URL(fileURLWithPath: args[1]), deviceID: "probe")
        print(String(decoding: try JSONEncoder().encode(result), as: UTF8.self))
      case "records":
        guard args.count == 2 else { throw ProbeError.usage }
        let archive = ActivityArchive(root: URL(fileURLWithPath: args[1]))
        print(String(decoding: try JSONEncoder().encode(await archive.allRecords()), as: UTF8.self))
      case "account":
        let client = GarminConnectClient()
        if args.count == 2 { try await client.importSession(from: URL(fileURLWithPath: args[1])) }
        print(await client.connectionStatus())
      case "upload":
        guard args.count == 2 else { throw ProbeError.usage }
        let bytes = try Data(contentsOf: URL(fileURLWithPath: args[1]))
        try FITInspector.validateActivity(bytes)
        let client = GarminConnectClient()
        let receipt = try await client.upload(file: URL(fileURLWithPath: args[1]))
        if let id = receipt.activityID { try await client.verify(activityID: id) }
        print("duplicate=\(receipt.duplicate) activityID=\(receipt.activityID ?? "unverified")")
      case "reconcile":
        guard args.count == 2 else { throw ProbeError.usage }
        let archive = ActivityArchive(root: URL(fileURLWithPath: args[1]))
        let client = GarminConnectClient()
        for record in try await archive.allRecords() where record.state != "confirmed" {
          let found = try await client.reconcile(
            file: URL(fileURLWithPath: record.localPath), filename: record.filename)
          if let id = found {
            try await archive.update(id: record.id, state: "confirmed", activityID: id)
          }
          FileHandle.standardOutput.write(
            Data("\(record.filename): \(found == nil ? "unresolved" : "confirmed")\n".utf8))
        }
      case "latency":
        let monitor = DeviceMonitor()
        var timings: [Double] = []
        var count = 0
        monitor.onChange = { count = $0.count }
        for _ in 0..<200 {
          let start = ContinuousClock.now
          monitor.refresh()
          guard count > 0 else { throw ProbeError.noDevice }
          let duration = start.duration(to: .now)
          timings.append(
            Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds)
              / 1e15)
        }
        timings.sort()
        print(
          "{\"metric\":\"registry_snapshot\",\"repetitions\":200,\"median_ms\":\((timings[99] + timings[100]) / 2),\"p95_ms\":\(timings[189]),\"max_ms\":\(timings[199])}"
        )
      case "watch":
        let seconds: UInt64
        if args.count == 2, let value = UInt64(args[1]), (1...300).contains(value) {
          seconds = value
        } else if args.count == 1 {
          seconds = 90
        } else {
          throw ProbeError.usage
        }
        let monitor = DeviceMonitor()
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        monitor.onObservation = { observation in
          do {
            var line = try encoder.encode(observation)
            line.append(0x0A)
            FileHandle.standardOutput.write(line)
          } catch {
            FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
          }
        }
        try monitor.start()
        try await Task.sleep(nanoseconds: seconds * 1_000_000_000)
        monitor.stop()
      default: throw ProbeError.usage
      }
    } catch {
      FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
      exit(1)
    }
  }
  enum ProbeError: LocalizedError {
    case usage, noDevice
    var errorDescription: String? {
      if case .noDevice = self { return "No Garmin device observed; latency sample is invalid." }
      return
        "Usage: tether-probe devices | latency | import FIT ARCHIVE | records ARCHIVE | account [SESSION_JSON] | upload FIT | watch [SECONDS:1...300]"
    }
  }
}
