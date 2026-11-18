import Darwin
import Foundation

public struct WatchActivity: Codable, Identifiable, Sendable {
  public let id: String
  public let name: String
  public let size: Int64
  public let modified: Date?
  public let storageID: UInt32
  public let sourcePath: String
  public let serial: String

}
public struct WatchListing: Codable, Sendable {
  public let model: String
  public let serial: String
  public let activities: [WatchActivity]
}
public struct TransportFailure: LocalizedError {
  public let message: String
  public var errorDescription: String? { message }
}

/// Each operation owns a child process, which bounds Go panics, USB stalls, and callback state.
public actor KalamTransport {
  private let expectedSerial: String
  private let helperURL: URL
  private var active: Process?
  public private(set) var deviceModel: String?

  public init(expectedSerial: String, helperURL: URL? = nil) {
    self.expectedSerial = expectedSerial
    self.helperURL =
      helperURL
      ?? (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0]))
      .deletingLastPathComponent().appendingPathComponent("tether-mtp")
  }
  public func enumerateActivities() async throws -> [WatchActivity] {
    let data = try await execute(["list", "--serial", expectedSerial])
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .custom { decoder in
      let value = try decoder.singleValueContainer().decode(String.self)
      let parser = ISO8601DateFormatter()
      parser.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
      guard let date = parser.date(from: value) else {
        throw TransportFailure(message: "Invalid activity modification time")
      }
      return date
    }
    let listing = try decoder.decode(WatchListing.self, from: data)
    guard listing.serial == expectedSerial else {
      throw TransportFailure(message: "Watch identity changed")
    }
    deviceModel = listing.model
    return listing.activities
  }
  public func download(_ activity: WatchActivity, to destination: URL) async throws {
    guard activity.serial == expectedSerial, destination.isFileURL else {
      throw TransportFailure(message: "Invalid activity identity or destination")
    }
    let data = try await execute([
      "download", "--serial", expectedSerial, "--storage", String(activity.storageID), "--path",
      activity.sourcePath, "--out", destination.path,
    ])
    struct Completion: Decodable { let ok: Bool }
    guard try JSONDecoder().decode(Completion.self, from: data).ok else {
      throw TransportFailure(message: "Transfer did not complete")
    }
    let values = try destination.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
    guard values.isRegularFile == true, values.fileSize.map(Int64.init) == activity.size else {
      throw TransportFailure(message: "Downloaded file size does not match the watch listing")
    }
  }
  public func cancel() { stopActive() }
  public func disconnect() {
    stopActive()
    deviceModel = nil
  }
  private func stopActive() {
    guard let process = active, process.isRunning else { return }
    process.terminate()
    Task {
      try? await Task.sleep(nanoseconds: 500_000_000)
      if process.isRunning { kill(process.processIdentifier, SIGKILL) }
    }
  }
  private func execute(_ arguments: [String]) async throws -> Data {
    try Task.checkCancellation()
    guard active == nil else {
      throw TransportFailure(message: "A watch operation is already running")
    }
    guard !expectedSerial.isEmpty else {
      throw TransportFailure(message: "A watch identity is required")
    }
    guard FileManager.default.isExecutableFile(atPath: helperURL.path) else {
      throw TransportFailure(message: "Missing tether-mtp helper at \(helperURL.path)")
    }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "tether-response-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let stdoutURL = directory.appendingPathComponent("stdout")
    let stderrURL = directory.appendingPathComponent("stderr")
    _ = FileManager.default.createFile(atPath: stdoutURL.path, contents: nil)
    _ = FileManager.default.createFile(atPath: stderrURL.path, contents: nil)
    let output = try FileHandle(forWritingTo: stdoutURL)
    let errors = try FileHandle(forWritingTo: stderrURL)
    defer {
      try? output.close()
      try? errors.close()
    }
    let process = Process()
    process.executableURL = helperURL
    process.arguments = arguments
    var environment = ProcessInfo.processInfo.environment
    environment["TMPDIR"] = directory.path + "/"
    process.environment = environment
    process.standardOutput = output
    process.standardError = errors
    active = process
    defer { active = nil }
    try process.run()
    let deadline = ContinuousClock.now.advanced(by: .seconds(20))
    do {
      while process.isRunning {
        try Task.checkCancellation()
        guard ContinuousClock.now < deadline else {
          throw TransportFailure(
            message: "Watch operation exceeded 20 seconds; the USB session was released")
        }
        try await Task.sleep(nanoseconds: 50_000_000)
      }
    } catch {
      stopActive()
      try? await Task.sleep(nanoseconds: 600_000_000)
      if process.isRunning { kill(process.processIdentifier, SIGKILL) }
      throw error
    }
    guard process.terminationStatus == 0 else {
      let message = String(decoding: try Data(contentsOf: stderrURL), as: UTF8.self)
      throw TransportFailure(
        message: message.isEmpty
          ? "MTP helper exited with status \(process.terminationStatus)"
          : String(message.suffix(3000)))
    }
    return try Data(contentsOf: stdoutURL)
  }
}
