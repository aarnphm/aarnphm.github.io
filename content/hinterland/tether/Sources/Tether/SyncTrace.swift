import Foundation

@MainActor
final class SyncTrace {
  private struct Entry: Encodable {
    let event: String
    let elapsedMilliseconds: Double
    let count: Int?
    let registryID: UInt64?

    enum CodingKeys: String, CodingKey {
      case event, count
      case elapsedMilliseconds = "elapsed_ms"
      case registryID = "registry_id"
    }
  }

  private let began = ContinuousClock.now
  private let output: FileHandle?

  init() {
    if let path = ProcessInfo.processInfo.environment["TETHER_TRACE_PATH"] {
      FileManager.default.createFile(
        atPath: path, contents: nil, attributes: [.posixPermissions: 0o600])
      output = try? FileHandle(forWritingTo: URL(fileURLWithPath: path))
    } else {
      output = nil
    }
  }

  func record(_ event: String, count: Int? = nil, registryID: UInt64? = nil) {
    guard let output else { return }
    let duration = began.duration(to: .now)
    let entry = Entry(
      event: event,
      elapsedMilliseconds: Double(duration.components.seconds) * 1000
        + Double(duration.components.attoseconds) / 1e15,
      count: count, registryID: registryID)
    guard var data = try? JSONEncoder().encode(entry) else { return }
    data.append(10)
    try? output.write(contentsOf: data)
  }

  deinit { try? output?.close() }
}
