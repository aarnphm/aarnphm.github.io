import CryptoKit
import Darwin
import Foundation

enum GarminReconciliation {
  static func findMatch(
    file: URL, filename: String,
    request: @escaping @Sendable (String) async throws -> Data
  ) async throws -> String? {
    guard
      filename.range(
        of: #"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.fit$"#, options: .regularExpression) != nil
    else { return nil }
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.dateFormat = "yyyy-MM-dd"
    formatter.isLenient = false
    guard let date = formatter.date(from: String(filename.prefix(10))) else { return nil }
    let start = formatter.string(from: date.addingTimeInterval(-86400))
    let end = formatter.string(from: date.addingTimeInterval(86400))
    let values = try file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
    guard values.isRegularFile == true, let size = values.fileSize, size > 0, size <= maximumBytes
    else {
      throw GarminConnectError.invalidFile
    }
    let local = try Data(contentsOf: file)
    guard local.count <= maximumBytes else { throw GarminConnectError.invalidFile }
    try FITInspector.validateActivity(local)
    let expected = SHA256.hash(data: local)
    return try await withThrowingTaskGroup(of: String?.self) { group in
      group.addTask {
        struct Activity: Decodable { let activityId: UInt64 }
        let listing = try await request(
          "/activitylist-service/activities/search/activities?start=0&limit=100&startDate=\(start)&endDate=\(end)"
        )
        let activities = try JSONDecoder().decode([Activity].self, from: listing)
        let ids = Array(Set(activities.map(\.activityId))).sorted()
        guard ids.count <= 30 else { return nil }
        var matching: String?
        for id in ids where id > 0 {
          try Task.checkCancellation()
          let archive = try await request("/download-service/files/activity/\(id)")
          guard archive.count <= maximumBytes else { throw GarminConnectError.malformedResponse }
          let original = try await originalFIT(archive)
          if SHA256.hash(data: original) == expected {
            guard matching == nil else { return nil }
            matching = String(id)
          }
        }
        return matching
      }
      group.addTask {
        try await Task.sleep(for: .seconds(210))
        throw GarminConnectError.uncertain
      }
      defer { group.cancelAll() }
      return try await group.next() ?? nil
    }
  }

  private static let maximumBytes = 100_000_000

  private static func originalFIT(_ archive: Data) async throws -> Data {
    let bytes = [UInt8](archive)
    func integer(_ offset: Int, _ length: Int) throws -> Int {
      guard offset >= 0, offset + length <= bytes.count else {
        throw GarminConnectError.malformedResponse
      }
      return (0..<length).reduce(0) { $0 | Int(bytes[offset + $1]) << ($1 * 8) }
    }
    guard bytes.count >= 22 else { throw GarminConnectError.malformedResponse }
    var end: Int?
    for offset in stride(from: bytes.count - 22, through: max(0, bytes.count - 65557), by: -1) {
      if try integer(offset, 4) == 0x0605_4b50,
        try offset + 22 + integer(offset + 20, 2) == bytes.count
      {
        end = offset
        break
      }
    }
    guard let end, try integer(end + 4, 2) == 0, try integer(end + 6, 2) == 0 else {
      throw GarminConnectError.malformedResponse
    }
    let count = try integer(end + 10, 2)
    guard count > 0, count <= 100, try integer(end + 8, 2) == count else {
      throw GarminConnectError.malformedResponse
    }
    var offset = try integer(end + 16, 4)
    var fits: [(name: String, size: Int)] = []
    for _ in 0..<count {
      guard try integer(offset, 4) == 0x0201_4b50 else {
        throw GarminConnectError.malformedResponse
      }
      let flags = try integer(offset + 8, 2)
      let compressed = try integer(offset + 20, 4)
      let size = try integer(offset + 24, 4)
      let nameLength = try integer(offset + 28, 2)
      let extraLength = try integer(offset + 30, 2)
      let commentLength = try integer(offset + 32, 2)
      let attributes = try integer(offset + 38, 4)
      let mode = attributes >> 16
      let next = offset + 46 + nameLength + extraLength + commentLength
      guard next <= end, compressed <= maximumBytes, size <= maximumBytes,
        let name = String(bytes: bytes[(offset + 46)..<(offset + 46 + nameLength)], encoding: .utf8)
      else {
        throw GarminConnectError.malformedResponse
      }
      if name.lowercased().hasSuffix(".fit") {
        guard flags & 1 == 0, attributes & 0x10 == 0,
          mode & 0xf000 == 0 || mode & 0xf000 == 0x8000,
          !name.contains("\0"), !name.contains("*"), !name.contains("?"), !name.contains("["),
          !name.hasPrefix("-"), size > 0
        else { throw GarminConnectError.malformedResponse }
        fits.append((name, size))
      }
      offset = next
    }
    guard fits.count == 1, let fit = fits.first else { throw GarminConnectError.malformedResponse }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
      "tether-reconcile-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: folder, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    defer { try? FileManager.default.removeItem(at: folder) }
    let path = folder.appendingPathComponent("original.zip")
    try archive.write(to: path)
    let extractor = ZIPReadProcess(path: path, member: fit.name, limit: fit.size)
    let result = try await extractor.read()
    guard result.count == fit.size else { throw GarminConnectError.malformedResponse }
    try FITInspector.validateActivity(result)
    return result
  }
}

private final class ZIPReadProcess: @unchecked Sendable {
  private let process = Process()
  private let output = Pipe()
  private let lock = NSLock()
  private var cancelled = false
  private let limit: Int

  init(path: URL, member: String, limit: Int) {
    self.limit = limit
    process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
    process.arguments = ["-p", path.path, member]
    process.standardOutput = output
    process.standardError = FileHandle.nullDevice
    process.standardInput = FileHandle.nullDevice
  }

  func read() async throws -> Data {
    try await withTaskCancellationHandler {
      let timeout = Task {
        try await Task.sleep(for: .seconds(20))
        self.cancel()
      }
      defer { timeout.cancel() }
      return try await Task.detached { try self.readBlocking() }.value
    } onCancel: {
      self.cancel()
    }
  }

  private func cancel() {
    lock.lock()
    defer { lock.unlock() }
    cancelled = true
    if process.isRunning { kill(process.processIdentifier, SIGKILL) }
  }

  private func readBlocking() throws -> Data {
    lock.lock()
    if cancelled {
      lock.unlock()
      throw CancellationError()
    }
    do { try process.run() } catch {
      lock.unlock()
      throw GarminConnectError.malformedResponse
    }
    lock.unlock()
    defer { try? output.fileHandleForReading.close() }
    var result = Data()
    do {
      while let chunk = try output.fileHandleForReading.read(upToCount: 65536), !chunk.isEmpty {
        guard result.count + chunk.count <= limit else {
          throw GarminConnectError.malformedResponse
        }
        result.append(chunk)
      }
    } catch {
      cancel()
      process.waitUntilExit()
      throw error
    }
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { throw GarminConnectError.malformedResponse }
    return result
  }
}
