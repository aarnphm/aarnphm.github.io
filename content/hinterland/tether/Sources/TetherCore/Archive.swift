import CryptoKit
import Foundation

public actor ActivityArchive {
  public nonisolated let root: URL
  private var records: [TransferRecord] = []
  private var loaded = false

  public init(root: URL? = nil) {
    self.root =
      root
      ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("app.aarnphm.tether", isDirectory: true)
  }

  public func allRecords() throws -> [TransferRecord] {
    try load()
    return records.sorted { ($0.sourceModified ?? $0.date) > ($1.sourceModified ?? $1.date) }
  }

  public func ingest(
    file: URL, deviceID: String, filename: String? = nil, sourcePath: String? = nil,
    modified: Date? = nil
  ) throws -> TransferRecord {
    try load()
    let values = try file.resourceValues(forKeys: [
      .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
    ])
    guard values.isRegularFile == true, values.isSymbolicLink != true,
      let size = values.fileSize, size <= 100_000_000
    else { throw ArchiveError.invalidSource }
    let bytes = try Data(contentsOf: file, options: .mappedIfSafe)
    try FITInspector.validateActivity(bytes)
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    if let index = records.firstIndex(where: { $0.id == digest }) {
      let destination = URL(fileURLWithPath: records[index].localPath)
      if (try? Data(contentsOf: destination, options: .mappedIfSafe)) != bytes {
        try makePrivateDirectory(destination.deletingLastPathComponent())
        try bytes.write(to: destination, options: .atomic)
        try FileManager.default.setAttributes(
          [.posixPermissions: 0o600], ofItemAtPath: destination.path)
      }
      if let sourcePath {
        records[index].deviceID = deviceID
        records[index].sourcePath = sourcePath
        records[index].sourceModified = modified
        if let filename { records[index].filename = filename }
        try save()
      }
      return records[index]
    }
    let folder = root.appendingPathComponent("activities", isDirectory: true)
    try makePrivateDirectory(folder)
    let destination = folder.appendingPathComponent(digest).appendingPathExtension("fit")
    try bytes.write(to: destination, options: .atomic)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o600], ofItemAtPath: destination.path)
    var record = TransferRecord(
      id: digest, deviceID: deviceID, filename: filename ?? file.lastPathComponent,
      bytes: Int64(bytes.count), state: "saved", localPath: destination.path)
    record.sourcePath = sourcePath
    record.sourceModified = modified
    records.append(record)
    do { try save() } catch {
      records.removeLast()
      throw error
    }
    return record
  }

  public func cachedActivity(deviceID: String, path: String, size: Int64, modified: Date?) throws
    -> TransferRecord?
  {
    try load()
    guard let modified else { return nil }
    return records.first { record in
      record.deviceID == deviceID && record.sourcePath == path && record.bytes == size
        && record.sourceModified == modified && archiveMatches(record)
    }
  }

  private func archiveMatches(_ record: TransferRecord) -> Bool {
    guard
      let bytes = try? Data(
        contentsOf: URL(fileURLWithPath: record.localPath), options: .mappedIfSafe),
      bytes.count == record.bytes
    else { return false }
    return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() == record.id
  }

  public func update(id: String, state: String, activityID: String? = nil) throws {
    try load()
    guard let index = records.firstIndex(where: { $0.id == id }) else {
      throw ArchiveError.missingRecord
    }
    let before = records[index]
    records[index].state = state
    if let activityID { records[index].activityID = activityID }
    do { try save() } catch {
      records[index] = before
      throw error
    }
  }

  private func load() throws {
    guard !loaded else { return }
    try makePrivateDirectory(root)
    let file = root.appendingPathComponent("receipts.json")
    if FileManager.default.fileExists(atPath: file.path) {
      records = try JSONDecoder().decode([TransferRecord].self, from: Data(contentsOf: file))
      // A process can stop after the server accepts a POST but before its receipt is saved.
      for index in records.indices where records[index].state == "uploading" {
        records[index].state = "uncertain"
      }
    }
    loaded = true
  }

  private func save() throws {
    let file = root.appendingPathComponent("receipts.json")
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(records).write(to: file, options: .atomic)
    try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
  }

  private func makePrivateDirectory(_ url: URL) throws {
    try FileManager.default.createDirectory(
      at: url, withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700])
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
  }

  public enum ArchiveError: LocalizedError {
    case invalidSource, missingRecord
    public var errorDescription: String? {
      switch self {
      case .invalidSource: return "Choose a regular FIT file smaller than 100 MB."
      case .missingRecord: return "The activity is absent from the local archive."
      }
    }
  }
}
