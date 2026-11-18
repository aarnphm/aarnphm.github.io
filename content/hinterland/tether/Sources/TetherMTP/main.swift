import Darwin
import Foundation

struct Failure: Error, CustomStringConvertible { let description: String }
struct Envelope<T: Decodable>: Decodable {
  let error: String
  let data: T?
}
struct Device: Decodable {
  let model: String
  let serialNumber: String
  enum CodingKeys: String, CodingKey {
    case model = "Model"
    case serialNumber = "SerialNumber"
  }
}
struct DeviceData: Decodable { let mtpDeviceInfo: Device }
struct Storage: Decodable {
  let sid: UInt32
  enum CodingKeys: String, CodingKey { case sid = "Sid" }
}
struct Entry: Decodable {
  let size: Int64
  let isFolder: Bool
  let dateAdded: String
  let name: String
  let path: String
}
struct Activity: Encodable {
  let id: String
  let name: String
  let size: Int64
  let modified: String?
  let storageID: UInt32
  let sourcePath: String
  let serial: String
}
struct Listing: Encodable {
  let model: String
  let serial: String
  let activities: [Activity]
}
struct WalkInput: Encodable {
  let storageId: UInt32
  let fullPath: String
  let recursive = false
  let skipDisallowedFiles = false
  let skipHiddenFiles = false
}
struct DownloadInput: Encodable {
  let storageId: UInt32
  let sources: [String]
  let destination: String
  let preprocessFiles = true
}

// Kalam invokes callbacks synchronously before returning; its progress callbacks may use Go threads.
private var callbackData: Data?
private let resultCallback: @convention(c) (UnsafeMutablePointer<CChar>?) -> Void = { text in
  guard let text else { return }
  callbackData = Data(String(cString: text).utf8)
  free(text)
}
private let discardCallback: @convention(c) (UnsafeMutablePointer<CChar>?) -> Void = { text in
  if let text { free(text) }
}
typealias Unary = @convention(c) (UnsafeMutableRawPointer?) -> Void
typealias JSONCall = @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void
typealias DownloadCall =
  @convention(c) (
    UnsafePointer<CChar>?, UnsafeMutableRawPointer?, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?
  ) -> Void

final class Kalam {
  private let handle: UnsafeMutableRawPointer
  private let initialize: Unary
  private let storages: Unary
  private let dispose: Unary
  private let walk: JSONCall
  private let download: DownloadCall
  private var opened = false
  private let callback = unsafeBitCast(resultCallback, to: UnsafeMutableRawPointer.self)
  private let discard = unsafeBitCast(discardCallback, to: UnsafeMutableRawPointer.self)

  init() throws {
    let base = "/Applications/OpenMTP.app/Contents/Resources/bin/arm64/"
    guard let library = dlopen(base + "kalam.dylib", RTLD_NOW | RTLD_LOCAL) else {
      throw Failure(description: "OpenMTP transport unavailable: \(String(cString: dlerror()))")
    }
    handle = library
    func symbol(_ name: String) throws -> UnsafeMutableRawPointer {
      guard let value = dlsym(library, name) else {
        throw Failure(description: "Missing Kalam symbol: \(name)")
      }
      return value
    }
    initialize = unsafeBitCast(try symbol("Initialize"), to: Unary.self)
    storages = unsafeBitCast(try symbol("FetchStorages"), to: Unary.self)
    dispose = unsafeBitCast(try symbol("Dispose"), to: Unary.self)
    walk = unsafeBitCast(try symbol("Walk"), to: JSONCall.self)
    download = unsafeBitCast(try symbol("DownloadFiles"), to: DownloadCall.self)
  }
  // Go's runtime owns background threads; do not dlclose its library.
  func close() {
    if opened {
      callbackData = nil
      dispose(callback)
      opened = false
    }
  }
  func decode<T: Decodable>(_ type: T.Type) throws -> T {
    guard let data = callbackData else {
      throw Failure(description: "MTP operation returned no response")
    }
    let response = try JSONDecoder().decode(Envelope<T>.self, from: data)
    guard response.error.isEmpty else {
      throw Failure(
        description:
          "MTP: \(response.error). If another application owns the watch, release it and retry.")
    }
    guard let result = response.data else {
      throw Failure(description: "MTP response omitted data")
    }
    return result
  }
  func open(serial: String) throws -> Device {
    callbackData = nil
    initialize(callback)
    opened = true
    let device = try decode(DeviceData.self).mtpDeviceInfo
    guard device.serialNumber.caseInsensitiveCompare(serial) == .orderedSame else {
      throw Failure(description: "Connected watch identity changed; refusing access")
    }
    return device
  }
  func storageList() throws -> [Storage] {
    callbackData = nil
    storages(callback)
    return try decode([Storage].self)
  }
  func entries(storage: UInt32, path: String) throws -> [Entry] {
    let json = String(
      decoding: try JSONEncoder().encode(WalkInput(storageId: storage, fullPath: path)),
      as: UTF8.self)
    callbackData = nil
    json.withCString { walk($0, callback) }
    // Kalam serializes an empty listing as null.
    guard let data = callbackData else {
      throw Failure(description: "MTP listing returned no response")
    }
    let result = try JSONDecoder().decode(Envelope<[Entry]>.self, from: data)
    guard result.error.isEmpty else { throw Failure(description: result.error) }
    return result.data ?? []
  }
  func copy(storage: UInt32, path: String, destination: URL) throws {
    let json = String(
      decoding: try JSONEncoder().encode(
        DownloadInput(storageId: storage, sources: [path], destination: destination.path)),
      as: UTF8.self)
    callbackData = nil
    json.withCString { download($0, discard, discard, callback) }
    guard try decode(Bool.self) else { throw Failure(description: "MTP transfer did not complete") }
  }
}

func checkedPath(_ path: String) throws {
  let parts = path.split(separator: "/")
  guard path.hasPrefix("/"), !path.contains("\0"), !parts.contains(".."), !parts.contains("."),
    parts.count == 3 || (parts.count == 4 && parts[2].lowercased() == "pendinghd"),
    parts[0].lowercased() == "garmin", parts[1].lowercased() == "activity",
    parts.last?.lowercased().hasSuffix(".fit") == true
  else {
    throw Failure(
      description: "Only FIT files in GARMIN/Activity or its PendingHD folder may be read")
  }
}
func run() throws {
  let args = Array(CommandLine.arguments.dropFirst())
  guard let command = args.first, ["list", "download"].contains(command), args.count % 2 == 1 else {
    throw Failure(
      description:
        "Usage: tether-mtp list --serial SERIAL | download --serial SERIAL --storage ID --path /GARMIN/Activity/FILE.fit --out /absolute/new/file.fit"
    )
  }
  var options: [String: String] = [:]
  for i in stride(from: 1, to: args.count, by: 2) {
    guard options[args[i]] == nil else { throw Failure(description: "Duplicate argument") }
    options[args[i]] = args[i + 1]
  }
  let allowed: Set<String> =
    command == "list" ? ["--serial"] : ["--serial", "--storage", "--path", "--out"]
  guard Set(options.keys) == allowed, let serial = options["--serial"], !serial.isEmpty else {
    throw Failure(description: "Invalid arguments or missing watch identity")
  }
  var downloadRequest: (UInt32, String, URL)?
  if command == "download" {
    guard let storageText = options["--storage"], let storage = UInt32(storageText),
      let path = options["--path"], let out = options["--out"], out.hasPrefix("/"),
      !out.contains("\0")
    else { throw Failure(description: "Invalid download arguments") }
    try checkedPath(path)
    let destination = URL(fileURLWithPath: out)
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      throw Failure(description: "Local destination already exists")
    }
    downloadRequest = (storage, path, destination)
  }
  let transport = try Kalam()
  defer { transport.close() }
  let device = try transport.open(serial: serial)
  if let (storage, path, destination) = downloadRequest {
    let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(
      ".tether-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: scratch) }
    try transport.copy(storage: storage, path: path, destination: scratch)
    let copied = scratch.appendingPathComponent(URL(fileURLWithPath: path).lastPathComponent)
    guard FileManager.default.fileExists(atPath: copied.path) else {
      throw Failure(description: "MTP reported success without creating the requested file")
    }
    try FileManager.default.moveItem(at: copied, to: destination)
    emit("{\"ok\":true}")
  } else {
    var activities: [Activity] = []
    var foundActivityFolder = false
    for storage in try transport.storageList() {
      let root = try transport.entries(storage: storage.sid, path: "/")
      guard let garmin = root.first(where: { $0.isFolder && $0.name.lowercased() == "garmin" })
      else { continue }
      let folders = try transport.entries(storage: storage.sid, path: garmin.path)
      guard
        let activity = folders.first(where: { $0.isFolder && $0.name.lowercased() == "activity" })
      else { continue }
      foundActivityFolder = true
      var entries = try transport.entries(storage: storage.sid, path: activity.path)
      for folder in entries where folder.isFolder && folder.name.lowercased() == "pendinghd" {
        entries.append(contentsOf: try transport.entries(storage: storage.sid, path: folder.path))
      }
      for entry in entries where !entry.isFolder && entry.name.lowercased().hasSuffix(".fit") {
        try checkedPath(entry.path)
        guard entry.size >= 0 else { throw Failure(description: "Invalid activity size") }
        activities.append(
          Activity(
            id: "\(serial):\(storage.sid):\(entry.path)", name: entry.name, size: entry.size,
            modified: entry.dateAdded, storageID: storage.sid, sourcePath: entry.path,
            serial: serial))
      }
    }
    guard foundActivityFolder else {
      throw Failure(description: "Watch did not expose GARMIN/Activity")
    }
    emit(
      String(
        decoding: try JSONEncoder().encode(
          Listing(model: device.model, serial: serial, activities: activities)), as: UTF8.self))
  }
}
// Preserve a separate response descriptor before Go can print diagnostics.
let outputFD = dup(STDOUT_FILENO)
dup2(STDERR_FILENO, STDOUT_FILENO)
func emit(_ text: String) {
  FileHandle(fileDescriptor: outputFD, closeOnDealloc: false).write(Data((text + "\n").utf8))
}
do { try run() } catch {
  fputs("\(error)\n", stderr)
  exit(1)
}
