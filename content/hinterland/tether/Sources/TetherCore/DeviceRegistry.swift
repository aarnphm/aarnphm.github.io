import Foundation

public actor DeviceRegistry {
  private let file: URL
  public init(root: URL) { file = root.appendingPathComponent("devices.json") }
  public func load() throws -> [GarminDevice] {
    guard FileManager.default.fileExists(atPath: file.path) else { return [] }
    return try JSONDecoder().decode([GarminDevice].self, from: Data(contentsOf: file)).filter {
      !$0.serial.isEmpty && $0.productID != 3
    }.map {
      var device = $0
      device.connected = false
      device.busyOwner = nil
      return device
    }
  }
  public func save(_ devices: [GarminDevice]) throws {
    try FileManager.default.createDirectory(
      at: file.deletingLastPathComponent(), withIntermediateDirectories: true,
      attributes: [.posixPermissions: 0o700])
    try JSONEncoder().encode(devices).write(to: file, options: .atomic)
    try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
  }
}
