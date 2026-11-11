import Foundation

public struct GarminDevice: Identifiable, Codable, Equatable, Sendable {
  public var id: String
  public var name: String
  public var serial: String
  public var productID: Int
  public var registryID: UInt64
  public var connected: Bool
  public var lastSeen: Date
  public var busyOwner: String?
  public var detectionMilliseconds: Double?

  public init(
    id: String, name: String, serial: String, productID: Int, registryID: UInt64,
    connected: Bool, lastSeen: Date = Date(), detectionMilliseconds: Double? = nil
  ) {
    self.id = id
    self.name = name
    self.serial = serial
    self.productID = productID
    self.registryID = registryID
    self.connected = connected
    self.lastSeen = lastSeen
    self.detectionMilliseconds = detectionMilliseconds
  }
}

public struct TransferRecord: Identifiable, Codable, Sendable {
  public var id: String
  public var deviceID: String
  public var filename: String
  public var bytes: Int64
  public var date: Date
  public var state: String
  public var activityID: String?
  public var localPath: String
  public var sourcePath: String?
  public var sourceModified: Date?

  public init(
    id: String, deviceID: String, filename: String, bytes: Int64, date: Date = Date(),
    state: String, activityID: String? = nil, localPath: String
  ) {
    self.id = id
    self.deviceID = deviceID
    self.filename = filename
    self.bytes = bytes
    self.date = date
    self.state = state
    self.activityID = activityID
    self.localPath = localPath
  }
}
