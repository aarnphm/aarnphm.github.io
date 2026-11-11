public struct ConnectionSyncQueue {
  private struct Connection: Hashable {
    let deviceID: String
    let registryID: UInt64
    init(_ device: GarminDevice) {
      deviceID = device.id
      registryID = device.registryID
    }
  }

  private var devices: [GarminDevice] = []
  private var attempted: Set<Connection> = []

  public init() {}

  public mutating func observe(_ snapshot: [GarminDevice]) {
    devices = snapshot.filter {
      $0.connected && !$0.serial.isEmpty && $0.productID != 3 && $0.productID != 0
    }
    attempted.formIntersection(Set(devices.map(Connection.init)))
  }

  public mutating func next() -> GarminDevice? {
    guard
      let device = devices.first(where: {
        $0.busyOwner == nil && !attempted.contains(Connection($0))
      })
    else { return nil }
    markStarted(device)
    return device
  }

  public mutating func markStarted(_ device: GarminDevice) {
    attempted.insert(Connection(device))
  }

  public mutating func retryConnected() {
    attempted.removeAll()
  }
}
