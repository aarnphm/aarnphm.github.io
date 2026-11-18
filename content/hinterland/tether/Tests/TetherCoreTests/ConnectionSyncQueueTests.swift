import XCTest

@testable import TetherCore

final class ConnectionSyncQueueTests: XCTestCase {
  private func watch(_ id: String = "watch", registry: UInt64 = 1) -> GarminDevice {
    GarminDevice(
      id: id, name: "Forerunner 970", serial: id, productID: 0x51d5,
      registryID: registry, connected: true)
  }

  func testRepeatedNotificationsStartOneSync() {
    var queue = ConnectionSyncQueue()
    let device = watch()
    queue.observe([device])
    XCTAssertEqual(queue.next()?.id, device.id)
    for _ in 0..<10 {
      queue.observe([device])
      XCTAssertNil(queue.next())
    }
  }

  func testDetachThenReconnectStartsAgain() {
    var queue = ConnectionSyncQueue()
    queue.observe([watch()])
    XCTAssertNotNil(queue.next())
    queue.observe([])
    XCTAssertNil(queue.next())
    queue.observe([watch(registry: 2)])
    XCTAssertEqual(queue.next()?.registryID, 2)
  }

  func testChangedRegistryStartsAgainWithoutEmptySnapshot() {
    var queue = ConnectionSyncQueue()
    queue.observe([watch()])
    XCTAssertNotNil(queue.next())
    queue.observe([watch(registry: 2)])
    XCTAssertEqual(queue.next()?.registryID, 2)
  }

  func testBusyDeviceWaitsUntilReleased() {
    var queue = ConnectionSyncQueue()
    var device = watch()
    device.busyOwner = "pid 123, Garmin Express"
    queue.observe([device])
    XCTAssertNil(queue.next())
    device.busyOwner = nil
    queue.observe([device])
    XCTAssertNotNil(queue.next())
  }

  func testBootstrapAndOfflineDevicesNeverSync() {
    var queue = ConnectionSyncQueue()
    var temporary = watch("temporary")
    temporary.serial = ""
    temporary.productID = 3
    var offline = watch("offline")
    offline.connected = false
    queue.observe([temporary, offline])
    XCTAssertNil(queue.next())
  }

  func testSecondWatchWaitsAndRemovedPendingWatchIsDiscarded() {
    var queue = ConnectionSyncQueue()
    let first = watch("first")
    let second = watch("second", registry: 2)
    queue.observe([first, second])
    XCTAssertEqual(queue.next()?.id, "first")
    queue.observe([first])
    XCTAssertNil(queue.next())
    queue.observe([first, second])
    XCTAssertEqual(queue.next()?.id, "second")
  }

  func testManualSyncAndCancellationDoNotRestartAutomatically() {
    var queue = ConnectionSyncQueue()
    let device = watch()
    queue.observe([device])
    queue.markStarted(device)
    queue.observe([device])
    XCTAssertNil(queue.next())
    queue.retryConnected()
    XCTAssertEqual(queue.next()?.id, device.id)
  }
}
