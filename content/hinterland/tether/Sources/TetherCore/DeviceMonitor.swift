import Foundation
import IOKit

public struct DeviceObservation: Codable, Sendable {
  public let event: String
  public let source: String
  public let observedAt: Date
  public let elapsedMilliseconds: Double
  public let registryID: UInt64
  public let attachmentElapsedMilliseconds: Double?
  public let productID: Int?
  public let serialPresent: Bool?
  public let messageType: UInt32?
  public let propertyReadMilliseconds: Double?
}

public final class DeviceMonitor {
  public var onChange: (([GarminDevice]) -> Void)?
  public var onPreparing: (() -> Void)?
  public var onObservation: ((DeviceObservation) -> Void)?
  private let began = ContinuousClock.now
  private var attachedAt: [UInt64: ContinuousClock.Instant] = [:]
  private var interests: [UInt64: io_object_t] = [:]
  // IOMessage.h composes these macros with iokit_common_msg, which Swift cannot import.
  private enum ServiceMessage: UInt32 {
    case terminated = 0xe000_0010
    case resumed = 0xe000_0030
    case closed = 0xe000_0110
    case propertiesChanged = 0xe000_0130
    case poweredOn = 0xe000_0230
  }
  private struct PublishedDevice: Equatable {
    let id: String
    let registryID: UInt64
    let busyOwner: String?
  }
  private var publishedDevices: [PublishedDevice]?
  private var publishedPreparing: Set<UInt64>?
  private var port: IONotificationPortRef?
  private var attachedIterator: io_iterator_t = 0
  private var removedIterator: io_iterator_t = 0
  private var devices: [UInt64: GarminDevice] = [:]
  private var preparing: Set<UInt64> = []
  private var settling: [UInt64: DispatchWorkItem] = [:]
  private var generations: [UInt64: UUID] = [:]

  public init() {}

  public func start() throws {
    guard port == nil else { return }
    guard let created = IONotificationPortCreate(kIOMainPortDefault) else {
      throw MonitorError.unavailable
    }
    port = created
    IONotificationPortSetDispatchQueue(created, DispatchQueue.main)
    let context = Unmanaged.passUnretained(self).toOpaque()
    let added = IOServiceAddMatchingNotification(
      created, kIOFirstMatchNotification,
      matching(),
      { context, iterator in
        guard let context else { return }
        Unmanaged<DeviceMonitor>.fromOpaque(context).takeUnretainedValue().drain(
          iterator, removed: false)
      }, context, &attachedIterator)
    guard added == KERN_SUCCESS else {
      stop()
      throw MonitorError.registration(added)
    }
    drain(attachedIterator, removed: false, source: "initial_snapshot")
    let removed = IOServiceAddMatchingNotification(
      created, kIOTerminatedNotification,
      matching(),
      { context, iterator in
        guard let context else { return }
        Unmanaged<DeviceMonitor>.fromOpaque(context).takeUnretainedValue().drain(
          iterator, removed: true)
      }, context, &removedIterator)
    guard removed == KERN_SUCCESS else {
      stop()
      throw MonitorError.registration(removed)
    }
    drain(removedIterator, removed: true, source: "initial_snapshot")
  }

  public func refresh() {
    var iterator: io_iterator_t = 0
    guard IOServiceGetMatchingServices(kIOMainPortDefault, matching(), &iterator) == KERN_SUCCESS
    else { return }
    defer { IOObjectRelease(iterator) }
    cancelSettling()
    devices.removeAll()
    preparing.removeAll()
    drain(iterator, removed: false, source: "refresh")
  }

  public func stop() {
    cancelSettling()
    if attachedIterator != 0 {
      IOObjectRelease(attachedIterator)
      attachedIterator = 0
    }
    if removedIterator != 0 {
      IOObjectRelease(removedIterator)
      removedIterator = 0
    }
    if let port { IONotificationPortDestroy(port) }
    port = nil
    devices.removeAll()
    preparing.removeAll()
    publishedDevices = nil
    publishedPreparing = nil
  }

  deinit { stop() }

  private func matching() -> CFDictionary {
    let dictionary = NSMutableDictionary(dictionary: [
      "IOProviderClass": "IOUSBHostDevice", "IOPropertyMatch": ["idVendor": 0x091e],
    ])
    return dictionary
  }

  private func property(_ service: io_service_t, _ name: String) -> Any? {
    IORegistryEntryCreateCFProperty(service, name as CFString, kCFAllocatorDefault, 0)?
      .takeRetainedValue()
  }

  private func cancelSettling() {
    for work in settling.values { work.cancel() }
    settling.removeAll()
    generations.removeAll()
    for interest in interests.values { IOObjectRelease(interest) }
    interests.removeAll()
    attachedAt.removeAll()
  }

  private func read(
    _ service: io_service_t, registryID: UInt64, source: String, messageType: UInt32? = nil
  ) {
    let start = ContinuousClock.now
    let serial =
      (property(service, "USB Serial Number") as? String
      ?? property(service, "kUSBSerialNumberString") as? String ?? "")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    let product = (property(service, "idProduct") as? NSNumber)?.intValue ?? 0
    // Garmin's vendor-specific USB interface (product 3) cannot expose MTP activities.
    guard !serial.isEmpty, product != 3, product != 0 else {
      devices.removeValue(forKey: registryID)
      preparing.insert(registryID)
      observe(
        "preparing", source: source, registryID: registryID, productID: product,
        serialPresent: !serial.isEmpty, messageType: messageType,
        propertyReadMilliseconds: milliseconds(start.duration(to: .now)))
      return
    }
    preparing.remove(registryID)
    let name =
      property(service, "USB Product Name") as? String
      ?? (product == 0x51d5 ? "Forerunner 970" : "Garmin device")
    let rawOwner = property(service, "UsbExclusiveOwner") as? String
    let owner = rawOwner.flatMap { value -> String? in
      guard value.hasPrefix("pid "),
        let pidText = value.dropFirst(4).split(separator: ",", maxSplits: 1).first,
        let pid = Int32(pidText.trimmingCharacters(in: .whitespaces)), pid > 0
      else { return nil }
      return value
    }
    let propertyReadMilliseconds = milliseconds(start.duration(to: .now))
    let firstReady = devices[registryID] == nil
    var device = GarminDevice(
      id: "garmin-\(product)-\(serial)", name: name,
      serial: serial, productID: product, registryID: registryID, connected: true,
      detectionMilliseconds: propertyReadMilliseconds)
    device.busyOwner = owner
    devices[registryID] = device
    observe(
      firstReady ? "ready" : "updated", source: source, registryID: registryID,
      productID: product, serialPresent: true, messageType: messageType,
      propertyReadMilliseconds: propertyReadMilliseconds)
  }

  private func milliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15
  }

  private func observe(
    _ event: String, source: String, registryID: UInt64,
    productID: Int? = nil, serialPresent: Bool? = nil, messageType: UInt32? = nil,
    propertyReadMilliseconds: Double? = nil
  ) {
    let now = ContinuousClock.now
    onObservation?(
      DeviceObservation(
        event: event, source: source, observedAt: Date(),
        elapsedMilliseconds: milliseconds(began.duration(to: now)), registryID: registryID,
        attachmentElapsedMilliseconds: attachedAt[registryID].map {
          milliseconds($0.duration(to: now))
        },
        productID: productID, serialPresent: serialPresent, messageType: messageType,
        propertyReadMilliseconds: propertyReadMilliseconds))
  }

  private func publish() {
    let snapshot = devices.values.sorted { $0.id < $1.id }
    let keys = snapshot.map {
      PublishedDevice(id: $0.id, registryID: $0.registryID, busyOwner: $0.busyOwner)
    }
    let changed = publishedDevices != keys
    let preparationChanged = publishedPreparing != preparing
    publishedDevices = keys
    publishedPreparing = preparing
    if changed || preparationChanged { onChange?(snapshot) }
    if preparationChanged && !preparing.isEmpty { onPreparing?() }
  }

  private func remove(registryID: UInt64, source: String) {
    observe("removed", source: source, registryID: registryID)
    devices.removeValue(forKey: registryID)
    preparing.remove(registryID)
    generations.removeValue(forKey: registryID)
    settling.removeValue(forKey: registryID)?.cancel()
    if let interest = interests.removeValue(forKey: registryID) { IOObjectRelease(interest) }
    attachedAt.removeValue(forKey: registryID)
  }

  private func registerInterest(_ service: io_service_t, registryID: UInt64) {
    guard interests[registryID] == nil, let port else { return }
    var notification: io_object_t = 0
    let result = IOServiceAddInterestNotification(
      port, service, kIOGeneralInterest,
      { context, service, messageType, _ in
        guard let context else { return }
        Unmanaged<DeviceMonitor>.fromOpaque(context).takeUnretainedValue()
          .interestChanged(service, messageType: messageType)
      }, Unmanaged.passUnretained(self).toOpaque(), &notification)
    if result == KERN_SUCCESS {
      interests[registryID] = notification
    } else {
      observe("interest_unavailable", source: "registration", registryID: registryID)
    }
  }

  private func interestChanged(_ service: io_service_t, messageType: UInt32) {
    var registryID: UInt64 = 0
    guard port != nil, IORegistryEntryGetRegistryEntryID(service, &registryID) == KERN_SUCCESS,
      let generation = generations[registryID]
    else { return }
    if messageType == ServiceMessage.terminated.rawValue {
      remove(registryID: registryID, source: "interest")
      publish()
      return
    }
    guard
      [ServiceMessage.propertiesChanged, .resumed, .poweredOn, .closed]
        .contains(where: { $0.rawValue == messageType })
    else { return }
    read(service, registryID: registryID, source: "interest", messageType: messageType)
    publish()
    guard port != nil, generations[registryID] == generation else { return }
    if preparing.contains(registryID) {
      if settling[registryID] == nil {
        scheduleRead(registryID: registryID, generation: generation)
      }
    } else {
      settling.removeValue(forKey: registryID)?.cancel()
    }
  }

  private func scheduleRead(registryID: UInt64, generation: UUID) {
    let elapsed = attachedAt[registryID].map { milliseconds($0.duration(to: .now)) } ?? 0
    // Poll only incomplete USB identities; driver messages can make them ready sooner.
    let delay = elapsed < 5000 ? 50 : 250
    let work = DispatchWorkItem { [weak self] in
      guard let self, self.port != nil, self.generations[registryID] == generation else { return }
      self.settling.removeValue(forKey: registryID)
      let service = IOServiceGetMatchingService(
        kIOMainPortDefault, IORegistryEntryIDMatching(registryID))
      guard service != 0 else {
        self.remove(registryID: registryID, source: "fallback")
        self.publish()
        return
      }
      defer { IOObjectRelease(service) }
      self.read(service, registryID: registryID, source: "fallback")
      self.publish()
      guard self.port != nil, self.generations[registryID] == generation else { return }
      if self.preparing.contains(registryID) {
        self.scheduleRead(registryID: registryID, generation: generation)
      }
    }
    settling[registryID] = work
    DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(delay), execute: work)
  }

  private func drain(_ iterator: io_iterator_t, removed: Bool, source: String = "matching") {
    while true {
      let service = IOIteratorNext(iterator)
      guard service != 0 else { break }
      defer { IOObjectRelease(service) }
      var registryID: UInt64 = 0
      guard IORegistryEntryGetRegistryEntryID(service, &registryID) == KERN_SUCCESS else {
        continue
      }
      if removed {
        if generations[registryID] != nil { remove(registryID: registryID, source: source) }
      } else {
        if generations[registryID] == nil {
          generations[registryID] = UUID()
          attachedAt[registryID] = .now
          observe(
            source == "matching" ? "attached" : "discovered", source: source, registryID: registryID
          )
        }
        registerInterest(service, registryID: registryID)
        read(service, registryID: registryID, source: source)
        if port != nil, preparing.contains(registryID), settling[registryID] == nil,
          let generation = generations[registryID]
        {
          scheduleRead(registryID: registryID, generation: generation)
        }
      }
    }
    publish()
  }

  public enum MonitorError: LocalizedError {
    case unavailable
    case registration(kern_return_t)
    public var errorDescription: String? {
      switch self {
      case .unavailable: return "macOS USB notifications are unavailable."
      case .registration(let code): return "Could not register USB notifications (\(code))."
      }
    }
  }
}
