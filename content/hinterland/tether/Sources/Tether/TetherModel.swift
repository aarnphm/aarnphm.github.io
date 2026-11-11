import AppKit
import Combine
import Foundation
import TetherCore
import UniformTypeIdentifiers

@MainActor
final class TetherModel: ObservableObject {
  @Published var devices: [GarminDevice] = []
  @Published var selectedDeviceID: String?
  @Published var records: [TransferRecord] = []
  @Published var status = "looking for a watch"
  @Published var isSyncing = false
  @Published var syncProgress: SyncProgress?
  @Published var accountStatus = "not connected"
  @Published var error: String?
  @Published var showAccount = false
  @Published var showAddDevice = false

  var selectedDevice: GarminDevice? { devices.first { $0.id == selectedDeviceID } }
  var selectedRecords: [TransferRecord] {
    guard let selectedDeviceID else { return records }
    return records.filter { $0.deviceID == selectedDeviceID || $0.deviceID == "imported" }
  }

  private let monitor = DeviceMonitor()
  private let archive = ActivityArchive()
  private let client = GarminConnectClient()
  private let registry: DeviceRegistry
  private var activeTransport: KalamTransport?
  private var syncTask: Task<Void, Never>?
  private var started = false
  private var registryLoaded = false
  private var syncingDevice: GarminDevice?
  private var automaticSync = ConnectionSyncQueue()
  private var transferProgress = SyncProgress(phase: .connecting)
  private var progressPublishedAt = ContinuousClock.now
  private var sessionRefresh: Task<String, Never>?
  private var progressDismissal: Task<Void, Never>?
  private let trace = SyncTrace()

  init() { registry = DeviceRegistry(root: archive.root) }

  func start() {
    guard !started else { return }
    monitor.onChange = { [weak self] connected in self?.receive(connected) }
    monitor.onObservation = { [weak self] observation in
      self?.trace.record("usb.\(observation.event)", registryID: observation.registryID)
    }
    monitor.onPreparing = { [weak self] in
      guard let self, !self.isSyncing else { return }
      self.status = "connecting"
      self.progressDismissal?.cancel()
      self.setProgress(.connecting, completed: 0, total: nil)
    }
    do {
      try monitor.start()
      started = true
    } catch {
      self.error = error.localizedDescription
      return
    }
    Task {
      do {
        let saved = try await registry.load()
        let known = Set(devices.map(\.id))
        devices.append(contentsOf: saved.filter { !known.contains($0.id) })
        if selectedDeviceID == nil { selectedDeviceID = devices.first?.id }
        registryLoaded = true
        try await registry.save(devices)
        records = try await archive.allRecords()
        startNextAutomaticSync()
      } catch { self.error = error.localizedDescription }
    }
    Task { accountStatus = await client.connectionStatus() }
  }

  private func receive(_ connected: [GarminDevice]) {
    let connected = connected.filter { !$0.serial.isEmpty && $0.productID != 3 }
    let present = Set(connected.map(\.id))
    let previous = Set(devices.filter(\.connected).map(\.id))
    var merged = devices.filter { !$0.serial.isEmpty && $0.productID != 3 }.map {
      item -> GarminDevice in
      var device = item
      device.connected = false
      device.busyOwner = nil
      return device
    }
    for device in connected {
      if let index = merged.firstIndex(where: { $0.id == device.id }) {
        merged[index] = device
      } else {
        merged.append(device)
      }
    }
    devices = merged
    if connected.isEmpty && !isSyncing { syncProgress = nil }
    automaticSync.observe(connected)
    if selectedDeviceID == nil { selectedDeviceID = connected.first?.id ?? merged.first?.id }
    if let syncingDevice,
      !connected.contains(where: {
        $0.id == syncingDevice.id && $0.registryID == syncingDevice.registryID
      })
    {
      cancelSync()
      status = "watch disconnected; saved files are safe"
    } else if !isSyncing && (present != previous || status == "looking for a watch") {
      status = connected.isEmpty ? "connect a watch with its USB cable" : "watch detected"
    }
    if registryLoaded {
      Task {
        do { try await registry.save(merged) } catch { self.error = error.localizedDescription }
      }
    }
    startNextAutomaticSync()
  }

  private func startNextAutomaticSync() {
    guard registryLoaded, !isSyncing, let device = automaticSync.next() else { return }
    selectedDeviceID = device.id
    sync()
  }

  func refreshConnection() {
    guard started else { return }
    monitor.refresh()
  }

  func reconnect() {
    guard !isSyncing else { return }
    error = nil
    status = "checking the USB connection"
    sessionRefresh = Task { await client.refreshSession() }
    automaticSync.retryConnected()
    monitor.refresh()
    if !isSyncing {
      status =
        devices.contains(where: \.connected)
        ? "watch detected" : "connect a watch with its USB cable"
    }
    startNextAutomaticSync()
  }

  private func setProgress(_ phase: SyncProgress.Phase, completed: Int? = nil, total: Int? = nil) {
    transferProgress.phase = phase
    if let completed { transferProgress.completed = completed }
    if let total { transferProgress.total = total }
    if phase == .connecting || phase == .reading { transferProgress.total = nil }
    let now = ContinuousClock.now
    let changedPhase = syncProgress?.phase != phase
    let terminal = phase == .complete || phase == .cancelled || phase == .failed
    guard changedPhase || terminal || progressPublishedAt.duration(to: now) >= .milliseconds(150)
    else { return }
    progressPublishedAt = now
    syncProgress = transferProgress
    if changedPhase {
      trace.record("progress.\(phase.rawValue)", count: transferProgress.completed)
    }
  }

  func sync() {
    guard !isSyncing, let device = selectedDevice, device.connected else { return }
    guard device.busyOwner == nil else {
      error = "The watch is in use by another app. Quit that app, then reconnect."
      return
    }
    guard !device.serial.isEmpty else {
      error = "This watch did not provide a USB serial number. Tether cannot safely select it."
      return
    }
    isSyncing = true
    progressDismissal?.cancel()
    setProgress(.reading, completed: 0)
    error = nil
    syncingDevice = device
    automaticSync.markStarted(device)
    let startedAt = ContinuousClock.now
    trace.record("sync.started", registryID: device.registryID)
    let transport = KalamTransport(expectedSerial: device.serial)
    activeTransport = transport
    syncTask = Task {
      defer { finishSync() }
      do {
        status = "reading the activity index"
        let activities = try await transport.enumerateActivities().sorted { $0.name > $1.name }
        trace.record("sync.listed", count: activities.count)
        setProgress(.checking, completed: 0, total: activities.count)
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
          "tether-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
          at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: folder) }
        for (index, activity) in activities.enumerated() {
          try Task.checkCancellation()
          setProgress(.checking, completed: index)
          let record: TransferRecord
          if let cached = try await archive.cachedActivity(
            deviceID: device.id, path: activity.sourcePath,
            size: activity.size, modified: activity.modified)
          {
            record = cached
          } else {
            setProgress(.downloading)
            let destination = folder.appendingPathComponent(UUID().uuidString)
              .appendingPathExtension("fit")
            try await transport.download(activity, to: destination)
            try Task.checkCancellation()
            record = try await archive.ingest(
              file: destination, deviceID: device.id, filename: activity.name,
              sourcePath: activity.sourcePath, modified: activity.modified)
            try? FileManager.default.removeItem(at: destination)
          }
          try await uploadIfReady(record)
          setProgress(.checking, completed: index + 1)
          if record.state != "confirmed" { records = try await archive.allRecords() }
        }
        let paths = Set(activities.map(\.sourcePath))
        let confirmed = try await archive.allRecords().filter {
          $0.deviceID == device.id && $0.sourcePath.map(paths.contains) == true
            && $0.state == "confirmed"
        }.count
        let elapsed = startedAt.duration(to: .now)
        let seconds =
          Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        let duration = String(format: "%.2f s", seconds)
        let pending = activities.count - confirmed
        setProgress(.complete, completed: activities.count)
        trace.record("sync.checked", count: activities.count)
        status =
          pending == 0
          ? "\(confirmed) synced · \(duration)" : "\(confirmed) synced · \(pending) pending"
        if pending > 0 {
          let hasSession = await client.hasSession()
          if !hasSession {
            self.error =
              "Garmin session expired or unavailable. Open the account menu to reconnect."
          }
        }
        await transport.disconnect()
      } catch is CancellationError {
        status = "sync cancelled; saved files are safe"
        setProgress(.cancelled)
      } catch {
        self.error = error.localizedDescription
        status = "sync stopped"
        setProgress(.failed)
      }
      records = (try? await archive.allRecords()) ?? records
      Task { accountStatus = await client.connectionStatus() }
    }
  }

  private func uploadIfReady(_ record: TransferRecord) async throws {
    if record.state == "confirmed" { return }
    if let sessionRefresh {
      accountStatus = await sessionRefresh.value
      self.sessionRefresh = nil
    }
    guard await client.hasSession() else { return }
    if let id = record.activityID {
      setProgress(.verifying)
      try await client.verify(activityID: id)
      try await archive.update(id: record.id, state: "confirmed")
      return
    }
    if record.state == "uncertain" || record.state == "duplicate, unverified" {
      setProgress(.verifying)
      status = "checking \(record.filename)"
      if let id = try await client.reconcile(
        file: URL(fileURLWithPath: record.localPath), filename: record.filename)
      {
        try await archive.update(id: record.id, state: "confirmed", activityID: id)
      }
      return
    }
    try Task.checkCancellation()
    status = "uploading \(record.filename)"
    setProgress(.uploading)
    try await archive.update(id: record.id, state: "uploading")
    let receipt: UploadReceipt
    do {
      receipt = try await client.upload(file: URL(fileURLWithPath: record.localPath))
    } catch {
      let state: String
      if let failure = error as? GarminConnectError, case .uncertain = failure {
        state = "uncertain"
      } else {
        state = "failed"
      }
      try? await archive.update(id: record.id, state: state)
      throw error
    }
    if let id = receipt.activityID {
      do {
        try await archive.update(id: record.id, state: "uploaded; verifying", activityID: id)
      } catch {
        try? await archive.update(id: record.id, state: "uncertain")
        throw error
      }
      setProgress(.verifying)
      try await client.verify(activityID: id)
      try await archive.update(id: record.id, state: "confirmed")
    } else if receipt.duplicate {
      try await archive.update(id: record.id, state: "duplicate, unverified")
    }
  }

  func cancelSync() {
    status = "cancelling"
    setProgress(.cancelling)
    syncTask?.cancel()
    if let activeTransport { Task { await activeTransport.cancel() } }
  }

  private func finishSync() {
    trace.record("sync.finished")
    syncingDevice = nil
    activeTransport = nil
    syncTask = nil
    let result = status
    monitor.refresh()
    status = result
    isSyncing = false
    startNextAutomaticSync()
    if !isSyncing, let progress = syncProgress,
      [.complete, .cancelled, .failed].contains(progress.phase)
    {
      progressDismissal = Task {
        do { try await Task.sleep(for: .milliseconds(350)) } catch { return }
        guard !isSyncing, syncProgress == progress else { return }
        syncProgress = nil
        trace.record("progress.dismissed")
      }
    }
  }

  func importAccount() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.json]
    panel.canChooseDirectories = false
    panel.message =
      "Choose your Garmin session JSON. Tether stores only its access token in Keychain."
    panel.begin { [weak self] result in
      guard result == .OK, let url = panel.url else { return }
      Task { @MainActor in
        guard let self else { return }
        do {
          try await self.client.importSession(from: url)
          self.accountStatus = await self.client.connectionStatus()
          self.showAccount = false
          self.error = nil
          self.automaticSync.retryConnected()
          self.startNextAutomaticSync()
        } catch { self.error = error.localizedDescription }
      }
    }
  }

  func authorizeAccount() {
    guard accountStatus != "waiting for keychain" else { return }
    accountStatus = "waiting for keychain"
    Task { accountStatus = await client.authorizeSession() }
  }

  func importActivities() {
    guard !isSyncing else { return }
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [UTType(filenameExtension: "fit") ?? .data]
    panel.allowsMultipleSelection = true
    panel.canChooseDirectories = false
    panel.begin { [weak self] result in
      guard result == .OK else { return }
      Task { @MainActor in self?.importFiles(panel.urls) }
    }
  }

  private func importFiles(_ urls: [URL]) {
    guard !isSyncing else {
      error = "A watch sync is running. Import files after it finishes."
      return
    }
    isSyncing = true
    progressDismissal?.cancel()
    setProgress(.checking, completed: 0, total: urls.count)
    error = nil
    syncTask = Task {
      defer { finishSync() }
      do {
        for (index, url) in urls.enumerated() {
          try Task.checkCancellation()
          let record = try await archive.ingest(file: url, deviceID: "imported")
          try await uploadIfReady(record)
          setProgress(.checking, completed: index + 1)
          records = try await archive.allRecords()
        }
        status = "\(urls.count) activity files checked"
        setProgress(.complete, completed: urls.count)
      } catch is CancellationError {
        status = "import cancelled"
        setProgress(.cancelled)
      } catch {
        self.error = error.localizedDescription
        status = "import stopped"
        setProgress(.failed)
      }
      records = (try? await archive.allRecords()) ?? records
    }
  }

  func forgetSelectedDevice() {
    guard !isSyncing, let selectedDeviceID else { return }
    devices.removeAll { $0.id == selectedDeviceID }
    self.selectedDeviceID = devices.first?.id
    let remaining = devices
    Task {
      do { try await registry.save(remaining) } catch { self.error = error.localizedDescription }
    }
  }

  func revealArchive() { NSWorkspace.shared.open(archive.root) }
}
