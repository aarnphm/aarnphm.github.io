import Foundation

actor HealthExportRuntime {
  static let shared = HealthExportRuntime()

  private enum ExportScope: Equatable {
    case full
    case recent
  }

  private struct InFlightExport {
    let id: Int
    let scope: ExportScope
    let task: Task<HealthExportResult, Error>
  }

  private let health = HealthKitService.shared
  private let writer = HealthExportWriter()
  private let stateStore = HealthExportStateStore()
  private var changeGeneration = 0
  private var cachedGeneration = -1
  private var cachedResult: HealthExportResult?
  private var inFlightExport: InFlightExport?
  private var nextExportID = 0
  private var observerExportScheduled = false
  private var observerCompletions: [HealthObserverCompletion] = []
  private static let recentDaysBack = 2
  private static let observerDebounceNanoseconds: UInt64 = 1_000_000_000

  private init() {}

  nonisolated static func startObservers(restart: Bool = false) {
    HealthKitService.shared.startObservers(restart: restart) { completion in
      Task {
        await Self.shared.healthChanged(completion: completion)
      }
    }
  }

  func requestAuthorization() async throws {
    try await health.requestAuthorization()
    Self.startObservers(restart: true)
  }

  private func healthChanged(completion: HealthObserverCompletion) {
    changeGeneration += 1
    observerCompletions.append(completion)
    if inFlightExport == nil {
      scheduleObserverExport()
    }
  }

  private func scheduleObserverExport() {
    guard !observerExportScheduled else { return }
    observerExportScheduled = true
    Task {
      try? await Task.sleep(nanoseconds: Self.observerDebounceNanoseconds)
      await Self.shared.runScheduledExport()
    }
  }

  private func runScheduledExport() async {
    observerExportScheduled = false
    let completions = observerCompletions
    observerCompletions.removeAll(keepingCapacity: true)
    _ = try? await export(scope: .recent)
    for completion in completions {
      completion.complete()
    }
    if !observerCompletions.isEmpty {
      scheduleObserverExport()
    }
  }

  func currentExport() -> HealthExportResult? {
    if let cachedResult { return cachedResult }
    guard let result = try? writer.read() else { return nil }
    cachedResult = result
    PhoneWatchSession.shared.publish(result)
    return result
  }

  func needsMigration() -> Bool {
    guard let result = currentExport() else { return false }
    return result.document.version != HealthExportDocument.currentVersion
  }

  func catchUpIfNeeded(maxAge: TimeInterval, now: Date = Date()) async throws {
    if needsMigration() {
      _ = try await export(scope: .full, force: true)
      return
    }
    guard stateStore.state(maxAge: maxAge, now: now) == .stale else { return }
    _ = try await export(scope: .recent, force: true)
  }

  func export(force: Bool = false) async throws -> HealthExportResult {
    try await export(scope: .full, force: force)
  }

  private func export(scope: ExportScope, force: Bool = false) async throws -> HealthExportResult {
    if !force,
      let cachedResult,
      cachedGeneration == changeGeneration,
      FileManager.default.fileExists(atPath: cachedResult.url.path)
    {
      PhoneWatchSession.shared.publish(cachedResult)
      return cachedResult
    }
    if let inFlightExport {
      let result = try await inFlightExport.task.value
      if scope == .full && inFlightExport.scope == .recent {
        return try await export(scope: .full, force: true)
      }
      return result
    }
    let generation = changeGeneration
    let id = nextExportID
    nextExportID += 1
    let task = Task {
      do {
        let result = try await Self.shared.performExport(scope: scope)
        await Self.shared.finishExport(id: id, generation: generation, result: result)
        return result
      } catch {
        await Self.shared.failExport(id: id)
        throw error
      }
    }
    inFlightExport = InFlightExport(id: id, scope: scope, task: task)
    return try await task.value
  }

  private func finishExport(id: Int, generation: Int, result: HealthExportResult) {
    guard inFlightExport?.id == id else { return }
    cachedResult = result
    cachedGeneration = generation
    stateStore.markSuccessful()
    inFlightExport = nil
    if changeGeneration != generation {
      scheduleObserverExport()
    }
  }

  private func failExport(id: Int) {
    guard inFlightExport?.id == id else { return }
    inFlightExport = nil
    if !observerCompletions.isEmpty {
      scheduleObserverExport()
    }
  }

  private func performExport(scope: ExportScope) async throws -> HealthExportResult {
    let now = Date()
    let previous = currentExport()
    let needsMigration = previous?.document.version != HealthExportDocument.currentVersion
    let daysBack = scope == .full || previous == nil || needsMigration ? 180 : Self.recentDaysBack
    var calendar = Calendar.autoupdatingCurrent
    calendar.timeZone = .autoupdatingCurrent
    let routeCutoff = previous == nil || needsMigration
      ? nil
      : calendar.date(
        byAdding: .day,
        value: -Self.recentDaysBack,
        to: calendar.startOfDay(for: now)
      )
    let update = try await health.export(daysBack: daysBack, routesSince: routeCutoff, now: now)
    let updateDocument = update.document.preservingWorkoutGPXFiles(from: previous?.document)
    let document: HealthExportDocument
    if scope == .recent, let previous, !needsMigration {
      let cutoff = calendar.date(
        byAdding: .day,
        value: -Self.recentDaysBack,
        to: calendar.startOfDay(for: now)
      ) ?? calendar.startOfDay(for: now)
      document = previous.document.replacingRecent(
        with: updateDocument,
        dayCutoff: HealthExporterFormat.dayString(cutoff, calendar: calendar),
        timestampCutoff: HealthExporterFormat.utcTimestampString(cutoff)
      )
      if document.hasSameHealthData(as: previous.document) {
        if !update.routes.isEmpty {
          _ = try writer.write(previous.document, routes: update.routes)
        }
        PhoneWatchSession.shared.publish(previous)
        return previous
      }
    } else {
      document = updateDocument
    }
    let url = try writer.write(document, routes: update.routes)
    let result = HealthExportResult(document: document, url: url)
    PhoneWatchSession.shared.publish(result)
    return result
  }
}
