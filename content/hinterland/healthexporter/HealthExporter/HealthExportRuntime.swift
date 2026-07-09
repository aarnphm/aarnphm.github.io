import Foundation

actor HealthExportRuntime {
  static let shared = HealthExportRuntime()

  private enum ExportScope {
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

  func catchUpIfNeeded(maxAge: TimeInterval, now: Date = Date()) async throws -> HealthExportResult {
    if let result = currentExport(), isCurrentExportFresh(maxAge: maxAge, now: now) {
      return result
    }
    return try await export(scope: .recent, force: true)
  }

  private func isCurrentExportFresh(maxAge: TimeInterval, now: Date = Date()) -> Bool {
    guard let result = currentExport(),
      let generatedAt = ISO8601DateFormatter().date(from: result.document.generatedAt)
    else { return false }
    let age = now.timeIntervalSince(generatedAt)
    return age >= 0 && age < maxAge
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
    let previous = scope == .recent ? currentExport() : nil
    let daysBack = previous == nil ? 180 : Self.recentDaysBack
    let update = try await health.export(daysBack: daysBack, now: now)
    let document: HealthExportDocument
    if let previous {
      var calendar = Calendar.autoupdatingCurrent
      calendar.timeZone = .autoupdatingCurrent
      let cutoff = calendar.date(
        byAdding: .day,
        value: -Self.recentDaysBack,
        to: calendar.startOfDay(for: now)
      ) ?? calendar.startOfDay(for: now)
      document = previous.document.replacingRecent(
        with: update,
        dayCutoff: HealthExporterFormat.dayString(cutoff, calendar: calendar),
        timestampCutoff: HealthExporterFormat.utcTimestampString(cutoff)
      )
      if document.hasSameHealthData(as: previous.document) {
        PhoneWatchSession.shared.publish(previous)
        return previous
      }
    } else {
      document = update
    }
    let url = try writer.write(document)
    let result = HealthExportResult(document: document, url: url)
    PhoneWatchSession.shared.publish(result)
    return result
  }
}
