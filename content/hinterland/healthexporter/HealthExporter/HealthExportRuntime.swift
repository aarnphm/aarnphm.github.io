import Foundation

actor HealthExportRuntime {
  static let shared = HealthExportRuntime()

  private let health = HealthKitService.shared
  private let writer = HealthExportWriter()
  private var observersStarted = false
  private var changeGeneration = 0
  private var cachedGeneration = -1
  private var cachedResult: HealthExportResult?
  private var exportTask: Task<HealthExportResult, Error>?
  private var observerExportScheduled = false
  private static let observerDebounceNanoseconds: UInt64 = 5_000_000_000

  private init() {}

  func requestAuthorization() async throws {
    try await health.requestAuthorization()
    startObservers()
  }

  private func startObservers() {
    guard !observersStarted else { return }
    observersStarted = true
    health.startObservers {
      Task {
        await Self.shared.healthChanged()
      }
    }
  }

  private func healthChanged() {
    changeGeneration += 1
    scheduleObserverExport()
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
    _ = try? await export()
  }

  func export(force: Bool = false) async throws -> HealthExportResult {
    if !force,
      let cachedResult,
      cachedGeneration == changeGeneration,
      FileManager.default.fileExists(atPath: cachedResult.url.path)
    {
      PhoneWatchSession.shared.publish(cachedResult)
      return cachedResult
    }
    if let exportTask {
      return try await exportTask.value
    }
    let generation = changeGeneration
    let task = Task {
      try await Self.shared.performExport()
    }
    exportTask = task
    do {
      let result = try await task.value
      cachedResult = result
      cachedGeneration = generation
      exportTask = nil
      if changeGeneration != generation {
        scheduleObserverExport()
      }
      return result
    } catch {
      exportTask = nil
      throw error
    }
  }

  private func performExport() async throws -> HealthExportResult {
    let document = try await health.export()
    let url = try writer.write(document)
    let result = HealthExportResult(document: document, url: url)
    PhoneWatchSession.shared.publish(result)
    return result
  }
}
