import Foundation

final class HealthExportRuntime {
  static let shared = HealthExportRuntime()

  private let health = HealthKitService.shared
  private let writer = HealthExportWriter()
  private var observersStarted = false

  private init() {}

  func requestAuthorization() async throws {
    try await health.requestAuthorization()
    startObservers()
  }

  func startObservers() {
    guard !observersStarted else { return }
    observersStarted = true
    health.startObservers {
      Task {
        _ = try? await Self.shared.export()
      }
    }
  }

  func export() async throws -> HealthExportResult {
    let document = try await health.export()
    let url = try writer.write(document)
    let result = HealthExportResult(document: document, url: url)
    PhoneWatchSession.shared.publish(result)
    return result
  }
}
