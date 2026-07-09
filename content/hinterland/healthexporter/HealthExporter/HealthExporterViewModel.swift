import Foundation

@MainActor
final class HealthExporterViewModel: ObservableObject {
  private static let catchUpInterval: TimeInterval = 60 * 60

  @Published var status = "Waiting for HealthKit permission"
  @Published var filePath = HealthExportWriter.visiblePath
  @Published var generatedAt = "Never"
  @Published var dayCount = 0
  @Published var swimCount = 0
  @Published var workoutCount = 0
  @Published var heartRateCount = 0
  @Published var isExporting = false

  func prepare() async {
    if let result = await HealthExportRuntime.shared.currentExport() {
      apply(result)
    }
    do {
      try await HealthExportRuntime.shared.requestAuthorization()
      HealthBackgroundScheduler.shared.schedule()
      let result = try await HealthExportRuntime.shared.catchUpIfNeeded(maxAge: Self.catchUpInterval)
      apply(result)
      status = "Ready. Background sync runs when Health changes."
    } catch {
      status = message(error)
    }
  }

  func exportNow() {
    guard !isExporting else { return }
    isExporting = true
    status = "Exporting"
    Task {
      do {
        let result = try await HealthExportRuntime.shared.export(force: true)
        apply(result)
        status = "Exported to iCloud Drive"
      } catch {
        status = message(error)
      }
      isExporting = false
    }
  }

  private func apply(_ result: HealthExportResult) {
    generatedAt = result.document.generatedAt
    dayCount = result.document.days.count
    swimCount = result.document.swims.count
    workoutCount = result.document.workouts.count
    heartRateCount = result.document.workouts.reduce(0) { $0 + $1.heartRate.count }
    filePath = HealthExportWriter.visiblePath
  }

  private func message(_ error: Error) -> String {
    if let error = error as? LocalizedError, let description = error.errorDescription {
      return description
    }
    return error.localizedDescription
  }
}
