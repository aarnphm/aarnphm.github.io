import Foundation

@MainActor
final class HealthExporterViewModel: ObservableObject {
  @Published var status = "Waiting for HealthKit permission"
  @Published var filePath = HealthExportWriter.visiblePath
  @Published var generatedAt = "Never"
  @Published var dayCount = 0
  @Published var swimCount = 0
  @Published var isExporting = false

  func prepare() async {
    do {
      try await HealthExportRuntime.shared.requestAuthorization()
      HealthBackgroundScheduler.shared.schedule()
      status = "Ready"
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
        let result = try await HealthExportRuntime.shared.export()
        generatedAt = result.document.generatedAt
        dayCount = result.document.days.count
        swimCount = result.document.swims.count
        filePath = HealthExportWriter.visiblePath
        status = "Exported to iCloud Drive"
      } catch {
        status = message(error)
      }
      isExporting = false
    }
  }

  private func message(_ error: Error) -> String {
    if let error = error as? LocalizedError, let description = error.errorDescription {
      return description
    }
    return error.localizedDescription
  }
}
