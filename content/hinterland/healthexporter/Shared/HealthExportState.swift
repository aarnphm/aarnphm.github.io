import Foundation

enum HealthExportState: String, Sendable {
  case exported = "Exported"
  case stale = "Stale"
  case exporting = "Exporting"
  case failed = "Failed"
}

struct HealthExportStateStore {
  private static let lastSuccessfulExportKey = "healthExporter.lastSuccessfulExport"

  private let defaults: UserDefaults

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
  }

  func state(maxAge: TimeInterval, now: Date = Date()) -> HealthExportState {
    guard let lastSuccessfulExport = defaults.object(
      forKey: Self.lastSuccessfulExportKey
    ) as? Date else {
      return .stale
    }
    let age = now.timeIntervalSince(lastSuccessfulExport)
    return age >= 0 && age < maxAge ? .exported : .stale
  }

  func markSuccessful(at date: Date = Date()) {
    defaults.set(date, forKey: Self.lastSuccessfulExportKey)
  }
}
