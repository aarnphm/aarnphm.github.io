import Foundation

enum HealthExporterError: LocalizedError {
  case healthDataUnavailable
  case missingQuantityType(String)
  case iCloudUnavailable(String)
  case exportRejected(String)

  var errorDescription: String? {
    switch self {
    case .healthDataUnavailable:
      return "Health data is unavailable on this device."
    case .missingQuantityType(let identifier):
      return "HealthKit quantity type is unavailable: \(identifier)."
    case .iCloudUnavailable(let container):
      return "iCloud container is unavailable: \(container)."
    case .exportRejected(let reason):
      return reason
    }
  }
}
