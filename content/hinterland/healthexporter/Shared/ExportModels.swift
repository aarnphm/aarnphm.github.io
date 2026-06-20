import Foundation

enum HealthMetricKind: String, Equatable {
  case activeEnergy
  case basalEnergy
  case dietaryEnergy
  case bodyMass
  case vo2Max
}

struct QuantitySampleValue: Equatable {
  let kind: HealthMetricKind
  let startDate: Date
  let endDate: Date
  let value: Double
}

enum SwimStrokeName: String, Codable, CaseIterable {
  case freestyle
  case breaststroke
  case backstroke
  case butterfly
  case mixed
  case kickboard
}

struct SwimSampleValue: Equatable {
  let startDate: Date
  let meters: Double
  let stroke: SwimStrokeName?
}

struct AppleHealthDay: Codable, Equatable, Identifiable {
  let date: String
  let burnKcal: Int?
  let activeKcal: Int?
  let intakeKcal: Int?
  let weightKg: Double?
  let vo2max: Double?

  var id: String { date }
}

struct AppleHealthSwim: Codable, Equatable, Identifiable {
  let date: String
  let totalM: Int
  let laps: Int
  let strokes: [String: Int]

  var id: String { date }
}

struct HealthExportDocument: Codable, Equatable {
  let version: Int
  let generatedAt: String
  let timezone: String
  let days: [AppleHealthDay]
  let swims: [AppleHealthSwim]
}

struct HealthExportResult: Equatable {
  let document: HealthExportDocument
  let url: URL
}

enum HealthExporterFormat {
  static func dayString(_ date: Date, calendar: Calendar) -> String {
    let formatter = DateFormatter()
    formatter.calendar = calendar
    formatter.timeZone = calendar.timeZone
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.dateFormat = "yyyy-MM-dd"
    return formatter.string(from: date)
  }

  static func timestampString(_ date: Date, timeZone: TimeZone) -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withColonSeparatorInTimeZone]
    formatter.timeZone = timeZone
    return formatter.string(from: date)
  }
}
