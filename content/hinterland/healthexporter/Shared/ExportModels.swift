import Foundation

enum HealthMetricKind: String, Equatable, Sendable {
  case activeEnergy
  case basalEnergy
  case dietaryEnergy
  case bodyMass
  case vo2Max
}

struct QuantitySampleValue: Equatable, Sendable {
  let kind: HealthMetricKind
  let startDate: Date
  let endDate: Date
  let value: Double
}

enum SwimStrokeName: String, Codable, CaseIterable, Sendable {
  case freestyle
  case breaststroke
  case backstroke
  case butterfly
  case mixed
  case kickboard
}

struct SwimSampleValue: Equatable, Sendable {
  let startDate: Date
  let meters: Double
  let stroke: SwimStrokeName?
}

struct AppleHealthDay: Codable, Equatable, Identifiable, Sendable {
  let date: String
  let burnKcal: Int?
  let activeKcal: Int?
  let intakeKcal: Int?
  let weightKg: Double?
  let vo2max: Double?

  var id: String { date }
}

struct AppleHealthSwim: Codable, Equatable, Identifiable, Sendable {
  let date: String
  let totalM: Int
  let laps: Int
  let strokes: [String: Int]

  var id: String { date }
}

struct AppleHealthHeartRate: Codable, Equatable, Sendable {
  let time: String
  let bpm: Int
}

struct AppleHealthWorkout: Codable, Equatable, Identifiable, Sendable {
  let id: String
  let activity: String
  let start: String
  let end: String
  let durationS: Int
  let heartRate: [AppleHealthHeartRate]
}

struct HealthExportDocument: Codable, Equatable, Sendable {
  let version: Int
  let generatedAt: String
  let timezone: String
  let days: [AppleHealthDay]
  let swims: [AppleHealthSwim]
  let workouts: [AppleHealthWorkout]
}

struct HealthExportResult: Equatable, Sendable {
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

  static func utcTimestampString(_ date: Date) -> String {
    let timeZone = TimeZone(secondsFromGMT: 0) ?? TimeZone(identifier: "UTC") ?? .current
    return timestampString(date, timeZone: timeZone)
  }
}
