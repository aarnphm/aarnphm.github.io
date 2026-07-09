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

  func replacingRecent(
    with recent: HealthExportDocument,
    dayCutoff: String,
    timestampCutoff: String
  ) -> HealthExportDocument {
    HealthExportDocument(
      version: recent.version,
      generatedAt: recent.generatedAt,
      timezone: recent.timezone,
      days: (days.filter { $0.date < dayCutoff } + recent.days).sorted { $0.date < $1.date },
      swims: (swims.filter { $0.date < dayCutoff } + recent.swims).sorted { $0.date < $1.date },
      workouts: workouts.filter { $0.start < timestampCutoff } + recent.workouts
    )
  }

  func hasSameHealthData(as other: HealthExportDocument) -> Bool {
    version == other.version
      && timezone == other.timezone
      && days == other.days
      && swims == other.swims
      && workouts == other.workouts
  }
}

struct HealthExportResult: Equatable, Sendable {
  let document: HealthExportDocument
  let url: URL
}

private final class HealthExporterFormatterCache: @unchecked Sendable {
  private let lock = NSLock()
  private var dayFormatters: [String: DateFormatter] = [:]
  private var timestampFormatters: [String: ISO8601DateFormatter] = [:]

  func dayString(_ date: Date, calendar: Calendar) -> String {
    let key = "\(calendar.identifier):\(calendar.timeZone.identifier)"
    lock.lock()
    let formatter: DateFormatter
    if let cached = dayFormatters[key] {
      formatter = cached
    } else {
      let created = DateFormatter()
      created.calendar = calendar
      created.timeZone = calendar.timeZone
      created.locale = Locale(identifier: "en_US_POSIX")
      created.dateFormat = "yyyy-MM-dd"
      dayFormatters[key] = created
      formatter = created
    }
    let result = formatter.string(from: date)
    lock.unlock()
    return result
  }

  func timestampString(_ date: Date, timeZone: TimeZone) -> String {
    let key = timeZone.identifier
    lock.lock()
    let formatter: ISO8601DateFormatter
    if let cached = timestampFormatters[key] {
      formatter = cached
    } else {
      let created = ISO8601DateFormatter()
      created.formatOptions = [.withInternetDateTime, .withColonSeparatorInTimeZone]
      created.timeZone = timeZone
      timestampFormatters[key] = created
      formatter = created
    }
    let result = formatter.string(from: date)
    lock.unlock()
    return result
  }
}

enum HealthExporterFormat {
  private static let cache = HealthExporterFormatterCache()

  static func dayString(_ date: Date, calendar: Calendar) -> String {
    cache.dayString(date, calendar: calendar)
  }

  static func timestampString(_ date: Date, timeZone: TimeZone) -> String {
    cache.timestampString(date, timeZone: timeZone)
  }

  static func utcTimestampString(_ date: Date) -> String {
    let timeZone = TimeZone(secondsFromGMT: 0) ?? TimeZone(identifier: "UTC") ?? .current
    return timestampString(date, timeZone: timeZone)
  }
}
