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

struct SwimSessionValue: Equatable, Sendable {
  let id: String
  let startDate: Date
  let endDate: Date
  let distanceMeters: Double?
  let activeTimeS: TimeInterval
  let strokeCount: Double?
  let strokeTimeS: TimeInterval?
  let lapCount: Int?
}

struct SwimStrokeIntervalValue: Equatable, Sendable {
  let startDate: Date
  let endDate: Date
  let count: Double
  let stroke: SwimStrokeName?
}

struct SwimStrokeMetricsValue: Equatable, Sendable {
  let count: Double
  let timeS: TimeInterval
}

struct SwimSampleValue: Equatable, Sendable {
  let workoutID: String
  let startDate: Date
  let endDate: Date
  let meters: Double
  let strokeCount: Double?
  let strokeTimeS: TimeInterval?
  let stroke: SwimStrokeName?

  init(
    workoutID: String,
    startDate: Date,
    endDate: Date,
    meters: Double,
    strokeCount: Double? = nil,
    strokeTimeS: TimeInterval? = nil,
    stroke: SwimStrokeName?
  ) {
    self.workoutID = workoutID
    self.startDate = startDate
    self.endDate = endDate
    self.meters = meters
    self.strokeCount = strokeCount
    self.strokeTimeS = strokeTimeS
    self.stroke = stroke
  }
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

struct AppleHealthSwimInterval: Codable, Equatable, Sendable {
  let start: String
  let end: String
  let startElapsedS: TimeInterval?
  let endElapsedS: TimeInterval?
  let distanceM: Double
  let durationS: TimeInterval?
  let strokeCount: Double?
  let strokeTimeS: TimeInterval?
  let stroke: SwimStrokeName?

  init(
    start: String,
    end: String,
    startElapsedS: TimeInterval? = nil,
    endElapsedS: TimeInterval? = nil,
    distanceM: Double,
    durationS: TimeInterval? = nil,
    strokeCount: Double? = nil,
    strokeTimeS: TimeInterval? = nil,
    stroke: SwimStrokeName? = nil
  ) {
    self.start = start
    self.end = end
    self.startElapsedS = startElapsedS
    self.endElapsedS = endElapsedS
    self.distanceM = distanceM
    self.durationS = durationS
    self.strokeCount = strokeCount
    self.strokeTimeS = strokeTimeS
    self.stroke = stroke
  }
}

struct AppleHealthSwim: Codable, Equatable, Identifiable, Sendable {
  let id: String
  let date: String
  let start: String?
  let end: String?
  let totalM: Int
  let laps: Int
  let activeTimeS: Int?
  let strokeCount: Int?
  let strokeTimeS: Int?
  let strokes: [String: Int]
  let intervals: [AppleHealthSwimInterval]

  init(
    id: String,
    date: String,
    start: String?,
    end: String?,
    totalM: Int,
    laps: Int,
    activeTimeS: Int?,
    strokeCount: Int?,
    strokeTimeS: Int?,
    strokes: [String: Int],
    intervals: [AppleHealthSwimInterval] = []
  ) {
    self.id = id
    self.date = date
    self.start = start
    self.end = end
    self.totalM = totalM
    self.laps = laps
    self.activeTimeS = activeTimeS
    self.strokeCount = strokeCount
    self.strokeTimeS = strokeTimeS
    self.strokes = strokes
    self.intervals = intervals
  }

  private enum CodingKeys: String, CodingKey {
    case id
    case date
    case start
    case end
    case totalM
    case laps
    case activeTimeS
    case strokeCount
    case strokeTimeS
    case strokes
    case intervals
  }

  init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    date = try values.decode(String.self, forKey: .date)
    id = try values.decodeIfPresent(String.self, forKey: .id) ?? date
    start = try values.decodeIfPresent(String.self, forKey: .start)
    end = try values.decodeIfPresent(String.self, forKey: .end)
    totalM = try values.decode(Int.self, forKey: .totalM)
    laps = try values.decode(Int.self, forKey: .laps)
    activeTimeS = try values.decodeIfPresent(Int.self, forKey: .activeTimeS)
    strokeCount = try values.decodeIfPresent(Int.self, forKey: .strokeCount)
    strokeTimeS = try values.decodeIfPresent(Int.self, forKey: .strokeTimeS)
    strokes = try values.decode([String: Int].self, forKey: .strokes)
    intervals = try values.decodeIfPresent([AppleHealthSwimInterval].self, forKey: .intervals) ?? []
  }
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
  static let currentVersion = 6

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
      swims: (swims.filter { $0.date < dayCutoff } + recent.swims).sorted {
        ($0.start ?? $0.date, $0.id) < ($1.start ?? $1.date, $1.id)
      },
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
