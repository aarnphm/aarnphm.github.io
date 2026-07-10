import Foundation

struct HealthAggregator {
  private struct LatestValue {
    let date: Date
    let value: Double
  }

  private struct DayBucket {
    var activeKcal: Double?
    var basalKcal: Double?
    var intakeKcal: Double?
    var weightKg: LatestValue?
    var vo2max: LatestValue?
  }

  private struct SwimBucket {
    let session: SwimSessionValue
    let date: String
    var samples: [SwimSampleValue] = []
  }

  private struct SwimSampleKey: Hashable {
    let startDate: Date
    let endDate: Date
  }

  static func document(
    quantitySamples: [QuantitySampleValue],
    swimSamples: [SwimSampleValue],
    swimSessions: [SwimSessionValue] = [],
    workouts: [AppleHealthWorkout] = [],
    generatedAt: Date,
    calendar: Calendar
  ) -> HealthExportDocument {
    let days = aggregateDays(quantitySamples: quantitySamples, calendar: calendar)
    let swims = aggregateSwims(
      swimSamples: swimSamples,
      swimSessions: swimSessions,
      calendar: calendar
    )
    return HealthExportDocument(
      version: HealthExportDocument.currentVersion,
      generatedAt: HealthExporterFormat.timestampString(generatedAt, timeZone: calendar.timeZone),
      timezone: calendar.timeZone.identifier,
      days: days,
      swims: swims,
      workouts: workouts
    )
  }

  static func aggregateDays(
    quantitySamples: [QuantitySampleValue],
    calendar: Calendar
  ) -> [AppleHealthDay] {
    var buckets: [String: DayBucket] = [:]
    for sample in quantitySamples {
      let key = HealthExporterFormat.dayString(sample.startDate, calendar: calendar)
      var bucket = buckets[key] ?? DayBucket()
      switch sample.kind {
      case .activeEnergy:
        bucket.activeKcal = (bucket.activeKcal ?? 0) + sample.value
      case .basalEnergy:
        bucket.basalKcal = (bucket.basalKcal ?? 0) + sample.value
      case .dietaryEnergy:
        bucket.intakeKcal = (bucket.intakeKcal ?? 0) + sample.value
      case .bodyMass:
        let latest = bucket.weightKg
        if latest.map({ sample.endDate >= $0.date }) ?? true {
          bucket.weightKg = LatestValue(date: sample.endDate, value: sample.value)
        }
      case .vo2Max:
        let latest = bucket.vo2max
        if latest.map({ sample.endDate >= $0.date }) ?? true {
          bucket.vo2max = LatestValue(date: sample.endDate, value: sample.value)
        }
      }
      buckets[key] = bucket
    }
    return buckets.keys.sorted().map { key in
      guard let bucket = buckets[key] else {
        return AppleHealthDay(
          date: key,
          burnKcal: nil,
          activeKcal: nil,
          intakeKcal: nil,
          weightKg: nil,
          vo2max: nil
        )
      }
      let active = roundedInt(bucket.activeKcal)
      let basal = roundedInt(bucket.basalKcal)
      let burn = active != nil || basal != nil ? (active ?? 0) + (basal ?? 0) : nil
      return AppleHealthDay(
        date: key,
        burnKcal: burn,
        activeKcal: active,
        intakeKcal: roundedInt(bucket.intakeKcal),
        weightKg: roundedTenth(bucket.weightKg?.value),
        vo2max: roundedTenth(bucket.vo2max?.value)
      )
    }
  }

  static func aggregateSwims(
    swimSamples: [SwimSampleValue],
    swimSessions: [SwimSessionValue],
    calendar: Calendar
  ) -> [AppleHealthSwim] {
    var buckets = Dictionary(uniqueKeysWithValues: swimSessions.map { session in
      (
        session.id,
        SwimBucket(
          session: session,
          date: HealthExporterFormat.dayString(session.startDate, calendar: calendar)
        )
      )
    })
    for sample in swimSamples {
      guard sample.meters > 0, var bucket = buckets[sample.workoutID] else { continue }
      bucket.samples.append(sample)
      buckets[sample.workoutID] = bucket
    }
    return buckets.values.compactMap { bucket in
      var samplesByKey: [SwimSampleKey: SwimSampleValue] = [:]
      for sample in bucket.samples {
        let key = SwimSampleKey(
          startDate: sample.startDate,
          endDate: sample.endDate
        )
        if let existing = samplesByKey[key] {
          samplesByKey[key] = SwimSampleValue(
            workoutID: existing.workoutID,
            startDate: existing.startDate,
            endDate: existing.endDate,
            meters: max(existing.meters, sample.meters),
            stroke: existing.stroke ?? sample.stroke
          )
          continue
        }
        samplesByKey[key] = sample
      }
      let samples = Array(samplesByKey.values)
      let totalM = bucket.session.distanceMeters ?? 0
      guard totalM > 0 else { return nil }
      var strokes: [String: Int] = [:]
      for stroke in SwimStrokeName.allCases {
        let meters = samples.filter { $0.stroke == stroke }.reduce(0) { $0 + $1.meters }
        let rounded = Int(meters.rounded())
        if rounded > 0 { strokes[stroke.rawValue] = rounded }
      }
      return AppleHealthSwim(
        id: bucket.session.id,
        date: bucket.date,
        start: HealthExporterFormat.utcTimestampString(bucket.session.startDate),
        end: HealthExporterFormat.utcTimestampString(bucket.session.endDate),
        totalM: Int(totalM.rounded()),
        laps: bucket.session.lapCount ?? samples.count,
        activeTimeS: Int(bucket.session.activeTimeS.rounded()),
        strokeCount: bucket.session.strokeCount.map { Int($0.rounded()) },
        strokeTimeS: bucket.session.strokeTimeS.map { Int($0.rounded()) },
        strokes: strokes
      )
    }.sorted {
      ($0.start ?? $0.date, $0.id) < ($1.start ?? $1.date, $1.id)
    }
  }

  static func strokeTime(strokeSamples: [SwimStrokeIntervalValue]) -> TimeInterval? {
    let sorted = strokeSamples.filter {
      $0.count > 0 && $0.stroke != .kickboard && $0.endDate > $0.startDate
    }.sorted {
      if $0.startDate != $1.startDate { return $0.startDate < $1.startDate }
      if $0.endDate != $1.endDate { return $0.endDate > $1.endDate }
      return $0.count > $1.count
    }
    guard !sorted.isEmpty else { return nil }
    var time: TimeInterval = 0
    var coveredUntil: Date?
    for sample in sorted {
      let uncoveredStart = max(sample.startDate, coveredUntil ?? sample.startDate)
      time += max(0, sample.endDate.timeIntervalSince(uncoveredStart))
      coveredUntil = max(coveredUntil ?? sample.endDate, sample.endDate)
    }
    return time
  }

  private static func roundedInt(_ value: Double?) -> Int? {
    guard let value, value.isFinite else { return nil }
    return Int(value.rounded())
  }

  private static func roundedTenth(_ value: Double?) -> Double? {
    guard let value, value.isFinite else { return nil }
    return (value * 10).rounded() / 10
  }
}
