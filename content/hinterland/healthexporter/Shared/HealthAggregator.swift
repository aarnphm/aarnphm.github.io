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

  static func document(
    quantitySamples: [QuantitySampleValue],
    swimSamples: [SwimSampleValue],
    workouts: [AppleHealthWorkout] = [],
    generatedAt: Date,
    calendar: Calendar
  ) -> HealthExportDocument {
    let days = aggregateDays(quantitySamples: quantitySamples, calendar: calendar)
    let swims = aggregateSwims(swimSamples: swimSamples, calendar: calendar)
    return HealthExportDocument(
      version: 2,
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
    calendar: Calendar
  ) -> [AppleHealthSwim] {
    var totalByDay: [String: Double] = [:]
    var lapsByDay: [String: Int] = [:]
    var strokesByDay: [String: [String: Double]] = [:]
    for sample in swimSamples {
      guard let stroke = sample.stroke, sample.meters > 0 else { continue }
      let key = HealthExporterFormat.dayString(sample.startDate, calendar: calendar)
      totalByDay[key] = (totalByDay[key] ?? 0) + sample.meters
      lapsByDay[key] = (lapsByDay[key] ?? 0) + 1
      var strokes = strokesByDay[key] ?? [:]
      strokes[stroke.rawValue] = (strokes[stroke.rawValue] ?? 0) + sample.meters
      strokesByDay[key] = strokes
    }
    return totalByDay.keys.sorted().compactMap { key in
      guard let total = totalByDay[key], let laps = lapsByDay[key], let strokeMeters = strokesByDay[key] else {
        return nil
      }
      var strokes: [String: Int] = [:]
      for (name, meters) in strokeMeters {
        let rounded = Int(meters.rounded())
        if rounded > 0 { strokes[name] = rounded }
      }
      if strokes.isEmpty { return nil }
      return AppleHealthSwim(date: key, totalM: Int(total.rounded()), laps: laps, strokes: strokes)
    }
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
