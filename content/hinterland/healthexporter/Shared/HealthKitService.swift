#if canImport(HealthKit)
import Foundation
import HealthKit

final class HealthKitService {
  static let shared = HealthKitService()

  private struct DistanceSample {
    let startDate: Date
    let meters: Double
  }

  private struct StrokeSample {
    let startDate: Date
    let stroke: SwimStrokeName
  }

  private struct HeartRateSample {
    let startDate: Date
    let value: AppleHealthHeartRate
  }

  private let store = HKHealthStore()
  private var observerQueries: [HKObserverQuery] = []

  private let cumulativeTypes: [(HealthMetricKind, HKQuantityTypeIdentifier, HKUnit)] = [
    (.activeEnergy, .activeEnergyBurned, .kilocalorie()),
    (.basalEnergy, .basalEnergyBurned, .kilocalorie()),
    (.dietaryEnergy, .dietaryEnergyConsumed, .kilocalorie()),
  ]

  private let latestTypes: [(HealthMetricKind, HKQuantityTypeIdentifier, HKUnit)] = [
    (.bodyMass, .bodyMass, .gramUnit(with: .kilo)),
    (.vo2Max, .vo2Max, HKUnit(from: "mL/kg*min")),
  ]

  private var allQuantityTypes: [HKQuantityType] {
    let identifiers = (cumulativeTypes + latestTypes).map(\.1) + [
      .distanceSwimming,
      .swimmingStrokeCount,
      .heartRate,
    ]
    return identifiers.compactMap { HKQuantityType.quantityType(forIdentifier: $0) }
  }

  private var allSampleTypes: [HKSampleType] {
    allQuantityTypes + [HKObjectType.workoutType()]
  }

  func requestAuthorization() async throws {
    guard HKHealthStore.isHealthDataAvailable() else {
      throw HealthExporterError.healthDataUnavailable
    }
    let types = Set(allSampleTypes.map { $0 as HKObjectType })
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
      store.requestAuthorization(toShare: [], read: types) { success, error in
        if let error {
          continuation.resume(throwing: error)
        } else if success {
          continuation.resume()
        } else {
          continuation.resume(throwing: HealthExporterError.exportRejected("HealthKit authorization was denied."))
        }
      }
    }
  }

  func startObservers(onChange: @escaping () -> Void) {
    guard observerQueries.isEmpty else { return }
    for type in allSampleTypes {
      let query = HKObserverQuery(sampleType: type, predicate: nil) { _, completion, _ in
        onChange()
        completion()
      }
      observerQueries.append(query)
      store.execute(query)
      store.enableBackgroundDelivery(for: type, frequency: .hourly) { _, _ in }
    }
  }

  func export(daysBack: Int = 180, now: Date = Date()) async throws -> HealthExportDocument {
    var currentCalendar = Calendar.autoupdatingCurrent
    currentCalendar.timeZone = .autoupdatingCurrent
    let calendar = currentCalendar
    let end = now
    let start = calendar.date(byAdding: .day, value: -daysBack, to: calendar.startOfDay(for: now))
      ?? calendar.startOfDay(for: now)
    async let activeEnergy = cumulativeSamples(
      kind: .activeEnergy,
      identifier: .activeEnergyBurned,
      unit: .kilocalorie(),
      start: start,
      end: end,
      calendar: calendar
    )
    async let basalEnergy = cumulativeSamples(
      kind: .basalEnergy,
      identifier: .basalEnergyBurned,
      unit: .kilocalorie(),
      start: start,
      end: end,
      calendar: calendar
    )
    async let dietaryEnergy = cumulativeSamples(
      kind: .dietaryEnergy,
      identifier: .dietaryEnergyConsumed,
      unit: .kilocalorie(),
      start: start,
      end: end,
      calendar: calendar
    )
    async let bodyMass = latestSamples(
      kind: .bodyMass,
      identifier: .bodyMass,
      unit: .gramUnit(with: .kilo),
      start: start,
      end: end
    )
    async let vo2Max = latestSamples(
      kind: .vo2Max,
      identifier: .vo2Max,
      unit: HKUnit(from: "mL/kg*min"),
      start: start,
      end: end
    )
    async let swimSamples = swimSamples(start: start, end: end, calendar: calendar)
    async let workouts = workoutHeartRates(start: start, end: end)
    var quantitySamples = try await activeEnergy
    quantitySamples += try await basalEnergy
    quantitySamples += try await dietaryEnergy
    quantitySamples += try await bodyMass
    quantitySamples += try await vo2Max
    return HealthAggregator.document(
      quantitySamples: quantitySamples,
      swimSamples: try await swimSamples,
      workouts: try await workouts,
      generatedAt: now,
      calendar: calendar
    )
  }

  private func quantityType(_ identifier: HKQuantityTypeIdentifier) throws -> HKQuantityType {
    guard let type = HKQuantityType.quantityType(forIdentifier: identifier) else {
      throw HealthExporterError.missingQuantityType(identifier.rawValue)
    }
    return type
  }

  private func cumulativeSamples(
    kind: HealthMetricKind,
    identifier: HKQuantityTypeIdentifier,
    unit: HKUnit,
    start: Date,
    end: Date,
    calendar: Calendar
  ) async throws -> [QuantitySampleValue] {
    let type = try quantityType(identifier)
    let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [.strictStartDate])
    var interval = DateComponents()
    interval.day = 1
    return try await withCheckedThrowingContinuation { continuation in
      let query = HKStatisticsCollectionQuery(
        quantityType: type,
        quantitySamplePredicate: predicate,
        options: .cumulativeSum,
        anchorDate: calendar.startOfDay(for: start),
        intervalComponents: interval
      )
      query.initialResultsHandler = { _, collection, error in
        if let error {
          continuation.resume(throwing: error)
          return
        }
        var samples: [QuantitySampleValue] = []
        collection?.enumerateStatistics(from: start, to: end) { statistics, _ in
          guard let quantity = statistics.sumQuantity() else { return }
          samples.append(
            QuantitySampleValue(
              kind: kind,
              startDate: statistics.startDate,
              endDate: statistics.endDate,
              value: quantity.doubleValue(for: unit)
            )
          )
        }
        continuation.resume(returning: samples)
      }
      store.execute(query)
    }
  }

  private func latestSamples(
    kind: HealthMetricKind,
    identifier: HKQuantityTypeIdentifier,
    unit: HKUnit,
    start: Date,
    end: Date
  ) async throws -> [QuantitySampleValue] {
    let type = try quantityType(identifier)
    let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [.strictStartDate])
    let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: true)
    return try await withCheckedThrowingContinuation { continuation in
      let query = HKSampleQuery(
        sampleType: type,
        predicate: predicate,
        limit: HKObjectQueryNoLimit,
        sortDescriptors: [sort]
      ) { _, samples, error in
        if let error {
          continuation.resume(throwing: error)
          return
        }
        let values = (samples ?? []).compactMap { sample -> QuantitySampleValue? in
          guard let sample = sample as? HKQuantitySample else { return nil }
          return QuantitySampleValue(
            kind: kind,
            startDate: sample.startDate,
            endDate: sample.endDate,
            value: sample.quantity.doubleValue(for: unit)
          )
        }
        continuation.resume(returning: values)
      }
      store.execute(query)
    }
  }

  private func swimSamples(start: Date, end: Date, calendar: Calendar) async throws -> [SwimSampleValue] {
    async let distances = distanceSamples(start: start, end: end)
    async let strokes = strokeSamples(start: start, end: end)
    let distanceValues = try await distances
    let strokeValues = try await strokes
    let distanceBySecond = Dictionary(uniqueKeysWithValues: distanceValues.map {
      (Int($0.startDate.timeIntervalSince1970.rounded()), $0.meters)
    })
    var distancesByDay: [String: [Double]] = [:]
    for sample in distanceValues {
      let key = HealthExporterFormat.dayString(sample.startDate, calendar: calendar)
      distancesByDay[key, default: []].append(sample.meters)
    }
    let medianByDay = distancesByDay.mapValues { values -> Double in
      let sorted = values.sorted()
      return sorted[sorted.count / 2]
    }
    return strokeValues.map { stroke in
      let second = Int(stroke.startDate.timeIntervalSince1970.rounded())
      let day = HealthExporterFormat.dayString(stroke.startDate, calendar: calendar)
      let meters = distanceBySecond[second] ?? medianByDay[day] ?? 25
      return SwimSampleValue(startDate: stroke.startDate, meters: meters, stroke: stroke.stroke)
    }
  }

  private func distanceSamples(start: Date, end: Date) async throws -> [DistanceSample] {
    let type = try quantityType(.distanceSwimming)
    let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [.strictStartDate])
    let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)
    return try await withCheckedThrowingContinuation { continuation in
      let query = HKSampleQuery(
        sampleType: type,
        predicate: predicate,
        limit: HKObjectQueryNoLimit,
        sortDescriptors: [sort]
      ) { _, samples, error in
        if let error {
          continuation.resume(throwing: error)
          return
        }
        let values = (samples ?? []).compactMap { sample -> DistanceSample? in
          guard let sample = sample as? HKQuantitySample else { return nil }
          return DistanceSample(
            startDate: sample.startDate,
            meters: sample.quantity.doubleValue(for: .meter())
          )
        }
        continuation.resume(returning: values)
      }
      store.execute(query)
    }
  }

  private func strokeSamples(start: Date, end: Date) async throws -> [StrokeSample] {
    let type = try quantityType(.swimmingStrokeCount)
    let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [.strictStartDate])
    let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)
    return try await withCheckedThrowingContinuation { continuation in
      let query = HKSampleQuery(
        sampleType: type,
        predicate: predicate,
        limit: HKObjectQueryNoLimit,
        sortDescriptors: [sort]
      ) { _, samples, error in
        if let error {
          continuation.resume(throwing: error)
          return
        }
        let values = (samples ?? []).compactMap { sample -> StrokeSample? in
          guard let sample = sample as? HKQuantitySample, let stroke = Self.strokeName(sample.metadata) else {
            return nil
          }
          return StrokeSample(startDate: sample.startDate, stroke: stroke)
        }
        continuation.resume(returning: values)
      }
      store.execute(query)
    }
  }

  private static func strokeName(_ metadata: [String: Any]?) -> SwimStrokeName? {
    let value: Int?
    if let number = metadata?[HKMetadataKeySwimmingStrokeStyle] as? NSNumber {
      value = number.intValue
    } else {
      value = metadata?[HKMetadataKeySwimmingStrokeStyle] as? Int
    }
    switch value {
    case HKSwimmingStrokeStyle.freestyle.rawValue:
      return .freestyle
    case HKSwimmingStrokeStyle.breaststroke.rawValue:
      return .breaststroke
    case HKSwimmingStrokeStyle.backstroke.rawValue:
      return .backstroke
    case HKSwimmingStrokeStyle.butterfly.rawValue:
      return .butterfly
    case HKSwimmingStrokeStyle.mixed.rawValue:
      return .mixed
    case HKSwimmingStrokeStyle.kickboard.rawValue:
      return .kickboard
    default:
      return nil
    }
  }

  private func workouts(start: Date, end: Date) async throws -> [HKWorkout] {
    let type = HKObjectType.workoutType()
    let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [.strictStartDate])
    let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)
    return try await withCheckedThrowingContinuation { continuation in
      let query = HKSampleQuery(
        sampleType: type,
        predicate: predicate,
        limit: HKObjectQueryNoLimit,
        sortDescriptors: [sort]
      ) { _, samples, error in
        if let error {
          continuation.resume(throwing: error)
          return
        }
        continuation.resume(returning: (samples ?? []).compactMap { $0 as? HKWorkout })
      }
      store.execute(query)
    }
  }

  private func heartRateSamples(start: Date, end: Date) async throws -> [HeartRateSample] {
    let type = try quantityType(.heartRate)
    let unit = HKUnit.count().unitDivided(by: .minute())
    let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [.strictStartDate])
    let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)
    return try await withCheckedThrowingContinuation { continuation in
      let query = HKSampleQuery(
        sampleType: type,
        predicate: predicate,
        limit: HKObjectQueryNoLimit,
        sortDescriptors: [sort]
      ) { _, samples, error in
        if let error {
          continuation.resume(throwing: error)
          return
        }
        let values = (samples ?? []).compactMap { sample -> HeartRateSample? in
          guard let sample = sample as? HKQuantitySample else { return nil }
          let bpm = Int(sample.quantity.doubleValue(for: unit).rounded())
          guard bpm > 0 else { return nil }
          return HeartRateSample(
            startDate: sample.startDate,
            value: AppleHealthHeartRate(
              time: HealthExporterFormat.utcTimestampString(sample.startDate),
              bpm: bpm
            )
          )
        }
        continuation.resume(returning: values)
      }
      store.execute(query)
    }
  }

  private static func lowerBoundHeartRate(_ samples: [HeartRateSample], start: Date) -> Int {
    var low = 0
    var high = samples.count
    while low < high {
      let mid = (low + high) / 2
      if samples[mid].startDate < start {
        low = mid + 1
      } else {
        high = mid
      }
    }
    return low
  }

  private func workoutHeartRates(start: Date, end: Date) async throws -> [AppleHealthWorkout] {
    async let workoutValues = workouts(start: start, end: end)
    async let heartRateValues = heartRateSamples(start: start, end: end)
    let workouts = try await workoutValues
    let heartRates = try await heartRateValues
    var exports: [AppleHealthWorkout] = []
    for workout in workouts {
      var heartRate: [AppleHealthHeartRate] = []
      var index = Self.lowerBoundHeartRate(heartRates, start: workout.startDate)
      while index < heartRates.count {
        let sample = heartRates[index]
        if sample.startDate > workout.endDate { break }
        heartRate.append(sample.value)
        index += 1
      }
      if heartRate.isEmpty { continue }
      exports.append(
        AppleHealthWorkout(
          id: workout.uuid.uuidString,
          activity: Self.workoutActivityName(workout.workoutActivityType),
          start: HealthExporterFormat.utcTimestampString(workout.startDate),
          end: HealthExporterFormat.utcTimestampString(workout.endDate),
          durationS: Int(workout.duration.rounded()),
          heartRate: heartRate
        )
      )
    }
    return exports
  }

  private static func workoutActivityName(_ type: HKWorkoutActivityType) -> String {
    switch type {
    case .cycling:
      return "cycling"
    case .running:
      return "running"
    case .walking:
      return "walking"
    case .swimming:
      return "swimming"
    case .traditionalStrengthTraining:
      return "strength"
    case .functionalStrengthTraining:
      return "functionalStrength"
    case .highIntensityIntervalTraining:
      return "hiit"
    case .yoga:
      return "yoga"
    case .coreTraining:
      return "core"
    case .flexibility:
      return "flexibility"
    case .other:
      return "other"
    default:
      return "other"
    }
  }
}
#endif
