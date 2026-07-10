#if canImport(HealthKit)
import Foundation
import HealthKit

final class HealthObserverCompletion: @unchecked Sendable {
  private let handler: () -> Void
  private let lock = NSLock()
  private var isCompleted = false

  init(_ handler: @escaping () -> Void) {
    self.handler = handler
  }

  func complete() {
    lock.lock()
    guard !isCompleted else {
      lock.unlock()
      return
    }
    isCompleted = true
    lock.unlock()
    handler()
  }
}

final class HealthKitService: @unchecked Sendable {
  static let shared = HealthKitService()

  private struct DistanceSample {
    let startDate: Date
    let endDate: Date
    let meters: Double
    let sourceBundleIdentifier: String
  }

  private struct StrokeSample {
    let startDate: Date
    let endDate: Date
    let count: Double
    let sourceBundleIdentifier: String
  }

  private struct SwimmingScope {
    let session: SwimSessionValue
    let workout: HKWorkout
    let events: [HKWorkoutEvent]
  }

  private struct SwimExportValues {
    let samples: [SwimSampleValue]
    let sessions: [SwimSessionValue]
  }

  private struct HeartRateSample {
    let startDate: Date
    let bpm: Int
  }

  private let store = HKHealthStore()
  private let observerLock = NSLock()
  private var observerQuery: HKObserverQuery?

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

  private var backgroundTriggerTypes: Set<HKSampleType> {
    let identifiers: [HKQuantityTypeIdentifier] = [
      .activeEnergyBurned,
      .dietaryEnergyConsumed,
      .bodyMass,
      .vo2Max,
    ]
    let quantities = identifiers.compactMap { HKQuantityType.quantityType(forIdentifier: $0) }
    return Set(quantities + [HKObjectType.workoutType()])
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

  func startObservers(
    restart: Bool = false,
    onChange: @escaping @Sendable (HealthObserverCompletion) -> Void
  ) {
    observerLock.lock()
    defer { observerLock.unlock() }
    if restart {
      if let query = observerQuery {
        store.stop(query)
      }
      observerQuery = nil
    }
    if observerQuery == nil {
      let descriptors = allSampleTypes.map { HKQueryDescriptor(sampleType: $0, predicate: nil) }
      let query = HKObserverQuery(queryDescriptors: descriptors) { _, _, completion, error in
        let observerCompletion = HealthObserverCompletion(completion)
        guard error == nil else {
          observerCompletion.complete()
          return
        }
        onChange(observerCompletion)
      }
      observerQuery = query
      store.execute(query)
    }
    let backgroundTriggerTypes = backgroundTriggerTypes
    for type in allSampleTypes {
      if backgroundTriggerTypes.contains(type) {
        store.enableBackgroundDelivery(for: type, frequency: .hourly) { _, _ in }
      } else {
        store.disableBackgroundDelivery(for: type) { _, _ in }
      }
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
    async let workoutSamples = workouts(start: start, end: end)
    let queriedWorkouts = try await workoutSamples
    let swimmingScopes = try swimmingScopes(workouts: queriedWorkouts)
    async let swimValues = swimSamples(start: start, end: end, scopes: swimmingScopes)
    async let workouts = workoutHeartRates(workouts: queriedWorkouts)
    var quantitySamples = try await activeEnergy
    quantitySamples += try await basalEnergy
    quantitySamples += try await dietaryEnergy
    quantitySamples += try await bodyMass
    quantitySamples += try await vo2Max
    let swims = try await swimValues
    return HealthAggregator.document(
      quantitySamples: quantitySamples,
      swimSamples: swims.samples,
      swimSessions: swims.sessions,
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

  private func swimmingScopes(workouts: [HKWorkout]) throws -> [SwimmingScope] {
    let distanceType = try quantityType(.distanceSwimming)
    let strokeType = try quantityType(.swimmingStrokeCount)
    var scopes: [SwimmingScope] = []
    for workout in workouts {
      switch workout.workoutActivityType {
      case .swimming:
        scopes.append(
          SwimmingScope(
            session: SwimSessionValue(
              id: workout.uuid.uuidString,
              startDate: workout.startDate,
              endDate: workout.endDate,
              distanceMeters: Self.statisticsValue(
                workout.statistics(for: distanceType),
                unit: .meter()
              ),
              activeTimeS: workout.duration,
              strokeCount: Self.statisticsValue(
                workout.statistics(for: strokeType),
                unit: .count()
              ),
              strokeTimeS: nil,
              lapCount: Self.lapCount(workout.workoutEvents ?? [])
            ),
            workout: workout,
            events: workout.workoutEvents ?? []
          )
        )
      case .swimBikeRun:
        for activity in workout.workoutActivities where
          activity.workoutConfiguration.activityType == .swimming
        {
          guard let endDate = activity.endDate else { continue }
          scopes.append(
            SwimmingScope(
              session: SwimSessionValue(
                id: activity.uuid.uuidString,
                startDate: activity.startDate,
                endDate: endDate,
                distanceMeters: Self.statisticsValue(
                  activity.statistics(for: distanceType),
                  unit: .meter()
                ),
                activeTimeS: activity.duration,
                strokeCount: Self.statisticsValue(
                  activity.statistics(for: strokeType),
                  unit: .count()
                ),
                strokeTimeS: nil,
                lapCount: Self.lapCount(activity.workoutEvents)
              ),
              workout: workout,
              events: activity.workoutEvents
            )
          )
        }
      default:
        continue
      }
    }
    return scopes
  }

  private static func statisticsValue(_ statistics: HKStatistics?, unit: HKUnit) -> Double? {
    guard let value = statistics?.sumQuantity()?.doubleValue(for: unit), value.isFinite, value >= 0 else {
      return nil
    }
    return value
  }

  private static func lapCount(_ events: [HKWorkoutEvent]) -> Int? {
    let count = events.filter { $0.type == .lap }.count
    return count > 0 ? count : nil
  }

  private func swimSamples(
    start: Date,
    end: Date,
    scopes: [SwimmingScope]
  ) async throws -> SwimExportValues {
    guard !scopes.isEmpty else { return SwimExportValues(samples: [], sessions: []) }
    let workouts = Array(Dictionary(uniqueKeysWithValues: scopes.map {
      ($0.workout.uuid, $0.workout)
    }).values)
    async let distances = distanceSamples(start: start, end: end, workouts: workouts)
    async let strokes = strokeSamples(start: start, end: end, workouts: workouts)
    let distanceValues = try await distances
    let strokeValues = try await strokes
    var strokesBySession: [String: [StrokeSample]] = [:]
    for stroke in strokeValues {
      guard
        let scope = Self.swimmingScope(
          startDate: stroke.startDate,
          endDate: stroke.endDate,
          sourceBundleIdentifier: stroke.sourceBundleIdentifier,
          scopes: scopes
        )
      else { continue }
      strokesBySession[scope.session.id, default: []].append(stroke)
    }
    let samples = distanceValues.compactMap { distance -> SwimSampleValue? in
      guard distance.meters > 0 else { return nil }
      guard
        let scope = Self.swimmingScope(
          startDate: distance.startDate,
          endDate: distance.endDate,
          sourceBundleIdentifier: distance.sourceBundleIdentifier,
          scopes: scopes
        )
      else { return nil }
      return SwimSampleValue(
        workoutID: scope.session.id,
        startDate: distance.startDate,
        endDate: distance.endDate,
        meters: distance.meters,
        stroke: Self.strokeName(
          startDate: distance.startDate,
          endDate: distance.endDate,
          events: scope.events
        )
      )
    }
    let sessions = scopes.map { scope in
      let session = scope.session
      return SwimSessionValue(
        id: session.id,
        startDate: session.startDate,
        endDate: session.endDate,
        distanceMeters: session.distanceMeters,
        activeTimeS: session.activeTimeS,
        strokeCount: session.strokeCount,
        strokeTimeS: HealthAggregator.strokeTime(
          strokeSamples: (strokesBySession[session.id] ?? []).map { stroke in
            SwimStrokeIntervalValue(
              startDate: stroke.startDate,
              endDate: stroke.endDate,
              count: stroke.count,
              stroke: Self.strokeName(
                startDate: stroke.startDate,
                endDate: stroke.endDate,
                events: scope.events
              )
            )
          }
        ),
        lapCount: session.lapCount
      )
    }
    return SwimExportValues(samples: samples, sessions: sessions)
  }

  private static func swimmingScope(
    startDate: Date,
    endDate: Date,
    sourceBundleIdentifier: String,
    scopes: [SwimmingScope]
  ) -> SwimmingScope? {
    var candidates = scopes.filter {
      $0.session.startDate <= startDate && $0.session.endDate >= endDate
    }
    if candidates.isEmpty {
      candidates = scopes.filter {
        $0.session.startDate < endDate && $0.session.endDate > startDate
      }
    }
    let sourceMatches = candidates.filter {
      $0.workout.sourceRevision.source.bundleIdentifier == sourceBundleIdentifier
    }
    let matches = sourceMatches.isEmpty ? candidates : sourceMatches
    return matches.min {
      let leftDuration = $0.session.endDate.timeIntervalSince($0.session.startDate)
      let rightDuration = $1.session.endDate.timeIntervalSince($1.session.startDate)
      if leftDuration != rightDuration { return leftDuration < rightDuration }
      return $0.session.id < $1.session.id
    }
  }

  private static func strokeName(
    startDate: Date,
    endDate: Date,
    events: [HKWorkoutEvent]
  ) -> SwimStrokeName? {
    let laps = events.filter { $0.type == .lap && strokeName($0.metadata) != nil }
    let containing = laps.filter {
      $0.dateInterval.start <= startDate && $0.dateInterval.end >= endDate
    }
    let overlapping = laps.filter {
      $0.dateInterval.start < endDate && $0.dateInterval.end > startDate
    }
    let candidates = containing.isEmpty ? overlapping : containing
    let matches = candidates.isEmpty ? laps.filter {
      abs($0.dateInterval.start.timeIntervalSince(startDate)) < 1
    } : candidates
    let event = matches.min {
      let leftDuration = $0.dateInterval.duration
      let rightDuration = $1.dateInterval.duration
      if leftDuration != rightDuration { return leftDuration < rightDuration }
      return $0.dateInterval.start < $1.dateInterval.start
    }
    return strokeName(event?.metadata)
  }

  private static func samplePredicate(
    start: Date,
    end: Date,
    workouts: [HKWorkout]
  ) -> NSPredicate {
    let range = HKQuery.predicateForSamples(
      withStart: start,
      end: end,
      options: [.strictStartDate]
    )
    let associations = NSCompoundPredicate(
      orPredicateWithSubpredicates: workouts.map { HKQuery.predicateForObjects(from: $0) }
    )
    return NSCompoundPredicate(andPredicateWithSubpredicates: [range, associations])
  }

  private func distanceSamples(
    start: Date,
    end: Date,
    workouts: [HKWorkout]
  ) async throws -> [DistanceSample] {
    let type = try quantityType(.distanceSwimming)
    let predicate = Self.samplePredicate(start: start, end: end, workouts: workouts)
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
            endDate: sample.endDate,
            meters: sample.quantity.doubleValue(for: .meter()),
            sourceBundleIdentifier: sample.sourceRevision.source.bundleIdentifier
          )
        }
        continuation.resume(returning: values)
      }
      store.execute(query)
    }
  }

  private func strokeSamples(
    start: Date,
    end: Date,
    workouts: [HKWorkout]
  ) async throws -> [StrokeSample] {
    let type = try quantityType(.swimmingStrokeCount)
    let predicate = Self.samplePredicate(start: start, end: end, workouts: workouts)
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
          guard let sample = sample as? HKQuantitySample else { return nil }
          return StrokeSample(
            startDate: sample.startDate,
            endDate: sample.endDate,
            count: sample.quantity.doubleValue(for: .count()),
            sourceBundleIdentifier: sample.sourceRevision.source.bundleIdentifier
          )
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

  private func heartRateSamples(workouts: [HKWorkout]) async throws -> [HeartRateSample] {
    guard !workouts.isEmpty else { return [] }
    let type = try quantityType(.heartRate)
    let unit = HKUnit.count().unitDivided(by: .minute())
    let predicate = NSCompoundPredicate(
      orPredicateWithSubpredicates: workouts.map {
        HKQuery.predicateForSamples(
          withStart: $0.startDate,
          end: $0.endDate,
          options: [.strictStartDate]
        )
      }
    )
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
            bpm: bpm
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

  private func workoutHeartRates(workouts: [HKWorkout]) async throws -> [AppleHealthWorkout] {
    let heartRates = try await heartRateSamples(workouts: workouts)
    var exports: [AppleHealthWorkout] = []
    for workout in workouts {
      var heartRate: [AppleHealthHeartRate] = []
      var index = Self.lowerBoundHeartRate(heartRates, start: workout.startDate)
      while index < heartRates.count {
        let sample = heartRates[index]
        if sample.startDate > workout.endDate { break }
        heartRate.append(
          AppleHealthHeartRate(
            time: HealthExporterFormat.utcTimestampString(sample.startDate),
            bpm: sample.bpm
          )
        )
        index += 1
      }
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
    case .swimBikeRun:
      return "swimBikeRun"
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
