#if canImport(HealthKit)
import CoreLocation
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

  private struct TimedQuantitySample {
    let startDate: Date
    let endDate: Date
    let value: Double
  }

  private struct WorkoutExportValues {
    let workouts: [AppleHealthWorkout]
    let routes: [WorkoutRouteValue]
  }

  private struct WorkoutRunningDynamics {
    let strideLengthM: [TimedQuantitySample]
    let groundContactTimeMs: [TimedQuantitySample]
    let verticalOscillationCm: [TimedQuantitySample]
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
      .distanceWalkingRunning,
      .stepCount,
      .runningPower,
      .runningStrideLength,
      .runningGroundContactTime,
      .runningVerticalOscillation,
    ]
    return identifiers.compactMap { HKQuantityType.quantityType(forIdentifier: $0) }
  }

  private var allSampleTypes: [HKSampleType] {
    allQuantityTypes + [HKObjectType.workoutType(), HKSeriesType.workoutRoute()]
  }

  private var backgroundTriggerTypes: Set<HKSampleType> {
    let identifiers: [HKQuantityTypeIdentifier] = [
      .activeEnergyBurned,
      .dietaryEnergyConsumed,
      .bodyMass,
      .vo2Max,
    ]
    let quantities = identifiers.compactMap { HKQuantityType.quantityType(forIdentifier: $0) }
    return Set(quantities + [HKObjectType.workoutType(), HKSeriesType.workoutRoute()])
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

  func export(
    daysBack: Int = 180,
    routesSince: Date? = nil,
    now: Date = Date()
  ) async throws -> HealthExportPayload {
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
    async let workoutValues = workoutExports(
      workouts: queriedWorkouts,
      routesSince: routesSince ?? start
    )
    var quantitySamples = try await activeEnergy
    quantitySamples += try await basalEnergy
    quantitySamples += try await dietaryEnergy
    quantitySamples += try await bodyMass
    quantitySamples += try await vo2Max
    let swims = try await swimValues
    let exportedWorkouts = try await workoutValues
    return HealthExportPayload(
      document: HealthAggregator.document(
        quantitySamples: quantitySamples,
        swimSamples: swims.samples,
        swimSessions: swims.sessions,
        workouts: exportedWorkouts.workouts,
        generatedAt: now,
        calendar: calendar
      ),
      routes: exportedWorkouts.routes
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
    var strokesBySession: [String: [SwimStrokeIntervalValue]] = [:]
    for stroke in strokeValues {
      guard
        let scope = Self.swimmingScope(
          startDate: stroke.startDate,
          endDate: stroke.endDate,
          sourceBundleIdentifier: stroke.sourceBundleIdentifier,
          scopes: scopes
      )
      else { continue }
      strokesBySession[scope.session.id, default: []].append(
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
      )
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
      let stroke = Self.strokeName(
        startDate: distance.startDate,
        endDate: distance.endDate,
        events: scope.events
      )
      let strokeMetrics: SwimStrokeMetricsValue? = stroke == .kickboard
        ? nil
        : HealthAggregator.intervalStrokeMetrics(
          startDate: distance.startDate,
          endDate: distance.endDate,
          strokeSamples: strokesBySession[scope.session.id] ?? []
        )
      return SwimSampleValue(
        workoutID: scope.session.id,
        startDate: distance.startDate,
        endDate: distance.endDate,
        meters: distance.meters,
        strokeCount: strokeMetrics?.count,
        strokeTimeS: strokeMetrics?.timeS,
        stroke: stroke
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
          strokeSamples: strokesBySession[session.id] ?? []
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

  private func quantitySamples(
    workouts: [HKWorkout],
    identifier: HKQuantityTypeIdentifier,
    unit: HKUnit
  ) async throws -> [TimedQuantitySample] {
    guard !workouts.isEmpty else { return [] }
    let type = try quantityType(identifier)
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
        let values = (samples ?? []).compactMap { sample -> TimedQuantitySample? in
          guard let sample = sample as? HKQuantitySample else { return nil }
          let value = sample.quantity.doubleValue(for: unit)
          guard value.isFinite else { return nil }
          return TimedQuantitySample(
            startDate: sample.startDate,
            endDate: sample.endDate,
            value: value
          )
        }
        continuation.resume(returning: values)
      }
      store.execute(query)
    }
  }

  private func quantitySeriesSamples(
    workout: HKWorkout,
    identifier: HKQuantityTypeIdentifier,
    unit: HKUnit
  ) async throws -> [TimedQuantitySample] {
    let type = try quantityType(identifier)
    let predicate = HKSamplePredicate.quantitySample(
      type: type,
      predicate: HKQuery.predicateForObjects(from: workout)
    )
    let descriptor = HKQuantitySeriesSampleQueryDescriptor(
      predicate: predicate,
      options: [.orderByQuantitySampleStartDate]
    )
    var values: [TimedQuantitySample] = []
    for try await result in descriptor.results(for: store) {
      let value = result.quantity.doubleValue(for: unit)
      guard value.isFinite, value > 0 else { continue }
      values.append(
        TimedQuantitySample(
          startDate: result.dateInterval.start,
          endDate: result.dateInterval.end,
          value: value
        )
      )
    }
    return values
  }

  private func runningDynamics(
    workout: HKWorkout
  ) async throws -> WorkoutRunningDynamics {
    async let strideLengthM = quantitySeriesSamples(
      workout: workout,
      identifier: .runningStrideLength,
      unit: .meter()
    )
    async let groundContactTimeMs = quantitySeriesSamples(
      workout: workout,
      identifier: .runningGroundContactTime,
      unit: .secondUnit(with: .milli)
    )
    async let verticalOscillationCm = quantitySeriesSamples(
      workout: workout,
      identifier: .runningVerticalOscillation,
      unit: .meterUnit(with: .centi)
    )
    return try await WorkoutRunningDynamics(
      strideLengthM: strideLengthM,
      groundContactTimeMs: groundContactTimeMs,
      verticalOscillationCm: verticalOscillationCm
    )
  }

  private func runningDynamics(
    workouts: [HKWorkout]
  ) async throws -> [UUID: WorkoutRunningDynamics] {
    try await withThrowingTaskGroup(
      of: (UUID, WorkoutRunningDynamics).self,
      returning: [UUID: WorkoutRunningDynamics].self
    ) { group in
      for workout in workouts {
        group.addTask {
          (workout.uuid, try await self.runningDynamics(workout: workout))
        }
      }
      var values: [UUID: WorkoutRunningDynamics] = [:]
      for try await (id, dynamics) in group {
        values[id] = dynamics
      }
      return values
    }
  }

  private static func runningDynamicsSamples(
    _ samples: [TimedQuantitySample]
  ) -> [AppleHealthRunningDynamicsSample] {
    samples.map {
      AppleHealthRunningDynamicsSample(
        time: HealthExporterFormat.utcTimestampString($0.startDate),
        value: $0.value
      )
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

  private static func heartRates(
    _ samples: [HeartRateSample],
    workout: HKWorkout
  ) -> [HeartRateSample] {
    var output: [HeartRateSample] = []
    var index = lowerBoundHeartRate(samples, start: workout.startDate)
    while index < samples.count {
      let sample = samples[index]
      if sample.startDate > workout.endDate { break }
      output.append(sample)
      index += 1
    }
    return output
  }

  private static func quantities(
    _ samples: [TimedQuantitySample],
    workout: HKWorkout
  ) -> [TimedQuantitySample] {
    samples.filter {
      $0.startDate >= workout.startDate && $0.startDate <= workout.endDate
    }
  }

  private static func sum(
    workout: HKWorkout,
    type: HKQuantityType,
    unit: HKUnit
  ) -> Double? {
    let value = workout.statistics(for: type)?.sumQuantity()?.doubleValue(for: unit)
    guard let value, value.isFinite, value >= 0 else { return nil }
    return value
  }

  private static func average(
    workout: HKWorkout,
    type: HKQuantityType,
    unit: HKUnit
  ) -> Double? {
    let value = workout.statistics(for: type)?.averageQuantity()?.doubleValue(for: unit)
    guard let value, value.isFinite, value >= 0 else { return nil }
    return value
  }

  private static func mean(_ values: [Double]) -> Double? {
    guard !values.isEmpty else { return nil }
    return values.reduce(0, +) / Double(values.count)
  }

  private func workoutRoutes(for workout: HKWorkout) async throws -> [HKWorkoutRoute] {
    let type = HKSeriesType.workoutRoute()
    let predicate = HKQuery.predicateForObjects(from: workout)
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
        continuation.resume(returning: (samples ?? []).compactMap { $0 as? HKWorkoutRoute })
      }
      store.execute(query)
    }
  }

  private func locations(for workout: HKWorkout) async throws -> [CLLocation] {
    var byTimestamp: [Date: CLLocation] = [:]
    for route in try await workoutRoutes(for: workout) {
      let locations = HKWorkoutRouteQueryDescriptor(route).results(for: store)
      for try await location in locations {
        byTimestamp[location.timestamp] = location
      }
    }
    return byTimestamp.values.sorted { $0.timestamp < $1.timestamp }
  }

  private static func nearestHeartRate(
    at date: Date,
    samples: [HeartRateSample]
  ) -> Int? {
    let index = lowerBoundHeartRate(samples, start: date)
    let candidates = [index - 1, index].compactMap { candidate -> HeartRateSample? in
      guard samples.indices.contains(candidate) else { return nil }
      return samples[candidate]
    }
    guard let nearest = candidates.min(by: {
      abs($0.startDate.timeIntervalSince(date)) < abs($1.startDate.timeIntervalSince(date))
    }), abs(nearest.startDate.timeIntervalSince(date)) <= 10
    else { return nil }
    return nearest.bpm
  }

  private static func quantity(
    at date: Date,
    samples: [TimedQuantitySample]
  ) -> TimedQuantitySample? {
    var low = 0
    var high = samples.count
    while low < high {
      let mid = (low + high) / 2
      if samples[mid].startDate < date {
        low = mid + 1
      } else {
        high = mid
      }
    }
    let candidates = [low - 1, low].compactMap { index -> TimedQuantitySample? in
      guard samples.indices.contains(index) else { return nil }
      return samples[index]
    }
    if let containing = candidates.filter({ $0.startDate <= date && $0.endDate >= date }).min(by: {
      $0.endDate.timeIntervalSince($0.startDate) < $1.endDate.timeIntervalSince($1.startDate)
    }) {
      return containing
    }
    return candidates.min(by: {
      abs($0.startDate.timeIntervalSince(date)) < abs($1.startDate.timeIntervalSince(date))
    }).flatMap {
      abs($0.startDate.timeIntervalSince(date)) <= 5 ? $0 : nil
    }
  }

  private static func routePoint(
    _ location: CLLocation,
    heartRates: [HeartRateSample],
    powers: [TimedQuantitySample],
    steps: [TimedQuantitySample]
  ) -> WorkoutRoutePointValue? {
    let latitude = location.coordinate.latitude
    let longitude = location.coordinate.longitude
    guard
      latitude.isFinite,
      longitude.isFinite,
      (-90...90).contains(latitude),
      (-180...180).contains(longitude)
    else { return nil }
    let stepSample = quantity(at: location.timestamp, samples: steps)
    let stepDuration = stepSample.map { $0.endDate.timeIntervalSince($0.startDate) }
    let cadence = stepSample.flatMap { sample -> Double? in
      guard let stepDuration, stepDuration > 0 else { return nil }
      let value = sample.value / stepDuration * 60
      return value.isFinite && value > 0 ? value : nil
    }
    let power = quantity(at: location.timestamp, samples: powers)?.value
    let altitude = location.verticalAccuracy >= 0 && location.altitude.isFinite
      ? location.altitude
      : nil
    return WorkoutRoutePointValue(
      date: location.timestamp,
      latitude: latitude,
      longitude: longitude,
      altitudeM: altitude,
      heartRateBpm: nearestHeartRate(at: location.timestamp, samples: heartRates),
      cadenceSpm: cadence,
      powerW: power.flatMap { $0 > 0 ? $0 : nil }
    )
  }

  private func workoutRoute(
    for workout: HKWorkout,
    heartRates: [HeartRateSample],
    powers: [TimedQuantitySample],
    steps: [TimedQuantitySample]
  ) async throws -> WorkoutRouteValue? {
    let points = try await locations(for: workout).compactMap {
      Self.routePoint($0, heartRates: heartRates, powers: powers, steps: steps)
    }
    guard !points.isEmpty else { return nil }
    return WorkoutRouteValue(
      workoutID: workout.uuid.uuidString,
      activity: Self.workoutActivityName(workout.workoutActivityType),
      start: workout.startDate,
      points: points
    )
  }

  private func workoutExports(
    workouts: [HKWorkout],
    routesSince: Date
  ) async throws -> WorkoutExportValues {
    let runningWorkouts = workouts.filter { $0.workoutActivityType == .running }
    async let heartRateValues = heartRateSamples(workouts: workouts)
    async let powerValues = quantitySamples(
      workouts: runningWorkouts,
      identifier: .runningPower,
      unit: .watt()
    )
    async let stepValues = quantitySamples(
      workouts: runningWorkouts,
      identifier: .stepCount,
      unit: .count()
    )
    async let runningDynamicsValues = runningDynamics(workouts: runningWorkouts)
    let heartRates = try await heartRateValues
    let powers = try await powerValues
    let steps = try await stepValues
    let dynamicsByWorkoutID = try await runningDynamicsValues
    let distanceType = try quantityType(.distanceWalkingRunning)
    let energyType = try quantityType(.activeEnergyBurned)
    let heartRateType = try quantityType(.heartRate)
    let powerType = try quantityType(.runningPower)
    let stepType = try quantityType(.stepCount)
    let heartRateUnit = HKUnit.count().unitDivided(by: .minute())
    var routes: [WorkoutRouteValue] = []
    for workout in runningWorkouts where workout.startDate >= routesSince {
      let workoutHeartRates = Self.heartRates(heartRates, workout: workout)
      let workoutPowers = Self.quantities(powers, workout: workout)
      let workoutSteps = Self.quantities(steps, workout: workout)
      if let route = try? await workoutRoute(
        for: workout,
        heartRates: workoutHeartRates,
        powers: workoutPowers,
        steps: workoutSteps
      ) {
        routes.append(route)
      }
    }
    let routesByWorkoutID = Dictionary(uniqueKeysWithValues: routes.map { ($0.workoutID, $0) })
    var exports: [AppleHealthWorkout] = []
    for workout in workouts {
      let workoutHeartRates = Self.heartRates(heartRates, workout: workout)
      let workoutPowers = Self.quantities(powers, workout: workout)
      let workoutSteps = Self.quantities(steps, workout: workout)
      let dynamics = dynamicsByWorkoutID[workout.uuid]
      let durationS = Int(workout.duration.rounded())
      let averageHeartRate = Self.average(
        workout: workout,
        type: heartRateType,
        unit: heartRateUnit
      ) ?? Self.mean(workoutHeartRates.map { Double($0.bpm) })
      let averagePower = Self.average(
        workout: workout,
        type: powerType,
        unit: .watt()
      ) ?? Self.mean(workoutPowers.map(\.value))
      let stepCount: Double? = Self.sum(workout: workout, type: stepType, unit: .count())
        ?? (!workoutSteps.isEmpty ? workoutSteps.reduce(0) { $0 + $1.value } : nil)
      let cadence = stepCount.flatMap { durationS > 0 ? $0 / Double(durationS) * 60 : nil }
      let lapCount = workout.workoutEvents?.filter { $0.type == .lap }.count ?? 0
      let route = routesByWorkoutID[workout.uuid.uuidString]
      exports.append(
        AppleHealthWorkout(
          id: workout.uuid.uuidString,
          activity: Self.workoutActivityName(workout.workoutActivityType),
          start: HealthExporterFormat.utcTimestampString(workout.startDate),
          end: HealthExporterFormat.utcTimestampString(workout.endDate),
          durationS: durationS,
          elapsedTimeS: Int(workout.endDate.timeIntervalSince(workout.startDate).rounded()),
          distanceM: Self.sum(workout: workout, type: distanceType, unit: .meter()),
          activeEnergyKcal: Self.sum(workout: workout, type: energyType, unit: .kilocalorie()),
          averageHeartRateBpm: averageHeartRate.map { Int($0.rounded()) },
          averageRunningPowerW: averagePower.map { Int($0.rounded()) },
          averageCadenceSpm: cadence.map { Int($0.rounded()) },
          lapCount: lapCount > 0 ? lapCount : nil,
          source: workout.sourceRevision.source.name,
          device: workout.device?.name ?? workout.device?.model,
          gpxFile: route?.relativePath,
          heartRate: workoutHeartRates.map {
            AppleHealthHeartRate(
              time: HealthExporterFormat.utcTimestampString($0.startDate),
              bpm: $0.bpm
            )
          },
          strideLengthM: Self.runningDynamicsSamples(dynamics?.strideLengthM ?? []),
          groundContactTimeMs: Self.runningDynamicsSamples(dynamics?.groundContactTimeMs ?? []),
          verticalOscillationCm: Self.runningDynamicsSamples(dynamics?.verticalOscillationCm ?? [])
        )
      )
    }
    return WorkoutExportValues(workouts: exports, routes: routes)
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
