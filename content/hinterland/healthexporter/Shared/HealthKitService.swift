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
    ]
    return identifiers.compactMap { HKQuantityType.quantityType(forIdentifier: $0) }
  }

  func requestAuthorization() async throws {
    guard HKHealthStore.isHealthDataAvailable() else {
      throw HealthExporterError.healthDataUnavailable
    }
    let types = Set(allQuantityTypes.map { $0 as HKObjectType })
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
    for type in allQuantityTypes {
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
    var calendar = Calendar.autoupdatingCurrent
    calendar.timeZone = .autoupdatingCurrent
    let end = now
    let start = calendar.date(byAdding: .day, value: -daysBack, to: calendar.startOfDay(for: now))
      ?? calendar.startOfDay(for: now)
    var quantitySamples: [QuantitySampleValue] = []
    for item in cumulativeTypes {
      quantitySamples += try await cumulativeSamples(
        kind: item.0,
        identifier: item.1,
        unit: item.2,
        start: start,
        end: end,
        calendar: calendar
      )
    }
    for item in latestTypes {
      quantitySamples += try await latestSamples(
        kind: item.0,
        identifier: item.1,
        unit: item.2,
        start: start,
        end: end
      )
    }
    let swimSamples = try await swimSamples(start: start, end: end, calendar: calendar)
    return HealthAggregator.document(
      quantitySamples: quantitySamples,
      swimSamples: swimSamples,
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
    let distances = try await distanceSamples(start: start, end: end)
    let strokes = try await strokeSamples(start: start, end: end)
    let distanceBySecond = Dictionary(uniqueKeysWithValues: distances.map {
      (Int($0.startDate.timeIntervalSince1970.rounded()), $0.meters)
    })
    var distancesByDay: [String: [Double]] = [:]
    for sample in distances {
      let key = HealthExporterFormat.dayString(sample.startDate, calendar: calendar)
      distancesByDay[key, default: []].append(sample.meters)
    }
    let medianByDay = distancesByDay.mapValues { values -> Double in
      let sorted = values.sorted()
      return sorted[sorted.count / 2]
    }
    return strokes.map { stroke in
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
}
#endif
