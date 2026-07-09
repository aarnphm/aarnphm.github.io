import Foundation
import XCTest

final class HealthAggregatorTests: XCTestCase {
  private var calendar: Calendar {
    var calendar = Calendar(identifier: .gregorian)
    calendar.timeZone = TimeZone(identifier: "America/Toronto")!
    return calendar
  }

  func testCumulativeDailyCaloriesRoundAndBurnTotals() {
    let day = date(2026, 6, 19, 8, 0)
    let samples = [
      QuantitySampleValue(kind: .activeEnergy, startDate: day, endDate: day, value: 120.4),
      QuantitySampleValue(kind: .activeEnergy, startDate: day, endDate: day, value: 80.2),
      QuantitySampleValue(kind: .basalEnergy, startDate: day, endDate: day, value: 1700.4),
      QuantitySampleValue(kind: .dietaryEnergy, startDate: day, endDate: day, value: 2600.5),
    ]

    XCTAssertEqual(
      HealthAggregator.aggregateDays(quantitySamples: samples, calendar: calendar),
      [
        AppleHealthDay(
          date: "2026-06-19",
          burnKcal: 1901,
          activeKcal: 201,
          intakeKcal: 2601,
          weightKg: nil,
          vo2max: nil
        ),
      ]
    )
  }

  func testLatestSameDayWeightAndVo2MaxWin() {
    let morning = date(2026, 6, 19, 7, 0)
    let night = date(2026, 6, 19, 22, 0)
    let samples = [
      QuantitySampleValue(kind: .bodyMass, startDate: morning, endDate: morning, value: 72.82),
      QuantitySampleValue(kind: .bodyMass, startDate: night, endDate: night, value: 72.34),
      QuantitySampleValue(kind: .vo2Max, startDate: morning, endDate: morning, value: 49.23),
      QuantitySampleValue(kind: .vo2Max, startDate: night, endDate: night, value: 50.16),
    ]

    XCTAssertEqual(
      HealthAggregator.aggregateDays(quantitySamples: samples, calendar: calendar),
      [
        AppleHealthDay(
          date: "2026-06-19",
          burnKcal: nil,
          activeKcal: nil,
          intakeKcal: nil,
          weightKg: 72.3,
          vo2max: 50.2
        ),
      ]
    )
  }

  func testMissingDataStaysNil() {
    let day = date(2026, 6, 19, 8, 0)
    let samples = [
      QuantitySampleValue(kind: .activeEnergy, startDate: day, endDate: day, value: 10),
    ]

    XCTAssertEqual(
      HealthAggregator.aggregateDays(quantitySamples: samples, calendar: calendar),
      [
        AppleHealthDay(
          date: "2026-06-19",
          burnKcal: 10,
          activeKcal: 10,
          intakeKcal: nil,
          weightKg: nil,
          vo2max: nil
        ),
      ]
    )
  }

  func testSwimExportRequiresStrokeData() {
    let start = date(2026, 6, 19, 7, 0)
    let samples = [
      SwimSampleValue(startDate: start, meters: 25, stroke: .freestyle),
      SwimSampleValue(startDate: start.addingTimeInterval(30), meters: 25, stroke: .breaststroke),
      SwimSampleValue(startDate: start.addingTimeInterval(60), meters: 25, stroke: nil),
    ]

    XCTAssertEqual(
      HealthAggregator.aggregateSwims(swimSamples: samples, calendar: calendar),
      [
        AppleHealthSwim(
          date: "2026-06-19",
          totalM: 50,
          laps: 2,
          strokes: ["breaststroke": 25, "freestyle": 25]
        ),
      ]
    )
  }

  func testDocumentKeepsTimezoneAndMetadata() {
    let generatedAt = date(2026, 6, 19, 7, 30)
    let document = HealthAggregator.document(
      quantitySamples: [],
      swimSamples: [],
      generatedAt: generatedAt,
      calendar: calendar
    )

    XCTAssertEqual(document.version, 2)
    XCTAssertEqual(document.generatedAt, "2026-06-19T07:30:00-04:00")
    XCTAssertEqual(document.timezone, "America/Toronto")
    XCTAssertEqual(document.days, [])
    XCTAssertEqual(document.swims, [])
    XCTAssertEqual(document.workouts, [])
  }

  func testDocumentCarriesWorkoutHeartRateStreams() {
    let generatedAt = date(2026, 6, 19, 7, 30)
    let workout = AppleHealthWorkout(
      id: "7E0BEF46-8C0E-4E08-8E2B-0F2E0A1C9E63",
      activity: "cycling",
      start: "2026-07-01T01:11:00Z",
      end: "2026-07-01T02:07:45Z",
      durationS: 3405,
      heartRate: [
        AppleHealthHeartRate(time: "2026-07-01T01:11:04Z", bpm: 118),
        AppleHealthHeartRate(time: "2026-07-01T01:11:09Z", bpm: 122),
      ]
    )
    let document = HealthAggregator.document(
      quantitySamples: [],
      swimSamples: [],
      workouts: [workout],
      generatedAt: generatedAt,
      calendar: calendar
    )

    XCTAssertEqual(document.workouts, [workout])
  }

  func testWriterRestoresTheLastExport() throws {
    let container = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: container) }
    let writer = HealthExportWriter(containerURL: container)
    let document = HealthAggregator.document(
      quantitySamples: [],
      swimSamples: [],
      generatedAt: date(2026, 6, 19, 7, 30),
      calendar: calendar
    )

    let writtenURL = try writer.write(document)
    let restored = try writer.read()

    XCTAssertEqual(restored?.document, document)
    XCTAssertEqual(restored?.url, writtenURL)
  }

  func testRecentExportReplacesOnlyTheRecentWindow() {
    let oldDay = AppleHealthDay(
      date: "2026-06-17",
      burnKcal: 1800,
      activeKcal: 200,
      intakeKcal: nil,
      weightKg: nil,
      vo2max: nil
    )
    let staleDay = AppleHealthDay(
      date: "2026-06-18",
      burnKcal: 1900,
      activeKcal: 300,
      intakeKcal: nil,
      weightKg: nil,
      vo2max: nil
    )
    let updatedDay = AppleHealthDay(
      date: "2026-06-18",
      burnKcal: 2100,
      activeKcal: 500,
      intakeKcal: 2300,
      weightKg: nil,
      vo2max: nil
    )
    let oldWorkout = AppleHealthWorkout(
      id: "old",
      activity: "running",
      start: "2026-06-17T12:00:00Z",
      end: "2026-06-17T13:00:00Z",
      durationS: 3600,
      heartRate: []
    )
    let staleWorkout = AppleHealthWorkout(
      id: "stale",
      activity: "cycling",
      start: "2026-06-18T12:00:00Z",
      end: "2026-06-18T13:00:00Z",
      durationS: 3600,
      heartRate: []
    )
    let updatedWorkout = AppleHealthWorkout(
      id: "updated",
      activity: "cycling",
      start: "2026-06-18T12:00:00Z",
      end: "2026-06-18T13:00:00Z",
      durationS: 3600,
      heartRate: []
    )
    let previous = HealthExportDocument(
      version: 2,
      generatedAt: "2026-06-18T12:00:00-04:00",
      timezone: "America/Toronto",
      days: [oldDay, staleDay],
      swims: [],
      workouts: [oldWorkout, staleWorkout]
    )
    let recent = HealthExportDocument(
      version: 2,
      generatedAt: "2026-06-19T12:00:00-04:00",
      timezone: "America/Toronto",
      days: [updatedDay],
      swims: [],
      workouts: [updatedWorkout]
    )

    let merged = previous.replacingRecent(
      with: recent,
      dayCutoff: "2026-06-18",
      timestampCutoff: "2026-06-18T04:00:00Z"
    )

    XCTAssertEqual(merged.generatedAt, recent.generatedAt)
    XCTAssertEqual(merged.days, [oldDay, updatedDay])
    XCTAssertEqual(merged.workouts, [oldWorkout, updatedWorkout])
  }

  private func date(_ year: Int, _ month: Int, _ day: Int, _ hour: Int, _ minute: Int) -> Date {
    calendar.date(from: DateComponents(year: year, month: month, day: day, hour: hour, minute: minute))!
  }
}
