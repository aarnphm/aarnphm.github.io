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

  func testSwimStatisticsStayAuthoritativeAndKickboardTimeIsExcludedFromStrokeRate() {
    let start = date(2026, 6, 19, 7, 0)
    let session = SwimSessionValue(
      id: "morning",
      startDate: start,
      endDate: start.addingTimeInterval(3600),
      distanceMeters: 100,
      activeTimeS: 75,
      strokeCount: 23,
      strokeTimeS: 40,
      lapCount: 4
    )
    let samples = [
      SwimSampleValue(
        workoutID: session.id,
        startDate: start,
        endDate: start.addingTimeInterval(30),
        meters: 25,
        strokeCount: 15,
        strokeTimeS: 30,
        stroke: .freestyle
      ),
      SwimSampleValue(
        workoutID: session.id,
        startDate: start,
        endDate: start.addingTimeInterval(30),
        meters: 20,
        strokeCount: 12,
        strokeTimeS: 30,
        stroke: nil
      ),
      SwimSampleValue(
        workoutID: session.id,
        startDate: start.addingTimeInterval(20),
        endDate: start.addingTimeInterval(40),
        meters: 25,
        strokeCount: 8,
        strokeTimeS: 20,
        stroke: nil
      ),
      SwimSampleValue(
        workoutID: session.id,
        startDate: start.addingTimeInterval(40),
        endDate: start.addingTimeInterval(75),
        meters: 25,
        stroke: .kickboard
      ),
    ]

    XCTAssertEqual(
      HealthAggregator.aggregateSwims(
        swimSamples: samples,
        swimSessions: [session],
        calendar: calendar
      ),
      [
        AppleHealthSwim(
          id: "morning",
          date: "2026-06-19",
          start: "2026-06-19T11:00:00Z",
          end: "2026-06-19T12:00:00Z",
          totalM: 100,
          laps: 4,
          activeTimeS: 75,
          strokeCount: 23,
          strokeTimeS: 40,
          strokes: ["freestyle": 25, "kickboard": 25],
          intervals: [
            AppleHealthSwimInterval(
              start: "2026-06-19T11:00:00Z",
              end: "2026-06-19T11:00:30Z",
              startElapsedS: 0,
              endElapsedS: 30,
              distanceM: 25,
              durationS: 30,
              strokeCount: 15,
              strokeTimeS: 30,
              stroke: .freestyle
            ),
            AppleHealthSwimInterval(
              start: "2026-06-19T11:00:20Z",
              end: "2026-06-19T11:00:40Z",
              startElapsedS: 20,
              endElapsedS: 40,
              distanceM: 25,
              durationS: 20,
              strokeCount: 8,
              strokeTimeS: 20,
              stroke: nil
            ),
            AppleHealthSwimInterval(
              start: "2026-06-19T11:00:40Z",
              end: "2026-06-19T11:01:15Z",
              startElapsedS: 40,
              endElapsedS: 75,
              distanceM: 25,
              durationS: 35,
              strokeCount: nil,
              stroke: .kickboard
            ),
          ]
        ),
      ]
    )
  }

  func testDistanceOnlySessionKeepsDistanceAndLeavesStrokeRateInputsMissing() {
    let start = date(2026, 6, 20, 7, 0)
    let session = SwimSessionValue(
      id: "distance-only",
      startDate: start,
      endDate: start.addingTimeInterval(120),
      distanceMeters: 200,
      activeTimeS: 120,
      strokeCount: nil,
      strokeTimeS: nil,
      lapCount: 2
    )
    let sample = SwimSampleValue(
      workoutID: session.id,
      startDate: start,
      endDate: start.addingTimeInterval(60),
      meters: 100,
      stroke: nil
    )

    XCTAssertEqual(
      HealthAggregator.aggregateSwims(
        swimSamples: [sample],
        swimSessions: [session],
        calendar: calendar
      ),
      [
        AppleHealthSwim(
          id: "distance-only",
          date: "2026-06-20",
          start: "2026-06-20T11:00:00Z",
          end: "2026-06-20T11:02:00Z",
          totalM: 200,
          laps: 2,
          activeTimeS: 120,
          strokeCount: nil,
          strokeTimeS: nil,
          strokes: [:],
          intervals: [
            AppleHealthSwimInterval(
              start: "2026-06-20T11:00:00Z",
              end: "2026-06-20T11:01:00Z",
              startElapsedS: 0,
              endElapsedS: 60,
              distanceM: 100,
              durationS: 60,
              strokeCount: nil,
              stroke: nil
            ),
          ]
        ),
      ]
    )
  }

  func testSessionStatisticsProduceSwimWithoutExpandedSamples() {
    let start = date(2026, 6, 21, 7, 0)
    let session = SwimSessionValue(
      id: "triathlon-swim-leg",
      startDate: start,
      endDate: start.addingTimeInterval(1800),
      distanceMeters: 1500,
      activeTimeS: 1750,
      strokeCount: 800,
      strokeTimeS: nil,
      lapCount: 60
    )

    XCTAssertEqual(
      HealthAggregator.aggregateSwims(
        swimSamples: [],
        swimSessions: [session],
        calendar: calendar
      ),
      [
        AppleHealthSwim(
          id: "triathlon-swim-leg",
          date: "2026-06-21",
          start: "2026-06-21T11:00:00Z",
          end: "2026-06-21T11:30:00Z",
          totalM: 1500,
          laps: 60,
          activeTimeS: 1750,
          strokeCount: 800,
          strokeTimeS: nil,
          strokes: [:]
        ),
      ]
    )
  }

  func testStrokeTimeUnionsPositiveIntervalsAndExcludesKickboard() {
    let start = date(2026, 6, 22, 7, 0)
    let intervals = [
      SwimStrokeIntervalValue(
        startDate: start,
        endDate: start.addingTimeInterval(30),
        count: 18,
        stroke: .freestyle
      ),
      SwimStrokeIntervalValue(
        startDate: start.addingTimeInterval(20),
        endDate: start.addingTimeInterval(40),
        count: 10,
        stroke: nil
      ),
      SwimStrokeIntervalValue(
        startDate: start.addingTimeInterval(40),
        endDate: start.addingTimeInterval(70),
        count: 1,
        stroke: .kickboard
      ),
      SwimStrokeIntervalValue(
        startDate: start.addingTimeInterval(70),
        endDate: start.addingTimeInterval(100),
        count: 0,
        stroke: nil
      ),
    ]

    XCTAssertEqual(HealthAggregator.strokeTime(strokeSamples: intervals), 40)
    XCTAssertNil(HealthAggregator.strokeTime(strokeSamples: Array(intervals.suffix(2))))
  }

  func testIntervalStrokeMetricsUsesExactBoundsAndDedupesCounts() {
    let start = date(2026, 6, 22, 7, 0)
    let samples = [
      SwimStrokeIntervalValue(
        startDate: start,
        endDate: start.addingTimeInterval(30),
        count: 18,
        stroke: .freestyle
      ),
      SwimStrokeIntervalValue(
        startDate: start,
        endDate: start.addingTimeInterval(30),
        count: 17,
        stroke: nil
      ),
      SwimStrokeIntervalValue(
        startDate: start.addingTimeInterval(5),
        endDate: start.addingTimeInterval(25),
        count: 40,
        stroke: .freestyle
      ),
    ]

    XCTAssertEqual(
      HealthAggregator.intervalStrokeMetrics(
        startDate: start,
        endDate: start.addingTimeInterval(30),
        strokeSamples: samples
      ),
      SwimStrokeMetricsValue(count: 18, timeS: 30)
    )
  }

  func testIntervalStrokeMetricsProratesContainingSample() {
    let start = date(2026, 6, 22, 7, 0)
    let sample = SwimStrokeIntervalValue(
      startDate: start.addingTimeInterval(-10),
      endDate: start.addingTimeInterval(40),
      count: 50,
      stroke: .freestyle
    )

    XCTAssertEqual(
      HealthAggregator.intervalStrokeMetrics(
        startDate: start,
        endDate: start.addingTimeInterval(30),
        strokeSamples: [sample]
      ),
      SwimStrokeMetricsValue(count: 30, timeS: 30)
    )
  }

  func testIntervalStrokeMetricsProratesPartialOverlapAndLeavesKickboardGap() {
    let start = date(2026, 6, 22, 7, 0)
    let samples = [
      SwimStrokeIntervalValue(
        startDate: start,
        endDate: start.addingTimeInterval(10),
        count: 10,
        stroke: .freestyle
      ),
      SwimStrokeIntervalValue(
        startDate: start,
        endDate: start.addingTimeInterval(10),
        count: 8,
        stroke: .freestyle
      ),
      SwimStrokeIntervalValue(
        startDate: start.addingTimeInterval(5),
        endDate: start.addingTimeInterval(20),
        count: 30,
        stroke: .freestyle
      ),
      SwimStrokeIntervalValue(
        startDate: start.addingTimeInterval(20),
        endDate: start.addingTimeInterval(25),
        count: 50,
        stroke: .kickboard
      ),
      SwimStrokeIntervalValue(
        startDate: start.addingTimeInterval(25),
        endDate: start.addingTimeInterval(35),
        count: 20,
        stroke: .breaststroke
      ),
    ]

    XCTAssertEqual(
      HealthAggregator.intervalStrokeMetrics(
        startDate: start,
        endDate: start.addingTimeInterval(30),
        strokeSamples: samples
      ),
      SwimStrokeMetricsValue(count: 40, timeS: 25)
    )
    XCTAssertNil(
      HealthAggregator.intervalStrokeMetrics(
        startDate: start.addingTimeInterval(20),
        endDate: start.addingTimeInterval(25),
        strokeSamples: samples
      )
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

    XCTAssertEqual(document.version, 6)
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
    let base = date(2026, 6, 19, 7, 30)
    let sessionStart = base.addingTimeInterval(0.24)
    let intervalStart = base.addingTimeInterval(5.91)
    let intervalEnd = intervalStart.addingTimeInterval(26.66)
    let document = HealthAggregator.document(
      quantitySamples: [],
      swimSamples: [
        SwimSampleValue(
          workoutID: "morning",
          startDate: intervalStart,
          endDate: intervalEnd,
          meters: 22.86,
          strokeCount: 13.36,
          strokeTimeS: 26.66,
          stroke: .freestyle
        ),
      ],
      swimSessions: [
        SwimSessionValue(
          id: "morning",
          startDate: sessionStart,
          endDate: intervalEnd,
          distanceMeters: 22.86,
          activeTimeS: 26.66,
          strokeCount: 13.36,
          strokeTimeS: 26.66,
          lapCount: 1
        ),
      ],
      generatedAt: sessionStart,
      calendar: calendar
    )

    let writtenURL = try writer.write(document)
    let restored = try writer.read()

    XCTAssertEqual(
      document.swims.first?.intervals,
      [
        AppleHealthSwimInterval(
          start: "2026-06-19T11:30:05Z",
          end: "2026-06-19T11:30:32Z",
          startElapsedS: 5.7,
          endElapsedS: 32.3,
          distanceM: 22.9,
          durationS: 26.7,
          strokeCount: 13.4,
          strokeTimeS: 26.7,
          stroke: .freestyle
        ),
      ]
    )
    XCTAssertEqual(restored?.document, document)
    XCTAssertEqual(restored?.url, writtenURL)
  }

  func testVersionTwoSwimsDecodeWithMigrationDefaults() throws {
    let json = """
      {
        "version": 2,
        "generatedAt": "2026-06-19T07:30:00-04:00",
        "timezone": "America/Toronto",
        "days": [],
        "swims": [
          {
            "date": "2026-06-19",
            "totalM": 1500,
            "laps": 60,
            "strokes": { "freestyle": 1500 }
          }
        ],
        "workouts": []
      }
      """

    let document = try JSONDecoder().decode(HealthExportDocument.self, from: Data(json.utf8))

    XCTAssertEqual(
      document.swims,
      [
        AppleHealthSwim(
          id: "2026-06-19",
          date: "2026-06-19",
          start: nil,
          end: nil,
          totalM: 1500,
          laps: 60,
          activeTimeS: nil,
          strokeCount: nil,
          strokeTimeS: nil,
          strokes: ["freestyle": 1500]
        ),
      ]
    )
  }

  func testVersionThreeSwimsDecodeWithoutIntervals() throws {
    let json = """
      {
        "version": 3,
        "generatedAt": "2026-06-19T07:30:00-04:00",
        "timezone": "America/Toronto",
        "days": [],
        "swims": [
          {
            "id": "morning",
            "date": "2026-06-19",
            "start": "2026-06-19T11:00:00Z",
            "end": "2026-06-19T12:00:00Z",
            "totalM": 1500,
            "laps": 60,
            "activeTimeS": 1800,
            "strokeCount": 960,
            "strokeTimeS": 1700,
            "strokes": { "freestyle": 1500 }
          }
        ],
        "workouts": []
      }
      """

    let document = try JSONDecoder().decode(HealthExportDocument.self, from: Data(json.utf8))

    XCTAssertEqual(document.swims.first?.intervals, [])
  }

  func testVersionFourIntervalsDecodeWithoutDuration() throws {
    let json = """
      {
        "version": 4,
        "generatedAt": "2026-06-19T07:30:00-04:00",
        "timezone": "America/Toronto",
        "days": [],
        "swims": [
          {
            "id": "morning",
            "date": "2026-06-19",
            "start": "2026-06-19T11:00:00Z",
            "end": "2026-06-19T12:00:00Z",
            "totalM": 1500,
            "laps": 60,
            "activeTimeS": 1800,
            "strokeCount": 960,
            "strokeTimeS": 1700,
            "strokes": { "freestyle": 1500 },
            "intervals": [
              {
                "start": "2026-06-19T11:00:00Z",
                "end": "2026-06-19T11:00:27Z",
                "distanceM": 25,
                "strokeCount": 14,
                "strokeTimeS": 27,
                "stroke": "freestyle"
              }
            ]
          }
        ],
        "workouts": []
      }
      """

    let document = try JSONDecoder().decode(HealthExportDocument.self, from: Data(json.utf8))

    XCTAssertEqual(
      document.swims.first?.intervals,
      [
        AppleHealthSwimInterval(
          start: "2026-06-19T11:00:00Z",
          end: "2026-06-19T11:00:27Z",
          distanceM: 25,
          durationS: nil,
          strokeCount: 14,
          strokeTimeS: 27,
          stroke: .freestyle
        ),
      ]
    )
  }

  func testVersionFiveIntervalsDecodeWithoutElapsedOffsets() throws {
    let json = """
      {
        "version": 5,
        "generatedAt": "2026-06-19T07:30:00-04:00",
        "timezone": "America/Toronto",
        "days": [],
        "swims": [
          {
            "id": "morning",
            "date": "2026-06-19",
            "start": "2026-06-19T11:00:00Z",
            "end": "2026-06-19T12:00:00Z",
            "totalM": 1500,
            "laps": 60,
            "activeTimeS": 1800,
            "strokeCount": 960,
            "strokeTimeS": 1700,
            "strokes": { "freestyle": 1500 },
            "intervals": [
              {
                "start": "2026-06-19T11:00:08Z",
                "end": "2026-06-19T11:00:35Z",
                "distanceM": 25,
                "durationS": 27,
                "strokeCount": 14,
                "strokeTimeS": 27,
                "stroke": "freestyle"
              }
            ]
          }
        ],
        "workouts": []
      }
      """

    let document = try JSONDecoder().decode(HealthExportDocument.self, from: Data(json.utf8))

    XCTAssertEqual(
      document.swims.first?.intervals,
      [
        AppleHealthSwimInterval(
          start: "2026-06-19T11:00:08Z",
          end: "2026-06-19T11:00:35Z",
          startElapsedS: nil,
          endElapsedS: nil,
          distanceM: 25,
          durationS: 27,
          strokeCount: 14,
          strokeTimeS: 27,
          stroke: .freestyle
        ),
      ]
    )
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

  func testExportStateUsesTheLastSuccessfulSyncWithoutReadingTheExport() {
    let suiteName = "HealthExportStateTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suiteName)!
    defer { defaults.removePersistentDomain(forName: suiteName) }
    let store = HealthExportStateStore(defaults: defaults)
    let now = date(2026, 7, 9, 12, 0)

    XCTAssertEqual(store.state(maxAge: 3600, now: now), .stale)

    store.markSuccessful(at: now.addingTimeInterval(-3599))
    XCTAssertEqual(store.state(maxAge: 3600, now: now), .exported)

    store.markSuccessful(at: now.addingTimeInterval(-3600))
    XCTAssertEqual(store.state(maxAge: 3600, now: now), .stale)
  }

  private func date(_ year: Int, _ month: Int, _ day: Int, _ hour: Int, _ minute: Int) -> Date {
    calendar.date(from: DateComponents(year: year, month: month, day: day, hour: hour, minute: minute))!
  }
}
