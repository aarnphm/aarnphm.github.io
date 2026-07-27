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
        )
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
        )
      ]
    )
  }

  func testMissingDataStaysNil() {
    let day = date(2026, 6, 19, 8, 0)
    let samples = [
      QuantitySampleValue(kind: .activeEnergy, startDate: day, endDate: day, value: 10)
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
        )
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
        )
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
            )
          ]
        )
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
      lapCount: 60,
      location: .openWater,
      waterTemperatureC: 14.4
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
          strokes: [:],
          location: .openWater,
          waterTemperatureC: 14.4
        )
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

    XCTAssertEqual(document.version, 10)
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
      ],
      strideLengthM: [
        AppleHealthRunningDynamicsSample(time: "2026-07-01T01:11:04Z", value: 1.18),
        AppleHealthRunningDynamicsSample(time: "2026-07-01T01:11:09Z", value: 1.21),
      ],
      groundContactTimeMs: [
        AppleHealthRunningDynamicsSample(time: "2026-07-01T01:11:04Z", value: 241)
      ],
      verticalOscillationCm: [
        AppleHealthRunningDynamicsSample(time: "2026-07-01T01:11:04Z", value: 9.8)
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

  func testHeartRateTimesPreserveSubsecondPrecision() {
    let start = date(2026, 7, 26, 11, 39)
    let heartRate = [
      AppleHealthHeartRate(
        time: HealthExporterFormat.utcFractionalTimestampString(
          start.addingTimeInterval(6.125)
        ),
        bpm: 82
      ),
      AppleHealthHeartRate(
        time: HealthExporterFormat.utcFractionalTimestampString(
          start.addingTimeInterval(6.875)
        ),
        bpm: 177
      ),
    ]

    XCTAssertEqual(heartRate[0].time, "2026-07-26T15:39:06.125Z")
    XCTAssertEqual(heartRate[1].time, "2026-07-26T15:39:06.875Z")
    XCTAssertNotEqual(heartRate[0].time, heartRate[1].time)
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
        )
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
        )
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
        )
      ]
    )
    XCTAssertEqual(restored?.document, document)
    XCTAssertEqual(restored?.url, writtenURL)
  }

  func testWriterRoundTripsNestedMultisportActivities() throws {
    let container = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: container) }
    let writer = HealthExportWriter(containerURL: container)
    let activities = [
      AppleHealthWorkout.Activity(
        id: "54A05FB9-60F9-4D04-A6B1-7239AAB19DCC",
        activity: "swimming",
        start: "2026-07-26T12:43:52Z",
        end: "2026-07-26T13:25:11Z",
        durationS: 2467,
        elapsedTimeS: 2479,
        distanceM: 438.8,
        activeEnergyKcal: 562.4,
        averageHeartRateBpm: 151,
        averageCadencePerMinute: 32,
        lapCount: 9,
        swimmingLocation: .openWater,
        waterTemperatureC: 14.4
      ),
      AppleHealthWorkout.Activity(
        id: "9C718390-60C2-47EC-A666-DCDBDD7FD3D1",
        activity: "transition",
        start: "2026-07-26T13:25:11Z",
        end: "2026-07-26T13:27:15Z",
        durationS: 124,
        elapsedTimeS: 124,
        activeEnergyKcal: 12.8,
        averageHeartRateBpm: 158
      ),
      AppleHealthWorkout.Activity(
        id: "F63E15EA-B312-4982-B08F-CBE758BEAD20",
        activity: "cycling",
        start: "2026-07-26T13:27:15Z",
        end: "2026-07-26T14:44:41Z",
        durationS: 4646,
        elapsedTimeS: 4646,
        distanceM: 39_473.8,
        activeEnergyKcal: 998.6,
        averageHeartRateBpm: 164,
        averagePowerW: 210,
        averageCadencePerMinute: 81,
        lapCount: 1
      ),
      AppleHealthWorkout.Activity(
        id: "CBF5531D-1D7D-4D20-AEAB-A517FC45AA2E",
        activity: "transition",
        start: "2026-07-26T14:44:41Z",
        end: "2026-07-26T14:47:40Z",
        durationS: 179,
        elapsedTimeS: 179,
        activeEnergyKcal: 18.2,
        averageHeartRateBpm: 166
      ),
      AppleHealthWorkout.Activity(
        id: "38AA52BA-D02A-4D93-9314-5025459103D1",
        activity: "running",
        start: "2026-07-26T14:47:40Z",
        end: "2026-07-26T15:41:19Z",
        durationS: 3219,
        elapsedTimeS: 3219,
        distanceM: 9495.1,
        activeEnergyKcal: 1195.4,
        averageHeartRateBpm: 169,
        averagePowerW: 273,
        averageCadencePerMinute: 160,
        lapCount: 1
      ),
    ]
    let workout = AppleHealthWorkout(
      id: "501220B5-1D70-4C31-94FB-0CB0F712740B",
      activity: "swimBikeRun",
      start: "2026-07-26T12:43:52Z",
      end: "2026-07-26T15:41:27Z",
      durationS: 10_502,
      elapsedTimeS: 10_654,
      activeEnergyKcal: 2787.5,
      averageHeartRateBpm: 164,
      source: "appl-watch-ultra-3",
      device: "Apple Watch",
      heartRate: [],
      activities: activities
    )
    let document = HealthExportDocument(
      version: HealthExportDocument.currentVersion,
      generatedAt: "2026-07-26T13:56:05-04:00",
      timezone: "America/Toronto",
      days: [],
      swims: [],
      workouts: [workout]
    )

    _ = try writer.write(document)
    let restored = try XCTUnwrap(writer.read()?.document)

    XCTAssertEqual(restored.workouts.first?.activities, activities)
    XCTAssertEqual(restored, document)
  }

  func testWriterExportsRunGPXWithSensorExtensions() throws {
    let container = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: container) }
    let writer = HealthExportWriter(containerURL: container)
    let start = date(2026, 7, 10, 18, 10)
    let workoutID = "7E0BEF46-8C0E-4E08-8E2B-0F2E0A1C9E63"
    let workout = AppleHealthWorkout(
      id: workoutID,
      activity: "running",
      start: "2026-07-10T22:10:00Z",
      end: "2026-07-10T22:10:01Z",
      durationS: 1,
      elapsedTimeS: 1,
      distanceM: 4.2,
      activeEnergyKcal: 1.2,
      averageHeartRateBpm: 156,
      averageRunningPowerW: 278,
      averageCadenceSpm: 159,
      lapCount: 1,
      source: "Strava",
      device: "Apple Watch Ultra 3 49mm",
      gpxFile: "GPX/\(workoutID).gpx",
      heartRate: []
    )
    let document = HealthAggregator.document(
      quantitySamples: [],
      swimSamples: [],
      workouts: [workout],
      generatedAt: start,
      calendar: calendar
    )
    let route = WorkoutRouteValue(
      workoutID: workoutID,
      activity: "running",
      start: start,
      points: [
        WorkoutRoutePointValue(
          date: start,
          latitude: 43.645581,
          longitude: -79.401239,
          altitudeM: 88.9,
          heartRateBpm: 156,
          cadenceRpm: 80,
          powerW: 278
        )
      ]
    )

    let documentURL = try writer.write(document, routes: [route])
    let gpxURL = documentURL.deletingLastPathComponent().appendingPathComponent(route.relativePath)
    let gpx = try String(contentsOf: gpxURL, encoding: .utf8)

    XCTAssertTrue(FileManager.default.fileExists(atPath: gpxURL.path))
    XCTAssertTrue(gpx.contains("<trkpt lat=\"43.6455810\" lon=\"-79.4012390\">"))
    XCTAssertTrue(gpx.contains("<gpxtpx:hr>156</gpxtpx:hr>"))
    XCTAssertTrue(gpx.contains("<gpxtpx:cad>80</gpxtpx:cad>"))
    XCTAssertTrue(gpx.contains("<power>278</power>"))
    XCTAssertTrue(XMLParser(data: Data(gpx.utf8)).parse())
  }

  func testWriterPreservesDistinctSubsecondRoutePointTimes() throws {
    let container = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: container) }
    let writer = HealthExportWriter(containerURL: container)
    let start = date(2026, 7, 10, 18, 10)
    let document = HealthAggregator.document(
      quantitySamples: [],
      swimSamples: [],
      generatedAt: start,
      calendar: calendar
    )
    let offsets = [0.125, 0.375, 0.625, 0.875]
    let route = WorkoutRouteValue(
      workoutID: "3B207025-810E-4C8A-8C26-E5FEDE0A934E",
      activity: "running",
      start: start,
      points: offsets.enumerated().map { index, offset in
        WorkoutRoutePointValue(
          date: start.addingTimeInterval(offset),
          latitude: 43.645581 + Double(index) / 1_000_000,
          longitude: -79.401239,
          altitudeM: 88.9,
          heartRateBpm: 156,
          cadenceRpm: 80,
          powerW: 278
        )
      }
    )

    let documentURL = try writer.write(document, routes: [route])
    let gpxURL = documentURL.deletingLastPathComponent().appendingPathComponent(route.relativePath)
    let gpx = try String(contentsOf: gpxURL, encoding: .utf8)

    XCTAssertTrue(gpx.contains("  <time>2026-07-10T22:10:00Z</time>"))
    XCTAssertTrue(gpx.contains("    <time>2026-07-10T22:10:00.125Z</time>"))
    XCTAssertTrue(gpx.contains("    <time>2026-07-10T22:10:00.375Z</time>"))
    XCTAssertTrue(gpx.contains("    <time>2026-07-10T22:10:00.625Z</time>"))
    XCTAssertTrue(gpx.contains("    <time>2026-07-10T22:10:00.875Z</time>"))
    XCTAssertTrue(XMLParser(data: Data(gpx.utf8)).parse())
  }

  func testWriterKeepsCyclingCadenceInRevolutionsPerMinute() throws {
    let container = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: container) }
    let writer = HealthExportWriter(containerURL: container)
    let start = date(2026, 7, 26, 9, 33)
    let document = HealthAggregator.document(
      quantitySamples: [],
      swimSamples: [],
      generatedAt: start,
      calendar: calendar
    )
    let route = WorkoutRouteValue(
      workoutID: "F63E15EA-B312-4982-B08F-CBE758BEAD20",
      activity: "cycling",
      start: start,
      points: [
        WorkoutRoutePointValue(
          date: start,
          latitude: 43.6426,
          longitude: -79.3868,
          altitudeM: 83.4,
          heartRateBpm: 164,
          cadenceRpm: 81,
          powerW: 210
        )
      ]
    )

    let documentURL = try writer.write(document, routes: [route])
    let gpxURL = documentURL.deletingLastPathComponent().appendingPathComponent(route.relativePath)
    let gpx = try String(contentsOf: gpxURL, encoding: .utf8)

    XCTAssertTrue(gpx.contains("<gpxtpx:cad>81</gpxtpx:cad>"))
    XCTAssertFalse(gpx.contains("<gpxtpx:cad>41</gpxtpx:cad>"))
    XCTAssertTrue(gpx.contains("<power>210</power>"))
    XCTAssertTrue(XMLParser(data: Data(gpx.utf8)).parse())
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
        )
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
        )
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
        )
      ]
    )
  }

  func testVersionSixWorkoutDecodesWithoutRunSummary() throws {
    let json = """
      {
        "version": 6,
        "generatedAt": "2026-07-10T20:00:00-04:00",
        "timezone": "America/Toronto",
        "days": [],
        "swims": [],
        "workouts": [
          {
            "id": "7E0BEF46-8C0E-4E08-8E2B-0F2E0A1C9E63",
            "activity": "running",
            "start": "2026-07-10T22:10:00Z",
            "end": "2026-07-10T23:12:58Z",
            "durationS": 3070,
            "heartRate": []
          }
        ]
      }
      """

    let document = try JSONDecoder().decode(HealthExportDocument.self, from: Data(json.utf8))
    let workout = try XCTUnwrap(document.workouts.first)

    XCTAssertNil(workout.elapsedTimeS)
    XCTAssertNil(workout.distanceM)
    XCTAssertNil(workout.averageRunningPowerW)
    XCTAssertNil(workout.gpxFile)
    XCTAssertEqual(workout.activities, [])
    XCTAssertEqual(workout.strideLengthM, [])
    XCTAssertEqual(workout.groundContactTimeMs, [])
    XCTAssertEqual(workout.verticalOscillationCm, [])
  }

  func testVersionEightWorkoutDecodesWithoutNestedActivities() throws {
    let json = """
      {
        "version": 8,
        "generatedAt": "2026-07-26T13:56:05-04:00",
        "timezone": "America/Toronto",
        "days": [],
        "swims": [],
        "workouts": [
          {
            "id": "501220B5-1D70-4C31-94FB-0CB0F712740B",
            "activity": "swimBikeRun",
            "start": "2026-07-26T12:43:52Z",
            "end": "2026-07-26T15:41:27Z",
            "durationS": 10502,
            "elapsedTimeS": 10654,
            "heartRate": []
          }
        ]
      }
      """

    let document = try JSONDecoder().decode(HealthExportDocument.self, from: Data(json.utf8))

    XCTAssertEqual(document.workouts.first?.activities, [])
  }

  func testWorkoutGPXPathSurvivesARecentSummaryRefresh() {
    let oldWorkout = AppleHealthWorkout(
      id: "run",
      activity: "running",
      start: "2026-07-10T22:10:00Z",
      end: "2026-07-10T23:12:58Z",
      durationS: 3070,
      gpxFile: "GPX/run.gpx",
      heartRate: []
    )
    let updatedWorkout = AppleHealthWorkout(
      id: "run",
      activity: "running",
      start: "2026-07-10T22:10:00Z",
      end: "2026-07-10T23:12:58Z",
      durationS: 3070,
      distanceM: 9214.5,
      heartRate: []
    )
    let previous = HealthExportDocument(
      version: 7,
      generatedAt: "2026-07-10T20:00:00-04:00",
      timezone: "America/Toronto",
      days: [],
      swims: [],
      workouts: [oldWorkout]
    )
    let update = HealthExportDocument(
      version: 7,
      generatedAt: "2026-07-10T20:05:00-04:00",
      timezone: "America/Toronto",
      days: [],
      swims: [],
      workouts: [updatedWorkout]
    )

    let preserved = update.preservingWorkoutGPXFiles(from: previous)

    XCTAssertEqual(preserved.workouts.first?.distanceM, 9214.5)
    XCTAssertEqual(preserved.workouts.first?.gpxFile, "GPX/run.gpx")
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
    calendar.date(
      from: DateComponents(year: year, month: month, day: day, hour: hour, minute: minute))!
  }
}
