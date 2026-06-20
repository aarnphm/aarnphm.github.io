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

    XCTAssertEqual(document.version, 1)
    XCTAssertEqual(document.generatedAt, "2026-06-19T07:30:00-04:00")
    XCTAssertEqual(document.timezone, "America/Toronto")
    XCTAssertEqual(document.days, [])
    XCTAssertEqual(document.swims, [])
  }

  private func date(_ year: Int, _ month: Int, _ day: Int, _ hour: Int, _ minute: Int) -> Date {
    calendar.date(from: DateComponents(year: year, month: month, day: day, hour: hour, minute: minute))!
  }
}
