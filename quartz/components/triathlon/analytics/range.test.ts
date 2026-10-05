import assert from 'node:assert/strict'
import test from 'node:test'
import type { OuraCache } from '../../../plugins/stores/oura'
import type { StravaRawCache } from '../../../plugins/stores/strava'
import { buildAnalytics, type AnalyticsInputs } from '../../../plugins/stores/analytics'
import { shiftIsoDay } from '../../../util/local-date'
import { analyticsForRange, defaultAnalyticsRange } from './range'

const fixture = (today = '2026-09-22', days = 90, inputs: AnalyticsInputs = {}) => {
  const dates = Array.from({ length: days }, (_, index) => shiftIsoDay(today, index - days + 1))
  const lastSync = Date.parse(`${today}T20:00:00Z`)
  const cache: StravaRawCache = {
    athleteId: 123,
    auth: { refreshToken: 'fixture', obtainedAt: 0 },
    lastSync,
    lastActivityStart: lastSync,
    activities: Object.fromEntries(
      dates.map((date, index) => [
        String(index + 1),
        {
          id: index + 1,
          name: `Run ${date}`,
          sportType: 'Run',
          distance: 5000,
          movingTime: 1800,
          elapsedTime: 1800,
          totalElevationGain: 20,
          startDate: `${date}T12:00:00Z`,
          startDateLocal: `${date}T08:00:00Z`,
          averageSpeed: 5000 / 1800,
          averageHeartrate: 150,
          maxHeartrate: 170,
        },
      ]),
    ),
  }
  const oura: OuraCache = {
    lastSync,
    days: Object.fromEntries(
      dates.map(date => [
        date,
        {
          date,
          readiness: 80,
          sleepScore: 80,
          hrv: 65,
          rhr: 50,
          sleepDurationS: 28_800,
          tempDeviationC: 0,
          totalCalories: 2500,
          activeCalories: 500,
        },
      ]),
    ),
  }
  return buildAnalytics(cache, {
    oura,
    weights: dates.map(date => ({
      date,
      weightKg: 75,
      weightLbs: null,
      windKph: null,
      windDir: null,
      race: false,
      event: null,
    })),
    ...inputs,
  })
}

test('analytics defaults to 60 days in the panel and all data on its dedicated page', () => {
  assert.equal(defaultAnalyticsRange(), '60d')
  assert.equal(defaultAnalyticsRange('analytics'), 'all')
})

test('60 days includes the reporting date and the preceding 59 calendar days', () => {
  const source = fixture()
  const snapshot = structuredClone(source)
  const selected = analyticsForRange(source, '60d')

  assert.equal(selected.meta.windowFrom, '2026-07-25')
  assert.equal(selected.meta.windowTo, source.meta.windowTo)
  assert.equal(selected.meta.today, '2026-09-22')
  assert.equal(selected.daily.length, 60)
  assert.equal(selected.daily[0].date, '2026-07-25')
  assert.equal(selected.daily.at(-1)?.date, '2026-09-22')
  assert.equal(selected.activities.length, 60)
  assert.equal(selected.meta.activityCount, 60)
  assert.equal(selected.body.series.length, 60)
  assert.equal(selected.recovery.series.length, 60)
  assert.equal(selected.heat.series.length, 60)
  assert.equal(selected.distributions.activities.length, 60)
  assert.ok(selected.weekly.every(point => point.weekStart >= '2026-07-25'))
  assert.ok(!selected.weekly.some(point => point.weekStart === '2026-07-20'))

  assert.deepEqual(selected.daily, source.daily.slice(-60))
  assert.equal(selected.risk, source.risk)
  assert.equal(selected.thresholds, source.thresholds)
  assert.equal(selected.trends, source.trends)
  assert.equal(selected.recovery.hrvBaseline, source.recovery.hrvBaseline)
  assert.equal(selected.powerCurve.year, source.powerCurve.year)
  assert.deepEqual(source, snapshot)
  assert.equal(analyticsForRange(source, 'all'), source)
  assert.equal(source.daily.length, 90)
})

test('sparse histories use calendar dates rather than taking the last 60 observations', () => {
  const source = fixture()
  const dates = ['2026-07-24', '2026-07-25', '2026-09-22', '2026-09-23']
  source.body.series = dates.map(date => ({ date, ts: Date.parse(date), kg: 75 }))
  source.body.bmrSeries = dates.map(date => ({ date, ts: Date.parse(date), bmr: 1900 }))
  source.body.ffmiSeries = dates.map(date => ({ date, ts: Date.parse(date), ffmi: 20 }))
  source.engine.vo2max.trend = dates.map(weekStart => ({ weekStart, vo2max: 50, method: 'garmin' }))
  source.engine.cardio.rhrSeries = dates.map(date => ({ date, rhr: 50 }))
  source.engine.cardio.hrvSeries = dates.map(date => ({ date, hrv: 65 }))
  source.engine.cardio.efSeries = dates.map(date => ({ date, ef: 1.5, sport: 'run' }))
  source.engine.cardio.decouplingSeries = dates.map(date => ({ date, pct: 3 }))

  const selected = analyticsForRange(source, '60d')
  const expected = ['2026-07-25', '2026-09-22']
  for (const points of [
    selected.body.series,
    selected.body.bmrSeries,
    selected.body.ffmiSeries,
    selected.engine.cardio.rhrSeries,
    selected.engine.cardio.hrvSeries,
    selected.engine.cardio.efSeries,
    selected.engine.cardio.decouplingSeries,
  ])
    assert.deepEqual(
      points.map(point => point.date),
      expected,
    )
  assert.deepEqual(
    selected.engine.vo2max.trend.map(point => point.weekStart),
    expected,
  )
})

test('body composition and its lab sessions retain all dates in the 60-day view', () => {
  const dates = ['2026-06-25', '2026-09-01']
  const source = fixture('2026-09-22', 90, {
    dexa: dates.map(date => ({
      date,
      totalLbs: 197.6,
      fatLbs: 54.2,
      leanLbs: 135.7,
      bmcLbs: 7.8,
      ffmLbs: 143.5,
      bodyFat: 27.4,
    })),
    vo2labs: dates.map(date => ({ date, value: 47.8, massKg: 88.9 })),
  })
  source.body.composition = dates.map(date => ({
    date,
    kg: 88.9,
    bmi: null,
    ffmi: null,
    bodyFatPct: 27.4,
    bodyWaterPct: null,
    muscleMassKg: null,
    boneMassKg: null,
  }))
  const selected = analyticsForRange(source, '60d')

  assert.equal(selected.meta.windowFrom, '2026-07-25')
  assert.deepEqual(
    selected.tests.dexa.map(scan => scan.date),
    dates,
  )
  assert.deepEqual(
    selected.tests.vo2max.map(session => session.date),
    dates,
  )
  assert.equal(selected.tests, source.tests)
  assert.equal(selected.body.composition, source.body.composition)
  assert.ok(selected.body.composition.some(day => day.date < selected.meta.windowFrom))
  assert.equal(selected.body.series.length, 60)
  assert.equal(selected.daily.length, 60)
})

test('the calendar window crosses daylight saving and year boundaries without losing a day', () => {
  for (const today of ['2026-03-09', '2026-11-02', '2026-01-10']) {
    const source = fixture(today)
    const selected = analyticsForRange(source, '60d')
    assert.equal(selected.daily.length, 60)
    assert.equal(selected.daily[0].date, shiftIsoDay(today, -59))
    assert.equal(selected.daily.at(-1)?.date, today)
  }
})

test('short and empty histories stay within the available data bounds', () => {
  const short = fixture('2026-09-22', 7)
  const selected = analyticsForRange(short, '60d')
  assert.equal(selected.meta.windowFrom, short.meta.windowFrom)
  assert.deepEqual(selected.daily, short.daily)
  const empty = buildAnalytics(null)
  assert.deepEqual(analyticsForRange(empty, '60d'), empty)
})
