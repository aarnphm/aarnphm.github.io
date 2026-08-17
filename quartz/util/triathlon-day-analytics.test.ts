import assert from 'node:assert/strict'
import test from 'node:test'
import type { OuraDayDetail } from '../plugins/stores/oura'
import { buildAnalytics } from '../plugins/stores/analytics'
import { buildTriathlonDailyAnalytics, isTriathlonDailyAnalytics } from './triathlon-day-analytics'

const date = '2026-08-16'

test('projects exact-date daily analytics without latest-value leakage', () => {
  const analytics = buildAnalytics(null)
  analytics.meta.today = '2026-08-17'
  analytics.daily = [
    {
      date,
      load: 87.3,
      effort: 12,
      swimLoad: 0,
      bikeLoad: 87.3,
      runLoad: 0,
      ctl: 132,
      atl: 244.8,
      tsb: -112.8,
      swimCtl: 0,
      bikeCtl: 132,
      runCtl: 0,
      readiness: 75,
      hrv: 54,
      rhr: 54,
      sleepScore: 84,
      sleepDurationS: 33_810,
      tempDevC: 0.39,
      weightKg: 99,
      totalCalories: null,
      intakeKcal: null,
      warmup: false,
    },
  ]
  analytics.body.ffmi = 18.42
  analytics.body.series = [{ date, ts: Date.parse(`${date}T12:00:00Z`), kg: 99 }]
  analytics.body.composition = [
    {
      date,
      kg: 86.06,
      bmi: 24.3,
      ffmi: 19.65,
      bodyFatPct: 19.3,
      bodyWaterPct: 58.9,
      muscleMassKg: 36.51,
      boneMassKg: 6.05,
    },
  ]
  analytics.recovery = {
    ...analytics.recovery,
    series: [
      {
        date,
        status: 'firm',
        baselineDays: 28,
        hrv: 54,
        hrvBaseline: 53.4,
        hrvZ: 0.1,
        rhr: 54,
        rhrBaseline: 55.2,
        rhrZ: -0.3,
        sleepS: 33_810,
        sleepBaselineS: 29_880,
        sleepTargetS: 30_600,
        sleepScore: 84,
        sleepDebtS: 17_040,
        readiness: 75,
        readinessBaseline: 77,
        tempDevC: 0.39,
        warmup: false,
      },
    ],
  }
  analytics.activities = [
    {
      id: 19771722076,
      date,
      sport: 'bike',
      name: 'Recovery Crit',
      distanceKm: 10,
      movingTimeS: 1_800,
      load: 87.3,
      paceIntensityFactor: null,
      effort: 12,
      cadence: 90,
      strokes: null,
      windKph: null,
      windDir: null,
      windGustKph: null,
    },
  ]
  analytics.engine.vo2max = {
    ...analytics.engine.vo2max,
    estimates: [{ method: 'garmin', vo2max: 55.2, conf: 'firm' }],
    trend: [{ weekStart: '2026-08-10', vo2max: 55.2, method: 'garmin' }],
  }
  analytics.heat = {
    ...analytics.heat,
    series: [
      {
        date,
        temperatureC: 37.8,
        heatStrainIndex: 0.9,
        source: 'core',
        observedMinutes: 74,
        hotMinutes: 0,
        dose: 0,
        acclimatisationPct: 100,
      },
    ],
    activities: [
      {
        id: 19771722076,
        date,
        startedAt: `${date}T18:00:00-04:00`,
        sport: 'bike',
        name: 'Recovery Crit',
        temperatureC: 37.8,
        heatStrainIndex: 0.9,
        source: 'core',
        coreOrigin: 'app',
        observedMinutes: 74,
        hotMinutes: 0,
        dose: 0,
      },
    ],
  }
  const sleep: OuraDayDetail = {
    date,
    bedtimeStart: `${date}T01:28:59-04:00`,
    bedtimeEnd: `${date}T12:25:05-04:00`,
    phase5Min: '442222333',
    efficiency: 86,
    latencyS: 2_100,
    timeInBedS: 39_366,
    totalSleepS: 33_810,
    deepS: 6_150,
    lightS: 21_720,
    remS: 5_940,
    awakeS: 5_556,
    avgBreath: 17.25,
    avgHr: 62.875,
    avgHrv: 54,
    lowestHr: 54,
    restlessPeriods: 201,
    hrv: { startTs: `${date}T01:28:59-04:00`, intervalS: 300, items: [20, 32, null, 54] },
    hr: { startTs: `${date}T01:28:59-04:00`, intervalS: 300, items: [64, 60, 56, 54] },
    readinessScore: 75,
    readinessContrib: { activity_balance: 34, hrv_balance: 82 },
    sleepScore: 84,
    sleepContrib: { deep_sleep: 96, latency: 46 },
  }

  const summary = buildTriathlonDailyAnalytics(analytics, { [date]: sleep })[date]

  assert.ok(summary)
  assert.equal(summary.body?.kg, 86.06)
  assert.equal(summary.body?.ffmi, 19.65)
  assert.equal(summary.recovery?.hrvBaseline, 53.4)
  assert.equal(summary.recovery?.sleepBaselineS, 29_880)
  assert.equal(summary.sleep?.timeInBedS, 39_366)
  assert.equal(summary.sleep?.restlessPeriods, 201)
  assert.equal(summary.sleep?.phase5Min, '442222333')
  assert.deepEqual(summary.sleep?.hrv?.items, [20, 32, null, 54])
  assert.deepEqual(summary.sleep?.heartRate?.items, [64, 60, 56, 54])
  assert.equal(summary.sleep?.sleepContrib?.deep_sleep, 96)
  assert.equal(summary.sleep?.readinessContrib?.hrv_balance, 82)
  assert.deepEqual(summary.training, {
    activityCount: 1,
    load: 87.3,
    relativeEffort: 12,
    ctl: 132,
    atl: 244.8,
    tsb: -112.8,
    garminTss: null,
    exerciseLoad: null,
    exerciseLoadSource: null,
    vo2max: { value: 55.2, method: 'garmin', confidence: 'firm', asOfDate: '2026-08-10' },
  })
  assert.equal(summary.heat?.heatStrainIndex, 0.9)
  assert.equal(summary.heat?.coreOrigin, 'app')
  assert.equal(isTriathlonDailyAnalytics({ [date]: summary }), true)
})

test('rejects malformed nested daily analytics', () => {
  assert.equal(
    isTriathlonDailyAnalytics({
      [date]: { date, body: {}, recovery: null, sleep: null, training: null, heat: null },
    }),
    false,
  )
})
