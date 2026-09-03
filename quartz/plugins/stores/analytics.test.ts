import assert from 'node:assert/strict'
import test from 'node:test'
import type { AppleCache } from './apple'
import type { CoreBodyTemperatureCache } from './core-body-temperature'
import type { GarminCache, GarminCyclingDynamics } from './garmin'
import type { OuraCache, OuraDaily } from './oura'
import type { TrackEntry } from './tracking'
import type { WeatherCache } from './weather'
import {
  ACTIVITY_FIELDS,
  ATHLETE,
  DAY_FIELDS,
  POWER_TO_WEIGHT_DURATIONS,
  WEEK_FIELDS,
  buildAnalytics,
  buildDataFeed,
  buildFtpPedalingEvidence,
  computeFtpHypothesisFromVo2,
  type PowerToWeightDurationS,
} from './analytics'
import { emptyGarminFueling, emptyGarminMetrics } from './garmin'
import {
  applyManualSauna,
  buildPayload,
  type RawStravaActivity,
  type StravaActivityDetail,
  type StravaRawCache,
  type StravaStreams,
} from './strava'

const DAY = 86_400_000

function iso(offset: number): string {
  return new Date(Date.parse('2026-05-12T00:00:00Z') + offset * DAY).toISOString().slice(0, 10)
}

function activity(
  id: number,
  sportType: string,
  day: string,
  movingTime: number,
  distance: number,
  extra: Partial<RawStravaActivity> = {},
): RawStravaActivity {
  return {
    id,
    name: `${sportType} ${id}`,
    sportType,
    distance,
    movingTime,
    elapsedTime: movingTime + 60,
    totalElevationGain: 120,
    startDate: `${day}T12:00:00Z`,
    startDateLocal: `${day}T08:00:00Z`,
    averageSpeed: distance / movingTime,
    averageHeartrate: 152,
    maxHeartrate: 178,
    averageCadence: sportType === 'Ride' ? 85 : 88,
    ...extra,
  }
}

function streams(n: number, mps: number, watts?: number[]): StravaStreams {
  return {
    latlng: Array.from({ length: n }, (_, i) => [43.6 + i * 1e-5, -79.4 + i * 1e-5]),
    altitude: Array.from({ length: n }, () => 100),
    distance: Array.from({ length: n }, (_, i) => i * mps),
    watts,
    heartrate: Array.from({ length: n }, (_, i) => 140 + Math.round((20 * i) / n)),
    cadence: Array.from({ length: n }, () => 85),
  }
}

function ouraDay(date: string, hrv: number): OuraDaily {
  return {
    date,
    readiness: 82,
    sleepScore: 78,
    hrv,
    rhr: 50,
    sleepDurationS: 27000,
    tempDeviationC: 0.1,
    totalCalories: 2600,
    activeCalories: 700,
  }
}

function fixtures(): {
  cache: StravaRawCache
  oura: OuraCache
  weights: TrackEntry[]
  weather: WeatherCache
} {
  const bikeDay = iso(20)
  const runDay = iso(22)
  const swimDay = iso(24)
  const cache: StravaRawCache = {
    athleteId: 123,
    auth: { refreshToken: 'super-secret-token', obtainedAt: 0 },
    lastSync: Date.parse('2026-06-11T10:00:00Z'),
    lastActivityStart: 0,
    activities: {
      '1': activity(1, 'Ride', bikeDay, 1500, 12000, {
        deviceWatts: true,
        weightedAverageWatts: 205,
        averageWatts: 200,
        maxWatts: 600,
      }),
      '2': activity(2, 'Run', runDay, 1600, 4800),
      '3': activity(3, 'Swim', swimDay, 1800, 1500, { averageCadence: undefined }),
    },
    streams: {
      '1': streams(
        1500,
        8,
        Array.from({ length: 1500 }, () => 200),
      ),
      '2': streams(1600, 3),
      '3': streams(360, 4.2),
    },
    zones: { hr: [127, 158, 174, 189], power: [110, 150, 180, 210, 240, 300], ftp: 213 },
  }
  const oura: OuraCache = { lastSync: cache.lastSync, days: {} }
  for (let i = 14; i <= 30; i++) {
    const date = iso(i)
    oura.days[date] = ouraDay(date, 80 + (i % 5))
  }
  const weights: TrackEntry[] = [
    {
      date: iso(15),
      weightLbs: 195,
      weightKg: 88.5,
      windKph: 15,
      windDir: 'NW',
      race: false,
      event: null,
    },
  ]
  const weather: WeatherCache = {
    version: 1,
    lastSync: cache.lastSync,
    current: null,
    activities: {
      '1': {
        activityId: 1,
        date: bikeDay,
        start: `${bikeDay}T12:00:00.000Z`,
        end: `${bikeDay}T12:26:00.000Z`,
        latitude: 43.6,
        longitude: -79.4,
        durationS: 1560,
        windKph: 18,
        windDir: 'W',
        windDirDeg: 270,
        windGustKph: 29,
        temperatureC: 22,
        source: 'weatherkit',
      },
    },
    days: {
      [bikeDay]: {
        date: bikeDay,
        activityCount: 1,
        durationS: 1560,
        windKph: 18,
        windDir: 'W',
        windDirDeg: 270,
        windGustKph: 29,
      },
    },
  }
  return { cache, oura, weights, weather }
}

interface PowerActivitySpec {
  id: number
  date: string
}

interface PowerDetailOptions {
  deviceWatts?: boolean
  sport?: StravaActivityDetail['sport']
  bestWatts?: Partial<Record<PowerToWeightDurationS, number>>
  curveWatts?: Partial<Record<PowerToWeightDurationS, number>>
  massKg?: number | null
  massDate?: string | null
}

function powerActivityCache(specs: readonly PowerActivitySpec[], today: string): StravaRawCache {
  const activities: Record<string, RawStravaActivity> = {}
  for (const spec of specs)
    activities[String(spec.id)] = activity(spec.id, 'Ride', spec.date, 1500, 12000, {
      deviceWatts: true,
      averageWatts: 250,
      weightedAverageWatts: 260,
      maxWatts: 900,
    })
  return {
    athleteId: 123,
    auth: { refreshToken: 'power-to-weight-test', obtainedAt: 0 },
    lastSync: Date.parse(`${today}T12:00:00Z`),
    lastActivityStart: 0,
    activities,
  }
}

function trackingWeight(date: string, weightKg: number): TrackEntry {
  return {
    date,
    weightKg,
    weightLbs: weightKg / 0.45359237,
    windKph: null,
    windDir: null,
    race: false,
    event: null,
  }
}

function powerDetail(
  id: number,
  date: string,
  options: PowerDetailOptions = {},
): StravaActivityDetail {
  const { cache } = fixtures()
  const generated = buildPayload(cache, null, null, '2026-05-12', null, ATHLETE.ftp).details['1']
  assert.ok(generated)
  const power = POWER_TO_WEIGHT_DURATIONS.flatMap(durationS => {
    const averageWatts = options.bestWatts?.[durationS]
    return averageWatts == null
      ? []
      : [{ durationS, averageWatts, wattsPerKg: null, averageHeartRate: null, elevationDeltaM: 0 }]
  })
  const powerCurve = POWER_TO_WEIGHT_DURATIONS.flatMap(durationS => {
    const watts = options.curveWatts?.[durationS]
    return watts == null ? [] : [{ s: durationS, w: watts }]
  })
  return {
    ...generated,
    id,
    date,
    start: `${date}T12:00:00Z`,
    sport: options.sport ?? 'bike',
    deviceWatts: options.deviceWatts ?? true,
    powerCurve: powerCurve.length > 0 ? powerCurve : null,
    bestEfforts: {
      weightKg: options.massKg ?? null,
      weightDate: options.massDate ?? null,
      distance: [],
      power,
      climbs: [],
    },
  }
}

test('distributions use payload heart-rate zones and canonical cadence and power units', () => {
  const { cache } = fixtures()
  const run = cache.activities['2']
  cache.activities['1'] = { ...cache.activities['1'], averageTemp: 42 }
  cache.activities['2'] = { ...run, averageWatts: 240, deviceWatts: false }
  cache.activityDetails = {
    '2': {
      calories: null,
      laps: [],
      segmentEfforts: [],
      splitsMetric: [
        {
          split: 1,
          distance: 1_000,
          elapsedTime: 410,
          movingTime: 400,
          averageSpeed: 2.5,
          elevationDifference: 0,
          paceZone: 2,
        },
        {
          split: 2,
          distance: 1_000,
          elapsedTime: 310,
          movingTime: 300,
          averageSpeed: 10 / 3,
          elevationDifference: 0,
          paceZone: 3,
        },
      ],
      splitsStandard: [],
    },
  }
  const heartRateZoneBounds = [130, 145, 160, 175]
  const payload = buildPayload(
    cache,
    null,
    null,
    '2026-05-12',
    null,
    ATHLETE.ftp,
    heartRateZoneBounds,
  )
  const swimDay = iso(24)
  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {},
    swims: {
      swim: {
        id: 'swim',
        date: swimDay,
        start: `${swimDay}T12:00:00Z`,
        end: `${swimDay}T12:30:00Z`,
        totalM: 1_500,
        laps: 30,
        activeTimeS: 1_500,
        strokeCount: 750,
        strokeTimeS: 1_500,
        strokes: { freestyle: 1_500 },
        location: 'pool',
        waterTemperatureC: null,
      },
    },
  }

  const distributions = buildAnalytics(cache, {
    apple,
    zones: payload.zones,
    activityDetails: payload.details,
    since: '2026-05-12',
  }).distributions

  assert.deepEqual(distributions.heartRateZoneBounds, heartRateZoneBounds)
  assert.deepEqual(distributions.powerZoneBounds, payload.zones.power)
  assert.equal(distributions.tenKmRaceTimeS, 50 * 60)
  assert.deepEqual(
    distributions.paceZoneBoundsSPerKm.map(value => Math.round(value)),
    [387, 334, 300, 280, 263],
  )
  assert.equal(distributions.activities.length, 3)
  for (const point of distributions.activities) {
    assert.equal(point.heartRateZoneSeconds?.length, heartRateZoneBounds.length + 1)
    assert.equal(
      point.heartRateZoneSeconds?.reduce((total, seconds) => total + seconds, 0),
      point.movingTimeS,
    )
  }
  const bike = distributions.activities.find(point => point.sport === 'bike')
  const runPoint = distributions.activities.find(point => point.sport === 'run')
  const swim = distributions.activities.find(point => point.sport === 'swim')
  assert.equal(bike?.averagePowerWatts, 200)
  assert.equal(bike?.powerSource, 'device')
  assert.equal(bike?.powerZoneSeconds?.length, payload.zones.power.length + 1)
  assert.equal(
    bike?.powerZoneSeconds?.reduce((total, seconds) => total + seconds, 0),
    bike?.movingTimeS,
  )
  assert.equal(bike?.paceZoneSeconds, null)
  assert.equal(bike?.cadence, 85)
  assert.equal(bike?.cadenceUnit, 'rpm')
  assert.equal(bike?.skinTemperatureC, null)
  assert.equal(bike?.heatStrainIndex, null)
  assert.equal(bike?.skinThermalSource, null)
  assert.equal(bike?.heatStrainThermalSource, null)
  assert.equal(runPoint?.averagePowerWatts, 240)
  assert.equal(runPoint?.powerSource, 'estimated')
  assert.equal(runPoint?.powerZoneSeconds, null)
  assert.deepEqual(runPoint?.paceZoneSeconds, [0, 0, 1599, 0, 0, 0])
  assert.equal(runPoint?.cadence, 176)
  assert.equal(runPoint?.cadenceUnit, 'spm')
  assert.equal(swim?.averagePowerWatts, null)
  assert.equal(swim?.powerSource, null)
  assert.equal(swim?.cadence, 30)
  assert.equal(swim?.cadenceUnit, 'str/min')
})

test('empty analytics include an empty distributions block', () => {
  const distributions = buildAnalytics(null).distributions
  assert.deepEqual(distributions, {
    heartRateZoneBounds: [],
    powerZoneBounds: [],
    paceZoneBoundsSPerKm: distributions.paceZoneBoundsSPerKm,
    tenKmRaceTimeS: 50 * 60,
    activities: [],
  })
  assert.deepEqual(
    distributions.paceZoneBoundsSPerKm.map(value => Math.round(value)),
    [387, 334, 300, 280, 263],
  )
})

test('activity summaries expose normalized pace intensity for run and swim', () => {
  const { cache } = fixtures()
  const analytics = buildAnalytics(cache, { since: '2026-05-12' })

  assert.equal(
    analytics.activities.find(activity => activity.sport === 'run')?.paceIntensityFactor,
    0.909,
  )
  assert.equal(
    analytics.activities.find(activity => activity.sport === 'swim')?.paceIntensityFactor,
    0.641,
  )
  assert.equal(
    analytics.activities.find(activity => activity.sport === 'bike')?.paceIntensityFactor,
    null,
  )
})

test('recovery block computes baselines, series, and flags from oura-merged daily', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  assert.equal(a.recovery.status, 'firm')
  assert.ok(a.recovery.baselineDays >= 14)
  assert.ok(a.recovery.hrvLatest != null && a.recovery.hrvLatest >= 80)
  assert.equal(a.recovery.rhrLatest, 50)
  assert.ok(a.recovery.series.length >= 16)
  assert.equal(a.recovery.sleepDebtS, 0)
  assert.ok(a.recovery.flags.every(f => ['info', 'watch', 'alert'].includes(f.severity)))
  assert.equal(a.recovery.thresholds.sleepTargetS, 25200)
  const day = a.daily.find(d => d.date === iso(20))
  assert.equal(day?.sleepDurationS, 27000)
  assert.equal(day?.tempDevC, 0.1)
})

test('carries build-time power curves and ranks exact durations at the latest measured mass', () => {
  const { cache, weights } = fixtures()
  const powerCurve = {
    sixWeeks: [
      { s: 15, w: 838 },
      { s: 300, w: 366 },
    ],
    year: [
      { s: 15, w: 945 },
      { s: 300, w: 419 },
    ],
    yearLabel: 2026,
    criticalPower: null,
    criticalPowerYear: null,
    ftp: 272,
    goalFtp: 350,
  }

  const ranked = buildAnalytics(cache, { powerCurve, weights, since: '2026-05-12' }).powerCurve
  assert.deepEqual(
    {
      sixWeeks: ranked.sixWeeks,
      year: ranked.year,
      yearLabel: ranked.yearLabel,
      criticalPower: ranked.criticalPower,
      criticalPowerYear: ranked.criticalPowerYear,
      ftp: ranked.ftp,
      goalFtp: ranked.goalFtp,
    },
    powerCurve,
  )
  assert.equal(ranked.ranking.massKg, 88.5)
  assert.equal(ranked.ranking.massDate, iso(15))
  assert.equal(ranked.ranking.massSource, 'tracking')
  assert.equal(ranked.ranking.intervals.length, 12)
  assert.equal(
    ranked.ranking.intervals.find(interval => interval.durationS === 15)?.efforts['six-weeks']
      ?.level,
    4,
  )
  assert.deepEqual(buildAnalytics(null).powerCurve, {
    sixWeeks: [],
    year: [],
    yearLabel: null,
    criticalPower: null,
    criticalPowerYear: null,
    ftp: null,
    goalFtp: null,
    ranking: {
      massKg: null,
      massDate: null,
      massSource: null,
      cohortEligible: false,
      reference: {
        source: 'strava-profile-snapshot',
        capturedDate: '2026-08-16',
        sex: 'M',
        ageMin: 24,
        ageMax: 29,
        massKg: 84.36818082,
      },
      intervals: [],
    },
    powerToWeight: {
      windowDays: 42,
      sex: 'M',
      age: 0,
      ageGroup: null,
      references: [],
      source: {
        url: 'https://www.youtube.com/watch?v=nwT8UtsaHds',
        label: 'GCN × Zwift data',
        population: 'Zwift riders',
        selectionBias:
          'Zwift riders are a self-selected cycling population, so these percentiles do not represent the general population.',
      },
      points: [],
    },
  })
})

test('keeps Apple-only ranking mass and measurement provenance together', () => {
  const { cache } = fixtures()
  const date = iso(27)
  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {
      [date]: {
        date,
        burnKcal: null,
        activeKcal: null,
        intakeKcal: null,
        weightKg: 84.4,
        vo2max: null,
      },
    },
  }
  const ranking = buildAnalytics(cache, { apple, since: '2026-05-12' }).powerCurve.ranking
  assert.equal(ranking.massKg, 84.4)
  assert.equal(ranking.massDate, date)
  assert.equal(ranking.massSource, 'apple')
})

test('power-to-weight extracts the four exact durations and current Zwift references', () => {
  const date = iso(20)
  const cache = powerActivityCache([{ id: 11, date }], date)
  const activityDetails = {
    '11': powerDetail(11, date, {
      bestWatts: { 5: 800, 60: 400 },
      curveWatts: { 300: 320, 1200: 280 },
    }),
  }
  const trend = buildAnalytics(cache, {
    activityDetails,
    weights: [trackingWeight(iso(15), 80)],
    since: date,
  }).powerCurve.powerToWeight
  const point = trend.points[0]

  assert.deepEqual(
    POWER_TO_WEIGHT_DURATIONS.map(durationS => ({
      durationS,
      watts: point.efforts[durationS]?.watts,
      wattsPerKg: point.efforts[durationS]?.wattsPerKg,
    })),
    [
      { durationS: 5, watts: 800, wattsPerKg: 10 },
      { durationS: 60, watts: 400, wattsPerKg: 5 },
      { durationS: 300, watts: 320, wattsPerKg: 4 },
      { durationS: 1200, watts: 280, wattsPerKg: 3.5 },
    ],
  )
  assert.equal(trend.windowDays, 42)
  assert.equal(trend.sex, 'M')
  assert.equal(trend.age, 25)
  assert.equal(trend.ageGroup, '20–29')
  assert.deepEqual(trend.references, [
    { durationS: 5, p10: 4.16, average: 8.75, p90: 14.06 },
    { durationS: 60, p10: 2.88, average: 4.75, p90: 6.82 },
    { durationS: 300, p10: 2.27, average: 3.63, p90: 5.1 },
    { durationS: 1200, p10: 1.95, average: 3.09, p90: 4.32 },
  ])
  assert.equal(trend.source.url, 'https://www.youtube.com/watch?v=nwT8UtsaHds')
  assert.equal(trend.source.label, 'GCN × Zwift data')
  assert.equal(trend.source.population, 'Zwift riders')
  assert.match(trend.source.selectionBias, /self-selected/)
})

test('power-to-weight rejects estimated power and non-bike details', () => {
  const date = iso(20)
  const cache = powerActivityCache(
    [
      { id: 21, date },
      { id: 22, date },
      { id: 23, date },
    ],
    date,
  )
  const trend = buildAnalytics(cache, {
    activityDetails: {
      '21': powerDetail(21, date, { bestWatts: { 5: 400 } }),
      '22': powerDetail(22, date, { deviceWatts: false, bestWatts: { 5: 1_200 } }),
      '23': powerDetail(23, date, { sport: 'run', bestWatts: { 5: 1_300 } }),
    },
    weights: [trackingWeight(date, 80)],
    since: date,
  }).powerCurve.powerToWeight

  assert.equal(trend.points[0].efforts[5]?.activityId, 21)
  assert.equal(trend.points[0].efforts[5]?.watts, 400)
})

test('power-to-weight preserves Garmin, tracking, and Apple mass provenance', () => {
  const firstDate = iso(20)
  const secondDate = iso(22)
  const cache = powerActivityCache(
    [
      { id: 31, date: firstDate },
      { id: 32, date: secondDate },
    ],
    secondDate,
  )
  const trend = buildAnalytics(cache, {
    activityDetails: {
      '31': powerDetail(31, firstDate, { bestWatts: { 5: 400 }, massKg: 80, massDate: firstDate }),
      '32': powerDetail(32, secondDate, { bestWatts: { 60: 450 }, massKg: 70, massDate: iso(23) }),
    },
    weights: [trackingWeight(iso(15), 90)],
    since: iso(15),
  }).powerCurve.powerToWeight
  const latest = trend.points[trend.points.length - 1]

  assert.deepEqual(
    {
      kg: latest.efforts[5]?.massKg,
      date: latest.efforts[5]?.massDate,
      source: latest.efforts[5]?.massSource,
    },
    { kg: 80, date: firstDate, source: 'garmin' },
  )
  assert.deepEqual(
    {
      kg: latest.efforts[60]?.massKg,
      date: latest.efforts[60]?.massDate,
      source: latest.efforts[60]?.massSource,
    },
    { kg: 90, date: iso(15), source: 'tracking' },
  )

  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {
      [iso(15)]: {
        date: iso(15),
        burnKcal: null,
        activeKcal: null,
        intakeKcal: null,
        weightKg: 84,
        vo2max: null,
      },
    },
  }
  const appleTrend = buildAnalytics(powerActivityCache([{ id: 33, date: firstDate }], firstDate), {
    activityDetails: { '33': powerDetail(33, firstDate, { bestWatts: { 300: 336 } }) },
    apple,
    since: iso(15),
  }).powerCurve.powerToWeight
  assert.deepEqual(
    {
      kg: appleTrend.points[appleTrend.points.length - 1].efforts[300]?.massKg,
      date: appleTrend.points[appleTrend.points.length - 1].efforts[300]?.massDate,
      source: appleTrend.points[appleTrend.points.length - 1].efforts[300]?.massSource,
    },
    { kg: 84, date: iso(15), source: 'apple' },
  )
})

test('power-to-weight never uses a future mass measurement', () => {
  const date = iso(20)
  const trend = buildAnalytics(powerActivityCache([{ id: 41, date }], iso(22)), {
    activityDetails: { '41': powerDetail(41, date, { bestWatts: { 5: 500 } }) },
    weights: [trackingWeight(iso(21), 75)],
    since: date,
  }).powerCurve.powerToWeight

  assert.deepEqual(trend.points, [])
})

test('power-to-weight selects same-day and per-duration winners independently', () => {
  const date = iso(20)
  const cache = powerActivityCache(
    [
      { id: 51, date },
      { id: 52, date },
    ],
    date,
  )
  const trend = buildAnalytics(cache, {
    activityDetails: {
      '51': powerDetail(51, date, { bestWatts: { 5: 800, 60: 400, 300: 300 } }),
      '52': powerDetail(52, date, { bestWatts: { 5: 700, 60: 480, 300: 320 } }),
    },
    weights: [trackingWeight(date, 80)],
    since: date,
  }).powerCurve.powerToWeight
  const efforts = trend.points[0].efforts

  assert.equal(efforts[5]?.activityId, 51)
  assert.equal(efforts[60]?.activityId, 52)
  assert.equal(efforts[300]?.activityId, 52)
  assert.equal(efforts[1200], null)
})

test('power-to-weight carries winners for 42 days and expires them on day 43', () => {
  const activityDate = iso(0)
  const day43 = iso(42)
  const trend = buildAnalytics(powerActivityCache([{ id: 61, date: activityDate }], day43), {
    activityDetails: { '61': powerDetail(61, activityDate, { bestWatts: { 5: 400 } }) },
    weights: [trackingWeight(activityDate, 80)],
    since: activityDate,
  }).powerCurve.powerToWeight

  assert.equal(trend.points.length, 43)
  assert.equal(trend.points[41].date, iso(41))
  assert.equal(trend.points[41].efforts[5]?.activityId, 61)
  assert.equal(trend.points[42].date, day43)
  assert.equal(trend.points[42].efforts[5], null)
})

test('empty power-to-weight state has no modelled values or cohort', () => {
  assert.deepEqual(buildAnalytics(null).powerCurve.powerToWeight, {
    windowDays: 42,
    sex: 'M',
    age: 0,
    ageGroup: null,
    references: [],
    source: {
      url: 'https://www.youtube.com/watch?v=nwT8UtsaHds',
      label: 'GCN × Zwift data',
      population: 'Zwift riders',
      selectionBias:
        'Zwift riders are a self-selected cycling population, so these percentiles do not represent the general population.',
    },
    points: [],
  })
})

test('volume improvement actions include CTL units', () => {
  const { cache, oura, weights } = fixtures()
  cache.activities['1'].distance = 40_000
  cache.activities['2'].distance = 10_000
  const actions = buildAnalytics(cache, { oura, weights, since: '2026-05-12' }).actions
  assert.equal(actions.length, 3)
  assert.ok(actions.every(action => action.sourceMetric.endsWith(' ctl')))
  assert.ok(actions.every(action => /^\d+(?:\.\d)? ctl$/.test(action.value)))
})

test('heat block combines WeatherKit and Strava exposure, excludes swims, and decays after three days', () => {
  const streamCache: Record<string, StravaStreams> = {}
  const cache: StravaRawCache = {
    athleteId: 123,
    auth: { refreshToken: 'heat-test-token', obtainedAt: 0 },
    lastSync: Date.parse(`${iso(30)}T18:00:00Z`),
    lastActivityStart: 0,
    activities: {},
    streams: streamCache,
  }
  const weather: WeatherCache = {
    version: 1,
    lastSync: cache.lastSync,
    current: null,
    activities: {},
    days: {},
  }

  for (let offset = 0; offset < 14; offset++) {
    const id = 100 + offset
    const date = iso(offset)
    cache.activities[String(id)] = activity(id, offset % 2 ? 'Run' : 'Ride', date, 3600, 10000)
    streamCache[String(id)] = streams(10, 3)
    weather.activities[String(id)] = {
      activityId: id,
      date,
      start: `${date}T12:00:00.000Z`,
      end: `${date}T13:01:00.000Z`,
      latitude: 43.6,
      longitude: -79.4,
      durationS: 3660,
      windKph: null,
      windDir: null,
      windDirDeg: null,
      windGustKph: null,
      temperatureC: 26,
      source: 'weatherkit',
    }
  }

  const fallbackDate = iso(14)
  cache.activities['114'] = activity(114, 'Run', fallbackDate, 3600, 10000, { averageTemp: 27 })
  streamCache['114'] = streams(10, 3)

  const coolDate = iso(29)
  cache.activities['129'] = activity(129, 'Ride', coolDate, 3600, 10000, { averageTemp: 35 })
  streamCache['129'] = streams(10, 3)
  weather.activities['129'] = {
    activityId: 129,
    date: coolDate,
    start: `${coolDate}T12:00:00.000Z`,
    end: `${coolDate}T13:01:00.000Z`,
    latitude: 43.6,
    longitude: -79.4,
    durationS: 3660,
    windKph: null,
    windDir: null,
    windDirDeg: null,
    windGustKph: null,
    temperatureC: 20,
    source: 'weatherkit',
  }

  const swimDate = iso(30)
  cache.activities['130'] = activity(130, 'Swim', swimDate, 3600, 1500, { averageTemp: 37 })
  streamCache['130'] = streams(10, 1)

  const heat = buildAnalytics(cache, { weather, since: iso(0) }).heat
  assert.equal(heat.currentPct, 72)
  assert.equal(heat.state, 'decaying')
  assert.equal(heat.confidence, 'moderate')
  assert.equal(heat.coveragePct, 100)
  assert.equal(heat.lastHeatDate, fallbackDate)
  assert.equal(heat.lastObservedDate, coolDate)
  assert.equal(heat.latestTemperatureC, 20)
  assert.equal(heat.heatDays14d, 0)
  assert.equal(heat.heatMinutes14d, 0)
  assert.deepEqual(heat.sourceCounts, { core: 0, weatherkit: 15, strava: 1 })
  assert.equal(heat.activities.length, 16)
  assert.deepEqual(
    heat.activities.find(activity => activity.id === 114),
    {
      id: 114,
      date: fallbackDate,
      startedAt: `${fallbackDate}T12:00:00Z`,
      sport: 'run',
      name: 'Run 114',
      temperatureC: 27,
      heatStrainIndex: null,
      source: 'strava',
      coreOrigin: null,
      observedMinutes: 61,
      hotMinutes: 61,
      dose: 1,
    },
  )
  assert.equal(heat.activities.find(activity => activity.id === 129)?.temperatureC, 20)
  assert.equal(heat.activities.find(activity => activity.id === 129)?.hotMinutes, 0)
  assert.equal(
    heat.activities.some(activity => activity.id === 130),
    false,
  )
  assert.equal(heat.series.find(day => day.date === fallbackDate)?.source, 'strava')
  assert.equal(heat.series.find(day => day.date === coolDate)?.temperatureC, 20)
  assert.equal(heat.series.find(day => day.date === swimDate)?.temperatureC, null)
})

test('heat block uses CORE heat strain before WeatherKit ambient temperature', () => {
  const date = iso(20)
  const rideActivity = activity(201, 'Ride', date, 3600, 20_000)
  const cache: StravaRawCache = {
    athleteId: 123,
    auth: { refreshToken: 'core-heat-test-token', obtainedAt: 0 },
    lastSync: Date.parse(`${date}T18:00:00Z`),
    lastActivityStart: 0,
    activities: { 201: rideActivity },
    streams: { 201: streams(10, 3) },
  }
  const weather: WeatherCache = {
    version: 1,
    lastSync: cache.lastSync,
    current: null,
    activities: {
      201: {
        activityId: 201,
        date,
        start: `${date}T12:00:00.000Z`,
        end: `${date}T13:01:00.000Z`,
        latitude: 43.6,
        longitude: -79.4,
        durationS: 3660,
        windKph: null,
        windDir: null,
        windDirDeg: null,
        windGustKph: null,
        temperatureC: 40,
        source: 'weatherkit',
      },
    },
    days: {},
  }
  const garmin: GarminCache = {
    version: 6,
    lastSync: cache.lastSync,
    activities: {
      core: {
        id: 'core',
        name: 'CORE ride',
        sport: 'bike',
        startDate: rideActivity.startDate,
        startDateLocal: rideActivity.startDateLocal,
        distanceM: rideActivity.distance,
        movingTimeS: rideActivity.movingTime,
        elapsedTimeS: rideActivity.elapsedTime,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
    streams: {
      core: {
        time: [0, 900, 1800, 2700, 3600],
        latlng: [],
        altitude: [],
        distance: [0, 5_000, 10_000, 15_000, 20_000],
        heatStrainIndex: [0, 2.9, 3, 4, 2],
        coreTemperatureC: [37, 37.1, 37.2, 37.3, 37.4],
        skinTemperatureC: [30, Number.NaN, 34, Number.NaN, 38],
      },
    },
  }

  const analytics = buildAnalytics(cache, { weather, garmin, since: date })
  const heat = analytics.heat
  assert.deepEqual(heat.sourceCounts, { core: 1, weatherkit: 0, strava: 0 })
  assert.deepEqual(heat.coreSourceCounts, { app: 0, fit: 1 })
  assert.equal(heat.latestTemperatureC, 37.15)
  assert.equal(heat.heatMinutes14d, 30)
  assert.deepEqual(heat.activities[0], {
    id: 201,
    date,
    startedAt: `${date}T12:00:00Z`,
    sport: 'bike',
    name: 'Ride 201',
    temperatureC: 37.15,
    heatStrainIndex: 2.5,
    source: 'core',
    coreOrigin: 'fit',
    observedMinutes: 61,
    hotMinutes: 30,
    dose: 0.5,
  })
  assert.equal(heat.series[0].source, 'core')
  assert.equal(heat.series[0].heatStrainIndex, 2.5)
  assert.deepEqual(analytics.distributions.activities[0], {
    id: 201,
    date,
    startedAt: `${date}T12:00:00Z`,
    sport: 'bike',
    name: 'Ride 201',
    movingTimeS: 3600,
    heartRateZoneSeconds: null,
    powerZoneSeconds: null,
    paceZoneSeconds: null,
    averagePowerWatts: null,
    powerSource: null,
    cadence: 85,
    cadenceUnit: 'rpm',
    skinTemperatureC: 32.19,
    heatStrainIndex: 2.5,
    skinThermalSource: 'core-fit',
    heatStrainThermalSource: 'core-fit',
    skinObservedSeconds: 1860,
    heatStrainObservedSeconds: 3660,
  })
})

test('heat block prefers CORE app onboard samples over CORE FIT telemetry', () => {
  const date = iso(20)
  const rideActivity = activity(201, 'Ride', date, 3600, 20_000)
  const cache: StravaRawCache = {
    athleteId: 123,
    auth: { refreshToken: 'core-app-test-token', obtainedAt: 0 },
    lastSync: Date.parse(`${date}T18:00:00Z`),
    lastActivityStart: 0,
    activities: { 201: rideActivity },
    streams: { 201: streams(10, 3) },
  }
  const core: CoreBodyTemperatureCache = {
    version: 1,
    lastSync: cache.lastSync,
    samples: Array.from({ length: 61 }, (_, index) => ({
      time: new Date(Date.parse(rideActivity.startDate) + index * 60_000).toISOString(),
      coreTemperatureC: 38 + index / 1_000,
      skinTemperatureC: 33,
      heatStrainIndex: 4,
      quality: 4,
      heartRate: 150,
    })),
  }
  const garmin: GarminCache = {
    version: 6,
    lastSync: cache.lastSync,
    activities: {
      core: {
        id: 'core',
        name: 'CORE ride',
        sport: 'bike',
        startDate: rideActivity.startDate,
        startDateLocal: rideActivity.startDateLocal,
        distanceM: rideActivity.distance,
        movingTimeS: rideActivity.movingTime,
        elapsedTimeS: rideActivity.elapsedTime,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
    streams: {
      core: {
        time: [0, 1800, 3600],
        latlng: [],
        altitude: [],
        distance: [0, 10_000, 20_000],
        heatStrainIndex: [1, 1, 1],
        coreTemperatureC: [37, 37, 37],
      },
    },
  }

  const analytics = buildAnalytics(cache, { core, garmin, since: date })
  const heat = analytics.heat
  assert.deepEqual(heat.sourceCounts, { core: 1, weatherkit: 0, strava: 0 })
  assert.deepEqual(heat.coreSourceCounts, { app: 1, fit: 0 })
  assert.equal(heat.activities[0].coreOrigin, 'app')
  assert.equal(heat.activities[0].temperatureC, 38.03)
  assert.equal(heat.activities[0].heatStrainIndex, 4)
  assert.equal(heat.activities[0].hotMinutes, 61)
  assert.equal(analytics.distributions.activities[0].skinTemperatureC, 33)
  assert.equal(analytics.distributions.activities[0].heatStrainIndex, 4)
  assert.equal(analytics.distributions.activities[0].skinThermalSource, 'core-app')
  assert.equal(analytics.distributions.activities[0].heatStrainThermalSource, 'core-app')
  assert.equal(analytics.distributions.activities[0].skinObservedSeconds, 3660)
  assert.equal(analytics.distributions.activities[0].heatStrainObservedSeconds, 3660)
})

test('activity telemetry resolves sparse CORE app and FIT metrics independently', () => {
  const date = iso(20)
  const rideActivity = activity(201, 'Ride', date, 3600, 20_000)
  const cache: StravaRawCache = {
    athleteId: 123,
    auth: { refreshToken: 'core-sparse-test-token', obtainedAt: 0 },
    lastSync: Date.parse(`${date}T18:00:00Z`),
    lastActivityStart: 0,
    activities: { 201: rideActivity },
    streams: { 201: streams(10, 3) },
  }
  const core: CoreBodyTemperatureCache = {
    version: 1,
    lastSync: cache.lastSync,
    samples: [
      {
        time: rideActivity.startDate,
        coreTemperatureC: null,
        skinTemperatureC: 34,
        heatStrainIndex: null,
        quality: 4,
        heartRate: null,
      },
    ],
  }
  const garmin: GarminCache = {
    version: 6,
    lastSync: cache.lastSync,
    activities: {
      core: {
        id: 'core',
        name: 'CORE ride',
        sport: 'bike',
        startDate: rideActivity.startDate,
        startDateLocal: rideActivity.startDateLocal,
        distanceM: rideActivity.distance,
        movingTimeS: rideActivity.movingTime,
        elapsedTimeS: rideActivity.elapsedTime,
        sourceDevice: 'Edge 1050',
        sourceFile: null,
        metrics: emptyGarminMetrics(),
        fueling: emptyGarminFueling('Edge 1050'),
      },
    },
    streams: {
      core: {
        time: [0, 1800, 3600],
        latlng: [],
        altitude: [],
        distance: [0, 10_000, 20_000],
        heatStrainIndex: [2, 3, 4],
        coreTemperatureC: [37, 37.1, 37.2],
      },
    },
  }

  const point = buildAnalytics(cache, { core, garmin, since: date }).distributions.activities[0]
  assert.equal(point.skinTemperatureC, 34)
  assert.equal(point.skinThermalSource, 'core-app')
  assert.equal(point.skinObservedSeconds, 60)
  assert.equal(point.heatStrainIndex, 2.5)
  assert.equal(point.heatStrainThermalSource, 'core-fit')
  assert.equal(point.heatStrainObservedSeconds, 3660)
})

test('engine block bases vo2max on the declared strava ftp and builds six radar axes', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  const v = a.engine.vo2max
  assert.equal(v.method, 'bike')
  assert.equal(v.conf, 'firm')
  assert.equal(v.bikeSource?.ftpW, 213)
  assert.equal(v.bikeSource?.ftpSource, 'strava')
  assert.equal(v.bikeSource?.mapW, 284)
  assert.ok(v.bikeSource?.weightKg != null)
  assert.ok(v.value != null && v.value > 25 && v.value < 50)
  assert.ok(v.fitnessAge != null && v.fitnessAge >= 20 && v.fitnessAge <= 80)
  assert.equal(v.chronoAge, 25)
  assert.equal(v.hrMax, ATHLETE.hrMax)
  assert.equal(v.hrMaxSource, 'declared')
  assert.ok(v.trend.length >= 1)
  assert.ok(a.engine.cardio.metrics.length === 4)
  assert.ok(a.engine.cardio.rhrSeries.length >= 16)
})

test('abilities block builds one radar per sport with per-discipline history', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  assert.deepEqual(
    a.engine.abilities.sports.map(s => s.sport),
    ['swim', 'bike', 'run'],
  )
  for (const s of a.engine.abilities.sports) {
    assert.deepEqual(
      s.axes.map(x => x.key),
      s.sport === 'run'
        ? ['sprint', 'threshold', 'endurance', 'stride', 'cadence', 'oscillation']
        : ['sprint', 'threshold', 'endurance', 'climb', 'cadence', 'recovery'],
    )
    assert.ok(s.history.length >= 2)
    assert.ok(s.area != null && s.area > 0)
    for (const x of s.axes) {
      if (x.score == null) assert.equal(x.proj, null)
      else assert.ok(x.proj != null && x.proj >= 0 && x.proj <= 100)
    }
  }
  const [swim, bike, run] = a.engine.abilities.sports
  const swimPace = swim.axes.find(x => x.key === 'climb')
  assert.equal(swimPace?.label, 'pace')
  assert.equal(swimPace?.rawUnit, 's/100m')
  assert.equal(swimPace?.rawValue, 120)
  assert.equal(swimPace?.score, 76)
  assert.equal(swim.axes.find(x => x.key === 'sprint')?.score, 76)
  assert.equal(swim.axes.find(x => x.key === 'cadence')?.score, null)
  assert.equal(swim.axes.find(x => x.key === 'sprint')?.rawUnit, 'm/s')
  assert.equal(bike.axes.find(x => x.key === 'sprint')?.rawUnit, 'w/kg')
  assert.equal(bike.axes.find(x => x.key === 'threshold')?.rawUnit, 'w/kg')
  assert.equal(bike.axes.find(x => x.key === 'climb')?.label, 'climb')
  assert.equal(bike.axes.find(x => x.key === 'cadence')?.label, 'cadence')
  assert.equal(run.axes.find(x => x.key === 'cadence')?.rawValue, 176)
  assert.equal(run.axes.find(x => x.key === 'stride')?.label, 'estimated stride length')
  assert.equal(run.axes.find(x => x.key === 'stride')?.rawValue, 1.02)
  assert.equal(run.axes.find(x => x.key === 'oscillation')?.rawValue, null)
  assert.equal(run.axes.find(x => x.key === 'cadence')?.label, 'cadence')
  assert.equal(run.axes.find(x => x.key === 'threshold')?.score, null)
  const bikeEnd = bike.axes.find(x => x.key === 'endurance')
  assert.equal(bikeEnd?.hi, 50)
  const runLast = run.history[run.history.length - 1]
  assert.ok(runLast.sprint != null)
  assert.equal(runLast.stride != null, true)
  assert.equal(runLast.oscillation, null)
  const swimLast = swim.history[swim.history.length - 1]
  assert.equal(swimLast.climb, 76)
  assert.equal(swimLast.sprint, 76)
})

test('swim sprint and threshold share the swim pace scale', () => {
  const { cache, oura, weights } = fixtures()
  for (let index = 0; index < 4; index++) {
    const id = 3 + index
    const day = iso(24 + index)
    cache.activities[String(id)] = activity(id, 'Swim', day, 1_000, 708, {
      averageCadence: undefined,
    })
  }

  const analytics = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  const swim = analytics.engine.abilities.sports.find(sport => sport.sport === 'swim')
  const sprint = swim?.axes.find(axis => axis.key === 'sprint')
  const threshold = swim?.axes.find(axis => axis.key === 'threshold')

  assert.equal(sprint?.rawValue, 0.71)
  assert.equal(sprint?.score, 69)
  assert.equal(threshold?.rawValue, 0.71)
  assert.equal(threshold?.score, 69)
  assert.equal(sprint?.lo, 100 / 360)
  assert.equal(sprint?.hi, 100 / 45)
  assert.equal(threshold?.lo, sprint?.lo)
  assert.equal(threshold?.hi, sprint?.hi)
  assert.equal(swim?.history.at(-1)?.sprint, 69)
  assert.equal(swim?.history.at(-1)?.threshold, 69)
})

test('run radar replaces climb and recovery with native stride and oscillation', () => {
  const { cache, oura, weights } = fixtures()
  const firstRun = cache.activities['2']
  const secondDay = iso(25)
  cache.activities['4'] = activity(4, 'Run', secondDay, 1_800, 6_000)
  assert.ok(cache.streams)
  cache.streams['4'] = streams(1_800, 10 / 3)
  const apple: AppleCache = {
    version: 9,
    lastSync: cache.lastSync,
    days: {},
    workouts: {
      first: {
        id: 'first',
        activity: 'running',
        start: firstRun.startDate,
        end: `${firstRun.startDate.slice(0, 11)}12:26:40Z`,
        durationS: firstRun.movingTime,
        distanceM: firstRun.distance,
        heartRate: [],
        strideLengthM: [
          { time: firstRun.startDate, value: 1 },
          { time: firstRun.startDate, value: 1.1 },
        ],
        verticalOscillationCm: [
          { time: firstRun.startDate, value: 10 },
          { time: firstRun.startDate, value: 10.2 },
        ],
      },
      second: {
        id: 'second',
        activity: 'running',
        start: cache.activities['4'].startDate,
        end: `${secondDay}T12:30:00Z`,
        durationS: 1_800,
        distanceM: 6_000,
        heartRate: [],
        strideLengthM: [
          { time: cache.activities['4'].startDate, value: 1.2 },
          { time: cache.activities['4'].startDate, value: 1.3 },
        ],
        verticalOscillationCm: [
          { time: cache.activities['4'].startDate, value: 8 },
          { time: cache.activities['4'].startDate, value: 8.2 },
        ],
      },
    },
  }

  const analytics = buildAnalytics(cache, { apple, oura, weights, since: '2026-05-12' })
  const run = analytics.engine.abilities.sports.find(sport => sport.sport === 'run')
  const stride = run?.axes.find(axis => axis.key === 'stride')
  const oscillation = run?.axes.find(axis => axis.key === 'oscillation')

  assert.equal(stride?.label, 'stride length')
  assert.equal(stride?.rawUnit, 'm')
  assert.equal(stride?.rawValue, 1.15)
  assert.equal(stride?.score, 50)
  assert.equal(oscillation?.label, 'vertical oscillation')
  assert.equal(oscillation?.rawUnit, 'cm')
  assert.equal(oscillation?.rawValue, 9.1)
  assert.equal(oscillation?.score, 50)
  assert.equal(run?.history.at(-1)?.stride, 50)
  assert.equal(run?.history.at(-1)?.oscillation, 50)
})

test('engine derives ftp from 20-min power only when strava declares none', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics({ ...cache, zones: undefined }, { oura, weights, since: '2026-05-12' })
  const v = a.engine.vo2max
  assert.equal(v.method, 'bike')
  assert.equal(v.conf, 'low')
  assert.equal(v.bikeSource?.ftpW, 190)
  assert.equal(v.bikeSource?.ftpSource, 'derived')
})

test('vo2 lab ftp hypothesis keeps the treadmill-to-bike estimate broad', () => {
  const h = computeFtpHypothesisFromVo2('2026-06-25', 47.8, 88.9)
  assert.ok(h)
  assert.equal(h.absoluteVo2, 4.25)
  assert.equal(h.cyclingVo2max, 3.91)
  assert.equal(h.thresholdVo2, 3.32)
  assert.equal(h.efficiencyFtp, 243)
  assert.equal(h.acsmFtp, 224)
  assert.equal(h.ftp, 230)
  assert.equal(h.low, 210)
  assert.equal(h.high, 260)
  assert.equal(h.wattsPerKg, 2.59)
  assert.equal(h.conf, 'low')
  assert.equal(h.massDate, '2026-06-25')
  assert.equal(h.massSource, 'lab')
  assert.equal(h.vo2maxDate, '2026-06-25')
  assert.equal(h.vo2maxSource, 'lab')
  assert.equal(h.vo2maxSport, 'running')
  assert.equal(h.defaultVo2max, 47.8)
  assert.equal(h.efficiency.source, 'literature-prior')
  assert.equal(h.efficiency.conf, 'prior')
})

test('bike vo2max uses the lab result and its derived ftp as the baseline', () => {
  const { cache, oura, weights } = fixtures()
  const analytics = buildAnalytics(cache, {
    oura,
    weights,
    ftp: 287,
    since: '2026-05-12',
    vo2labs: [{ date: iso(20), value: 47.8, massKg: 88.9 }],
  })

  assert.equal(analytics.engine.vo2max.method, 'bike')
  assert.equal(analytics.engine.vo2max.value, 57.2)
  assert.equal(analytics.engine.vo2max.trend.at(-1)?.vo2max, 57.2)
  assert.deepEqual(analytics.engine.vo2max.bikeSource?.labBaseline, {
    date: iso(20),
    vo2max: 47.8,
    ftpW: 230,
    weightKg: 88.9,
  })
})

test('pedaling evidence aggregates ride medians after per-side coverage checks', () => {
  const dynamics = (
    leftSmoothness: (number | null)[],
    rightSmoothness: (number | null)[],
    leftTorque: (number | null)[],
    rightTorque: (number | null)[],
  ): GarminCyclingDynamics => ({
    time: [0, 1, 2, 3, 4],
    distance: [0, 8, 16, 24, 32],
    leftPedalSmoothness: leftSmoothness,
    rightPedalSmoothness: rightSmoothness,
    leftTorqueEffectiveness: leftTorque,
    rightTorqueEffectiveness: rightTorque,
    leftPowerPhaseStart: [],
    leftPowerPhaseEnd: [],
    rightPowerPhaseStart: [],
    rightPowerPhaseEnd: [],
    positionChanges: [],
    seatedTimeS: null,
    standingTimeS: null,
  })
  const bike = (id: string, date: string): GarminCache['activities'][string] => ({
    id,
    name: `Ride ${id}`,
    sport: 'bike',
    startDate: `${date}T12:00:00Z`,
    startDateLocal: `${date}T08:00:00Z`,
    distanceM: 40_000,
    movingTimeS: 3600,
    elapsedTimeS: 3660,
    sourceDevice: 'Edge 1050',
    sourceFile: `${id}.fit`,
    metrics: emptyGarminMetrics(),
    fueling: emptyGarminFueling('Edge 1050'),
  })
  const garmin: GarminCache = {
    lastSync: Date.parse(`${iso(30)}T12:00:00Z`),
    activities: {
      one: bike('one', iso(20)),
      two: bike('two', iso(25)),
      sparse: bike('sparse', iso(27)),
    },
    cyclingDynamics: {
      one: dynamics(
        [20, 20, 20, 20, 20],
        [18, 18, 18, 18, 18],
        [80, 80, 80, 80, 80],
        [78, 78, 78, 78, 78],
      ),
      two: dynamics(
        [24, 24, 24, 24, 24],
        [22, 22, 22, 22, 22],
        [84, 84, 84, 84, 84],
        [82, 82, 82, 82, 82],
      ),
      sparse: dynamics(
        [30, 30, 30, 30, 30],
        [30, 30, 30, 30, 30],
        [90, 90, 90, 90, 90],
        [90, 90, null, null, null],
      ),
    },
  }
  const evidence = buildFtpPedalingEvidence(garmin, iso(30))
  assert.deepEqual(evidence, {
    windowFrom: iso(20),
    windowTo: iso(25),
    activityCount: 2,
    sampleCount: 10,
    coveragePct: 100,
    leftPedalSmoothnessPct: 22,
    rightPedalSmoothnessPct: 20,
    leftTorqueEffectivenessPct: 82,
    rightTorqueEffectivenessPct: 80,
  })
})

test('vo2 lab profile samples survive analytics parsing', () => {
  const { cache } = fixtures()
  const a = buildAnalytics(cache, {
    since: '2026-05-12',
    vo2labs: [
      {
        date: '2026-06-25',
        value: 47.8,
        massKg: 88.9,
        profile: {
          durationSec: 20,
          warmupEndSec: 10,
          cooldownStartSec: 18,
          vt1Sec: 12,
          vo2maxSec: 16,
          stats: { vo2: [6.7, 50.5, 35.9], hr: [72, 182, 139] },
          targetKmh: [
            [0, 5],
            [10, 7],
            [18, 5],
          ],
          samples: [
            [0, 9.6, 73, 31.6, 16.9, 1.86],
            [10, 21.3, 112, 34.9, 19.1, 1.98],
            [20, null, 158, null, null, null],
          ],
        },
      },
    ],
  })

  const lab = a.tests.vo2max[0]
  assert.equal(lab.profile?.durationSec, 20)
  assert.equal(lab.profile?.targetKmh[1].kmh, 7)
  assert.equal(lab.profile?.samples[2].vo2, null)
  assert.equal(lab.profile?.samples[2].hr, 158)
  assert.equal(a.engine.ftpHypothesis?.ftp, 230)
})

test('athlete ftp override drives analytics when supplied by the emitter', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, ftp: 230, since: '2026-05-12' })
  const v = a.engine.vo2max
  assert.equal(v.method, 'bike')
  assert.equal(v.conf, 'low')
  assert.equal(v.bikeSource?.ftpW, 230)
  assert.equal(v.bikeSource?.ftpSource, 'athlete')
  const bike = a.engine.abilities.sports.find(s => s.sport === 'bike')
  const threshold = bike?.axes.find(axis => axis.key === 'threshold')
  assert.equal(threshold?.rawValue, 2.6)
})

test('garmin vo2max outranks every other estimate', () => {
  const { cache, oura, weights } = fixtures()
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {},
    vo2max: {
      [iso(29)]: { date: iso(29), generic: 54, cycling: 49.8 },
      [iso(26)]: { date: iso(26), generic: 53.5, cycling: null },
    },
  }
  const a = buildAnalytics(cache, { oura, garmin, weights, since: '2026-05-12' })
  assert.equal(a.engine.vo2max.method, 'garmin')
  assert.equal(a.engine.vo2max.value, 54)
  assert.equal(a.engine.vo2max.conf, 'firm')
  assert.ok(a.engine.vo2max.estimates.some(e => e.method === 'bike'))
  assert.ok(a.engine.vo2max.fitnessAge != null && a.engine.vo2max.fitnessAge < 25)
})

test('garmin readings drive the vo2 trend and the bike proxy only fills earlier weeks', () => {
  const { cache, oura, weights } = fixtures()
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {},
    vo2max: { [iso(29)]: { date: iso(29), generic: 54, cycling: 49.8 } },
  }
  const a = buildAnalytics(cache, { oura, garmin, weights, since: '2026-05-12' })
  const trend = a.engine.vo2max.trend
  assert.equal(trend.length, 2)
  assert.equal(trend[0].weekStart, iso(20))
  assert.equal(trend[0].method, 'bike')
  assert.ok(trend[0].vo2max > 25 && trend[0].vo2max < 50)
  assert.equal(trend[1].weekStart, iso(27))
  assert.equal(trend[1].method, 'garmin')
  assert.equal(trend[1].vo2max, 54)
})

test('newer Apple Watch vo2max replaces stale Garmin within the same week', () => {
  const { cache, oura, weights } = fixtures()
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {},
    vo2max: { [iso(26)]: { date: iso(26), generic: 54, cycling: 49.8 } },
  }
  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {
      [iso(28)]: {
        date: iso(28),
        burnKcal: null,
        activeKcal: null,
        intakeKcal: null,
        weightKg: null,
        vo2max: 55.2,
      },
    },
  }
  const analytics = buildAnalytics(cache, { oura, apple, garmin, weights, since: '2026-05-12' })

  assert.equal(analytics.engine.vo2max.method, 'apple')
  assert.equal(analytics.engine.vo2max.value, 55.2)
  assert.equal(analytics.engine.vo2max.trend.at(-1)?.method, 'apple')
  assert.equal(analytics.engine.ftpHypothesis?.vo2max, 55.2)
  assert.equal(analytics.engine.ftpHypothesis?.vo2maxDate, iso(28))
  assert.equal(analytics.engine.ftpHypothesis?.vo2maxSource, 'apple')
  assert.equal(analytics.engine.ftpHypothesis?.vo2maxSport, 'unknown')
})

test('Garmin vo2max keeps precedence over Apple Watch on an equal date', () => {
  const { cache, oura, weights } = fixtures()
  const date = iso(28)
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {},
    vo2max: { [date]: { date, generic: 54, cycling: 49.8 } },
  }
  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {
      [date]: {
        date,
        burnKcal: null,
        activeKcal: null,
        intakeKcal: null,
        weightKg: null,
        vo2max: 55.2,
      },
    },
  }
  const analytics = buildAnalytics(cache, { oura, apple, garmin, weights, since: '2026-05-12' })

  assert.equal(analytics.engine.vo2max.method, 'garmin')
  assert.equal(analytics.engine.vo2max.value, 54)
  assert.equal(analytics.engine.ftpHypothesis?.vo2max, 54)
  assert.equal(analytics.engine.ftpHypothesis?.vo2maxSource, 'garmin')
})

test('vo2 trend summary uses comparable observed weeks without carrying stale values forward', () => {
  const { cache, oura, weights } = fixtures()
  const readings = [48, 49, 50, 51, 52]
  const vo2max = Object.fromEntries(
    readings.map((generic, index) => {
      const date = iso(1 + index * 7)
      return [date, { date, generic, cycling: null }]
    }),
  )
  const garmin: GarminCache = { lastSync: cache.lastSync, activities: {}, vo2max }

  const analytics = buildAnalytics(cache, { oura, garmin, weights, since: '2026-05-12' })
  const observed = analytics.engine.vo2max.trend.filter(point => point.method === 'garmin')
  const summary = analytics.engine.vo2max.trendSummary

  assert.equal(observed.length, readings.length)
  assert.deepEqual(
    observed.map(point => point.vo2max),
    readings,
  )
  assert.equal(summary.method, 'garmin')
  assert.equal(summary.sampleSize, readings.length)
  assert.equal(summary.spanDays, 28)
  assert.equal(summary.slopePerWeek, 1)
  assert.equal(summary.change28d, 4)
  assert.equal(summary.direction, 'improving')
})

test('lab test outranks a garmin reading in the same trend week', () => {
  const { cache, oura, weights } = fixtures()
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {},
    vo2max: {
      [iso(26)]: { date: iso(26), generic: 53.5, cycling: null },
      [iso(29)]: { date: iso(29), generic: 54, cycling: 49.8 },
    },
  }
  const a = buildAnalytics(cache, {
    oura,
    garmin,
    weights,
    since: '2026-05-12',
    vo2labs: [{ date: iso(30), value: 47.8, massKg: 88.9 }],
  })
  const trend = a.engine.vo2max.trend
  assert.equal(trend.length, 2)
  assert.equal(trend[0].method, 'garmin')
  assert.equal(trend[0].vo2max, 53.5)
  assert.equal(trend[1].weekStart, iso(27))
  assert.equal(trend[1].method, 'lab')
  assert.equal(trend[1].vo2max, 47.8)
  assert.equal(a.engine.vo2max.method, 'lab')
  assert.equal(a.engine.vo2max.value, 47.8)
})

test('vo2 headline follows the latest garmin mark after an older lab test', () => {
  const { cache, oura, weights } = fixtures()
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {},
    vo2max: { [iso(29)]: { date: iso(29), generic: 48.1, cycling: 47.5 } },
  }
  const a = buildAnalytics(cache, {
    oura,
    garmin,
    weights,
    since: '2026-05-12',
    vo2labs: [{ date: iso(23), value: 47.8, massKg: 88.9 }],
  })
  const latest = a.engine.vo2max.trend[a.engine.vo2max.trend.length - 1]
  assert.equal(latest.method, 'garmin')
  assert.equal(latest.vo2max, 48.1)
  assert.equal(a.engine.vo2max.method, 'garmin')
  assert.equal(a.engine.vo2max.value, 48.1)
  assert.equal(a.engine.vo2max.conf, 'firm')
  assert.equal(a.engine.vo2max.bikeSource, null)
})

test('ftp hypothesis preserves unknown Garmin provenance and removes the discount for cycling-only VO2max', () => {
  const { cache, oura, weights } = fixtures()
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {},
    vo2max: { [iso(29)]: { date: iso(29), generic: 50.5, cycling: 49.2 } },
    weight: [
      {
        ts: Date.parse(`${iso(28)}T07:00:00.000Z`),
        date: iso(28),
        weightKg: 86.8,
        bmi: 26.7,
        bodyFatPct: 21.1,
        bodyWaterPct: 55.3,
        muscleMassKg: 35.4,
        boneMassKg: 3.7,
      },
    ],
  }
  const live = buildAnalytics(cache, {
    oura,
    garmin,
    weights,
    since: '2026-05-12',
    vo2labs: [{ date: iso(23), value: 47.8, massKg: 88.9 }],
  }).engine.ftpHypothesis
  assert.ok(live)
  assert.equal(live.massKg, 86.8)
  assert.equal(live.massDate, iso(28))
  assert.equal(live.massSource, 'daily')
  assert.equal(live.vo2max, 50.5)
  assert.equal(live.vo2maxDate, iso(29))
  assert.equal(live.vo2maxSource, 'garmin')
  assert.equal(live.vo2maxSport, 'unknown')
  assert.equal(live.defaultVo2max, ATHLETE.vo2max)
  assert.equal(live.crossModalDiscountPct, 8)
  assert.equal(live.ftp, 240)

  const cyclingGarmin: GarminCache = {
    ...garmin,
    vo2max: { [iso(29)]: { date: iso(29), generic: null, cycling: 49.2 } },
  }
  const cycling = buildAnalytics(cache, {
    oura,
    garmin: cyclingGarmin,
    weights,
    since: '2026-05-12',
    vo2labs: [{ date: iso(23), value: 47.8, massKg: 88.9 }],
  }).engine.ftpHypothesis
  assert.ok(cycling)
  assert.equal(cycling.vo2max, 49.2)
  assert.equal(cycling.vo2maxSport, 'cycling')
  assert.equal(cycling.crossModalDiscountPct, 0)
  assert.ok(cycling.ftp > live.ftp)

  const fallback = buildAnalytics(cache, { oura, weights, since: '2026-05-12' }).engine
    .ftpHypothesis
  assert.ok(fallback)
  assert.equal(fallback.massKg, 88.5)
  assert.equal(fallback.massDate, iso(15))
  assert.equal(fallback.massSource, 'daily')
  assert.equal(fallback.vo2max, ATHLETE.vo2max)
  assert.equal(fallback.vo2maxSource, 'default')
  assert.equal(fallback.vo2maxSport, 'running')
})

test('calibration tracks newest pace and volume deltas against the prior window', () => {
  const { cache } = fixtures()
  cache.lastSync = Date.parse('2026-06-11T10:00:00Z')
  cache.activities = {
    '10': activity(10, 'Run', iso(-16), 1800, 5000, { totalElevationGain: 0 }),
    '11': activity(11, 'Run', iso(-6), 1800, 5000, { totalElevationGain: 0 }),
    '12': activity(12, 'Run', iso(20), 1500, 5000, { totalElevationGain: 0 }),
    '13': activity(13, 'Run', iso(26), 1500, 5000, { totalElevationGain: 0 }),
  }
  cache.streams = {}

  const a = buildAnalytics(cache, { since: '2026-04-01' })
  const run = a.calibration.paces.find(p => p.sport === 'run')
  assert.ok(run)
  assert.equal(a.calibration.asOf, '2026-06-11')
  assert.equal(a.calibration.windowDays, 28)
  assert.equal(a.calibration.projectionDays, 14)
  assert.equal(run.sampleSize, 2)
  assert.equal(run.previousSampleSize, 2)
  assert.equal(run.average, 300)
  assert.equal(run.previous, 360)
  assert.equal(run.direction, 'faster')
  assert.equal(run.deltaPct, 16.7)
  assert.ok(run.projected != null && run.projected < run.average)
  assert.ok(run.projectedDeltaPct != null && run.projectedDeltaPct > 0)
  assert.equal(a.calibration.volume.currentKm, 10)
  assert.equal(a.calibration.volume.previousKm, 10)
  assert.ok(a.calibration.volume.deltaHours < 0)
  const runVolume = a.calibration.volume.sports.find(s => s.sport === 'run')
  assert.equal(runVolume?.currentKm, 10)
  assert.equal(runVolume?.previousKm, 10)
  const activeWeek = a.weekly.find(w => w.sessions === 2 && w.runKm === 10)
  assert.ok(activeWeek)
  assert.equal(activeWeek.runHours, 0.8)
})

test('personal bests reject implausible swim results', () => {
  const { cache } = fixtures()
  cache.activities = {
    '10': activity(10, 'Swim', iso(18), 1200, 1000, { totalElevationGain: 0 }),
    '11': activity(11, 'Swim', iso(20), 306, 2300, { totalElevationGain: 0 }),
    '12': activity(12, 'Swim', iso(22), 1000, 1000, { totalElevationGain: 0 }),
  }
  cache.streams = {}

  const swim = buildAnalytics(cache, { since: '2026-05-01' }).bests.find(
    best => best.sport === 'swim',
  )

  assert.equal(swim?.fastestRate, 100)
  assert.deepEqual(swim?.bestToDate, [
    { date: iso(18), rate: 120 },
    { date: iso(22), rate: 100 },
  ])
})

test('lactate threshold projection stays a low-confidence training proxy with its model band', () => {
  const { cache } = fixtures()
  const durations = [1800, 1740, 1680, 1620, 1560, 1500]
  cache.activities = Object.fromEntries(
    durations.map((movingTime, index) => {
      const id = 20 + index
      const day = iso(-6 + index * 7)
      return [String(id), activity(id, 'Run', day, movingTime, 5000, { totalElevationGain: 0 })]
    }),
  )
  cache.streams = {}

  const analytics = buildAnalytics(cache, { since: '2026-05-01' })
  const projection = analytics.engine.lactateThreshold.sports.find(sport => sport.sport === 'run')

  assert.deepEqual(analytics.engine.lactateThreshold.heartRate, {
    value: ATHLETE.lt,
    unit: 'bpm',
    source: 'declared',
  })
  assert.ok(projection)
  assert.equal(projection.source, 'training-pace-trend')
  assert.equal(projection.method, 'ols')
  assert.equal(projection.conf, 'low')
  assert.equal(projection.horizonDays, 14)
  assert.equal(projection.points.length, 15)
  assert.ok(projection.projected != null && projection.projected < projection.current)
  assert.ok(projection.deltaPct != null && projection.deltaPct > 0)
  assert.ok(projection.low != null && projection.projected >= projection.low)
  assert.ok(projection.high != null && projection.projected <= projection.high)
})

test('suffer score flows into daily effort, activity summaries, and weekly totals', () => {
  const { cache, oura, weights } = fixtures()
  cache.activities['1'].sufferScore = 96
  cache.activities['2'].sufferScore = 41
  cache.activities['4'] = activity(4, 'Walk', iso(21), 1200, 1500, { sufferScore: 8 })
  cache.activities['5'] = activity(5, 'Yoga', iso(23), 1800, 0, { sufferScore: 16 })
  const a = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  assert.equal(a.daily.find(d => d.date === iso(20))?.effort, 96)
  assert.equal(a.daily.find(d => d.date === iso(21))?.effort, 8)
  assert.equal(a.daily.find(d => d.date === iso(22))?.effort, 41)
  assert.equal(a.daily.find(d => d.date === iso(23))?.effort, 16)
  assert.equal(a.daily.find(d => d.date === iso(24))?.effort, 0)
  assert.equal(a.activities.find(x => x.id === 1)?.effort, 96)
  assert.equal(a.activities.find(x => x.id === 2)?.effort, 41)
  assert.equal(a.activities.find(x => x.id === 3)?.effort, null)
  assert.equal(a.activities.find(x => x.id === 4)?.effort, 8)
  assert.equal(a.activities.find(x => x.id === 5)?.effort, 16)
  assert.equal(
    a.weekly.reduce((s, w) => s + w.effort, 0),
    161,
  )
  const scoredWeek = a.weekly.find(w => w.effort === 161)
  assert.ok(scoredWeek)
  assert.equal(scoredWeek.effortSessions, 4)
  const weekEnd = new Date(Date.parse(`${scoredWeek.weekStart}T00:00:00Z`) + 6 * DAY)
    .toISOString()
    .slice(0, 10)
  assert.equal(
    a.daily
      .filter(d => d.date >= scoredWeek.weekStart && d.date <= weekEnd)
      .reduce((sum, day) => sum + day.effort, 0),
    scoredWeek.effort,
  )
})

test('strength and yoga heart-rate load contributes to daily and weekly TSS', () => {
  const { cache, oura, weights } = fixtures()
  const strengthDay = iso(25)
  const yogaDay = iso(26)
  const missingHeartRateDay = iso(27)
  cache.activities['4'] = activity(4, 'WeightTraining', strengthDay, 1_200, 0, {
    averageHeartrate: undefined,
    maxHeartrate: undefined,
  })
  cache.activities['5'] = activity(5, 'Yoga', yogaDay, 1_500, 0, {
    averageHeartrate: undefined,
    maxHeartrate: undefined,
  })
  cache.activities['6'] = activity(6, 'WeightTraining', missingHeartRateDay, 1_800, 0, {
    averageHeartrate: undefined,
    maxHeartrate: undefined,
  })
  delete oura.days[strengthDay]
  const activityDetails = buildPayload(cache, null, null, '2026-05-12').details
  const withoutHeartRate = buildAnalytics(cache, {
    oura,
    weights,
    activityDetails,
    since: '2026-05-12',
  })
  const strengthDetail = activityDetails['4']
  assert.ok(strengthDetail)
  activityDetails['4'] = { ...strengthDetail, avgHr: 120.5, maxHr: 141 }
  cache.activities['5'].averageHeartrate = 83
  cache.activities['5'].maxHeartrate = 113

  const analytics = buildAnalytics(cache, { oura, weights, activityDetails, since: '2026-05-12' })

  assert.equal(analytics.activities.find(item => item.id === 4)?.load, 9.6)
  assert.equal(analytics.activities.find(item => item.id === 5)?.load, 3.4)
  assert.equal(analytics.activities.find(item => item.id === 6)?.load, 0)
  assert.equal(analytics.daily.find(day => day.date === strengthDay)?.load, 9.6)
  assert.equal(analytics.daily.find(day => day.date === yogaDay)?.load, 3.4)
  assert.equal(analytics.daily.find(day => day.date === missingHeartRateDay)?.load, 0)
  assert.equal(
    Math.round(analytics.weekly.reduce((total, week) => total + week.load, 0) * 10),
    Math.round(analytics.daily.reduce((total, day) => total + day.load, 0) * 10),
  )
  const withoutHeartRateState = withoutHeartRate.daily.at(-1)
  const state = analytics.daily.at(-1)
  assert.ok(withoutHeartRateState)
  assert.ok(state)
  assert.notDeepEqual(
    [state.ctl, state.atl, state.tsb],
    [withoutHeartRateState.ctl, withoutHeartRateState.atl, withoutHeartRateState.tsb],
  )
})

test('analytics emits scored non-tri weeks and calendar gaps', () => {
  const { cache } = fixtures()
  cache.lastSync = Date.parse('2026-06-11T10:00:00Z')
  cache.activities = {
    '10': activity(10, 'Yoga', '2026-05-15', 1800, 0, { sufferScore: 30 }),
    '11': activity(11, 'Yoga', '2026-06-01', 1800, 0, { sufferScore: 60 }),
  }
  cache.streams = {}

  const a = buildAnalytics(cache, { since: '2026-05-15' })

  assert.deepEqual(
    a.weekly.map(w => [w.weekStart, w.complete, w.sessions, w.load, w.effort]),
    [
      ['2026-05-11', false, 1, 0, 30],
      ['2026-05-18', true, 0, 0, 0],
      ['2026-05-25', true, 0, 0, 0],
      ['2026-06-01', true, 1, 0, 60],
      ['2026-06-08', false, 0, 0, 0],
    ],
  )
  assert.equal(a.daily.find(d => d.date === '2026-05-15')?.effort, 30)
  assert.equal(a.daily.find(d => d.date === '2026-06-01')?.effort, 60)
})

test('analytics treats late evening syncs as the local calendar day', () => {
  const env = {
    health: process.env.HEALTH_TIMEZONE,
    local: process.env.LOCAL_TIMEZONE,
    tz: process.env.TZ,
  }
  const { cache } = fixtures()
  cache.lastSync = Date.parse('2026-07-01T02:45:00.000Z')
  cache.activities = {
    '10': activity(10, 'Ride', '2026-06-30', 1800, 12000, {
      startDate: '2026-07-01T01:11:01Z',
      startDateLocal: '2026-06-30T21:11:01',
    }),
  }
  cache.streams = {}

  try {
    delete process.env.HEALTH_TIMEZONE
    delete process.env.LOCAL_TIMEZONE
    process.env.TZ = 'UTC'
    const a = buildAnalytics(cache, { since: '2026-06-01' })

    assert.equal(a.meta.today, '2026-06-30')
    assert.equal(a.meta.windowTo, '2026-06-30')
    assert.equal(a.daily.at(-1)?.date, '2026-06-30')
    assert.ok((a.daily.at(-1)?.load ?? 0) > 0)
    assert.equal(a.activities[0]?.date, '2026-06-30')
  } finally {
    if (env.health == null) delete process.env.HEALTH_TIMEZONE
    else process.env.HEALTH_TIMEZONE = env.health
    if (env.local == null) delete process.env.LOCAL_TIMEZONE
    else process.env.LOCAL_TIMEZONE = env.local
    if (env.tz == null) delete process.env.TZ
    else process.env.TZ = env.tz
  }
})

test('provider refreshes share one generated timestamp and current daily point', () => {
  const { cache, oura } = fixtures()
  const nextDay = '2026-06-12'
  const generatedAt = Date.parse(`${nextDay}T10:30:00Z`)
  oura.lastSync = generatedAt
  oura.days[nextDay] = ouraDay(nextDay, 91)

  const payload = buildPayload(cache, oura, null, '2026-05-12')
  const analytics = buildAnalytics(cache, { oura, since: '2026-05-12' })
  const feed = buildDataFeed(cache, analytics, { oura })
  const rows = feed
    .trimEnd()
    .split('\n')
    .map(line => JSON.parse(line))

  assert.equal(payload.generatedAt, generatedAt)
  assert.equal(payload.days.at(-1)?.date, nextDay)
  assert.equal(analytics.meta.today, nextDay)
  assert.equal(analytics.meta.windowTo, nextDay)
  assert.equal(analytics.daily.at(-1)?.date, nextDay)
  assert.equal(analytics.daily.at(-1)?.hrv, 91)
  assert.equal(rows.find(row => row.kind === 'meta')?.generatedAt, generatedAt)
})

test('garmin scale drives body composition, multi-weigh-in series, weight merge, and goal', () => {
  const { cache, oura, weights } = fixtures()
  const at = (offset: number, h: number): number =>
    Date.parse(`${iso(offset)}T${String(h).padStart(2, '0')}:00:00.000Z`)
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {},
    weight: [
      {
        ts: at(25, 7),
        date: iso(25),
        weightKg: 87.2,
        bmi: 26.9,
        bodyFatPct: 21.5,
        bodyWaterPct: 55.3,
        muscleMassKg: 35.4,
        boneMassKg: 3.7,
      },
      {
        ts: at(28, 7),
        date: iso(28),
        weightKg: 87,
        bmi: 26.8,
        bodyFatPct: 21.3,
        bodyWaterPct: null,
        muscleMassKg: null,
        boneMassKg: null,
      },
      {
        ts: at(28, 21),
        date: iso(28),
        weightKg: 86.8,
        bmi: 26.7,
        bodyFatPct: 21.1,
        bodyWaterPct: null,
        muscleMassKg: null,
        boneMassKg: null,
      },
    ],
  }
  const a = buildAnalytics(cache, { oura, garmin, weights, since: '2026-05-12' })
  const b = a.body
  assert.equal(b.latestKg, 86.8)
  assert.equal(b.goalKg != null && Math.round(b.goalKg), 77)
  assert.equal(b.goalLbs, 170)
  assert.equal(b.goalDeltaKg, 9.7)
  assert.ok(b.trendKgPerWeek != null && b.trendKgPerWeek < 0)
  assert.ok(b.goalEtaWeeks != null && b.goalEtaWeeks > 0 && b.goalEtaWeeks <= 104)
  assert.equal(b.bodyFatPct, 21.1)
  assert.equal(b.bodyWaterPct, 55.3)
  assert.equal(b.muscleMassKg, 35.4)
  assert.equal(b.boneMassKg, 3.7)
  assert.equal(b.bmi, 26.7)
  assert.equal(b.ffmi, 19.38)
  assert.equal(b.series.length, 4)
  assert.deepEqual(
    b.ffmiSeries.map(p => p.ffmi),
    [19.37, 19.37, 19.38],
  )
  const day28 = b.series.filter(p => p.date === iso(28))
  assert.equal(day28.length, 2)
  assert.ok(day28[0].ts < day28[1].ts)
  assert.deepEqual(
    b.series.map(p => p.kg),
    [88.5, 87.2, 87, 86.8],
  )
  assert.equal(b.composition.length, 3)
  assert.equal(b.composition[0].ffmi, 19.37)
  const day = a.daily.find(d => d.date === iso(26))
  assert.equal(day?.weightKg, 87.2)
  const feed = buildDataFeed(cache, a, { oura, garmin, weights, zones: cache.zones })
  const rows = feed
    .trimEnd()
    .split('\n')
    .map(l => JSON.parse(l))
  assert.equal(rows[0].athlete.weightGoalKg != null && Math.round(rows[0].athlete.weightGoalKg), 77)
  const scaleDay = rows.find(r => r.kind === 'day' && r.date === iso(25))
  assert.equal(scaleDay.bmi, 26.9)
  assert.equal(scaleDay.ffmi, 19.37)
  assert.equal(scaleDay.bodyFatPct, 21.5)
  const plainDay = rows.find(r => r.kind === 'day' && r.date === iso(20))
  assert.equal(plainDay.bmi, null)
  assert.equal(plainDay.ffmi, null)
  assert.equal(plainDay.muscleMassKg, null)
})

test('body block reports goal-weight bmr and ffmi from dexa fat-free mass', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics(cache, {
    oura,
    weights,
    since: '2026-05-12',
    dexa: [
      {
        date: '2026-06-25',
        totalLbs: 197.6,
        fatLbs: 54.2,
        leanLbs: 135.7,
        bmcLbs: 7.8,
        ffmLbs: 143.5,
        bodyFat: 27.4,
      },
    ],
  })
  assert.equal(a.body.goalBmr, 1826)
  assert.equal(a.body.goalLeanBmr, 1776)
  assert.equal(a.tests.dexa[0].ffmi, 18.42)
  assert.equal(a.body.ffmi, 18.42)
})

test('apple vo2max wins the estimate priority when present', () => {
  const { cache, oura, weights } = fixtures()
  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {
      [iso(28)]: {
        date: iso(28),
        burnKcal: null,
        activeKcal: null,
        intakeKcal: null,
        weightKg: null,
        vo2max: 45.2,
      },
    },
  }
  const a = buildAnalytics(cache, { oura, apple, weights, since: '2026-05-12' })
  assert.equal(a.engine.vo2max.method, 'apple')
  assert.equal(a.engine.vo2max.value, 45.2)
  assert.ok(a.engine.vo2max.estimates.length >= 2)
})

test('apple swim strokes flow into activity summaries and data feed', () => {
  const { cache, oura, weights } = fixtures()
  const swimDay = iso(24)
  const strokes = { freestyle: 1200, breaststroke: 300 }
  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {},
    swims: {
      [swimDay]: {
        id: null,
        date: swimDay,
        start: null,
        end: null,
        totalM: 1500,
        laps: 60,
        activeTimeS: null,
        strokeCount: null,
        strokeTimeS: null,
        strokes,
        location: 'pool',
        waterTemperatureC: 27.8,
      },
    },
  }
  const a = buildAnalytics(cache, { oura, apple, weights, since: '2026-05-12' })
  const swim = a.activities.find(r => r.sport === 'swim')
  const run = a.activities.find(r => r.sport === 'run')
  assert.deepEqual(swim?.strokes, strokes)
  assert.equal(run?.strokes, null)

  const feed = buildDataFeed(cache, a, { oura, apple, weights, zones: cache.zones })
  const feedSwim = feed
    .trimEnd()
    .split('\n')
    .map(l => JSON.parse(l))
    .find(r => r.kind === 'activity' && r.sport === 'swim')
  assert.deepEqual(feedSwim?.strokes, strokes)
})

test('swim radar uses separate same-day activity pace and stroke rate samples', () => {
  const { cache, oura, weights } = fixtures()
  const swimDay = iso(24)
  cache.activities['4'] = activity(4, 'Swim', swimDay, 1500, 1000, {
    startDate: `${swimDay}T16:00:00Z`,
    startDateLocal: `${swimDay}T12:00:00Z`,
    averageCadence: undefined,
  })
  const firstStrokes = { freestyle: 600 }
  const secondStrokes = { freestyle: 300 }
  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {},
    swims: {
      first: {
        id: 'first',
        date: swimDay,
        start: `${swimDay}T12:00:00Z`,
        end: `${swimDay}T12:30:00Z`,
        totalM: 1500,
        laps: 60,
        activeTimeS: 1800,
        strokeCount: 600,
        strokeTimeS: 1200,
        strokes: firstStrokes,
        location: 'pool',
        waterTemperatureC: 27.8,
      },
      second: {
        id: 'second',
        date: swimDay,
        start: `${swimDay}T16:00:00Z`,
        end: `${swimDay}T16:25:00Z`,
        totalM: 1000,
        laps: 40,
        activeTimeS: 1500,
        strokeCount: 300,
        strokeTimeS: 900,
        strokes: secondStrokes,
        location: 'pool',
        waterTemperatureC: 27.8,
      },
    },
  }

  const a = buildAnalytics(cache, { oura, apple, weights, since: '2026-05-12' })
  const swim = a.engine.abilities.sports.find(sport => sport.sport === 'swim')
  const pace = swim?.axes.find(axis => axis.key === 'climb')
  const strokeRate = swim?.axes.find(axis => axis.key === 'cadence')

  assert.equal(pace?.label, 'pace')
  assert.equal(pace?.rawUnit, 's/100m')
  assert.equal(pace?.rawValue, 120)
  assert.equal(pace?.score, 76)
  assert.equal(strokeRate?.label, 'stroke rate')
  assert.equal(strokeRate?.rawUnit, 'str/min')
  assert.equal(strokeRate?.rawValue, 25)
  assert.equal(strokeRate?.score, 67)
  assert.equal(swim?.history.at(-1)?.climb, 76)
  assert.equal(swim?.history.at(-1)?.cadence, 67)
  assert.deepEqual(a.activities.find(row => row.id === 3)?.strokes, firstStrokes)
  assert.deepEqual(a.activities.find(row => row.id === 4)?.strokes, secondStrokes)
})

test('swim radar rejects invalid Apple metrics and falls back to Strava activity pace', () => {
  const { cache, oura, weights } = fixtures()
  const swimDay = iso(24)
  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {},
    swims: {
      invalid: {
        id: 'invalid',
        date: swimDay,
        start: `${swimDay}T12:00:00Z`,
        end: `${swimDay}T12:30:00Z`,
        totalM: 1500,
        laps: 60,
        activeTimeS: 100,
        strokeCount: 600,
        strokeTimeS: 0,
        strokes: {},
        location: 'pool',
        waterTemperatureC: 27.8,
      },
    },
  }

  const a = buildAnalytics(cache, { oura, apple, weights, since: '2026-05-12' })
  const swim = a.engine.abilities.sports.find(sport => sport.sport === 'swim')
  const pace = swim?.axes.find(axis => axis.key === 'climb')
  const strokeRate = swim?.axes.find(axis => axis.key === 'cadence')

  assert.equal(pace?.rawValue, 120)
  assert.equal(pace?.score, 76)
  assert.equal(strokeRate?.rawValue, null)
  assert.equal(strokeRate?.score, null)
})

test('data feed emits meta, ordered kinds, fixed fields, and explicit nulls', () => {
  const { cache, oura, weights, weather } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, weather, since: '2026-05-12' })
  const feed = buildDataFeed(cache, a, {
    oura,
    weather,
    weights,
    trainingExclusions: [{ date: iso(20), activityId: 1 }],
    zones: cache.zones,
  })
  assert.ok(feed.endsWith('\n'))
  const lines = feed.trimEnd().split('\n')
  const rows = lines.map(l => JSON.parse(l))
  assert.equal(rows[0].kind, 'meta')
  assert.equal(rows[0].v, 5)
  assert.equal(rows[0].criticalPower, null)
  assert.equal(rows[0].criticalPowerYear, null)
  assert.deepEqual(rows[0].fields.day, [...DAY_FIELDS])
  assert.deepEqual(rows[0].fields.activity, [...ACTIVITY_FIELDS])
  assert.deepEqual(rows[0].fields.week, [...WEEK_FIELDS])
  assert.equal(rows[0].counts.activity, 3)
  assert.equal(rows[0].athlete.sex, 'M')
  assert.equal(rows[0].athlete.born, '2001-03')
  assert.equal(rows[0].athlete.ageYears, 25)
  assert.equal(rows[0].athlete.heightCm, ATHLETE.heightCm)
  assert.equal(rows[0].athlete.hrMaxEst, ATHLETE.hrMax)
  const kinds = rows.map(r => r.kind)
  const order = ['meta', 'day', 'activity', 'week']
  assert.deepEqual([...new Set(kinds)], order)
  let lastRank = 0
  for (const k of kinds) {
    const rank = order.indexOf(k)
    assert.ok(rank >= lastRank)
    lastRank = rank
  }
  const days = rows.filter(r => r.kind === 'day')
  for (let i = 1; i < days.length; i++) assert.ok(days[i].date > days[i - 1].date)
  for (const d of days) {
    assert.ok('intakeKcal' in d)
    assert.ok('windDir' in d)
    assert.ok('windGustKph' in d)
    assert.ok('readinessNext' in d)
  }
  const windDay = days.find(d => d.date === iso(15))
  assert.equal(windDay?.windKph, 15)
  assert.equal(windDay?.windDir, 'NW')
  const weatherDay = days.find(d => d.date === iso(20))
  assert.equal(weatherDay?.windKph, 18)
  assert.equal(weatherDay?.windDir, 'W')
  assert.equal(weatherDay?.windGustKph, 29)
  const trained = days.find(d => d.date === iso(20))
  assert.equal(trained?.sessions, 1)
  assert.ok(trained?.hrv != null)
  assert.ok(trained?.readinessNext != null)
  const week = rows.find(r => r.kind === 'week')
  assert.ok('sessions' in week)
  assert.ok('swimKm' in week)
  assert.ok('bikeHours' in week)
  assert.ok('runHours' in week)
  const ride = rows.find(r => r.kind === 'activity' && r.id === 1)
  assert.equal(ride?.windKph, 18)
  assert.equal(ride?.windDir, 'W')
  assert.equal(ride?.windGustKph, 29)
  assert.equal(ride?.avgTemp, 22)
  assert.equal(ride?.skipTraining, true)
  const run = rows.find(r => r.kind === 'activity' && r.id === 2)
  assert.equal(run?.skipTraining, false)
})

test('data feed prefers Garmin run heart rate over Strava heart rate', () => {
  const { cache, oura, weights } = fixtures()
  const run = cache.activities['2']
  assert.ok(run)
  const metrics = emptyGarminMetrics()
  metrics.avgHeartRate = 141
  metrics.maxHeartRate = 169
  const garmin: GarminCache = {
    lastSync: cache.lastSync,
    activities: {
      run: {
        id: 'run',
        name: 'Run 2',
        sport: 'run',
        startDate: run.startDate,
        startDateLocal: run.startDateLocal,
        distanceM: run.distance * 2,
        movingTimeS: run.movingTime * 2,
        elapsedTimeS: run.elapsedTime * 2,
        sourceDevice: null,
        sourceFile: null,
        metrics,
        fueling: emptyGarminFueling(),
      },
    },
  }

  const a = buildAnalytics(cache, { oura, garmin, weights, since: '2026-05-12' })
  const feed = buildDataFeed(cache, a, { oura, garmin, weights, zones: cache.zones })
  const rows = feed
    .trimEnd()
    .split('\n')
    .map(l => JSON.parse(l))
  const runRow = rows.find(row => row.kind === 'activity' && row.id === 2)
  const bikeRow = rows.find(row => row.kind === 'activity' && row.id === 1)
  assert.equal(runRow?.avgHr, 141)
  assert.equal(runRow?.maxHr, 169)
  assert.equal(bikeRow?.avgHr, 152)
})

test('data feed preserves Apple daily fallback values', () => {
  const { cache, oura } = fixtures()
  const day = iso(22)
  const o = oura.days[day]
  assert.ok(o)
  oura.days[day] = { ...o, totalCalories: null, activeCalories: null }
  const apple: AppleCache = {
    lastSync: cache.lastSync,
    days: {
      [day]: {
        date: day,
        burnKcal: 2310,
        activeKcal: 410,
        intakeKcal: 2800,
        weightKg: 87.2,
        vo2max: null,
      },
    },
  }
  const a = buildAnalytics(cache, { oura, apple, since: '2026-05-12' })
  const feed = buildDataFeed(cache, a, { oura, apple, zones: cache.zones })
  const row = feed
    .trimEnd()
    .split('\n')
    .map(l => JSON.parse(l))
    .find(r => r.kind === 'day' && r.date === day)

  assert.equal(row?.totalCalories, 2310)
  assert.equal(row?.activeCalories, 410)
  assert.equal(row?.intakeKcal, 2800)
  assert.equal(row?.weightKg, 87.2)
})

test('data feed derives stream features on 1hz activities and nulls them on swims', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  const feed = buildDataFeed(cache, a, { oura, weights, zones: cache.zones })
  const rows = feed
    .trimEnd()
    .split('\n')
    .map(l => JSON.parse(l))
  const bike = rows.find(r => r.kind === 'activity' && r.sport === 'bike')
  const run = rows.find(r => r.kind === 'activity' && r.sport === 'run')
  const swim = rows.find(r => r.kind === 'activity' && r.sport === 'swim')
  assert.equal(bike.pp30, 200)
  assert.equal(bike.pp1200, 200)
  assert.equal(bike.deviceWatts, true)
  assert.ok(bike.decoupling != null)
  assert.ok(bike.ef != null)
  assert.ok(run.ps30 != null)
  assert.ok(run.decoupling != null)
  assert.equal(swim.pp30, null)
  assert.equal(swim.ps30, null)
  assert.equal(swim.decoupling, null)
})

test('manual sauna sessions appear in analytics and the data feed without entering sport load', () => {
  const { cache, oura } = fixtures()
  const date = iso(25)
  const payload = buildPayload(cache, oura, null, '2026-05-12')
  applyManualSauna(
    payload,
    [
      {
        id: 8_202_606_061_830,
        stravaActivityId: null,
        title: 'Untangle',
        date,
        time: '18:30',
        durationS: 4_500,
        temperatureC: 91.111,
        humidityPct: 11,
        cooldown: 'cold plunge',
        heatTrainingLoad: 7.7,
      },
    ],
    [],
    'America/Toronto',
  )
  const analytics = buildAnalytics(cache, {
    oura,
    activityDetails: payload.details,
    since: '2026-05-12',
  })
  const summary = analytics.activities.find(activity => activity.sport === 'sauna')
  assert.deepEqual(summary, {
    id: 8_202_606_061_830,
    date,
    sport: 'sauna',
    name: 'Untangle',
    distanceKm: 0,
    movingTimeS: 4_500,
    load: 0,
    paceIntensityFactor: null,
    effort: null,
    cadence: null,
    strokes: null,
    windKph: null,
    windDir: null,
    windGustKph: null,
  })

  const feed = buildDataFeed(cache, analytics, {
    oura,
    zones: cache.zones,
    activityDetails: payload.details,
  })
  const rows = feed
    .trimEnd()
    .split('\n')
    .map(line => JSON.parse(line))
  const activity = rows.find(row => row.kind === 'activity' && row.sport === 'sauna')
  assert.equal(activity?.name, 'Untangle')
  assert.equal(activity?.skipTraining, true)
  assert.equal(activity?.load, 0)
  assert.deepEqual(activity?.sauna, payload.details[String(activity.id)].sauna)
  assert.equal(rows.find(row => row.kind === 'day' && row.date === date)?.sessions, 1)
  assert.equal(rows.find(row => row.kind === 'meta')?.v, 5)
})

test('attached sauna activity replaces its raw Strava classification in analytics', () => {
  const { cache, oura, weather } = fixtures()
  const date = iso(25)
  cache.activities['4'] = activity(4, 'Workout', date, 4_500, 0, { name: 'Sauna', sufferScore: 30 })
  weather.activities['4'] = {
    activityId: 4,
    date,
    start: `${date}T22:30:00Z`,
    end: `${date}T23:45:00Z`,
    latitude: 43.6,
    longitude: -79.4,
    durationS: 4_500,
    windKph: 10,
    windDir: 'NNW',
    windDirDeg: 338,
    windGustKph: 16,
    temperatureC: 27,
    source: 'weatherkit',
  }
  const payload = buildPayload(cache, oura, null, '2026-05-12')
  applyManualSauna(
    payload,
    [
      {
        id: 8_202_606_061_830,
        stravaActivityId: 4,
        title: 'Untangle',
        date,
        time: '18:30',
        durationS: 4_500,
        temperatureC: 91.111,
        humidityPct: 11,
        cooldown: 'cold plunge',
        heatTrainingLoad: 7.7,
      },
    ],
    [],
    'America/Toronto',
    weather,
  )
  const analytics = buildAnalytics(cache, {
    oura,
    weather,
    activityDetails: payload.details,
    since: '2026-05-12',
  })
  const matching = analytics.activities.filter(candidate => candidate.id === 4)

  assert.deepEqual(
    matching.map(candidate => candidate.sport),
    ['sauna'],
  )
  assert.equal(analytics.meta.activityCount, analytics.activities.length)
  assert.equal(analytics.daily.find(candidate => candidate.date === date)?.load, 0)
  assert.deepEqual(
    matching.map(candidate => [candidate.windKph, candidate.windDir, candidate.windGustKph]),
    [[10, 'NNW', 16]],
  )

  const feed = buildDataFeed(cache, analytics, {
    oura,
    weather,
    zones: cache.zones,
    activityDetails: payload.details,
  })
  const rows = feed
    .trimEnd()
    .split('\n')
    .map(line => JSON.parse(line))
  assert.equal(rows.filter(row => row.kind === 'activity' && row.id === 4).length, 1)
  assert.equal(rows.find(row => row.kind === 'activity' && row.id === 4)?.sport, 'sauna')
  assert.equal(rows.find(row => row.kind === 'activity' && row.id === 4)?.avgTemp, 27)
  assert.equal(rows.find(row => row.kind === 'activity' && row.id === 4)?.windKph, 10)
  assert.equal(rows.find(row => row.kind === 'activity' && row.id === 4)?.windDir, 'NNW')
  assert.equal(rows.find(row => row.kind === 'activity' && row.id === 4)?.windGustKph, 16)
  assert.equal(rows.find(row => row.kind === 'day' && row.date === date)?.sessions, 1)
})

test('data feed never leaks coordinates or secrets', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  const feed = buildDataFeed(cache, a, { oura, weights, zones: cache.zones })
  assert.doesNotMatch(feed, /latlng|polyline|refreshToken|"lat"|"lng"|43\.6|-79\.4/)
})

test('data feed degrades to a single meta line without a cache', () => {
  const feed = buildDataFeed(null, buildAnalytics(null), {})
  const lines = feed.trimEnd().split('\n')
  assert.equal(lines.length, 1)
  const meta = JSON.parse(lines[0])
  assert.equal(meta.kind, 'meta')
  assert.deepEqual(meta.counts, { day: 0, activity: 0, week: 0 })
  assert.equal(meta.athlete.ageYears, null)
})
