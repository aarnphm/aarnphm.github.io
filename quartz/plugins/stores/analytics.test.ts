import assert from 'node:assert/strict'
import test from 'node:test'
import type { AppleCache } from './apple'
import type { GarminCache } from './garmin'
import type { OuraCache, OuraDaily } from './oura'
import type { RawStravaActivity, StravaRawCache, StravaStreams } from './strava'
import type { TrackEntry } from './tracking'
import type { WeatherCache } from './weather'
import {
  ACTIVITY_FIELDS,
  DAY_FIELDS,
  WEEK_FIELDS,
  buildAnalytics,
  buildDataFeed,
} from './analytics'
import { emptyGarminFueling, emptyGarminMetrics } from './garmin'

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

test('recovery block computes baselines, series, and flags from oura-merged daily', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  assert.equal(a.recovery.status, 'firm')
  assert.ok(a.recovery.baselineDays >= 14)
  assert.ok(a.recovery.hrvLatest != null && a.recovery.hrvLatest >= 80)
  assert.equal(a.recovery.rhrLatest, 50)
  assert.ok(a.recovery.series.length >= 16)
  assert.ok(a.recovery.sleepDebtS > 0)
  assert.ok(a.recovery.flags.every(f => ['info', 'watch', 'alert'].includes(f.severity)))
  assert.equal(a.recovery.thresholds.sleepTargetS, 28800)
  const day = a.daily.find(d => d.date === iso(20))
  assert.equal(day?.sleepDurationS, 27000)
  assert.equal(day?.tempDevC, 0.1)
})

test('engine block bases vo2max on the declared strava ftp and builds six radar axes', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  const v = a.engine.vo2max
  assert.equal(v.method, 'bike')
  assert.equal(v.conf, 'firm')
  assert.ok(v.note.includes('ftp 213w (strava)'))
  assert.ok(v.value != null && v.value > 25 && v.value < 50)
  assert.ok(v.fitnessAge != null && v.fitnessAge >= 20 && v.fitnessAge <= 80)
  assert.equal(v.chronoAge, 25)
  assert.equal(v.hrMax, 190)
  assert.equal(v.hrMaxSource, 'declared')
  assert.ok(v.trend.length >= 1)
  assert.equal(a.engine.abilities.axes.length, 6)
  const keys = a.engine.abilities.axes.map(x => x.key)
  assert.deepEqual(keys, ['sprint', 'threshold', 'endurance', 'climb', 'cadence', 'recovery'])
  assert.ok(a.engine.cardio.metrics.length === 4)
  assert.ok(a.engine.cardio.rhrSeries.length >= 16)
})

test('engine derives ftp from 20-min power only when strava declares none', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics({ ...cache, zones: undefined }, { oura, weights, since: '2026-05-12' })
  const v = a.engine.vo2max
  assert.equal(v.method, 'bike')
  assert.equal(v.conf, 'low')
  assert.ok(v.note.includes('ftp 190w ·'))
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
  assert.equal(b.goalKg != null && Math.round(b.goalKg), 82)
  assert.equal(b.goalLbs, 180)
  assert.equal(b.goalDeltaKg, 5.2)
  assert.ok(b.trendKgPerWeek != null && b.trendKgPerWeek < 0)
  assert.ok(b.goalEtaWeeks != null && b.goalEtaWeeks > 0 && b.goalEtaWeeks <= 104)
  assert.equal(b.bodyFatPct, 21.1)
  assert.equal(b.bodyWaterPct, 55.3)
  assert.equal(b.muscleMassKg, 35.4)
  assert.equal(b.boneMassKg, 3.7)
  assert.equal(b.bmi, 26.7)
  assert.equal(b.series.length, 4)
  const day28 = b.series.filter(p => p.date === iso(28))
  assert.equal(day28.length, 2)
  assert.ok(day28[0].ts < day28[1].ts)
  assert.deepEqual(
    b.series.map(p => p.kg),
    [88.5, 87.2, 87, 86.8],
  )
  assert.equal(b.composition.length, 3)
  const day = a.daily.find(d => d.date === iso(26))
  assert.equal(day?.weightKg, 87.2)
  const feed = buildDataFeed(cache, a, { oura, garmin, weights, zones: cache.zones })
  const rows = feed
    .trimEnd()
    .split('\n')
    .map(l => JSON.parse(l))
  assert.equal(rows[0].athlete.weightGoalKg != null && Math.round(rows[0].athlete.weightGoalKg), 82)
  const scaleDay = rows.find(r => r.kind === 'day' && r.date === iso(25))
  assert.equal(scaleDay.bmi, 26.9)
  assert.equal(scaleDay.bodyFatPct, 21.5)
  const plainDay = rows.find(r => r.kind === 'day' && r.date === iso(20))
  assert.equal(plainDay.bmi, null)
  assert.equal(plainDay.muscleMassKg, null)
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

test('data feed emits meta, ordered kinds, fixed fields, and explicit nulls', () => {
  const { cache, oura, weights, weather } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, weather, since: '2026-05-12' })
  const feed = buildDataFeed(cache, a, { oura, weather, weights, zones: cache.zones })
  assert.ok(feed.endsWith('\n'))
  const lines = feed.trimEnd().split('\n')
  const rows = lines.map(l => JSON.parse(l))
  assert.equal(rows[0].kind, 'meta')
  assert.equal(rows[0].v, 1)
  assert.deepEqual(rows[0].fields.day, [...DAY_FIELDS])
  assert.deepEqual(rows[0].fields.activity, [...ACTIVITY_FIELDS])
  assert.deepEqual(rows[0].fields.week, [...WEEK_FIELDS])
  assert.equal(rows[0].counts.activity, 3)
  assert.equal(rows[0].athlete.sex, 'M')
  assert.equal(rows[0].athlete.born, '2001-03')
  assert.equal(rows[0].athlete.ageYears, 25)
  assert.equal(rows[0].athlete.hrMaxEst, 190)
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
  const ride = rows.find(r => r.kind === 'activity' && r.id === 1)
  assert.equal(ride?.windKph, 18)
  assert.equal(ride?.windDir, 'W')
  assert.equal(ride?.windGustKph, 29)
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
