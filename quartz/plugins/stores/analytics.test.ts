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
  computeFtpHypothesisFromVo2,
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
  assert.equal(a.recovery.sleepDebtS, 0)
  assert.ok(a.recovery.flags.every(f => ['info', 'watch', 'alert'].includes(f.severity)))
  assert.equal(a.recovery.thresholds.sleepTargetS, 25200)
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
  assert.equal(v.hrMax, 184)
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
      ['sprint', 'threshold', 'endurance', 'climb', 'cadence', 'recovery'],
    )
    assert.ok(s.history.length >= 2)
    assert.ok(s.area != null && s.area > 0)
    for (const x of s.axes) {
      if (x.score == null) assert.equal(x.proj, null)
      else assert.ok(x.proj != null && x.proj >= 0 && x.proj <= 100)
    }
  }
  const [swim, bike, run] = a.engine.abilities.sports
  assert.equal(swim.axes.find(x => x.key === 'climb')?.score, null)
  assert.equal(swim.axes.find(x => x.key === 'cadence')?.score, null)
  assert.equal(swim.axes.find(x => x.key === 'sprint')?.rawUnit, 'm/s')
  assert.equal(bike.axes.find(x => x.key === 'sprint')?.rawUnit, 'w/kg')
  assert.equal(bike.axes.find(x => x.key === 'threshold')?.rawUnit, 'w/kg')
  assert.equal(run.axes.find(x => x.key === 'cadence')?.rawValue, 176)
  assert.equal(run.axes.find(x => x.key === 'threshold')?.score, null)
  const bikeEnd = bike.axes.find(x => x.key === 'endurance')
  assert.equal(bikeEnd?.hi, 50)
  const runLast = run.history[run.history.length - 1]
  assert.ok(runLast.sprint != null)
  assert.equal(runLast.climb != null, true)
  const swimLast = swim.history[swim.history.length - 1]
  assert.equal(swimLast.climb, null)
})

test('engine derives ftp from 20-min power only when strava declares none', () => {
  const { cache, oura, weights } = fixtures()
  const a = buildAnalytics({ ...cache, zones: undefined }, { oura, weights, since: '2026-05-12' })
  const v = a.engine.vo2max
  assert.equal(v.method, 'bike')
  assert.equal(v.conf, 'low')
  assert.ok(v.note.includes('ftp 190w ·'))
})

test('vo2 lab ftp hypothesis keeps the treadmill-to-bike estimate broad', () => {
  const h = computeFtpHypothesisFromVo2('2026-06-25', 47.8, 88.9)
  assert.ok(h)
  assert.equal(h.absoluteRunningVo2, 4.25)
  assert.equal(h.cyclingVo2max, 3.91)
  assert.equal(h.thresholdVo2, 3.32)
  assert.equal(h.efficiencyFtp, 243)
  assert.equal(h.acsmFtp, 224)
  assert.equal(h.ftp, 230)
  assert.equal(h.low, 210)
  assert.equal(h.high, 260)
  assert.equal(h.wattsPerKg, 2.59)
  assert.equal(h.conf, 'low')
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
  assert.ok(v.note.includes('ftp 230w (athlete)'))
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
  const lastWeek = a.weekly[a.weekly.length - 1]
  assert.equal(lastWeek.sessions, 2)
  assert.equal(lastWeek.runKm, 10)
  assert.equal(lastWeek.runHours, 0.8)
})

test('suffer score flows into daily effort, activity summaries, and weekly totals', () => {
  const { cache, oura, weights } = fixtures()
  cache.activities['1'].sufferScore = 96
  cache.activities['2'].sufferScore = 41
  const a = buildAnalytics(cache, { oura, weights, since: '2026-05-12' })
  assert.equal(a.daily.find(d => d.date === iso(20))?.effort, 96)
  assert.equal(a.daily.find(d => d.date === iso(22))?.effort, 41)
  assert.equal(a.daily.find(d => d.date === iso(24))?.effort, 0)
  assert.equal(a.activities.find(x => x.id === 1)?.effort, 96)
  assert.equal(a.activities.find(x => x.id === 2)?.effort, 41)
  assert.equal(a.activities.find(x => x.id === 3)?.effort, null)
  assert.equal(
    a.weekly.reduce((s, w) => s + w.effort, 0),
    137,
  )
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
  assert.equal(b.goalKg != null && Math.round(b.goalKg), 73)
  assert.equal(b.goalLbs, 160)
  assert.equal(b.goalDeltaKg, 14.2)
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
  assert.equal(rows[0].athlete.weightGoalKg != null && Math.round(rows[0].athlete.weightGoalKg), 73)
  const scaleDay = rows.find(r => r.kind === 'day' && r.date === iso(25))
  assert.equal(scaleDay.bmi, 26.9)
  assert.equal(scaleDay.bodyFatPct, 21.5)
  const plainDay = rows.find(r => r.kind === 'day' && r.date === iso(20))
  assert.equal(plainDay.bmi, null)
  assert.equal(plainDay.muscleMassKg, null)
})

test('body block reports goal-weight bmr estimates from athlete goal and dexa lean mass', () => {
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
  assert.equal(a.body.goalBmr, 1781)
  assert.equal(a.body.goalLeanBmr, 1776)
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
    swims: { [swimDay]: { date: swimDay, totalM: 1500, laps: 60, strokes } },
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

test('data feed emits meta, ordered kinds, fixed fields, and explicit nulls', () => {
  const { cache, oura, weights, weather } = fixtures()
  const a = buildAnalytics(cache, { oura, weights, weather, since: '2026-05-12' })
  const feed = buildDataFeed(cache, a, { oura, weather, weights, zones: cache.zones })
  assert.ok(feed.endsWith('\n'))
  const lines = feed.trimEnd().split('\n')
  const rows = lines.map(l => JSON.parse(l))
  assert.equal(rows[0].kind, 'meta')
  assert.equal(rows[0].v, 2)
  assert.deepEqual(rows[0].fields.day, [...DAY_FIELDS])
  assert.deepEqual(rows[0].fields.activity, [...ACTIVITY_FIELDS])
  assert.deepEqual(rows[0].fields.week, [...WEEK_FIELDS])
  assert.equal(rows[0].counts.activity, 3)
  assert.equal(rows[0].athlete.sex, 'M')
  assert.equal(rows[0].athlete.born, '2001-03')
  assert.equal(rows[0].athlete.ageYears, 25)
  assert.equal(rows[0].athlete.hrMaxEst, 184)
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
