import {
  Encoder,
  Profile,
  type FileIdMesg,
  type RecordMesg,
  type EventMesg,
  type SessionMesg,
} from '@garmin/fitsdk'
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, utimesSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test, { type TestContext } from 'node:test'
import type { ChangeEvent } from '../types/plugin'
import type { FilePath, FullSlug, SimpleSlug } from './path'
import { isActivityDetail } from '../components/triathlon/activity/data'
import { emptyGarminMetrics, emptyGarminFueling, type GarminCache } from '../plugins/stores/garmin'
import { applyActivityTracking, buildPayload, type StravaRawCache } from '../plugins/stores/strava'
import { parseTrackingBlock } from '../plugins/stores/tracking'
import { defaultProcessedContent } from '../plugins/vfile'
import { moreStatRows, activityTableRows } from './triathlon-card'
import {
  loadTrackedWahooFits,
  trackedWahooChangeEvents,
  wahooTrackingStamp,
} from './wahoo-tracking'

const START = new Date('2026-09-06T16:00:00Z')
const FIT_PATH = 'triathlon/wahoo/ride.fit'
const entry = parseTrackingBlock(
  null,
  `activity: 101\ngarmin: 55\nvirtual: true\nwahoo: [[${FIT_PATH}]]`,
)!.activity!
function fit(power = 200, sport: 'cycling' | 'running' = 'cycling'): Uint8Array {
  const encoder = new Encoder()
  encoder.onMesg(Profile.MesgNum.FILE_ID, {
    type: 'activity',
    manufacturer: 'wahooFitness',
    productName: 'ELEMNT BOLT',
    timeCreated: START,
  } as FileIdMesg)
  for (const [index, second] of [0, 1, 2, 9, 11].entries())
    encoder.onMesg(Profile.MesgNum.RECORD, {
      timestamp: new Date(START.getTime() + second * 1000),
      distance: second * 1200,
      ...(index === 2 ? {} : { power: index === 1 ? 0 : power + second * 10 }),
      heartRate: 130 + second,
      cadence: index === 1 ? 0 : 80,
      temperature: 18 + index,
      leftTorqueEffectiveness: 80,
      rightTorqueEffectiveness: 82,
    } as RecordMesg)
  encoder.onMesg(Profile.MesgNum.EVENT, {
    timestamp: new Date(START.getTime() + 6000),
    event: 'rearGearChange',
    eventType: 'marker',
    frontGearNum: 2,
    frontGear: 54,
    rearGearNum: 5,
    rearGear: 21,
  } as EventMesg)
  encoder.onMesg(Profile.MesgNum.SESSION, {
    timestamp: new Date(START.getTime() + 11000),
    startTime: START,
    sport,
    totalDistance: 12000,
    totalTimerTime: 10,
    totalElapsedTime: 11,
    avgPower: power,
    normalizedPower: power + 20,
    maxPower: 400,
    avgHeartRate: 135,
    maxHeartRate: 145,
    avgCadence: 80,
    totalCalories: 25,
    totalWork: 2000,
    intensityFactor: 0.7,
    trainingStressScore: 0,
  } as SessionMesg)
  return encoder.close()
}
function sources() {
  const time = Array.from({ length: 11 }, (_, i) => i)
  const strava: StravaRawCache = {
    athleteId: 1,
    auth: { refreshToken: '', obtainedAt: 0 },
    lastSync: START.getTime(),
    lastActivityStart: 0,
    activities: {
      '101': {
        id: 101,
        name: 'Rouvy',
        sportType: 'VirtualRide',
        startDate: new Date(START.getTime() + 1000).toISOString(),
        startDateLocal: '2026-09-06T12:00:01Z',
        distance: 10000,
        movingTime: 10,
        elapsedTime: 10,
        totalElevationGain: 80,
        averageSpeed: 1000,
        averageWatts: 150,
        weightedAverageWatts: 170,
        averageHeartrate: 120,
        averageCadence: 60,
      },
    },
    streams: {
      '101': {
        time,
        latlng: time.map(i => [45 + i / 1000, 6 + i / 1000]),
        altitude: time.map(i => 100 + i),
        distance: time.map(i => i * 1000),
        watts: time.map(i => 100 + i),
        heartrate: time.map(() => 120),
        cadence: time.map(() => 60),
      },
    },
  }
  const garmin: GarminCache = {
    lastSync: START.getTime(),
    activities: {
      'connect:55': {
        id: 'connect:55',
        name: 'Rouvy',
        sport: 'bike',
        startDate: strava.activities['101'].startDate,
        startDateLocal: strava.activities['101'].startDateLocal,
        distanceM: 11000,
        movingTimeS: 10,
        elapsedTimeS: 10,
        sourceDevice: null,
        sourceFile: null,
        metrics: {
          ...emptyGarminMetrics(),
          totalAscentM: 100,
          normalizedPower: 180,
          intensityFactor: 1.1,
          trainingStressScore: 80,
          aerobicTrainingEffect: 2.8,
          exerciseLoad: 70.2,
        },
        fueling: emptyGarminFueling(),
      },
    },
    streams: {
      'connect:55': {
        ...strava.streams!['101'],
        distance: time.map(i => i * 1100),
        watts: time.map(() => 250),
      },
    },
  }
  return { strava, garmin }
}
function temporary(t: TestContext) {
  const dir = mkdtempSync(join(tmpdir(), 'wahoo-tracking-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  mkdirSync(join(dir, 'triathlon/wahoo'), { recursive: true })
  writeFileSync(join(dir, FIT_PATH), fit())
  return dir
}

test('loads an uncached linked FIT and merges telemetry onto the virtual course without changing providers', t => {
  const dir = temporary(t)
  const { strava, garmin } = sources()
  const before = structuredClone({ strava, garmin })
  const wahoo = loadTrackedWahooFits(null, strava, [entry], dir)!
  const wahooBefore = structuredClone(wahoo)
  const dynamics = Object.values(wahoo.cyclingDynamics)[0]
  garmin.cyclingDynamics = {
    'connect:55': {
      ...structuredClone(dynamics),
      time: dynamics.time.map(value => Math.max(0, value - 1)),
      leftTorqueEffectiveness: dynamics.time.map(() => 20),
      leftPowerPhaseStart: dynamics.time.map(() => 350),
    },
  }
  before.garmin.cyclingDynamics = structuredClone(garmin.cyclingDynamics)
  strava.activityDetails = {
    '101': {
      calories: 10,
      laps: [
        {
          id: 'lap1',
          name: 'Effort',
          distance: 10000,
          movingTime: 10,
          elapsedTime: 10,
          startDate: strava.activities['101'].startDate,
          startIndex: 0,
          endIndex: 10,
          totalElevationGain: 80,
          averageSpeed: 1000,
          averageWatts: 1,
          averageHeartrate: 1,
          averageCadence: 1,
        },
      ],
      segmentEfforts: [],
      splitsMetric: [],
      splitsStandard: [],
    },
  }
  before.strava.activityDetails = structuredClone(strava.activityDetails)
  const projected = applyActivityTracking(strava, garmin, [entry], wahoo)!
  assert.equal(projected.activities['101'].distance, 11000)
  assert.equal(projected.activities['101'].elapsedTime, 10)
  assert.equal(projected.streams!['101'].watts![0], 0)
  assert.equal(projected.streams!['101'].cadence![0], 0)
  assert.equal(projected.streams!['101'].watts![1], 101)
  assert.equal(projected.streams!['101'].heartrate![1], 132)
  assert.equal(projected.streams!['101'].watts![5], 105)
  assert.equal(projected.streams!['101'].watts![8], 290)
  const detail = buildPayload(
    strava,
    null,
    garmin,
    undefined,
    null,
    287,
    undefined,
    undefined,
    wahoo,
    190,
    166,
    undefined,
    [entry],
  ).details['101']
  assert.equal(detail.distanceKm, 11)
  assert.equal(detail.distanceSource, 'garmin')
  assert.equal(detail.elevationM, 100)
  assert.equal(detail.avgWatts, 200)
  assert.equal(detail.npWatts, 220)
  assert.equal(detail.avgHr, 135)
  assert.equal(detail.calories, 25)
  assert.equal(detail.wahoo?.startOffsetS, -1)
  assert.equal(detail.wahoo?.stravaNormalizedPower, 170)
  assert.equal(detail.wahoo?.metrics.trainingStressScore, 0)
  assert.equal(detail.garmin?.normalizedPower, 180)
  assert.equal(detail.garmin?.aerobicTrainingEffect, 2.8)
  assert.equal(detail.garmin?.exerciseLoad, 70.2)
  assert.equal(detail.gearShifts[0].elapsedS, 5)
  assert.equal(detail.gearShifts[0].distanceKm, 5.5)
  assert.ok(detail.cyclingDynamics?.distanceKm.every(value => value <= 11))
  assert.ok(detail.cyclingDynamics?.leftTorqueEffectiveness.some(value => value === 80))
  assert.ok(detail.cyclingDynamics?.leftPowerPhaseStart.some(value => value === 350))
  assert.equal(detail.route[0].tempC, 19)
  assert.equal(detail.analysisRanges[0].distanceKm, 11)
  assert.equal(detail.analysisRanges[0].averageWatts, 168.5)
  assert.ok(detail.route.some(point => point.w === 0))
  assert.equal(isActivityDetail(JSON.parse(JSON.stringify(detail))), true)
  assert.equal(isActivityDetail({ ...detail, wahoo: { ...detail.wahoo, metrics: {} } }), false)
  const presentation = { locale: 'en', distance: 'metric', powerSamples: 'recorded' } as const
  assert.ok(
    moreStatRows(presentation, detail).some(
      ([key, value]) => key === 'telemetry source' && value === 'Wahoo FIT',
    ),
  )
  assert.ok(
    moreStatRows(presentation, detail).some(
      ([key, value]) => key === 'computer' && value === 'ELEMNT BOLT',
    ),
  )
  assert.ok(
    activityTableRows(presentation, detail).some(
      ([key, value]) => key === 'intensity factor' && value === '0.700',
    ),
  )
  assert.deepEqual({ strava, garmin }, before)
  assert.deepEqual(wahoo, wahooBefore)
})

test('uses Strava course without Garmin and Garmin UTC timeline without Strava streams', t => {
  const dir = temporary(t)
  const { strava, garmin } = sources()
  const wahoo = loadTrackedWahooFits(null, strava, [entry], dir)!
  const detail = buildPayload(
    strava,
    null,
    null,
    undefined,
    null,
    287,
    undefined,
    undefined,
    wahoo,
    190,
    166,
    undefined,
    [entry],
  ).details['101']
  assert.equal(detail.distanceKm, 10)
  assert.equal(detail.distanceSource, 'strava')
  assert.equal(detail.gearShifts[0].distanceKm, 5)
  delete strava.streams
  garmin.activities['connect:55'].startDate = START.toISOString()
  const projected = applyActivityTracking(strava, garmin, [entry], wahoo)!
  assert.equal(projected.streams!['101'].time![0], 0)
  assert.equal(projected.streams!['101'].watts![0], 0)
})

test('deduplicates an identical session re-encoded by Wahoo and detects replaced files', t => {
  const dir = temporary(t)
  const { strava } = sources()
  const cache = loadTrackedWahooFits(null, strava, [entry], dir)!
  const id = Object.keys(cache.activities)[0]
  cache.activities[id].sourceFile.sha256 = 'a'.repeat(64)
  const before = structuredClone(cache)
  const again = loadTrackedWahooFits(cache, strava, [entry], dir)!
  assert.equal(Object.keys(again.activities).length, 1)
  assert.notEqual(again.activities[id].sourceFile.sha256, 'a'.repeat(64))
  assert.deepEqual(cache, before)
  const stamp = wahooTrackingStamp([entry], dir)
  writeFileSync(join(dir, FIT_PATH), fit(210))
  utimesSync(join(dir, FIT_PATH), new Date(), new Date(Date.now() + 2000))
  assert.notEqual(wahooTrackingStamp([entry], dir), stamp)
  const updated = loadTrackedWahooFits(null, strava, [entry], dir)!
  assert.equal(Object.values(updated.activities)[0].metrics.avgPower, 210)
})

test('rejects missing, corrupt, wrong-sport, and wrong-session linked files', t => {
  const dir = temporary(t)
  const { strava } = sources()
  assert.throws(
    () =>
      loadTrackedWahooFits(
        null,
        strava,
        [{ ...entry, wahooFitPath: 'triathlon/wahoo/missing.fit' }],
        dir,
      ),
    /ENOENT/,
  )
  writeFileSync(join(dir, FIT_PATH), 'not FIT')
  assert.throws(() => loadTrackedWahooFits(null, strava, [entry], dir), /FIT/)
  writeFileSync(join(dir, FIT_PATH), fit(200, 'running'))
  assert.throws(() => loadTrackedWahooFits(null, strava, [entry], dir), /cycling/)
  writeFileSync(join(dir, FIT_PATH), fit())
  strava.activities['101'].startDate = '2026-09-07T16:00:00Z'
  assert.throws(() => loadTrackedWahooFits(null, strava, [entry], dir), /does not match/)
})

test('FIT changes rebuild the tracking owner and its embedding pages', () => {
  const owner = defaultProcessedContent({
    slug: 'triathlon' as FullSlug,
    filePath: 'content/triathlon.md' as FilePath,
    tracking: { activities: [entry] } as NonNullable<
      ReturnType<typeof defaultProcessedContent>[1]['data']['tracking']
    >,
  })
  const stream = defaultProcessedContent({
    slug: 'stream' as FullSlug,
    filePath: 'content/stream.md' as FilePath,
    links: ['triathlon' as SimpleSlug],
  })
  const unrelated = defaultProcessedContent({
    slug: 'other' as FullSlug,
    filePath: 'content/other.md' as FilePath,
  })
  const events: ChangeEvent[] = [{ type: 'change', path: `content/${FIT_PATH}` as FilePath }]
  assert.deepEqual(
    trackedWahooChangeEvents([owner, stream, unrelated], events, 'content').map(
      event => event.path,
    ),
    [`content/${FIT_PATH}`, 'content/triathlon.md', 'content/stream.md'],
  )
  const other: ChangeEvent[] = [
    { type: 'change', path: 'content/triathlon/wahoo/other.fit' as FilePath },
  ]
  assert.equal(trackedWahooChangeEvents([owner, stream], other, 'content'), other)
  const detached = defaultProcessedContent({
    slug: 'triathlon' as FullSlug,
    filePath: 'content/triathlon.md' as FilePath,
  })
  const removal: ChangeEvent[] = [
    {
      type: 'change',
      path: 'content/triathlon.md' as FilePath,
      file: detached[1],
      previousFile: owner[1],
    },
  ]
  assert.deepEqual(
    trackedWahooChangeEvents([detached, stream], removal, 'content').map(event => event.path),
    ['content/triathlon.md', 'content/stream.md'],
  )
})
