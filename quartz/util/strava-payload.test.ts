import assert from 'node:assert/strict'
import test from 'node:test'
import type { AppleCache, AppleSwim } from '../plugins/stores/apple'
import {
  emptyPayload,
  type StravaActivityDetail,
  type StravaPayload,
} from '../plugins/stores/strava'
import { enrichSwimMetrics } from './strava-payload'

const detail = (values: Partial<StravaActivityDetail> = {}): StravaActivityDetail => ({
  id: 1,
  sport: 'swim',
  name: 'Pool swim',
  date: '2026-07-09',
  start: '2026-07-09T20:13:31Z',
  distanceKm: 1,
  movingTimeS: 1_590,
  elevationM: 0,
  avgHr: 140,
  maxHr: 160,
  avgWatts: null,
  npWatts: null,
  maxWatts: null,
  kilojoules: null,
  deviceWatts: false,
  avgCadence: null,
  sufferScore: null,
  calories: null,
  avgTemp: null,
  windKph: null,
  windDir: null,
  windDirDeg: null,
  windGustKph: null,
  location: null,
  fueling: null,
  garmin: null,
  route: [],
  minAlt: 0,
  maxAlt: 0,
  descentM: 0,
  hrZones: null,
  powerZones: null,
  powerHist: null,
  powerCurve: null,
  bestEfforts: null,
  strokes: null,
  strokeCount: null,
  strokeRateSpm: null,
  swimPaceSPer100m: null,
  ...values,
})

const appleSwim = (values: Partial<AppleSwim> = {}): AppleSwim => ({
  id: 'apple-swim',
  date: '2026-07-09',
  start: '2026-07-09T20:13:30Z',
  end: '2026-07-09T20:40:10Z',
  activeTimeS: 1_600,
  totalM: 1_000,
  laps: 40,
  strokes: { freestyle: 600, breaststroke: 400 },
  strokeCount: 700,
  strokeTimeS: 1_500,
  ...values,
})

const payloadWith = (...details: StravaActivityDetail[]): StravaPayload => {
  const payload = emptyPayload(1)
  for (const item of details) payload.details[String(item.id)] = item
  return payload
}

test('enriches swim detail and trend with Apple count, rate, and active-time pace', () => {
  const payload = payloadWith(detail())
  const swim = appleSwim()
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { [swim.id ?? swim.date]: swim },
    workouts: {},
  }

  enrichSwimMetrics(payload, apple)

  assert.deepEqual(payload.details['1'].strokes, swim.strokes)
  assert.equal(payload.details['1'].strokeCount, 700)
  assert.equal(payload.details['1'].strokeRateSpm, 28)
  assert.equal(payload.details['1'].swimPaceSPer100m, 160)
  assert.deepEqual(payload.swimTrend, [
    {
      id: 1,
      date: '2026-07-09',
      start: '2026-07-09T20:13:31Z',
      paceSPer100m: 160,
      strokeRateSpm: 28,
    },
  ])
})

test('keeps two same-date swim activities as separate trend observations', () => {
  const morningActivity = detail({
    id: 1,
    start: '2026-07-09T10:01:00Z',
    distanceKm: 0.5,
    movingTimeS: 620,
  })
  const eveningActivity = detail({
    id: 2,
    start: '2026-07-09T18:02:00Z',
    distanceKm: 1,
    movingTimeS: 1_520,
  })
  const morning = appleSwim({
    id: 'morning',
    start: '2026-07-09T10:00:00Z',
    end: '2026-07-09T10:10:00Z',
    totalM: 500,
    activeTimeS: 600,
    strokeCount: 300,
    strokeTimeS: 600,
  })
  const evening = appleSwim({
    id: 'evening',
    start: '2026-07-09T18:00:00Z',
    end: '2026-07-09T18:25:00Z',
    totalM: 1_000,
    activeTimeS: 1_500,
    strokeCount: 560,
    strokeTimeS: 1_200,
  })
  const payload = payloadWith(eveningActivity, morningActivity)
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { evening, morning },
    workouts: {},
  }

  enrichSwimMetrics(payload, apple)

  assert.deepEqual(payload.swimTrend, [
    {
      id: 1,
      date: '2026-07-09',
      start: '2026-07-09T10:01:00Z',
      paceSPer100m: 120,
      strokeRateSpm: 30,
    },
    {
      id: 2,
      date: '2026-07-09',
      start: '2026-07-09T18:02:00Z',
      paceSPer100m: 150,
      strokeRateSpm: 28,
    },
  ])
})

test('keeps valid Strava pace history and drops an implausible GPS swim', () => {
  const payload = payloadWith(
    detail({ id: 1, date: '2026-07-08', start: '2026-07-08T20:00:00Z' }),
    detail({
      id: 2,
      date: '2026-07-09',
      distanceKm: 2.3,
      movingTimeS: 306,
      start: '2026-07-09T20:00:00Z',
    }),
  )

  enrichSwimMetrics(payload, null)

  assert.equal(payload.details['1'].swimPaceSPer100m, 159)
  assert.equal(payload.details['2'].swimPaceSPer100m, null)
  assert.deepEqual(payload.swimTrend, [
    {
      id: 1,
      date: '2026-07-08',
      start: '2026-07-08T20:00:00Z',
      paceSPer100m: 159,
      strokeRateSpm: null,
    },
  ])
})

test('keeps a valid stroke-rate observation when pace is unavailable', () => {
  const payload = payloadWith(detail({ movingTimeS: 20 }))
  const swim = appleSwim({ activeTimeS: 20 })
  const apple: AppleCache = {
    version: 4,
    lastSync: 1,
    days: {},
    swims: { [swim.id ?? swim.date]: swim },
    workouts: {},
  }

  enrichSwimMetrics(payload, apple)

  assert.equal(payload.details['1'].swimPaceSPer100m, null)
  assert.equal(payload.details['1'].strokeRateSpm, 28)
  assert.deepEqual(payload.swimTrend, [
    {
      id: 1,
      date: '2026-07-09',
      start: '2026-07-09T20:13:31Z',
      paceSPer100m: null,
      strokeRateSpm: 28,
    },
  ])
})
