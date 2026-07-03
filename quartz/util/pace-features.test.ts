import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  PACE_FEATURE_DIM,
  PACE_FEATURE_NAMES,
  PACE_INPUT_DIM,
  type PaceContext,
  type PaceDayState,
  type PaceLegSpec,
  buildFeatureVector,
  contextFromMetaRow,
  dayStateFromFeedRow,
  legSpecFromActivityRow,
  parsePaceFeed,
} from './pace-features'

const day: PaceDayState = {
  date: '2026-06-01',
  ctl: 50,
  atl: 40,
  tsb: 10,
  swimCtl: 5,
  bikeCtl: 20,
  runCtl: 25,
  hrv: 60,
  rhr: 48,
  readiness: 80,
  sleepDurationS: 28800,
  tempDeviationC: -0.25,
  weightKg: 88,
}

const ctx: PaceContext = { vThrBySport: { swim: 1.25, bike: 8, run: 3.5 }, hrMax: 182 }

const runSpec: PaceLegSpec = {
  sport: 'run',
  distanceKm: 10,
  elevationM: 120,
  tempC: 18,
  windKph: 12,
}

test('feature schema dims are coherent', () => {
  assert.equal(PACE_FEATURE_DIM, 20)
  assert.equal(PACE_FEATURE_NAMES.length, PACE_FEATURE_DIM)
  assert.equal(PACE_INPUT_DIM, 40)
  assert.equal(new Set(PACE_FEATURE_NAMES).size, PACE_FEATURE_NAMES.length)
})

test('fully populated vector matches expected order', () => {
  const { raw, presence } = buildFeatureVector(day, runSpec, ctx)
  assert.deepEqual(Array.from(raw), [
    0,
    0,
    1,
    10,
    120,
    18,
    12,
    50,
    40,
    10,
    25,
    60,
    48,
    80,
    28800,
    -0.25,
    88,
    3.5,
    182,
    Math.fround(0.92),
  ])
  assert.deepEqual(Array.from(presence), Array<number>(PACE_FEATURE_DIM).fill(1))
})

test('effort pins to race fraction and overrides via ctx', () => {
  const def = buildFeatureVector(day, runSpec, ctx)
  assert.equal(def.raw[19], Math.fround(0.92))
  assert.equal(def.presence[19], 1)
  const swim = buildFeatureVector(day, { ...runSpec, sport: 'swim' }, ctx)
  assert.equal(swim.raw[19], Math.fround(0.88))
  const pinned = buildFeatureVector(day, runSpec, { ...ctx, effortFrac: 0.75 })
  assert.equal(pinned.raw[19], Math.fround(0.75))
})

test('sport one-hot and sport_ctl select by sport', () => {
  const swim = buildFeatureVector({ ...day }, { ...runSpec, sport: 'swim' }, ctx)
  assert.deepEqual(Array.from(swim.raw).slice(0, 3), [1, 0, 0])
  assert.equal(swim.raw[10], day.swimCtl)
  assert.equal(swim.raw[17], ctx.vThrBySport.swim)

  const bike = buildFeatureVector({ ...day }, { ...runSpec, sport: 'bike' }, ctx)
  assert.deepEqual(Array.from(bike.raw).slice(0, 3), [0, 1, 0])
  assert.equal(bike.raw[10], day.bikeCtl)
})

test('missing optionals zero the value and clear presence', () => {
  const sparseDay: PaceDayState = { ...day, hrv: null, readiness: null, weightKg: null }
  const sparseSpec: PaceLegSpec = { ...runSpec, tempC: null, windKph: null }
  const { raw, presence } = buildFeatureVector(sparseDay, sparseSpec, { ...ctx, hrMax: null })
  for (const i of [5, 6, 11, 13, 16, 18]) {
    assert.equal(presence[i], 0, `presence[${i}] should be 0`)
    assert.equal(raw[i], 0, `raw[${i}] should be 0`)
  }
  for (const i of [0, 3, 7, 10, 12, 14, 15, 17]) assert.equal(presence[i], 1)
})

test('dayStateFromFeedRow reads day fields and nulls', () => {
  const row = {
    kind: 'day',
    date: '2026-06-01',
    ctl: 50,
    atl: 40,
    tsb: 10,
    swimCtl: 5,
    bikeCtl: 20,
    runCtl: 25,
    hrv: null,
    rhr: 48,
    readiness: 80,
    sleepDurationS: 28800,
    tempDeviationC: -0.25,
    weightKg: 88,
  }
  const state = dayStateFromFeedRow(row)
  assert.ok(state)
  assert.equal(state.date, '2026-06-01')
  assert.equal(state.ctl, 50)
  assert.equal(state.runCtl, 25)
  assert.equal(state.hrv, null)
  assert.equal(state.rhr, 48)
  assert.equal(dayStateFromFeedRow(42), null)
})

test('legSpecFromActivityRow maps geometry and conditions', () => {
  const spec = legSpecFromActivityRow({
    kind: 'activity',
    sport: 'bike',
    distanceKm: 40,
    elevationM: 350,
    avgTemp: 22,
    windKph: null,
  })
  assert.ok(spec)
  assert.equal(spec.sport, 'bike')
  assert.equal(spec.distanceKm, 40)
  assert.equal(spec.tempC, 22)
  assert.equal(spec.windKph, null)
  assert.equal(legSpecFromActivityRow({ sport: 'yoga' }), null)
})

test('contextFromMetaRow reads thresholds and hrMax', () => {
  const meta = {
    kind: 'meta',
    thresholds: [
      { sport: 'run', vThr: 3.5 },
      { sport: 'bike', vThr: 8 },
    ],
    athlete: { hrMaxEst: 182 },
  }
  const built = contextFromMetaRow(meta)
  assert.equal(built.vThrBySport.run, 3.5)
  assert.equal(built.vThrBySport.bike, 8)
  assert.equal(built.vThrBySport.swim, 0)
  assert.equal(built.hrMax, 182)
})

test('parsePaceFeed splits row kinds', () => {
  const feed = parsePaceFeed(
    [
      JSON.stringify({ kind: 'meta', thresholds: [] }),
      JSON.stringify({ kind: 'day', date: '2026-06-01', ctl: 1 }),
      JSON.stringify({ kind: 'activity', sport: 'run' }),
      '',
      'not json',
    ].join('\n'),
  )
  assert.ok(feed.meta)
  assert.equal(feed.days.length, 1)
  assert.equal(feed.activities.length, 1)
})
