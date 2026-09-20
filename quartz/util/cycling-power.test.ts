import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildCyclingPowerTrace,
  CYCLING_POWER_MAX_POINTS,
  type CyclingPowerPoint,
} from './cycling-power'

function trace(
  time: number[],
  watts: (number | null)[],
  elapsedTimeS = time.at(-1) ?? 0,
  startOffsetS = 0,
) {
  const result = buildCyclingPowerTrace({
    source: 'wahoo',
    streams: {
      time,
      watts,
      distance: time.map(value => value * 10),
      altitude: time.map(() => 100),
    },
    startOffsetS,
    elapsedTimeS,
  })
  assert.ok(result)
  return result
}

test('requires full trailing windows and includes measured zero power', () => {
  const time = Array.from({ length: 361 }, (_, index) => index)
  const result = trace(
    time,
    time.map(second => (second < 30 ? 0 : 200)),
  )
  assert.equal(result.method, 'recorded-power-average-v1')
  assert.equal(result.source, 'wahoo')
  assert.equal(result.points[0].cumulativePowerWatts, null)
  assert.equal(result.points[29].power30sWatts, null)
  assert.equal(result.points[30].power30sWatts, 0)
  assert.equal(result.points[45].power30sWatts, 100)
  assert.equal(result.points[60].power30sWatts, 200)
  assert.equal(result.points[60].cumulativePowerWatts, 100)
  assert.equal(result.points[299].power5mWatts, null)
  assert.equal(result.points[300].power5mWatts, 180)
  assert.equal(result.points[360].power5mWatts, 200)
  assert.equal(result.points[360].cumulativePowerWatts, 183.333)
  assert.equal(result.points[360].distanceKm, 3.6)
})

test('weights native fractional sample intervals by their recorded duration', () => {
  const result = trace([0, 0.5, 1.25, 2, 2.5, 3], [100, 200, 300, 0, 100, 100])
  assert.equal(result.points.find(point => point.elapsedS === 2)?.cumulativePowerWatts, 212.5)
  assert.equal(result.points.at(-1)?.cumulativePowerWatts, 158.333)
})

test('two-second timestamp jumps preserve paused time and reset trailing windows', () => {
  const time = Array.from({ length: 400 }, (_, index) => (index < 60 ? index : index + 1))
  const result = trace(
    time,
    time.map(second => (second < 60 ? 100 : 300)),
  )
  const at = (elapsedS: number) => result.points.find(point => point.elapsedS === elapsedS)
  assert.equal(at(59)?.power30sWatts, 100)
  assert.equal(at(60)?.cumulativePowerWatts, 100)
  assert.equal(at(61)?.cumulativePowerWatts, 100)
  assert.equal(at(61)?.power30sWatts, null)
  assert.equal(at(62)?.cumulativePowerWatts, 103.279)
  assert.equal(at(90)?.power30sWatts, null)
  assert.equal(at(91)?.power30sWatts, 300)
  assert.equal(at(360)?.power5mWatts, null)
  assert.equal(at(361)?.power5mWatts, 300)
})

test('half-second recordings keep a skipped half second as a gap', () => {
  const time = Array.from({ length: 150 }, (_, index) => (index < 80 ? index : index + 1) / 2)
  const result = trace(
    time,
    time.map(() => 200),
  )
  assert.equal(result.points.find(point => point.elapsedS === 40)?.cumulativePowerWatts, 200)
  assert.equal(result.points.find(point => point.elapsedS === 40)?.power30sWatts, null)
  assert.equal(result.points.find(point => point.elapsedS === 70)?.power30sWatts, null)
  assert.equal(result.points.find(point => point.elapsedS === 70.5)?.power30sWatts, 200)
})

test('missing and invalid power interrupt trailing averages while preserving recorded average', () => {
  const time = Array.from({ length: 401 }, (_, index) => index)
  const result = trace(
    time,
    time.map(second => (second === 60 ? null : second === 61 ? -1 : second === 62 ? NaN : 200)),
  )
  for (const elapsedS of [60, 61, 62]) {
    const point = result.points.find(point => point.elapsedS === elapsedS)
    assert.equal(point?.cumulativePowerWatts, 200)
    assert.equal(point?.power30sWatts, null)
  }
  assert.equal(result.points.find(point => point.elapsedS === 92)?.power30sWatts, null)
  assert.equal(result.points.find(point => point.elapsedS === 93)?.power30sWatts, 200)
  assert.equal(result.points.at(-1)?.cumulativePowerWatts, 200)
})

test('clips recording offsets and never holds a final sample across missing time', () => {
  const time = Array.from({ length: 61 }, (_, index) => index)
  const shifted = trace(
    time,
    time.map(() => 200),
    70,
    5,
  )
  assert.equal(shifted.points[0].elapsedS, 0)
  assert.equal(shifted.points[0].elevationM, null)
  assert.equal(shifted.points[0].cumulativePowerWatts, null)
  assert.equal(shifted.points.find(point => point.elapsedS === 35)?.power30sWatts, 200)
  assert.equal(shifted.points.find(point => point.elapsedS === 66)?.cumulativePowerWatts, 200)
  assert.equal(shifted.points.find(point => point.elapsedS === 66)?.power30sWatts, null)
  assert.equal(shifted.points.find(point => point.elapsedS === 66)?.elevationM, null)
  assert.equal(shifted.points.at(-1)?.elapsedS, 70)
  assert.equal(shifted.points.at(-1)?.cumulativePowerWatts, 200)
  assert.equal(shifted.points.at(-1)?.elevationM, null)
  const clipped = trace(
    time,
    time.map(second => (second < 5 ? 900 : 200)),
    55,
    -5,
  )
  assert.equal(clipped.points[0].elapsedS, 0)
  assert.equal(clipped.points.find(point => point.elapsedS === 30)?.power30sWatts, 200)
  assert.equal(clipped.points.at(-1)?.cumulativePowerWatts, 200)
})

test('bounds serialized output while keeping terrain extrema and missing-power breaks', () => {
  const time = Array.from({ length: 10_801 }, (_, index) => index)
  const result = buildCyclingPowerTrace({
    source: 'strava',
    streams: {
      time,
      watts: time.map(second => (second === 5_045 ? null : second < 5_000 ? 100 : 200)),
      distance: time.map(second => second * 10),
      altitude: time.map(second =>
        second === 7_005 ? null : second === 4_005 ? 2_000 : second === 6_005 ? -20 : 100,
      ),
    },
    startOffsetS: 0,
    elapsedTimeS: 10_800,
  })
  assert.ok(result)
  assert.ok(result.points.length <= CYCLING_POWER_MAX_POINTS)
  assert.equal(Math.max(...result.points.map(point => point.elevationM ?? -Infinity)), 2_000)
  assert.equal(Math.min(...result.points.map(point => point.elevationM ?? Infinity)), -20)
  assert.equal(result.points.at(-1)?.cumulativePowerWatts, 153.699)
  const before = result.points.findLastIndex(point => point.elapsedS < 5_045)
  assert.notEqual(result.points[before + 1].cumulativePowerWatts, null)
  assert.equal(result.points[before + 1].power30sWatts, null)
  assert.equal(result.points[before + 1].elevationM, 100)
  const beforeMissingElevation = result.points.findLastIndex(point => point.elapsedS < 7_005)
  assert.equal(result.points[beforeMissingElevation + 1].elevationM, null)
  assert.ok(
    result.points.every(
      (point, index) => index === 0 || point.elapsedS > result.points[index - 1].elapsedS,
    ),
  )
})

test('keeps terrain and recorded average continuous through a long pause without filling watts', () => {
  const time = [
    ...Array.from({ length: 60 }, (_, index) => index),
    ...Array.from({ length: 361 }, (_, index) => index + 3_600),
  ]
  const result = buildCyclingPowerTrace({
    source: 'wahoo',
    streams: {
      time,
      watts: time.map(second => (second < 3_600 ? 100 : 300)),
      altitude: time.map(second => (second < 3_600 ? 120 : 125 + (second - 3_600) / 60)),
    },
    startOffsetS: 0,
    elapsedTimeS: 3_960,
  })
  assert.ok(result)
  const at = (elapsedS: number) => result.points.find(point => point.elapsedS === elapsedS)
  assert.ok(result.points.every(point => point.elevationM != null))
  assert.equal(at(60)?.elevationM, 120)
  assert.equal(at(3_600)?.elevationM, 125)
  assert.equal(at(60)?.cumulativePowerWatts, 100)
  assert.equal(at(3_600)?.cumulativePowerWatts, 100)
  assert.equal(at(60)?.power30sWatts, null)
  assert.equal(at(3_629)?.power30sWatts, null)
  assert.equal(at(3_630)?.power30sWatts, 300)
  assert.equal(at(3_899)?.power5mWatts, null)
  assert.equal(at(3_900)?.power5mWatts, 300)
  assert.equal(at(3_960)?.cumulativePowerWatts, 271.429)
})

test('retains actual missing elevation independently of available power', () => {
  const result = buildCyclingPowerTrace({
    source: 'wahoo',
    streams: {
      time: [0, 1, 2, 100, 101, 102],
      watts: [200, 200, 200, 200, 200, 200],
      altitude: [100, NaN, null, 110, 111, 112],
    },
    startOffsetS: 0,
    elapsedTimeS: 102,
  })
  assert.ok(result)
  for (const elapsedS of [1, 2, 3]) {
    const point: CyclingPowerPoint | undefined = result.points.find(
      point => point.elapsedS === elapsedS,
    )
    assert.equal(point?.elevationM, null)
    assert.equal(point?.cumulativePowerWatts, 200)
  }
  assert.equal(result.points.find(point => point.elapsedS === 100)?.elevationM, 110)
  assert.equal(result.points.at(-1)?.elevationM, 112)
  assert.equal(result.points.at(-1)?.cumulativePowerWatts, 200)
})

test('rejects absent power, invalid ordering, and unbounded activity timelines', () => {
  for (const streams of [
    undefined,
    { time: [0, 1, 2], watts: [null, null, null] },
    { time: [0, 1, 1], watts: [200, 200, 200] },
    { time: [0, 2, 1], watts: [200, 200, 200] },
    { time: [0, NaN, 2], watts: [200, 200, 200] },
    { time: [0, 1, 2], watts: [200, 200] },
  ])
    assert.equal(
      buildCyclingPowerTrace({ source: 'wahoo', streams, startOffsetS: 0, elapsedTimeS: 2 }),
      null,
    )
  assert.equal(
    buildCyclingPowerTrace({
      source: 'strava',
      streams: { time: [0, 1], watts: [200, 200] },
      startOffsetS: 0,
      elapsedTimeS: 48 * 3_600 + 1,
    }),
    null,
  )
})
