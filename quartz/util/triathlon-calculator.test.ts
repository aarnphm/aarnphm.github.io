import assert from 'node:assert/strict'
import test from 'node:test'
import {
  computeTriathlonCalcTimes,
  deriveZoneBands,
  formatDurationClock,
  parseClockSeconds,
  projectZoneTimes,
  solveTriathlonCalcTarget,
  type SportThresholdVel,
  type TriathlonCalcInput,
} from './triathlon-calculator'

const olympicInput: TriathlonCalcInput = {
  swimKm: 1.5,
  bikeKm: 40,
  runKm: 10,
  swimPaceSec: 120,
  t1Sec: 120,
  bikeMph: 18,
  t2Sec: 90,
  runPaceSec: 540,
}

test('parses and formats calculator clock values', () => {
  assert.equal(parseClockSeconds('2:00'), 120)
  assert.equal(parseClockSeconds('1:16:52'), 4612)
  assert.equal(formatDurationClock(4612), '1:16:52')
  assert.equal(formatDurationClock(532), '8:52')
})

test('computes triathlon calculator leg and total times', () => {
  const times = computeTriathlonCalcTimes(olympicInput)

  assert.equal(times.swimSec, 1800)
  assert.equal(times.t1Sec, 120)
  assert.equal(times.t2Sec, 90)
  assert.equal(Math.round(times.bikeSec), 4971)
  assert.equal(Math.round(times.runSec), 3355)
  assert.equal(formatDurationClock(times.totalSec), '2:52:16')
})

test('solves target finish time by scaling sport paces around fixed transitions', () => {
  const targetTotalSec = parseClockSeconds('2:45:00')
  const paces = solveTriathlonCalcTarget(olympicInput, targetTotalSec)
  assert.ok(paces)

  const solved = computeTriathlonCalcTimes({
    ...olympicInput,
    swimPaceSec: paces.swimPaceSec,
    bikeMph: paces.bikeMph,
    runPaceSec: paces.runPaceSec,
  })

  assert.ok(Math.abs(solved.totalSec - targetTotalSec) < 0.000001)
  assert.equal(solved.t1Sec, olympicInput.t1Sec)
  assert.equal(solved.t2Sec, olympicInput.t2Sec)
  assert.ok(paces.swimPaceSec < olympicInput.swimPaceSec)
  assert.ok(paces.bikeMph > olympicInput.bikeMph)
  assert.ok(paces.runPaceSec < olympicInput.runPaceSec)
})

test('rejects target times shorter than transitions', () => {
  assert.equal(solveTriathlonCalcTarget(olympicInput, 120), null)
})

const lab = { zonesKmh: [7, 8, 10.5, 14], zonesHr: [112, 121, 142, 171], maxKmh: 15, hrMax: 182 }

test('derives zone speed bands from lab anchors', () => {
  const bands = deriveZoneBands(lab)
  assert.equal(bands.length, 4)
  const z2 = bands[1]
  assert.equal(z2.index, 2)
  assert.equal(z2.key, 'fat burning')
  assert.equal(z2.hrLo, 121)
  assert.equal(z2.hrHi, 141)
  assert.equal(z2.vMinKmh, 8)
  assert.equal(z2.vMaxKmh, 10.5)
  assert.equal(bands[3].hrHi, 182)
  assert.equal(bands[3].vMaxKmh, 15)
})

test('ignores lab data without two valid speed anchors', () => {
  assert.deepEqual(
    deriveZoneBands({ zonesKmh: [10], zonesHr: [120], maxKmh: null, hrMax: null }),
    [],
  )
})

const thresholds: SportThresholdVel = { swim: 0.7, bike: 8, run: 3 }

test('projects a zone into per-leg ranges that scale by each sport threshold', () => {
  const z2 = deriveZoneBands(lab)[1]
  const proj = projectZoneTimes(olympicInput, z2, thresholds)
  assert.ok(proj)

  const runThrKmh = thresholds.run * 3.6
  assert.ok(Math.abs(proj.ifMin - z2.vMinKmh / runThrKmh) < 1e-9)
  assert.ok(Math.abs(proj.ifMax - z2.vMaxKmh / runThrKmh) < 1e-9)

  assert.ok(Math.abs(proj.run.vMinKmh - z2.vMinKmh) < 1e-9)
  assert.ok(Math.abs(proj.run.vMaxKmh - z2.vMaxKmh) < 1e-9)

  assert.ok(
    Math.abs(proj.bike.vMaxKmh / proj.run.vMaxKmh - thresholds.bike / thresholds.run) < 1e-9,
  )
  assert.ok(
    Math.abs(proj.swim.vMaxKmh / proj.run.vMaxKmh - thresholds.swim / thresholds.run) < 1e-9,
  )

  for (const leg of [proj.swim, proj.bike, proj.run]) assert.ok(leg.fastSec < leg.slowSec)
  assert.ok(proj.fastSec < proj.slowSec)
  assert.ok(
    Math.abs(
      proj.fastSec -
        (proj.swim.fastSec + proj.t1Sec + proj.bike.fastSec + proj.t2Sec + proj.run.fastSec),
    ) < 1e-9,
  )
  assert.equal(proj.t1Sec, olympicInput.t1Sec)
  assert.equal(proj.t2Sec, olympicInput.t2Sec)
})

test('rejects projection when run threshold is missing', () => {
  const z2 = deriveZoneBands(lab)[1]
  assert.equal(projectZoneTimes(olympicInput, z2, { swim: 0.7, bike: 8, run: 0 }), null)
})
