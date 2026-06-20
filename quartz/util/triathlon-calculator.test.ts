import assert from 'node:assert/strict'
import test from 'node:test'
import {
  computeTriathlonCalcTimes,
  formatDurationClock,
  parseClockSeconds,
  solveTriathlonCalcTarget,
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
