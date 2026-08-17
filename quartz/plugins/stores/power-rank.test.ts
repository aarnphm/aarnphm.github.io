import assert from 'node:assert/strict'
import test from 'node:test'
import type { PowerRankAthlete, PowerRankMass } from './power-rank'
import { buildPowerRank, POWER_SKILL_DURATIONS } from './power-rank'

const referenceMassKg = 84.36818082
const athlete: PowerRankAthlete = { sex: 'M', age: 25 }
const mass = (kg: number, date: string): PowerRankMass => ({ kg, date, source: 'garmin' })

test('builds the twelve weight-adjusted power intervals from exact curve durations', () => {
  const sixWeeks = POWER_SKILL_DURATIONS.map(durationS => ({
    s: durationS,
    w: durationS === 15 ? 838 : durationS === 30 ? 624 : durationS === 60 ? 399 : 280,
  }))
  const rank = buildPowerRank(sixWeeks, [], mass(referenceMassKg, '2026-08-16'), athlete)

  assert.equal(rank.intervals.length, 12)
  assert.equal(rank.massDate, '2026-08-16')
  assert.deepEqual(
    rank.intervals.slice(0, 4).map(interval => interval.skill),
    ['sprint', 'sprint', 'sprint', 'attack'],
  )
  assert.deepEqual(
    rank.intervals.slice(0, 3).map(interval => interval.efforts['six-weeks']?.level),
    [5, 3, 2],
  )
  assert.equal(rank.intervals[0].efforts['six-weeks']?.levelName, 'elite')
  assert.equal(rank.intervals[0].efforts['six-weeks']?.percentile, 72.4)
  assert.equal(rank.intervals[0].efforts['six-weeks']?.nextWatts, 945)
  assert.equal(rank.intervals[0].efforts['six-weeks']?.wattsToNext, 107)
})

test('preserves rankings when mass and power scale together', () => {
  const durationS = 300
  const base = buildPowerRank(
    [{ s: durationS, w: 366 }],
    [],
    mass(referenceMassKg, '2026-08-16'),
    athlete,
  )
  const doubled = buildPowerRank(
    [{ s: durationS, w: 732 }],
    [],
    mass(referenceMassKg * 2, '2026-08-17'),
    athlete,
  )
  const baseEffort = base.intervals.find(interval => interval.durationS === durationS)?.efforts[
    'six-weeks'
  ]
  const doubledEffort = doubled.intervals.find(interval => interval.durationS === durationS)
    ?.efforts['six-weeks']

  assert.equal(baseEffort?.level, 5)
  assert.equal(doubledEffort?.level, 5)
  assert.equal(baseEffort?.percentile, doubledEffort?.percentile)
  assert.equal(baseEffort?.wattsPerKg, doubledEffort?.wattsPerKg)
})

test('returns no ranked intervals without a valid current mass', () => {
  assert.deepEqual(buildPowerRank([{ s: 15, w: 838 }], [], null, athlete).intervals, [])
  assert.deepEqual(
    buildPowerRank([{ s: 15, w: 838 }], [], mass(0, '2026-08-16'), athlete).intervals,
    [],
  )
})

test('classifies with unrounded watts per kilogram and rejects an unmatched cohort', () => {
  const justBelowElite = buildPowerRank([{ s: 15, w: 842 }], [], mass(86.06, '2026-08-16'), athlete)
  const effort = justBelowElite.intervals[0].efforts['six-weeks']
  assert.equal(effort?.level, 4)
  assert.equal(effort?.nextWatts, 843)
  assert.equal(effort?.wattsToNext, 1)

  const unmatched = buildPowerRank([{ s: 15, w: 842 }], [], mass(86.06, '2026-08-16'), {
    sex: 'F',
    age: 25,
  })
  assert.equal(unmatched.cohortEligible, false)
  assert.deepEqual(unmatched.intervals, [])
})
