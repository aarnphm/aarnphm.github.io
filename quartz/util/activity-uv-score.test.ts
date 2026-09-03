import assert from 'node:assert/strict'
import test from 'node:test'
import type { UvSeverity } from './activity-provider-reports'
import {
  auditGardenUvCalibration,
  calibrateGardenUvScore,
  gardenUvScoreFromDose,
  gardenUvSeverity,
  parseGardenUvCalibrationArtifact,
  type GardenUvCalibrationPair,
} from './activity-uv-score'

const doseForScore = (score: number): number => -10 * Math.log(1 - score / 100)

const scores = [5, 18, 42, 70, 83, 90, 98]

const pairs = (count: number, offset = 0): GardenUvCalibrationPair[] =>
  Array.from({ length: count }, (_, index) => {
    const score = scores[(index + offset) % scores.length]
    const dose = doseForScore(score)
    return {
      activityId: 10_000 + offset * 100 + index,
      date: new Date(Date.UTC(2025, 0, 1 + offset + index)).toISOString().slice(0, 10),
      score,
      severity: gardenUvSeverity(score),
      elapsedSed: dose,
      movingTelemetrySed: dose * 1.3,
    }
  })

test('matches the exponential score anchors', () => {
  assert.equal(gardenUvScoreFromDose(doseForScore(83), 10), 83)
  assert.equal(gardenUvScoreFromDose(doseForScore(18), 10), 18)
  assert.equal(gardenUvSeverity(10), 'negligible')
  assert.equal(gardenUvSeverity(11), 'low')
  assert.equal(gardenUvSeverity(31), 'moderate')
  assert.equal(gardenUvSeverity(61), 'high')
  assert.equal(gardenUvSeverity(86), 'serious')
  assert.equal(gardenUvSeverity(96), 'extreme')
})

test('requires thirty pairs and four represented provider bands', () => {
  assert.equal(calibrateGardenUvScore(pairs(29)).status, 'insufficient')
  const low: UvSeverity = 'low'
  const oneBand = pairs(30).map(pair => ({
    ...pair,
    score: 18,
    severity: low,
    elapsedSed: doseForScore(18),
    movingTelemetrySed: doseForScore(18),
  }))
  assert.equal(calibrateGardenUvScore(oneBand).status, 'insufficient')
})

test('freezes the training choice before a passing ten-pair holdout', () => {
  const artifact = calibrateGardenUvScore(pairs(30))

  assert.equal(artifact.status, 'active')
  assert.equal(artifact.doseClock, 'elapsed')
  assert(Math.abs((artifact.coefficientSed ?? 0) - 10) < 0.02)
  assert.equal(artifact.training?.pairCount, 20)
  assert.equal(artifact.holdout?.pairCount, 10)
  assert.equal(artifact.holdout?.mae, 0)
  assert.equal(artifact.holdout?.bandAgreementPct, 100)
})

test('holdout changes can reject a model without changing its trained coefficient', () => {
  const baseline = pairs(30)
  const first = calibrateGardenUvScore(baseline)
  const negligible: UvSeverity = 'negligible'
  const changed = baseline.map((pair, index) =>
    index < 20 ? pair : { ...pair, score: 0, severity: negligible },
  )
  const rejected = calibrateGardenUvScore(changed)

  assert.equal(rejected.status, 'rejected')
  assert.equal(rejected.coefficientSed, first.coefficientSed)
  assert.equal(rejected.doseClock, first.doseClock)
})

test('ten new failing pairs suspend an active frozen calibration', () => {
  const artifact = calibrateGardenUvScore(pairs(30))
  const negligible: UvSeverity = 'negligible'
  const failures = pairs(10, 40).map(pair => ({ ...pair, score: 0, severity: negligible }))
  const pending = auditGardenUvCalibration(artifact, failures.slice(0, 9))
  const suspended = auditGardenUvCalibration(artifact, failures)

  assert.equal(pending.status, 'active')
  assert.equal(suspended.status, 'suspended')
  assert.equal(suspended.auditedActivityIds.length, 10)
})

test('rejects stale formula versions and internally inconsistent calibration states', () => {
  const active = calibrateGardenUvScore(pairs(30))

  assert.deepEqual(parseGardenUvCalibrationArtifact(active), active)
  assert.equal(parseGardenUvCalibrationArtifact({ ...active, formulaVersion: 2 }), null)
  assert.equal(parseGardenUvCalibrationArtifact({ ...active, doseClock: null }), null)
  assert.equal(parseGardenUvCalibrationArtifact({ ...active, coefficientSed: null }), null)
  assert.equal(
    parseGardenUvCalibrationArtifact({ ...active, auditedActivityIds: [active.activityIds[0]] }),
    null,
  )
})
