import assert from 'node:assert/strict'
import test from 'node:test'
import { calculateFtpHypothesis, FTP_HYPOTHESIS_DEFAULTS } from './ftp-hypothesis'

test('calculates the established treadmill VO2max hypothesis', () => {
  const hypothesis = calculateFtpHypothesis(47.8, 88.9)
  assert.ok(hypothesis)
  assert.equal(hypothesis.absoluteVo2, 4.25)
  assert.equal(hypothesis.cyclingVo2max, 3.91)
  assert.equal(hypothesis.thresholdVo2, 3.32)
  assert.equal(hypothesis.efficiencyFtp, 243)
  assert.equal(hypothesis.acsmFtp, 224)
  assert.equal(hypothesis.ftp, 230)
  assert.equal(hypothesis.low, 210)
  assert.equal(hypothesis.high, 260)
  assert.equal(hypothesis.wattsPerKg, 2.59)
})

test('uses cycling VO2max without a cross-modal discount', () => {
  const hypothesis = calculateFtpHypothesis(47.8, 88.9, {
    ...FTP_HYPOTHESIS_DEFAULTS,
    crossModalDiscountPct: 0,
  })
  assert.ok(hypothesis)
  assert.equal(hypothesis.absoluteVo2, hypothesis.cyclingVo2max)
  assert.equal(hypothesis.crossModalDiscountPct, 0)
  assert.equal(hypothesis.ftp, 260)
})

test('rejects non-positive physiological inputs', () => {
  assert.equal(calculateFtpHypothesis(0, 88.9), null)
  assert.equal(calculateFtpHypothesis(47.8, 0), null)
})
