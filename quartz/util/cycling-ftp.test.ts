import assert from 'node:assert/strict'
import test from 'node:test'
import { estimateFtpFromPowerCurve } from './cycling-ftp'

test('estimates FTP from the observed 20-minute effort and retains its source ride', () => {
  const anchor = { s: 1200, w: 243, activityId: 20188550065, activityDate: '2026-09-15' }
  assert.deepEqual(estimateFtpFromPowerCurve([{ s: 300, w: 400 }, anchor]), {
    watts: 231,
    method: '95%-20-minute-power',
    confidence: 'provisional',
    anchor,
  })
})

test('requires an exact 20-minute observation instead of interpolating or extrapolating', () => {
  for (const curve of [
    [],
    [{ s: 720, w: 300 }],
    [
      { s: 1199, w: 300 },
      { s: 1201, w: 290 },
    ],
  ])
    assert.equal(estimateFtpFromPowerCurve(curve), null)
})

test('rejects invalid powers while preserving valid observations', () => {
  for (const w of [NaN, Infinity, -Infinity, 0, -1, 0.1])
    assert.equal(estimateFtpFromPowerCurve([{ s: 1200, w }]), null)
  assert.equal(
    estimateFtpFromPowerCurve([
      { s: 1200, w: NaN },
      { s: 1200, w: 300 },
    ])?.watts,
    285,
  )
})

test('selects the best exact effort without changing the input curve', () => {
  const curve = Object.freeze([
    Object.freeze({ s: 1200, w: 280 }),
    Object.freeze({ s: 1200, w: 320 }),
    Object.freeze({ s: 1200, w: 300 }),
  ])
  assert.equal(estimateFtpFromPowerCurve(curve)?.watts, 304)
  assert.equal(estimateFtpFromPowerCurve(curve)?.anchor, curve[1])
})
