import assert from 'node:assert/strict'
import test from 'node:test'
import {
  parseTriathlonTraceSettings,
  serializeTriathlonTraceSettings,
  triathlonTraceEnabled,
  triathlonTraceName,
} from './triathlon-trace-settings'

test('parses and serializes kebab-case trace settings', () => {
  const settings = parseTriathlonTraceSettings(
    'settings=matched-rides:false&matched-runs:true&power-balance:false',
  )
  assert.deepEqual(settings, {
    'matched-rides': false,
    'matched-runs': true,
    'power-balance': false,
  })
  assert.ok(settings)
  assert.equal(
    serializeTriathlonTraceSettings(settings),
    'matched-rides:false&matched-runs:true&power-balance:false',
  )
})

test('rejects malformed, duplicate, and non-kebab trace settings', () => {
  assert.equal(parseTriathlonTraceSettings(undefined), null)
  assert.equal(parseTriathlonTraceSettings('settings='), null)
  assert.equal(parseTriathlonTraceSettings('matched rides:false'), null)
  assert.equal(parseTriathlonTraceSettings('matched-rides:0'), null)
  assert.equal(parseTriathlonTraceSettings('matched-rides:false&'), null)
  assert.equal(parseTriathlonTraceSettings('matched-rides:false&matched-rides:true'), null)
})

test('normalizes rendered trace labels and disables only explicit false settings', () => {
  const settings = { 'core-temperature': false, 'matched-rides': true }
  assert.equal(triathlonTraceName('CORE temperature'), 'core-temperature')
  assert.equal(triathlonTraceEnabled(settings, 'CORE temperature'), false)
  assert.equal(triathlonTraceEnabled(settings, 'matched-rides'), true)
  assert.equal(triathlonTraceEnabled(settings, 'power'), true)
})
