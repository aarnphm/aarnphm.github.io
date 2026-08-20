import assert from 'node:assert/strict'
import test from 'node:test'
import {
  parseTriathlonTraceSettings,
  serializeTriathlonTraceSettings,
  TRIATHLON_TRACE_DISPLAY_SETTINGS,
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

test('expands simplified display settings and treats detailed display as the default', () => {
  assert.deepEqual(TRIATHLON_TRACE_DISPLAY_SETTINGS.simplified, {
    'power-balance': false,
    'torque-effectiveness': false,
    'pedal-smoothness': false,
    'power-phase': false,
    'rider-position': false,
    stamina: false,
    'electronic-shifting': false,
    'stroke-rate': false,
    'matched-rides': false,
    '25w-power-distribution': false,
    'power-curve': false,
    'power-zones': false,
    'heart-rate-zones': false,
    'heat-strain-index': false,
    'core-temperature': false,
    'skin-temperature': false,
    temperature: false,
  })
  assert.deepEqual(
    parseTriathlonTraceSettings('settings=display:simplified'),
    TRIATHLON_TRACE_DISPLAY_SETTINGS.simplified,
  )
  assert.deepEqual(
    parseTriathlonTraceSettings('settings=display:detailed'),
    TRIATHLON_TRACE_DISPLAY_SETTINGS.detailed,
  )
  assert.deepEqual(parseTriathlonTraceSettings('settings=display:simplified&expanded:true'), {
    ...TRIATHLON_TRACE_DISPLAY_SETTINGS.simplified,
    expanded: true,
  })
  assert.deepEqual(parseTriathlonTraceSettings('settings=expanded:false&display:detailed'), {
    expanded: false,
  })
  assert.equal(serializeTriathlonTraceSettings(TRIATHLON_TRACE_DISPLAY_SETTINGS.detailed), '')
})

test('rejects malformed, duplicate, and non-kebab trace settings', () => {
  assert.equal(parseTriathlonTraceSettings(undefined), null)
  assert.equal(parseTriathlonTraceSettings('settings='), null)
  assert.equal(parseTriathlonTraceSettings('matched rides:false'), null)
  assert.equal(parseTriathlonTraceSettings('matched-rides:0'), null)
  assert.equal(parseTriathlonTraceSettings('matched-rides:false&'), null)
  assert.equal(parseTriathlonTraceSettings('matched-rides:false&matched-rides:true'), null)
  assert.equal(parseTriathlonTraceSettings('display:unknown'), null)
  assert.equal(parseTriathlonTraceSettings('display:simplified&matched-rides:false'), null)
  assert.equal(parseTriathlonTraceSettings('display:simplified&expanded:0'), null)
  assert.equal(parseTriathlonTraceSettings('display:simplified&display:detailed'), null)
})

test('normalizes rendered trace labels and disables only explicit false settings', () => {
  const settings = { 'core-temperature': false, 'matched-rides': true }
  assert.equal(triathlonTraceName('CORE temperature'), 'core-temperature')
  assert.equal(triathlonTraceEnabled(settings, 'CORE temperature'), false)
  assert.equal(triathlonTraceEnabled(settings, 'matched-rides'), true)
  assert.equal(triathlonTraceEnabled(settings, 'power'), true)
})
