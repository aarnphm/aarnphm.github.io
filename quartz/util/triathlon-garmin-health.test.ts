import type { Element } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h, s } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { GarminEnduranceScore } from '../plugins/stores/garmin-health'
import type { TriNodeFactory } from './triathlon-card'
import { health } from './fixtures/garmin-health'
import { garminHealthFetchError, garminHealthFetchResult } from './garmin-health'
import {
  buildGarminHealth,
  buildGarminRecovery,
  garminLoadRange,
  garminHealthSummary,
} from './triathlon-garmin-health'
import { DEFAULT_TRIATHLON_PRESENTATION } from './triathlon-presentation'

const factory: TriNodeFactory<Element> = {
  presentation: DEFAULT_TRIATHLON_PRESENTATION,
  el: (tag, cls, text, attrs) => h(tag, { class: cls, ...attrs }, text ?? []),
  math: (cls, text) => h('span', { class: cls }, text),
  svg: (tag, attrs) => s(tag, attrs),
  add: (parent, ...children) => parent.children.push(...children),
}

test('daily health renders compact recovery meters, accessible hovers and native load ranges', () => {
  const recovery = buildGarminRecovery(factory, health)
  const training = buildGarminHealth(factory, health)
  assert.ok(recovery)
  assert.ok(training)
  const html = toHtml(h('div', [recovery, training]))
  assert.deepEqual(
    garminHealthSummary(health).map(row => row.value),
    ['92', '78', '6h 12m', 'maintaining', '7918', '37'],
  )
  for (const text of [
    'Load Focus',
    '12:03',
    '20:28',
    '674–1109',
    'hill strength',
    'hill endurance',
    'expert',
    '2419',
    '15h 32m',
  ])
    assert.ok(html.includes(text), text)
  assert.equal((html.match(/role="slider"/g) ?? []).length, 0)
  assert.equal((html.match(/role="meter"/g) ?? []).length, 7)
  assert.match(html, /aria-label="Body Battery"[^>]*aria-valuenow="92"/)
  assert.match(html, /aria-label="training readiness"[^>]*aria-valuenow="78"/)
  assert.match(html, /role="tooltip">Garmin/)
  assert.match(html, /class="tri-health-load-marker"/)
  assert.doesNotMatch(
    html,
    />Garmin (recovery|training|performance)|>Garmin Training Readiness|<svg|NaN|undefined/,
  )
})

test('Garmin rendering keeps unavailable and failed readings distinct from measured zero', () => {
  const day = structuredClone(health)
  day.bodyBattery = garminHealthFetchError(day.bodyBattery, Date.now())
  day.enduranceScore = garminHealthFetchResult<GarminEnduranceScore>(null, Date.now())
  const hill = day.hillScore.value
  assert.ok(hill)
  hill.score = 0
  const summary = garminHealthSummary(day)
  assert.deepEqual(summary[0], {
    label: 'Body Battery',
    value: '92',
    detail: 'cached · refresh failed',
  })
  assert.deepEqual(summary[4], { label: 'Endurance Score', value: '—', detail: 'unavailable' })
  assert.equal(summary[5].value, '0')
  assert.equal(buildGarminHealth(factory, null), null)
})

test('load range geometry keeps the native targets and loads on a shared scale', () => {
  assert.deepEqual(garminLoadRange(2400, 600, 1100, 3000), {
    max: 3000,
    low: 20,
    high: (1100 / 3000) * 100,
    load: 80,
  })
  assert.deepEqual(garminLoadRange(0, 600, 1100, 3000), {
    max: 3000,
    low: 20,
    high: (1100 / 3000) * 100,
    load: 0,
  })
  assert.deepEqual(garminLoadRange(null, null, null, 3000), {
    max: 3000,
    low: null,
    high: null,
    load: null,
  })
  assert.deepEqual(garminLoadRange(200, 1000, 500, 3000), {
    max: 3000,
    low: null,
    high: null,
    load: (200 / 3000) * 100,
  })
  assert.equal(garminLoadRange(4000, 600, 1100, 3000).max, 4000)
  assert.equal(garminLoadRange(Number.NaN, 0, 100, 3000).load, null)
})
