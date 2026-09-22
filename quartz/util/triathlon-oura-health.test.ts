import type { Element } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h, s } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { GarminSleepSummary } from '../plugins/stores/garmin'
import type { OuraDayDetail, OuraDaily } from '../plugins/stores/oura'
import { buildAnalytics } from '../plugins/stores/analytics'
import { applyOuraSleepRows } from '../scripts/sync-oura'
import { applyOuraHealthRow, emptyOuraHealth } from './oura-health'
import { resolveSleepMetrics } from './sleep-metrics'
import {
  buildDaySleepAnalytics,
  sleepSupplementMetrics,
  type TriNodeFactory,
} from './triathlon-card'
import { buildTriathlonDailyAnalytics, isTriathlonDailyAnalytics } from './triathlon-day-analytics'
import { buildOuraHealth } from './triathlon-oura-health'
import { DEFAULT_TRIATHLON_PRESENTATION } from './triathlon-presentation'

const factory: TriNodeFactory<Element> = {
  presentation: DEFAULT_TRIATHLON_PRESENTATION,
  el: (tag, cls, text, attrs) => h(tag, { class: cls, ...attrs }, text ?? []),
  math: (cls, text) => h('span', { class: cls }, text),
  svg: (tag, attrs) => s(tag, attrs),
  add: (parent, ...children) => parent.children.push(...children),
}
const date = '2026-09-21'

test('Oura bars preserve zero, cap goal progress, explain baseline and omit missing groups', () => {
  const day = emptyOuraHealth(date)
  assert.deepEqual(buildOuraHealth(factory, day, null), [])
  applyOuraHealthRow(day, 'daily_stress', { stress_high: 0, recovery_high: 5400 })
  applyOuraHealthRow(day, 'daily_activity', {
    score: 96,
    steps: 0,
    active_calories: 1100,
    target_calories: 550,
    resting_time: 3600,
    sedentary_time: 3600,
    non_wear_time: 0,
  })
  applyOuraHealthRow(day, 'sleep_time', {
    optimal_bedtime: { start_offset: -1800, end_offset: 2700, day_tz: -14400 },
  })
  const html = toHtml(h('div', buildOuraHealth(factory, day, { seconds: 900, days: 7 })))
  assert.match(html, /23:30–00:45/)
  assert.match(html, /aria-valuenow="0"/)
  assert.match(html, /aria-valuenow="50"/)
  assert.match(html, /aria-valuenow="100" aria-valuetext="200%"/)
  assert.match(html, /Garden calculation/)
  assert.match(html, /n=7/)
  assert.match(html, /role="tooltip"/)
  assert.doesNotMatch(html, /NaN|Infinity|undefined|Pulse Ox|data-oura-health="resilience"/)
})

test('daily details use one fine hypnogram, movement and a single oxygen row without nap leakage', () => {
  const details: Record<string, OuraDayDetail> = {}
  const days: Record<string, OuraDaily> = {}
  const row = {
    id: 'night',
    type: 'long_sleep',
    day: date,
    bedtime_start: `${date}T01:00:15-04:00`,
    bedtime_end: `${date}T01:04:15-04:00`,
    total_sleep_duration: 24000,
    sleep_phase_5_min: '1234',
    sleep_phase_30_sec: '44332211',
    movement_30_sec: '11223344',
  }
  applyOuraSleepRows(
    [
      row,
      {
        ...row,
        id: 'nap',
        type: 'sleep',
        bedtime_start: `${date}T14:00:15-04:00`,
        bedtime_end: `${date}T14:04:15-04:00`,
      },
    ],
    days,
    details,
    date,
    date,
  )
  const health = {
    ...emptyOuraHealth(date),
    spo2: { averagePct: 95.4, breathingDisturbanceIndex: 0 },
    stress: { stressS: 0, restoredS: 3600, summary: null },
  }
  details[date].health = health
  const summary = buildTriathlonDailyAnalytics(buildAnalytics(null), details)[date]
  assert.equal(isTriathlonDailyAnalytics(JSON.parse(JSON.stringify({ [date]: summary }))), true)
  const render = () => {
    const node = buildDaySleepAnalytics(factory, summary)
    assert.ok(node)
    return toHtml(node)
  }
  const html = render()
  assert.equal((html.match(/data-oura-health="daytime stress"/g) ?? []).length, 1)
  assert.equal((html.match(/data-day-sleep-series="stages"/g) ?? []).length, 2)
  assert.equal((html.match(/data-day-sleep-series="movement"/g) ?? []).length, 2)
  assert.equal((html.match(/>95.4%/g) ?? []).length, 1)
  assert.equal((html.match(/data-day-sleep-interval="30"/g) ?? []).length, 4)
  assert.match(html, /01:03:45 · deep/)
  assert.match(html, /14:03:45 · active/)
  assert.ok(summary.sleep)
  summary.sleep.phase30Sec = null
  assert.match(render(), /data-day-sleep-interval="300"/)
})

test('supplemental oxygen rendering has one preferred row and preserves Garmin minimum provenance', () => {
  const health = {
    ...emptyOuraHealth(date),
    spo2: { averagePct: 95.4, breathingDisturbanceIndex: null },
  }
  const garmin: GarminSleepSummary = {
    source: 'garmin',
    date,
    startTime: null,
    endTime: null,
    averageBreathsPerMinute: null,
    lowestBreathsPerMinute: null,
    highestBreathsPerMinute: null,
    averageSpO2: 97,
    lowestSpO2: 91,
    bodyBatteryStart: null,
    bodyBatteryEnd: null,
    bodyBatteryChange: null,
    averageStress: null,
    restlessMoments: null,
  }
  for (const averageSpO2 of [97, null]) {
    const metrics = resolveSleepMetrics({ avgBreath: null, health }, { ...garmin, averageSpO2 })
    const rows = sleepSupplementMetrics(DEFAULT_TRIATHLON_PRESENTATION, metrics).filter(
      row => row.label === 'Pulse Ox',
    )
    assert.equal(rows.length, 1)
    assert.equal(rows[0].value, averageSpO2 == null ? '95.4%' : '97.0%')
    assert.match(
      rows[0].detail ?? '',
      averageSpO2 == null
        ? /Oura · Garmin unavailable\nGarmin: lowest Pulse Ox 91%/
        : /Garmin\nlowest Pulse Ox 91%/,
    )
  }
})
