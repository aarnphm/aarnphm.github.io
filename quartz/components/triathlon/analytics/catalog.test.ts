import assert from 'node:assert/strict'
import test from 'node:test'
import type { DailyPoint } from '../../../plugins/stores/analytics'
import { buildAnalytics, computeFtpHypothesisFromVo2 } from '../../../plugins/stores/analytics'
import { DEFAULT_TRIATHLON_FORMATTER } from '../runtime/formatter'
import { analyticsPanelDefinition } from './catalog'
import { GLOSS_CHART, SEARCH_SECTIONS } from './search'

const daily: DailyPoint = {
  date: '2026-08-11',
  load: 72.4,
  effort: 0,
  swimLoad: 12.4,
  bikeLoad: 40,
  runLoad: 20,
  ctl: 81.2,
  atl: 96.7,
  tsb: -15.5,
  swimCtl: 10,
  bikeCtl: 50,
  runCtl: 21.2,
  readiness: null,
  hrv: null,
  rhr: null,
  sleepScore: null,
  sleepDurationS: null,
  tempDevC: null,
  weightKg: null,
  totalCalories: null,
  intakeKcal: null,
  warmup: false,
}

test('performance management exposes daily TSS with CTL, ATL, and TSB', () => {
  const empty = buildAnalytics(null)
  const analytics = {
    ...empty,
    daily: [daily],
    risk: { ...empty.risk, ctl: daily.ctl, atl: daily.atl, tsb: daily.tsb },
  }
  const panel = analyticsPanelDefinition('pmc')
  assert.ok(panel)

  const summary = panel.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
  assert.equal(summary.title, 'fitness · fatigue · form · TSS')
  assert.deepEqual(summary.values, [
    { label: 'fitness', value: '81.2' },
    { label: 'fatigue', value: '96.7' },
    { label: 'form', value: '-15.5' },
    { label: 'TSS', value: '72.4' },
  ])
  assert.deepEqual(summary.series, [
    { label: 'fitness', values: [81.2] },
    { label: 'fatigue', values: [96.7] },
    { label: 'form', values: [-15.5] },
    { label: 'TSS', values: [72.4] },
  ])
})

test('TSS search and glossary navigation resolve to performance management', () => {
  const section = SEARCH_SECTIONS.find(item => item.chart === 'pmc')
  assert.ok(section?.hay.includes('tss'))
  assert.equal(GLOSS_CHART.tss, 'pmc')
})

test('critical power search and calendar-year SSR resolve to the power curve', () => {
  assert.equal(GLOSS_CHART.cp, 'power')
  assert.equal(GLOSS_CHART.wprime, 'power')
  const section = SEARCH_SECTIONS.find(item => item.chart === 'power')
  assert.ok(section?.hay.includes('critical power'))
  assert.ok(section?.hay.includes('power rank'))
  const analytics = buildAnalytics(null)
  analytics.powerCurve.criticalPowerYear = {
    criticalPowerWatts: 249,
    wPrimeJoules: 10_300,
    method: 'two-parameter-power-space',
    window: 'calendar-year',
    windowFrom: '2026-01-01',
    windowTo: '2026-08-13',
    anchors: [],
    independentEffortCount: 2,
    rmseWatts: 1.2,
    normalizedRmse: 0.004,
    confidence: 'provisional',
  }
  const panel = analyticsPanelDefinition('power')
  assert.ok(panel)
  const summary = panel.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
  assert.deepEqual(summary.values.slice(0, 3), [
    { label: 'FTP', value: '—' },
    { label: 'eCP', value: '249 W' },
    { label: 'eW′', value: '10.3 kJ' },
  ])
})

test('ftp summary carries provenance and observed cycling evidence', () => {
  const analytics = buildAnalytics(null)
  const hypothesis = computeFtpHypothesisFromVo2('2026-08-16', 55.2, 87.4)
  assert.ok(hypothesis)
  analytics.engine.ftpHypothesis = {
    ...hypothesis,
    power: {
      criticalPowerWatts: 249,
      modeled60MinuteWatts: 252,
      confidence: 'provisional',
      independentEffortCount: 2,
      declaredFtpWatts: 260,
    },
    pedaling: {
      windowFrom: '2026-08-14',
      windowTo: '2026-08-16',
      activityCount: 4,
      sampleCount: 3600,
      coveragePct: 96.9,
      leftPedalSmoothnessPct: 21.5,
      rightPedalSmoothnessPct: 24,
      leftTorqueEffectivenessPct: 69,
      rightTorqueEffectivenessPct: 68,
    },
  }
  const panel = analyticsPanelDefinition('ftp')
  assert.ok(panel)
  const summary = panel.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
  assert.deepEqual(summary.values.slice(3), [
    { label: 'VO2 source', value: 'running · lab' },
    { label: 'efficiency', value: '21.0% · literature-prior' },
    { label: 'eCP', value: '249 W' },
    { label: 'modeled 60 min', value: '252 W' },
    { label: 'pedal evidence', value: '4 rides · 3600 samples' },
  ])
})
