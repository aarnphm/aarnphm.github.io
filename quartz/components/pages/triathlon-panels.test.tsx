import assert from 'node:assert/strict'
import test from 'node:test'
import render from 'preact-render-to-string'
import { AnalyticsPanel, MapPanel, TrainingPanel } from './triathlon-panels'

test('triathlon panels share one dialog shell', () => {
  const cases = [
    {
      html: render(<AnalyticsPanel page />),
      rootClass: 'tri-analytics tri-analytics--page',
      scrimClass: 'tri-analytics-scrim',
      closeClass: 'tri-ana-close',
      label: 'triathlon analytics',
    },
    {
      html: render(<MapPanel page />),
      rootClass: 'tri-map tri-map--page',
      scrimClass: 'tri-map-scrim',
      closeClass: 'tri-ana-close tri-map-close',
      label: 'triathlon route maps',
    },
    {
      html: render(<TrainingPanel page />),
      rootClass: 'tri-training tri-training--page',
      scrimClass: 'tri-training-scrim',
      closeClass: 'tri-ana-close tri-training-close',
      label: 'triathlon training plan',
    },
  ]

  for (const { html, rootClass, scrimClass, closeClass, label } of cases) {
    assert.equal(html.includes(`class="${scrimClass}" aria-hidden="true"`), true)
    assert.equal(html.includes(`class="${rootClass}" aria-hidden="false"`), true)
    assert.equal(html.includes(`role="dialog" aria-label="${label}"`), true)
    assert.equal(html.includes(`class="${closeClass}" type="button" aria-label="Close"`), true)
    assert.equal(html.split('role="dialog"').length - 1, 1)
  }
})

test('analytics reserves one heat chart between effort and readiness', () => {
  const html = render(<AnalyticsPanel page />)
  const effort = html.indexOf('data-chart="effort"')
  const heat = html.indexOf('data-chart="heat"')
  const readiness = html.indexOf('data-chart="readiness"')

  assert.equal(html.split('data-chart="heat"').length - 1, 1)
  assert.ok(effort >= 0)
  assert.ok(heat > effort)
  assert.ok(readiness > heat)
})

test('analytics reserves one synchronized lab history mount', () => {
  const html = render(<AnalyticsPanel page />)
  const dexa = html.indexOf('data-chart="dexa"')
  const gauge = html.indexOf('data-chart="gauge"')

  assert.equal(html.split('data-chart="dexa"').length - 1, 1)
  assert.equal(html.includes('data-chart="vo2test"'), false)
  assert.ok(dexa >= 0)
  assert.ok(gauge > dexa)
})
