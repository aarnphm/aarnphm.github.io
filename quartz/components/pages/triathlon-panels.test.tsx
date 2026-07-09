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
