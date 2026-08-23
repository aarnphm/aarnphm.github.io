import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import {
  buildAnalytics,
  type PowerToWeightDurationS,
  type PowerToWeightEffort,
} from '../../../plugins/stores/analytics'
import { DEFAULT_TRIATHLON_FORMATTER } from '../runtime/formatter'
import { ANALYTICS_CATALOG, ANALYTICS_PANEL_ORDER } from './catalog'
import { analyticsChartPath, AnalyticsServerPanel } from './render'

test('analytics catalog is complete and preserves the dedicated route order', () => {
  assert.deepEqual(
    ANALYTICS_CATALOG.map(panel => panel.key),
    ANALYTICS_PANEL_ORDER,
  )
  assert.equal(new Set(ANALYTICS_PANEL_ORDER).size, ANALYTICS_PANEL_ORDER.length)
})

test('every analytics panel produces meaningful server markup from the real analytics model', () => {
  const analytics = buildAnalytics(null)
  for (const definition of ANALYTICS_CATALOG) {
    const content = definition.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
    assert.ok(content.title.length > 0, definition.key)
    assert.ok(content.values.length > 0, definition.key)
    const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
    assert.match(html, /data-tri-ssr="true"/)
    assert.match(html, new RegExp(`data-tri-server-panel="${definition.key}"`))
    assert.match(html, /<dl/)
  }
})

test('server analytics markup draws source-backed series when observations exist', () => {
  const analytics = buildAnalytics(null)
  analytics.body.latestKg = 70
  analytics.body.series = [
    { date: '2026-08-01', ts: 1, kg: 71 },
    { date: '2026-08-08', ts: 2, kg: 70 },
  ]
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'body')
  assert.ok(definition)
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /class="tri-ana-ssr-chart"/)
  assert.match(html, /data-series="weight"/)
  assert.match(html, /M0\.00 3\.00 L100\.00 29\.00/)
})

test('power-to-weight server series share one zero-based scale', () => {
  const analytics = buildAnalytics(null)
  const effort = (
    durationS: PowerToWeightDurationS,
    wattsPerKg: number,
    date: string,
  ): PowerToWeightEffort => ({
    durationS,
    watts: Math.round(wattsPerKg * 80),
    wattsPerKg,
    massKg: 80,
    massDate: date,
    massSource: 'tracking',
    activityId: durationS,
    activityDate: date,
  })
  analytics.powerCurve.powerToWeight.points = [
    {
      date: '2026-08-01',
      efforts: {
        5: effort(5, 10, '2026-08-01'),
        60: effort(60, 6, '2026-08-01'),
        300: effort(300, 4, '2026-08-01'),
        1200: effort(1200, 2, '2026-08-01'),
      },
    },
    {
      date: '2026-08-02',
      efforts: {
        5: effort(5, 12, '2026-08-02'),
        60: effort(60, 7, '2026-08-02'),
        300: effort(300, 5, '2026-08-02'),
        1200: effort(1200, 3, '2026-08-02'),
      },
    },
  ]
  const definition = ANALYTICS_CATALOG.find(panel => panel.key === 'power')
  assert.ok(definition)
  const content = definition.server(analytics, DEFAULT_TRIATHLON_FORMATTER)
  assert.equal(content.seriesDomain, 'shared-zero')
  assert.deepEqual(
    content.series?.map(series => series.label),
    ['5s', '1m', '5m', '20m'],
  )
  const html = renderToString(<AnalyticsServerPanel definition={definition} data={analytics} />)
  assert.match(html, /data-tri-series-count="4"/)
  assert.match(html, /data-tri-series-domain="shared-zero"/)
  assert.match(html, /data-series="20m"/)
  assert.equal(analyticsChartPath([10, 12], { minimum: 0, maximum: 12 }), 'M0.00 7.33 L100.00 3.00')
  assert.equal(analyticsChartPath([2, 3], { minimum: 0, maximum: 12 }), 'M0.00 24.67 L100.00 22.50')
})
