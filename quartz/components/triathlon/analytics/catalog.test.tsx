import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import { buildAnalytics } from '../../../plugins/stores/analytics'
import { DEFAULT_TRIATHLON_FORMATTER } from '../runtime/formatter'
import { ANALYTICS_CATALOG, ANALYTICS_PANEL_ORDER } from './catalog'
import { AnalyticsServerPanel } from './render'

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
