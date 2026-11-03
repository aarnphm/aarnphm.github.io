import { fromHtml } from 'hast-util-from-html'
import { toText } from 'hast-util-to-text'
import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import { visit } from 'unist-util-visit'
import { emptyPayload } from '../../plugins/stores/strava'
import { triText } from '../../util/triathlon-i18n'
import { SeasonSummary } from './SeasonSummary'

test('season summary appends walk distance and sauna and physiotherapy durations with counts', () => {
  const payload = emptyPayload()
  const html = renderToString(
    <SeasonSummary
      totals={payload.totals}
      strengthTotal={{ count: 2, movingTimeS: 3_600 }}
      activities={[
        { sport: 'walk', distanceKm: 1.24, movingTimeS: 900 },
        { sport: 'walk', distanceKm: 2.34, movingTimeS: 1_800 },
        { sport: 'sauna', distanceKm: 0, movingTimeS: 1_800 },
        { sport: 'sauna', distanceKm: 0, movingTimeS: 2_700 },
        { sport: 'treatment', distanceKm: 0, movingTimeS: 3_600 },
        { sport: 'yoga', distanceKm: 0, movingTimeS: 1_200 },
      ]}
    />,
  )
  const root = fromHtml(html, { fragment: true })
  const rows: string[] = []
  const defaultRows: string[] = []
  const extraRows: string[] = []
  let footExpanded: string | undefined
  let toggleExpanded: string | undefined
  let toggleHasArrowSvg = false
  visit(root, 'element', node => {
    const cls = node.properties.className?.toString() ?? ''
    if (cls === 'tri-foot') {
      footExpanded = String(node.properties.dataExpanded ?? '')
    }
    if (cls === 'tri-leg') {
      const text = toText(node)
      rows.push(text)
      if (node.properties.dataTriExtra === 'true') extraRows.push(text)
      else defaultRows.push(text)
    }
    if (cls === 'tri-foot-more') {
      toggleExpanded = String(node.properties.ariaExpanded ?? '')
      visit(node, 'element', child => {
        if (child.tagName === 'path' && typeof child.properties.d === 'string') {
          toggleHasArrowSvg = true
        }
      })
    }
    if (node.properties.dataKind === 'walk') {
      assert.equal(Number(node.properties.dataKm), 3.58)
      assert.equal(node.properties.dataGloss, 'legdist')
      assert.equal(node.properties.tabIndex, 0)
    }
  })
  assert.equal(footExpanded, 'false')
  assert.equal(toggleExpanded, 'false')
  assert.equal(toggleHasArrowSvg, true)
  assert.deepEqual(defaultRows, ['swim · 0 m · 0', 'bike · 0.0 mi · 0', 'run · 0.0 mi · 0'])
  assert.deepEqual(extraRows, [
    "strength · 1h00' · 2",
    'walk · 2.2 mi · 2',
    "sauna · 1h15' · 2",
    "physiotherapy · 1h00' · 1",
  ])
  assert.deepEqual(rows, [
    'swim · 0 m · 0',
    'bike · 0.0 mi · 0',
    'run · 0.0 mi · 0',
    "strength · 1h00' · 2",
    'walk · 2.2 mi · 2',
    "sauna · 1h15' · 2",
    "physiotherapy · 1h00' · 1",
  ])
})

test('season summary hides absent categories and retains recorded zero values', () => {
  const payload = emptyPayload()
  const renderHtml = (activities: Parameters<typeof SeasonSummary>[0]['activities']) =>
    renderToString(
      <SeasonSummary
        totals={payload.totals}
        strengthTotal={payload.strengthTotal}
        activities={activities}
      />,
    )
  const render = (activities: Parameters<typeof SeasonSummary>[0]['activities']) =>
    toText(fromHtml(renderHtml(activities), { fragment: true }))
  assert.equal(render([]), 'swim · 0 m · 0bike · 0.0 mi · 0run · 0.0 mi · 0')
  assert.doesNotMatch(renderHtml([]), /tri-foot-more/)
  const text = render([
    { sport: 'walk', distanceKm: 0, movingTimeS: 0 },
    { sport: 'sauna', distanceKm: 0, movingTimeS: 0 },
    { sport: 'treatment', distanceKm: 0, movingTimeS: 0 },
  ])
  assert.match(text, /walk · 0.0 mi · 1/)
  assert.match(text, /sauna · 0' · 1/)
  assert.match(text, /physiotherapy · 0' · 1/)
  assert.equal(triText('fr', 'physiotherapy'), 'physiothérapie')
  assert.equal(triText('fr', 'more activities'), "plus d'activités")
  assert.equal(triText('fr', 'fewer activities'), "moins d'activités")
})
