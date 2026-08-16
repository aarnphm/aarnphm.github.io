import type { Element, Root, RootContent } from 'hast'
import type { VNode } from 'preact'
import { fromHtml } from 'hast-util-from-html'
import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import type { TrainingPlan } from '../../plugins/stores/training'
import type { TriathlonMaintenance } from '../../util/triathlon-maintenance'
import type { TriathlonRenderData } from '../triathlon/render-data'
import { buildAnalytics } from '../../plugins/stores/analytics'
import { ANALYTICS_CATALOG } from '../triathlon/analytics/catalog'
import { AnalyticsPanel, GearPanel, MapPanel, ToolsPanel, TrainingPanel } from './triathlon-panels'

const plans: TrainingPlan[] = [
  {
    id: 'olympic-2026',
    meta: 'Toronto Olympic',
    distance: 'olympic',
    date: '2026-07-26',
    target: 'sub-3',
    author: 'Coach One',
    html: '<h2>Build</h2><h3>Week one</h3><p>Ride and run.</p>',
  },
  {
    id: 'sprint-2026',
    meta: 'Montreal Sprint',
    distance: 'sprint',
    date: '2026-06-10',
    target: 'finish',
    author: '',
    html: '<h2>Base</h2>',
  },
]

const renderData: TriathlonRenderData = { analytics: buildAnalytics(null), plans }

const maintenance: TriathlonMaintenance = {
  chains: [
    { id: '3', distance: null, lubricant: 'UFO Wax Drip-On', since: '2026-08-10', waxed: true },
  ],
  wheels: [],
}

const elements = (root: Root, predicate: (element: Element) => boolean): Element[] => {
  const found: Element[] = []
  const visit = (nodes: RootContent[]): void => {
    for (const node of nodes) {
      if (node.type !== 'element') continue
      if (predicate(node)) found.push(node)
      visit(node.children)
    }
  }
  visit(root.children)
  return found
}

const classes = (element: Element): string[] => {
  const value = element.properties?.className
  return Array.isArray(value) ? value.map(String) : []
}

const rendered = (node: VNode): Root => fromHtml(renderToString(node), { fragment: true })

test('dedicated analytics markup contains one populated server panel for every catalog key', () => {
  const root = rendered(<AnalyticsPanel page renderData={renderData} />)
  const blocks = elements(root, element => classes(element).includes('tri-ana-block'))
  assert.equal(blocks.length, ANALYTICS_CATALOG.length)
  for (const block of blocks) {
    assert.equal(
      elements(
        { type: 'root', children: block.children },
        element => element.properties?.dataTriSsr === 'true',
      ).length,
      1,
    )
    assert.equal(
      elements({ type: 'root', children: block.children }, element =>
        classes(element).includes('tri-ana-ssr-values'),
      ).length,
      1,
    )
  }
})

test('overview analytics markup retains empty lazy placeholders', () => {
  const root = rendered(<AnalyticsPanel />)
  assert.equal(elements(root, element => element.properties?.dataTriSsr === 'true').length, 0)
  assert.equal(
    elements(root, element => classes(element).includes('tri-ana-block')).length,
    ANALYTICS_CATALOG.length,
  )
})

test('map controls expose an SVG 3D terrain and buildings toggle', () => {
  const root = rendered(<MapPanel />)
  const buttons = elements(root, element => classes(element).includes('tri-map-3d'))
  assert.equal(buttons.length, 1)
  assert.equal(buttons[0].properties?.ariaPressed, 'false')
  assert.equal(buttons[0].properties?.ariaLabel, '3D terrain and buildings')
  assert.equal(
    elements({ type: 'root', children: buttons[0].children }, element => element.tagName === 'path')
      .length,
    3,
  )
})

test('dedicated training markup contains plan rows, selected document, and heading tree', () => {
  const root = rendered(<TrainingPanel page renderData={renderData} />)
  assert.equal(elements(root, element => element.properties?.dataTriSsr === 'true').length, 3)
  assert.equal(elements(root, element => element.properties?.dataPlan != null).length, plans.length)
  assert.equal(
    elements(root, element => element.properties?.id === 'tri-h-olympic-2026-0').length,
    1,
  )
  assert.equal(
    elements(root, element => element.properties?.dataTarget === 'tri-h-olympic-2026-0').length,
    1,
  )
})

test('overview training markup keeps its list, tree, and document empty', () => {
  const root = rendered(<TrainingPanel />)
  assert.equal(elements(root, element => element.properties?.dataTriSsr === 'true').length, 0)
  assert.equal(elements(root, element => element.properties?.dataPlan != null).length, 0)
})

test('gear popover and tools page place shared maintenance after the bike inventory', () => {
  const popover = renderToString(<GearPanel maintenance={maintenance} />)
  const tools = renderToString(<ToolsPanel maintenance={maintenance} />)

  for (const html of [popover, tools]) {
    assert.match(html, /class="tri-maintenance"/)
    assert.match(html, /UFO Wax Drip-On/)
    assert.match(html, /2026-08-10/)

    const sections = [
      html.indexOf('class="tri-ratio"'),
      html.indexOf('Cervélo Soloist'),
      html.indexOf('Canyon Speedmax CFR Di2 2026'),
      html.indexOf('class="tri-maintenance"'),
      html.indexOf('data-i18n="running"'),
    ]
    assert.ok(sections.every(section => section >= 0))
    assert.deepEqual(
      sections,
      sections.toSorted((left, right) => left - right),
    )
  }
})
