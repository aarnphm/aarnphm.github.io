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
import { triathlonDateTree } from '../../util/triathlon-date-route'
import { ANALYTICS_CATALOG } from '../triathlon/analytics/catalog'
import {
  AnalyticsPanel,
  GearPanel,
  MapPanel,
  OnTreePanel,
  PacePanel,
  ToolsPanel,
  TrainingPanel,
} from './triathlon-panels'

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

const analytics = buildAnalytics(null)
analytics.body.composition.push({
  date: '2026-08-16',
  kg: 86.06,
  bmi: 24.3,
  ffmi: 19.65,
  bodyFatPct: 19.3,
  bodyWaterPct: 58.9,
  muscleMassKg: 36.51,
  boneMassKg: 6.05,
})
const renderData: TriathlonRenderData = {
  analytics,
  plans,
  weather: {
    forecastStart: '2026-08-17T13:00:00.000Z',
    latitude: 43.64,
    longitude: -79.4,
    temperatureC: 18.5,
    conditionCode: 'LightRain',
    precipitationChance: 0.7,
    precipitationType: 'rain',
    source: 'weatherkit',
  },
}

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

test('gear surfaces place daily tire pressure after bike inventory and before maintenance', () => {
  const popover = renderToString(<GearPanel maintenance={maintenance} renderData={renderData} />)
  const tools = renderToString(<ToolsPanel maintenance={maintenance} renderData={renderData} />)

  for (const html of [popover, tools]) {
    assert.match(html, /class="tri-maintenance"/)
    assert.match(html, /UFO Wax Drip-On/)
    assert.match(html, /2026-08-10/)
    assert.match(html, /data-rider-kg="86.06"/)
    assert.match(html, /data-pressure-output="front">76.5</)
    assert.match(html, /data-pressure-output="rear">83</)
    assert.match(html, /Pirelli P Zero Race SL-R/)
    assert.match(html, /P Zero Race TLR SL-R/)
    assert.match(html, /custom bike/)
    assert.match(html, /custom wheelset/)
    assert.equal(html.match(/data-pressure-field="bikeMass"/g)?.length, 3)
    assert.equal(html.match(/data-pressure-field="balance"/g)?.length, 4)
    assert.equal(html.match(/data-pressure-field="customWheelWidth"/g)?.length, 2)
    assert.equal(html.match(/data-pressure-field="weightUnit"/g)?.length, 2)
    assert.match(html, /data-weight-unit="kg"/)
    assert.match(html, /<time class="tri-pressure-date" datetime="2026-08-16">2026-08-16<\/time>/)
    assert.doesNotMatch(html, /Garmin morning/)
    assert.match(html, /value="86.06" data-pressure-field="riderMass"/)
    assert.match(html, /data-pressure-bike="cervelo" inputmode="decimal"/)
    assert.match(html, /data-pressure-bike="custom" inputmode="decimal"/)
    assert.match(html, /type="text" value="19.5" data-pressure-field="speed"/)
    assert.match(html, /WeatherKit · 2026-08-17 13:00 UTC/)
    assert.match(html, /18.5 °C/)
    assert.match(html, /light rain/)
    assert.match(html, /70% precipitation/)
    assert.match(html, /mixed<\/strong><span>−3 PSI/)
    assert.match(html, /wet<\/strong><span>−8 PSI/)
    assert.match(html, /most Toronto roads, aged asphalt and seams/)
    assert.equal(html.match(/data-pressure-surface-tip/g)?.length, 1)
    assert.equal(html.match(/data-pressure-surface-option/g)?.length, 4)
    assert.doesNotMatch(html, /tri-pressure-spec/)
    assert.doesNotMatch(html, /tri-pressure-speed-presets/)

    const sections = [
      html.indexOf('class="tri-ratio"'),
      html.indexOf('Cervélo Soloist'),
      html.indexOf('Canyon Speedmax CFR Di2 2026'),
      html.indexOf('class="tri-pressure"'),
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

test('server-rendered unit surfaces start in imperial', () => {
  const pace = renderToString(<PacePanel page />)
  assert.match(pace, /<button class="tri-pace-unit" type="button">mph<\/button>/)
  assert.match(pace, /data-kph="16.1" data-mph="10.0">10.0<\/span>/)

  const tree = renderToString(
    <OnTreePanel
      root="/triathlon/on"
      tree={triathlonDateTree({
        activity: { date: '2026-08-16', sport: 'run', distanceKm: 10, movingTimeS: 3_600 },
      })}
    />,
  )
  assert.match(tree, /data-km="10" data-kind="combined">6 mi<\/span>/)
})
