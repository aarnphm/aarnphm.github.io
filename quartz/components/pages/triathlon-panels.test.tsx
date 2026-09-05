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
  CalcPanel,
  FeedPanel,
  GEAR,
  GearPanel,
  MapPanel,
  OnTreePanel,
  PacePanel,
  ToolsPanel,
  TrainingPanel,
  TriathlonSubnav,
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
  services: [],
  components: [],
  chains: [
    {
      id: '3',
      distanceMiles: null,
      lubricant: 'UFO Wax Drip-On',
      since: '2026-08-10',
      waxed: true,
    },
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

test('subpage navigation links remain native pointer targets for the shared bracket cursor', () => {
  const root = rendered(<TriathlonSubnav active="analytics" root="" />)
  const links = elements(root, element => classes(element).includes('tri-subnav-link'))
  assert.equal(links.length, 7)
  assert.ok(links.every(link => link.tagName === 'a' && typeof link.properties?.href === 'string'))
  assert.ok(links.every(link => !('dataSiteCursorAction' in (link.properties ?? {}))))
  assert.equal(links.filter(link => link.properties?.ariaCurrent === 'page').length, 1)
})

test('calculator copy control exposes its SVG states as one magnetic cursor action', () => {
  const root = rendered(<CalcPanel page />)
  const buttons = elements(root, element => classes(element).includes('tri-calc-copy'))
  assert.equal(buttons.length, 1)
  assert.ok('dataSiteCursorAction' in (buttons[0].properties ?? {}))
  const sources = elements(root, element => classes(element).includes('tri-calc-source'))
  assert.equal(sources.length, 1)
  const sourceControls = elements(
    { type: 'root', children: sources[0].children },
    element => element.tagName === 'button',
  )
  assert.equal(sourceControls.length, 4)
  assert.ok(classes(sourceControls[3]).includes('tri-calc-copy'))
  const tablists = elements(
    { type: 'root', children: sources[0].children },
    element => element.properties?.role === 'tablist',
  )
  assert.equal(tablists.length, 1)
  assert.equal(
    elements({ type: 'root', children: tablists[0].children }, element =>
      classes(element).includes('tri-calc-copy'),
    ).length,
    0,
  )
  const icons = elements(
    { type: 'root', children: buttons[0].children },
    element => element.tagName === 'svg',
  )
  assert.equal(icons.length, 2)
  assert.ok(icons.every(icon => 'dataSiteCursorIcon' in (icon.properties ?? {})))

  const modal = rendered(<CalcPanel />)
  const modalSources = elements(modal, element => classes(element).includes('tri-calc-source'))
  assert.equal(modalSources.length, 1)
  assert.equal(
    elements({ type: 'root', children: modalSources[0].children }, element =>
      classes(element).includes('tri-calc-copy'),
    ).length,
    1,
  )
})

test('calculator defaults both transition times to five minutes', () => {
  const root = rendered(<CalcPanel page />)
  const transitions = elements(root, element =>
    ['t1', 't2'].includes(String(element.properties?.dataK)),
  )

  assert.deepEqual(
    transitions.map(element => [element.properties?.dataK, element.properties?.value]),
    [
      ['t1', '5:00'],
      ['t2', '5:00'],
    ],
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

test('Soloist inventory includes the Reserve 40|44 wheelset rotation', () => {
  const inventory = GEAR.find(([name]) => name === 'Cervélo Soloist')?.[1]

  assert.ok(inventory)
  assert.ok(
    inventory.includes('Front Wheel: Reserve 40, 12x100mm, 24H, centerlock, tubeless compatible'),
  )
  assert.ok(
    inventory.includes('Rear Wheel: Reserve 44, 12x142mm, 24H, centerlock, tubeless compatible'),
  )
})

test('gear surfaces keep inventory and maintenance without calculators', () => {
  const popover = renderToString(<GearPanel maintenance={maintenance} />)
  const tools = renderToString(<ToolsPanel maintenance={maintenance} />)

  for (const html of [popover, tools]) {
    assert.match(html, /class="tri-maintenance"/)
    assert.match(html, /UFO Wax Drip-On/)
    assert.match(html, /2026-08-10/)
    assert.match(html, /Pirelli P Zero Race SL-R/)
    assert.match(html, /P Zero Race TLR SL-R/)
    assert.match(html, /HUNT 54 Aerodynamicist UD Carbon Spoke/)
    assert.match(html, /HUNT 58 Aerodynamicist UD Carbon Spoke/)
    assert.doesNotMatch(html, /class="tri-ratio"/)
    assert.doesNotMatch(html, /class="tri-pressure"/)

    const sections = [
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

test('dedicated subpages omit route titles while modal panels retain them', () => {
  const pagePanels = [
    <ToolsPanel maintenance={maintenance} />,
    <CalcPanel page renderData={renderData} />,
    <AnalyticsPanel page renderData={renderData} />,
    <MapPanel page />,
    <TrainingPanel page renderData={renderData} />,
    <FeedPanel />,
    <OnTreePanel root="/triathlon/on" tree={[]} />,
  ]

  for (const panel of pagePanels) {
    const root = rendered(panel)
    assert.equal(
      elements(root, element =>
        classes(element).some(className =>
          ['tri-tools-h', 'tri-calc-title', 'tri-ana-title'].includes(className),
        ),
      ).length,
      0,
    )
  }

  const modalPanels = [
    <CalcPanel renderData={renderData} />,
    <AnalyticsPanel renderData={renderData} />,
    <MapPanel />,
    <TrainingPanel renderData={renderData} />,
  ]

  for (const panel of modalPanels) {
    const root = rendered(panel)
    assert.equal(
      elements(root, element =>
        classes(element).some(className => ['tri-calc-title', 'tri-ana-title'].includes(className)),
      ).length,
      1,
    )
  }
})

test('calculator page tabs own race, gear ratio, and daily tire pressure calculators', () => {
  const panel = <CalcPanel page renderData={renderData} />
  const html = renderToString(panel)
  const shortcuts = elements(
    rendered(panel),
    element => typeof element.properties?.dataCalcTab === 'string',
  ).map(element => element.properties?.ariaKeyShortcuts)

  assert.equal(html.match(/data-calc-tab=/g)?.length, 3)
  assert.deepEqual(shortcuts, ['r', 'c', 't'])
  assert.equal(html.match(/role="tabpanel"/g)?.length, 3)
  assert.match(html, /data-calc-tab="race"/)
  assert.match(html, /data-calc-tab="gear-ratios"/)
  assert.match(html, /data-calc-tab="tire-pressure"/)
  assert.match(html, /class="tri-ratio"/)
  assert.match(html, /id="tri-ratio-chainring-preset-menu"/)
  assert.equal(html.match(/data-chainring-preset-id/g)?.length, 3)
  assert.match(html, /data-chainring-preset-id="54-40"/)
  assert.match(html, /data-chainring-preset-id="53-39"/)
  assert.match(html, /data-chainring-preset-id="52-36"/)
  assert.match(
    html,
    /value="54" min="24" max="64" step="1" inputmode="numeric" aria-label="chainring 1 teeth"/,
  )
  assert.match(
    html,
    /value="40" min="24" max="64" step="1" inputmode="numeric" aria-label="chainring 2 teeth"/,
  )
  assert.match(html, /<output class="tri-ratio-range" aria-live="polite">1\.18–4\.91<\/output>/)
  assert.match(html, /data-rider-kg="86.06"/)
  assert.match(html, /data-pressure-output="front">64</)
  assert.match(html, /data-pressure-output="rear">81</)
  assert.match(html, />Custom<\/span>/)
  assert.match(html, /data-wheel="reserve-40-44"/)
  assert.match(html, /value="reserve-40-44"[^>]*checked/)
  assert.match(html, /Reserve 40\|44 Road/)
  assert.match(html, /HUNT 54_58 Aerodynamicist UD/)
  assert.match(html, />Custom Wheelset<\/span>/)
  assert.equal(html.match(/data-pressure-field="bikeMass"/g)?.length, 3)
  assert.equal(html.match(/data-pressure-field="balance"/g)?.length, 4)
  assert.equal(html.match(/data-pressure-field="customWheelWidth"/g)?.length, 2)
  assert.equal(html.match(/data-pressure-field="measuredTireWidth"/g)?.length, 2)
  assert.equal(html.match(/data-pressure-field="weightUnit"/g)?.length, 2)
  assert.match(html, /data-weight-unit="kg"/)
  assert.match(html, /<time class="tri-pressure-date" datetime="2026-08-16">2026-08-16<\/time>/)
  assert.doesNotMatch(html, /Garmin morning/)
  assert.match(html, /value="86.06" data-pressure-field="riderMass"/)
  assert.match(html, /data-pressure-bike="cervelo" inputmode="decimal"/)
  assert.match(html, /data-pressure-bike="custom" inputmode="decimal"/)
  assert.match(
    html,
    /value="32" data-pressure-field="measuredTireWidth" data-pressure-axle="front" inputmode="numeric"/,
  )
  assert.match(
    html,
    /value="28" data-pressure-field="measuredTireWidth" data-pressure-axle="rear" inputmode="numeric"/,
  )
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
        rest: { date: '2026-08-17', durationS: 0, items: [] },
      })}
    />,
  )
  assert.match(tree, /data-km="10" data-kind="combined">6 mi<\/span>/)
  assert.match(tree, /class="tri-tree-day-sports" data-i18n="rest">rest<\/span>/)
})
