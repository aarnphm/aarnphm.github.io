import assert from 'node:assert/strict'
import test from 'node:test'
import render from 'preact-render-to-string'
import { CERAMICSPEED_CROSS_CHAIN_RESEARCH } from '../../util/triathlon-gear-ratio'
import {
  AnalyticsPanel,
  CalcPanel,
  DISPATCH_ICON,
  FeedPanel,
  GearPanel,
  MapPanel,
  PacePanel,
  TrainingPanel,
  ToolsPanel,
} from './triathlon-panels'

test('calculator defaults to the configured race distance with an olympic fallback', () => {
  const half = render(<CalcPanel defaultDistance="70.3" />)

  assert.equal(
    half.includes(
      'aria-label="triathlon calculator" data-swim="1.9" data-bike="90" data-run="21.1"',
    ),
    true,
  )
  assert.equal(
    half.includes(
      'class="tri-calc-preset tri-calc-preset--on" type="button" data-swim="1.9" data-bike="90" data-run="21.1">70.3</button>',
    ),
    true,
  )

  const fallback = render(<CalcPanel defaultDistance="unsupported" />)
  assert.equal(
    fallback.includes(
      'aria-label="triathlon calculator" data-swim="1.5" data-bike="40" data-run="10"',
    ),
    true,
  )
  assert.equal(
    fallback.includes(
      'class="tri-calc-preset tri-calc-preset--on" type="button" data-swim="1.5" data-bike="40" data-run="10">olympic</button>',
    ),
    true,
  )
})

test('feed exposes the shared activity query controls', () => {
  const html = render(<FeedPanel />)

  assert.equal(html.includes('class="tri-feed-search-wrap"'), true)
  assert.equal(html.includes('aria-label="search activities"'), true)
  assert.equal(html.includes('aria-controls="tri-feed-results" aria-expanded="false"'), true)
  assert.equal(
    html.includes('placeholder="search (filter:bike|run|swim|walk, sort:distance|cadence|pace)"'),
    true,
  )
  assert.equal(
    html.includes(
      'id="tri-feed-results" class="tri-ana-results tri-feed-results" aria-hidden="true"',
    ),
    true,
  )
})

test('triathlon navigation controls expose their locale keys', () => {
  const html = render(
    <>
      <GearPanel />
      <PacePanel />
    </>,
  )

  assert.match(html, /class="tri-gear-btn"[^>]*data-i18n="gear"/)
  assert.match(html, /class="tri-pace-btn"[^>]*data-i18n="pace"/)
  assert.match(html, /class="tri-pace-sec" data-i18n="run"/)
  assert.match(html, /class="tri-pace-sec" data-i18n="swim"/)
  assert.match(html, /class="tri-pace-sec" data-i18n="bike"/)
})

test('gear ratios render the declared drivetrain and cassette families', () => {
  const html = render(<GearPanel />)

  assert.equal(html.split('class="tri-ratio-table"').length - 1, 1)
  assert.equal(html.includes('aria-expanded="false" aria-controls="tri-gear-panel"'), true)
  assert.equal(html.includes('id="tri-gear-panel" class="tri-gear" aria-hidden="true"'), true)
  assert.equal(html.includes('data-ratio-chainring="52" data-ratio-cog="11"'), true)
  assert.equal(html.includes('data-ratio-value="4.73"'), true)
  assert.equal(html.includes('data-ratio-chainring="36" data-ratio-cog="34"'), true)
  assert.equal(html.includes('data-ratio-value="1.06"'), true)
  assert.equal(html.split('class="tri-ratio-efficiency-row').length - 1, 2)
  assert.equal(html.includes('est. vs. ideal · CeramicSpeed'), true)
  for (const source of CERAMICSPEED_CROSS_CHAIN_RESEARCH.sources) {
    assert.equal(html.includes(`href="${source.url}"`), true)
    assert.equal(html.includes(`>[${source.id}]</a>`), true)
  }
  assert.equal(html.includes('· 250 W · 95 rpm · 385 mm chainstay'), true)
  assert.equal(html.includes('class="tri-math"'), true)
  assert.equal(html.includes('class="katex"'), true)
  assert.equal(html.includes('data-efficiency-chainring="52" data-efficiency-cog="11"'), true)
  assert.equal(html.includes('data-efficiency-delta="-3.520"'), true)
  assert.equal(html.includes('tri-ratio-efficiency-value--full'), true)
  assert.equal(html.includes('tri-ratio-efficiency-value--compact'), true)
  assert.equal(html.split('tabindex="0"').length - 1, 1)
  assert.equal(html.includes('watts drivetrain loss;'), true)
  assert.equal(html.includes('watts cross-chain loss'), true)
  assert.equal(html.includes('data-cross-chain-loss-watts="'), true)
  assert.equal(
    html.indexOf('class="tri-ratio-row tri-ratio-row--1"') <
      html.indexOf('class="tri-ratio-efficiency-row tri-ratio-row--1"') &&
      html.indexOf('class="tri-ratio-efficiency-row tri-ratio-row--1"') <
        html.indexOf('class="tri-ratio-row tri-ratio-row--2"'),
    true,
  )
  assert.equal(html.includes('<select'), false)
  assert.equal(
    html.indexOf('class="tri-ratio-ring-inputs"') < html.indexOf('class="tri-ratio-cassette"') &&
      html.indexOf('class="tri-ratio-cassette"') < html.indexOf('class="tri-ratio-layout"'),
    true,
  )
  assert.equal(
    html.includes(
      'class="tri-ratio-cassette-trigger" type="button" aria-labelledby="tri-ratio-cassette-label tri-ratio-cassette-value" aria-haspopup="listbox" aria-expanded="false" aria-controls="tri-ratio-cassette-menu"',
    ),
    true,
  )
  assert.equal(
    html.includes(
      'id="tri-ratio-cassette-menu" class="tri-ratio-cassette-menu" role="listbox" aria-labelledby="tri-ratio-cassette-label" hidden',
    ),
    true,
  )
  assert.equal(
    html.includes(
      'role="option" aria-selected="true" data-cassette-id="shimano-ultegra-r8100-11-34"',
    ),
    true,
  )
  assert.equal(html.includes('Ultegra R8000 · 11–32 · 11s'), true)
  assert.equal(html.includes('Dura-Ace R9200 · 11–34 · 12s'), true)
  assert.equal(html.includes('RED XG-1290 · 10–36 · 12s'), true)
  assert.equal(html.includes('RED XPLR XG-1391 · 10–46 · 13s'), true)
  assert.equal(html.includes('Super Record 13 · 10–29 · 13s'), true)
  assert.equal(html.includes('Super Record Wireless · 10–29 · 12s'), true)
  assert.equal(html.includes('Chorus · 11–34 · 12s'), true)
  assert.equal(html.includes('Ekar · 9–42 · 13s'), true)
})

test('tools exposes gear and pace controls outside hidden dropdowns', () => {
  const html = render(<ToolsPanel />)

  assert.equal(html.split('class="tri-ratio-table"').length - 1, 1)
  assert.equal(html.includes('class="tri-gear-btn"'), false)
  assert.equal(html.includes('class="tri-pace-btn"'), false)
  assert.equal(html.includes('class="tri-gear" aria-hidden="false"'), true)
  assert.equal(html.includes('class="tri-pace" aria-hidden="false"'), true)
})

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

  assert.equal(html.includes('class="tri-analytics-search-wrap"'), true)
  assert.equal(
    html.includes(
      'class="tri-ana-search" type="search" placeholder="search (filter:bike|run|swim|walk, sort:distance|cadence|pace)" aria-label="search analytics" autocomplete="off"',
    ),
    true,
  )
  assert.equal(html.split('class="tri-ana-compare-toggle"').length - 1, 1)
  assert.equal(
    html.includes(
      'class="tri-ana-compare-toggle" type="button" aria-pressed="false" aria-label="compare activities" aria-controls="tri-analytics-results" data-i18n-aria-label="compare activities"',
    ),
    true,
  )
  assert.equal(html.split('class="tri-ana-compare-icon"').length - 1, 1)
  const toggleStart = html.indexOf('<button class="tri-ana-compare-toggle"')
  const toggleBodyStart = html.indexOf('>', toggleStart) + 1
  const toggleBodyEnd = html.indexOf('</button>', toggleBodyStart)
  const toggleBody = html.slice(toggleBodyStart, toggleBodyEnd)
  const expectedIcon =
    '<svg class="tri-ana-compare-icon" viewBox="0 0 1000 1000" aria-hidden="true">' +
    `<path d="${DISPATCH_ICON}"></path>` +
    '</svg>'
  assert.equal(toggleBody, expectedIcon)
  assert.equal(html.includes('class="tri-ana-compare-label"'), false)
  assert.equal(html.split('id="tri-analytics-results"').length - 1, 1)
  assert.equal(
    html.includes('id="tri-analytics-results" class="tri-ana-results" aria-hidden="true"'),
    true,
  )
  assert.equal(html.split('id="tri-analytics-detail"').length - 1, 1)
  assert.equal(
    html.includes('id="tri-analytics-detail" class="tri-ana-detail" aria-hidden="true"'),
    true,
  )
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

test('analytics reserves the best-efforts power curve after lactate threshold', () => {
  const html = render(<AnalyticsPanel page />)
  const lactate = html.indexOf('data-chart="lactate"')
  const power = html.indexOf('data-chart="power"')
  const abilities = html.indexOf('data-chart="abilities"')

  assert.equal(html.split('data-chart="power"').length - 1, 1)
  assert.ok(lactate >= 0)
  assert.ok(power > lactate)
  assert.ok(abilities > power)
})

test('map reserves one hidden activity selector over its canvas', () => {
  const html = render(<MapPanel page />)
  const canvas = html.indexOf('class="tri-map-canvas"')
  const selection = html.indexOf('class="tri-map-selection" aria-hidden="true"')
  const tip = html.indexOf('class="tri-map-tip" aria-hidden="true"')
  const sportOffsets = ['bike', 'run', 'swim', 'walk'].map(sport =>
    html.indexOf(`data-sport="${sport}"`),
  )

  assert.equal(html.split('class="tri-map-selection"').length - 1, 1)
  assert.equal(html.split('class="tri-map-sport"').length - 1, 4)
  assert.ok(sportOffsets.every(offset => offset >= 0))
  assert.ok(sportOffsets.every((offset, index) => index === 0 || offset > sportOffsets[index - 1]))
  assert.ok(canvas >= 0)
  assert.ok(selection > canvas)
  assert.ok(tip > selection)
})
