import type { Element, ElementContent } from 'hast'
import { h, s } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { StravaActivityDetail } from '../plugins/stores/strava'
import {
  buildActivity,
  buildCyclingBestEfforts,
  buildElevation,
  buildPowerCurve,
  buildTrace,
  clock,
  formatAltitude,
  powerCurveFraction,
  powerCurveHoverAt,
  setDistanceUnit,
  zoneDuo,
  type DetailCtx,
  type TriNodeFactory,
} from './triathlon-card'

const factory: TriNodeFactory<Element> = {
  el: (tag, cls, text, attrs) =>
    h(tag, { ...(cls ? { class: cls } : {}), ...attrs }, text === undefined ? [] : [text]),
  svg: (tag, attrs) => s(tag, attrs),
  add: (parent, ...children) => parent.children.push(...children),
}

test('carries rounded pace seconds into the next minute', () => {
  assert.equal(clock(539.6), '9:00')
})

test('resolves an exact power duration and its six-week reference value', () => {
  const curve = [
    { s: 1, w: 700 },
    { s: 2, w: 660 },
    { s: 3, w: 635 },
    { s: 5, w: 590 },
    { s: 60, w: 350 },
  ]
  const reference = [
    { s: 1, w: 1_060 },
    { s: 2, w: 1_034 },
    { s: 3, w: 1_016 },
    { s: 5, w: 983 },
    { s: 60, w: 396 },
  ]
  const fraction = powerCurveFraction(3, 1, 60)
  assert.deepEqual(powerCurveHoverAt(curve, reference, fraction), {
    index: 2,
    durationS: 3,
    watts: 635,
    referenceWatts: 1_016,
    xPct: fraction * 100,
  })
})

const classNames = (element: Element): string[] => {
  const value = element.properties.className
  if (Array.isArray(value)) return value.map(String)
  return typeof value === 'string' ? value.split(/\s+/) : []
}

const descendants = (root: Element, predicate: (element: Element) => boolean): Element[] => {
  const matches: Element[] = []
  const visit = (children: ElementContent[]): void => {
    for (const child of children) {
      if (child.type !== 'element') continue
      if (predicate(child)) matches.push(child)
      visit(child.children)
    }
  }
  if (predicate(root)) matches.push(root)
  visit(root.children)
  return matches
}

const byClass = (root: Element, cls: string): Element[] =>
  descendants(root, element => classNames(element).includes(cls))

const byTag = (root: Element, tag: string): Element[] =>
  descendants(root, element => element.tagName === tag)

const text = (root: Element): string => {
  let value = ''
  const visit = (children: ElementContent[]): void => {
    for (const child of children) {
      if (child.type === 'text') value += child.value
      else if (child.type === 'element') visit(child.children)
    }
  }
  visit(root.children)
  return value
}

const table = (root: Element, kind: string): Element => {
  const result = byClass(root, `tri-effort-table--${kind}`)[0]
  assert.ok(result)
  return result
}

const headerText = (root: Element): string[] => {
  const head = byTag(root, 'thead')[0]
  assert.ok(head)
  return byTag(head, 'th').map(text)
}

const bodyRows = (root: Element): string[][] => {
  const body = byTag(root, 'tbody')[0]
  assert.ok(body)
  return byTag(body, 'tr').map(row =>
    row.children.filter((child): child is Element => child.type === 'element').map(text),
  )
}

const detail = (overrides: Partial<StravaActivityDetail> = {}): StravaActivityDetail => ({
  id: 101,
  sport: 'bike',
  name: 'Threshold ride',
  date: '2026-07-09',
  distanceKm: 30,
  movingTimeS: 4_800,
  elevationM: 100,
  avgHr: 148,
  maxHr: 171,
  avgWatts: 188,
  npWatts: 205,
  maxWatts: 565,
  kilojoules: 900,
  deviceWatts: true,
  avgCadence: 88,
  sufferScore: null,
  calories: 960,
  avgTemp: null,
  windKph: null,
  windDir: null,
  windDirDeg: null,
  windGustKph: null,
  location: 'Toronto',
  fueling: null,
  garmin: null,
  route: [
    { x: 0, y: 0, d: 0, alt: 75, w: 160, hr: 130, cad: 82, lat: 43.6, lng: -79.4 },
    { x: 0.34, y: 0.4, d: 10, alt: 89, w: 200, hr: 145, cad: 86, lat: 43.7, lng: -79.3 },
    { x: 0.67, y: 0.8, d: 20, alt: 103, w: 215, hr: 153, cad: 90, lat: 43.8, lng: -79.2 },
    { x: 1, y: 1, d: 30, alt: 110, w: 175, hr: 149, cad: 84, lat: 43.9, lng: -79.1 },
  ],
  minAlt: 75,
  maxAlt: 110,
  descentM: 20,
  hrZones: null,
  powerZones: null,
  powerHist: null,
  powerCurve: null,
  bestEfforts: {
    weightKg: 87.55,
    weightDate: '2026-07-09',
    distance: [
      {
        label: '10K',
        targetDistanceM: 10_000,
        elapsedTimeS: 1_471,
        averageSpeedKph: 24.5,
        averageHeartRate: 151,
        elevationDeltaM: -30,
      },
    ],
    power: [
      {
        durationS: 5,
        averageWatts: 565,
        wattsPerKg: 6.45,
        averageHeartRate: 150,
        elevationDeltaM: 4,
      },
    ],
    climbs: [
      {
        name: 'Snake Road',
        durationS: 480,
        distanceM: 2_500,
        elevationGainM: 120,
        averageGradePct: 4.8,
        averageSpeedKph: 18.8,
        averageHeartRate: 155,
        averageWatts: 240,
        wattsPerKg: 2.74,
        vamMPerHour: 900,
      },
    ],
  },
  ...overrides,
})

test('builds semantic distance, power, and climbing tables in metric units', () => {
  setDistanceUnit(false)
  const rendered = buildCyclingBestEfforts(factory, detail())
  assert.ok(rendered)

  assert.equal(rendered.tagName, 'section')
  assert.equal(rendered.properties.ariaLabel, 'Cycling best efforts')
  assert.equal(byTag(rendered, 'caption').length, 0)
  assert.deepEqual(
    byClass(rendered, 'tri-effort-title').map(title => [title.tagName, text(title)]),
    [
      ['div', 'Distance'],
      ['div', 'Power'],
      ['div', 'Climbing'],
    ],
  )
  assert.deepEqual(
    byClass(rendered, 'tri-effort-block').map(block => block.tagName),
    ['div', 'div', 'div'],
  )
  assert.equal(byClass(rendered, 'tri-effort-viewport').length, 3)
  for (const scroll of byClass(rendered, 'tri-effort-scroll'))
    assert.equal(byClass(scroll, 'tri-effort-title').length, 0)
  assert.deepEqual(
    byClass(rendered, 'tri-effort-scroll').map(scroll => [
      scroll.properties.role,
      scroll.properties.ariaLabel,
      scroll.properties.tabIndex,
    ]),
    [
      ['region', 'Distance efforts', 0],
      ['region', 'Power efforts', 0],
      ['region', 'Climbing efforts', 0],
    ],
  )

  const distance = table(rendered, 'distance')
  assert.equal(distance.properties.ariaLabel, 'Distance efforts')
  assert.deepEqual(headerText(distance), ['Distance', 'Time', 'Speed', 'Heart rate', 'Elev'])
  assert.deepEqual(bodyRows(distance), [['10K', '24:31', '24.5 km/h', '151 bpm', '-30 m']])

  const power = table(rendered, 'power')
  assert.deepEqual(headerText(power), ['Time', 'Power', 'W/kg', 'Heart rate', 'Elev'])
  assert.deepEqual(bodyRows(power), [['5 sec', '565 W', '6.45 W/kg', '150 bpm', '4 m']])

  const climbing = table(rendered, 'climbing')
  assert.deepEqual(headerText(climbing), [
    'Climb',
    'Time',
    'Distance',
    'Gain',
    'Grade',
    'Speed',
    'Heart rate',
    'Power',
    'W/kg',
    'VAM',
  ])
  assert.deepEqual(bodyRows(climbing), [
    [
      'Snake Road',
      '8:00',
      '2.50 km',
      '120 m',
      '4.8%',
      '18.8 km/h',
      '155 bpm',
      '240 W',
      '2.74 W/kg',
      '900 m/h',
    ],
  ])

  const note = byClass(rendered, 'tri-effort-note')[0]
  assert.ok(note)
  assert.equal(text(note), 'W/kg from 87.55 kg Garmin weight · Jul 9')
  for (const heading of byTag(rendered, 'thead').flatMap(head => byTag(head, 'th')))
    assert.equal(heading.properties.scope, 'col')
  for (const body of byTag(rendered, 'tbody')) {
    const rowHeading = byTag(body, 'th')[0]
    assert.ok(rowHeading)
    assert.equal(rowHeading.properties.scope, 'row')
  }
})

test('renders cycling efforts in the expanded shared activity section', () => {
  setDistanceUnit(false)
  const rendered = buildActivity(factory, detail(), true)
  assert.equal(byClass(rendered, 'tri-act--expanded').length, 1)
  assert.equal(byClass(rendered, 'tri-act-more').length, 1)
  assert.equal(byClass(rendered, 'tri-efforts').length, 1)
})

test('marks every routed sport for the shared desktop figure split', () => {
  const routedSports: StravaActivityDetail['sport'][] = ['bike', 'run', 'walk']
  for (const sport of routedSports) {
    const rendered = buildActivity(factory, detail({ sport }), true)
    assert.equal(byClass(rendered, 'tri-act-figs--route').length, 1)
    assert.equal(byClass(rendered, 'tri-act-figs--split').length, 1)
  }

  const swim = buildActivity(
    factory,
    detail({ sport: 'swim', strokes: { freestyle: 1_500 } }),
    true,
  )
  assert.equal(byClass(swim, 'tri-act-figs--route').length, 1)
  assert.equal(byClass(swim, 'tri-act-figs--split').length, 1)

  const routeOnlySwim = buildActivity(factory, detail({ sport: 'swim', strokes: null }), true)
  assert.equal(byClass(routeOnlySwim, 'tri-act-figs--route').length, 1)
  assert.equal(byClass(routeOnlySwim, 'tri-act-figs--split').length, 0)
})

test('renders imperial effort values and elevation axes with feet grid increments', () => {
  setDistanceUnit(true)
  try {
    assert.equal(formatAltitude(-0.1), '0 ft')
    const ride = detail()
    const efforts = buildCyclingBestEfforts(factory, ride)
    assert.ok(efforts)
    assert.deepEqual(bodyRows(table(efforts, 'distance')), [
      ['10K', '24:31', '15.2 mph', '151 bpm', '-98 ft'],
    ])
    assert.deepEqual(bodyRows(table(efforts, 'climbing')), [
      [
        'Snake Road',
        '8:00',
        '1.55 mi',
        '394 ft',
        '4.8%',
        '11.7 mph',
        '155 bpm',
        '240 W',
        '2.74 W/kg',
        '2,953 ft/h',
      ],
    ])

    const elevation = buildElevation(factory, ride)
    assert.equal(byClass(elevation, 'tri-cax-frame').length, 1)
    assert.deepEqual(byClass(elevation, 'tri-cax-yt').map(text).filter(Boolean), [
      '260 ft',
      '280 ft',
      '300 ft',
      '320 ft',
      '340 ft',
      '360 ft',
    ])
    assert.deepEqual(byClass(elevation, 'tri-cax-xt').map(text), ['5 mi', '10 mi', '15 mi'])
    assert.equal(byClass(elevation, 'tri-elev-grid').length, 6)
    assert.deepEqual(
      byClass(elevation, 'tri-elev-cap')
        .flatMap(cap => byTag(cap, 'span'))
        .map(text),
      ['+328 ft', '−66 ft', '246 ft–361 ft'],
    )
  } finally {
    setDistanceUnit(false)
  }
})

const zonedDetail = (): StravaActivityDetail =>
  detail({
    hrZones: [600, 1_200, 900, 300, 60],
    powerZones: [400, 900, 1_100, 700, 300, 120, 40],
    powerHist: [30, 300, 600, 420, 60],
    powerCurve: [
      { s: 1, w: 565 },
      { s: 5, w: 540 },
      { s: 60, w: 320 },
      { s: 300, w: 250 },
      { s: 1_200, w: 230 },
      { s: 3_600, w: 210 },
    ],
  })

const ctx = (overrides: Partial<DetailCtx> = {}): DetailCtx => ({
  zones: { hr: [120, 140, 160, 180], power: [150, 200, 250, 300, 350, 400], ftp: 260 },
  curveRef: [],
  ftp: 260,
  goalFtp: 280,
  vt1: 150,
  ...overrides,
})

test('renders traces with numbered value and distance axes', () => {
  setDistanceUnit(false)
  const trace = buildTrace(
    factory,
    detail(),
    p => p.hr,
    'hr',
    max => `${max} bpm peak`,
    value => `${Math.round(value)}bpm`,
  )
  assert.equal(byClass(trace, 'tri-cax-frame').length, 1)
  assert.deepEqual(byClass(trace, 'tri-cax-yt').map(text), ['0', '50bpm', '100bpm', '150bpm'])
  assert.deepEqual(byClass(trace, 'tri-cax-xt').map(text), ['10 km', '20 km'])
  assert.equal(byClass(trace, 'tri-elev-grid').length, 4)
  assert.equal(byClass(trace, 'tri-cax-ax').length, 2)
  assert.deepEqual(
    byClass(trace, 'tri-elev-cap')
      .flatMap(cap => byTag(cap, 'span'))
      .map(text),
    ['hr', '153 bpm peak'],
  )
})

test('pairs hr/power zones and curve/hist into duos with aligned captions', () => {
  setDistanceUnit(false)
  const rendered = buildActivity(factory, zonedDetail(), true, ctx())
  const duos = byClass(rendered, 'tri-zone-duo')
  assert.equal(duos.length, 2)
  assert.deepEqual(byClass(duos[0], 'tri-zone-title').map(text), [
    'heart rate zones',
    'power zones',
  ])
  assert.deepEqual(byClass(duos[1], 'tri-zone-title').map(text), [
    'power curve',
    '25W power distribution',
  ])
  assert.deepEqual(byClass(duos[0], 'tri-zone-cap').map(text), [
    'based on vt1 150 bpm',
    'based on FTP 260 W',
  ])
})

test('places cycling efforts after the expanded charts', () => {
  const rendered = buildActivity(factory, zonedDetail(), true, ctx())
  const more = byClass(rendered, 'tri-act-more')[0]
  assert.ok(more)
  const children = more.children.filter((child): child is Element => child.type === 'element')
  const last = children[children.length - 1]
  assert.ok(last)
  assert.equal(classNames(last).includes('tri-efforts'), true)
})

test('scales power curve y axis with nice watt ticks', () => {
  setDistanceUnit(false)
  const curve = buildPowerCurve(factory, zonedDetail(), ctx())
  assert.ok(curve)
  assert.deepEqual(byClass(curve, 'tri-cax-yt').map(text), [
    '0',
    '100w',
    '200w',
    '300w',
    '400w',
    '500w',
  ])
})

test('renders a keyboard-focusable comparison readout for the power curve', () => {
  const curve = buildPowerCurve(
    factory,
    zonedDetail(),
    ctx({
      curveRef: [
        { s: 1, w: 700 },
        { s: 5, w: 650 },
        { s: 60, w: 400 },
        { s: 300, w: 300 },
        { s: 1_200, w: 260 },
        { s: 3_600, w: 220 },
      ],
    }),
  )
  assert.ok(curve)
  const svg = byClass(curve, 'tri-curve-svg')[0]
  assert.ok(svg)
  assert.equal(svg.properties.role, 'slider')
  assert.equal(svg.properties.tabIndex, 0)
  assert.equal(svg.properties.ariaValueMin, 1)
  assert.equal(svg.properties.ariaValueMax, 3_600)
  const readout = byClass(curve, 'tri-curve-readout')[0]
  assert.ok(readout)
  assert.deepEqual(byClass(readout, 'tri-curve-readout-label').map(text), [
    'this ride',
    '6-week best',
  ])
})

test('zoneDuo unwraps when one side is missing', () => {
  const solo = factory.el('div', 'tri-zone')
  assert.equal(zoneDuo(factory, solo, null), solo)
  assert.equal(zoneDuo(factory, null, solo), solo)
  assert.equal(zoneDuo(factory, null, null), null)
})

test('renders metric elevation ticks, distance ticks, and dotted grid nodes', () => {
  setDistanceUnit(false)
  assert.equal(formatAltitude(-0.1), '0 m')
  const elevation = buildElevation(factory, detail())
  assert.deepEqual(byClass(elevation, 'tri-cax-yt').map(text).filter(Boolean), [
    '80 m',
    '90 m',
    '100 m',
    '110 m',
  ])
  assert.deepEqual(byClass(elevation, 'tri-cax-xt').map(text), ['10 km', '20 km'])
  assert.equal(byClass(elevation, 'tri-elev-grid').length, 4)
})
