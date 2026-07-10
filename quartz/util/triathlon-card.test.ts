import type { Element, ElementContent } from 'hast'
import { h, s } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import type { StravaActivityDetail, SwimTrendPoint } from '../plugins/stores/strava'
import {
  buildActivity,
  buildCyclingBestEfforts,
  buildDayCard,
  buildElevation,
  buildPowerCurve,
  buildSwimTrends,
  buildTrace,
  clock,
  decodePowerCurve,
  encodePowerCurve,
  formatAltitude,
  powerCurveFraction,
  powerCurveHoverAt,
  setDistanceUnit,
  swimTrendHoverAt,
  zoneDuo,
  type SwimTrendChartPoint,
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

test('keeps a long-duration hover honest when its reference point is missing', () => {
  const curve = [
    { s: 60, w: 350 },
    { s: 2_340, w: 166 },
    { s: 3_600, w: 150 },
  ]
  const reference = [
    { s: 60, w: 400 },
    { s: 3_600, w: 170 },
  ]
  const fraction = powerCurveFraction(2_340, 60, 3_600)
  assert.deepEqual(powerCurveHoverAt(curve, reference, fraction), {
    index: 1,
    durationS: 2_340,
    watts: 166,
    referenceWatts: null,
    xPct: fraction * 100,
  })
})

test('round trips dense and sparse power curve attributes', () => {
  const dense = [
    { s: 1, w: 700 },
    { s: 2, w: 660 },
    { s: 3, w: 635 },
  ]
  const sparse = [
    { s: 1, w: 700 },
    { s: 5, w: 590 },
    { s: 60, w: 350 },
  ]
  assert.deepEqual(decodePowerCurve(encodePowerCurve(dense)), dense)
  assert.deepEqual(decodePowerCurve(encodePowerCurve(sparse)), sparse)
  assert.deepEqual(decodePowerCurve('d|1|700,nope'), [])
  assert.deepEqual(decodePowerCurve('d|1|700,'), [])
  assert.deepEqual(decodePowerCurve('s|1:700,2:'), [])
})

test('selects the nearest serialized swim trend point and clamps the scrub range', () => {
  const points: SwimTrendChartPoint[] = [
    {
      activityId: 1,
      date: '2026-07-01',
      start: '2026-07-01T12:00:00Z',
      value: 112,
      xPct: 0,
      yPct: 80,
    },
    {
      activityId: 2,
      date: '2026-07-03',
      start: '2026-07-03T12:00:00Z',
      value: 108,
      xPct: 40,
      yPct: 40,
    },
    {
      activityId: 3,
      date: '2026-07-09',
      start: '2026-07-09T12:00:00Z',
      value: 100,
      xPct: 100,
      yPct: 0,
    },
  ]

  assert.deepEqual(swimTrendHoverAt(points, 0.51), {
    index: 1,
    activityId: 2,
    date: '2026-07-03',
    start: '2026-07-03T12:00:00Z',
    value: 108,
    xPct: 40,
    yPct: 40,
  })
  assert.equal(swimTrendHoverAt(points, -1)?.index, 0)
  assert.equal(swimTrendHoverAt(points, 2)?.index, 2)
  assert.equal(swimTrendHoverAt(points, Number.NaN)?.index, 0)
  assert.equal(swimTrendHoverAt([], 0.5), null)
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
  start: '2026-07-09T12:00:00Z',
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
  strokeCount: null,
  strokeRateSpm: null,
  swimPaceSPer100m: null,
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

test('labels the activity disclosure and exposes its expanded state and controlled panel', () => {
  const collapsed = buildActivity(factory, detail({ id: 42 }))
  const collapsedToggle = byClass(collapsed, 'tri-act-toggle')[0]
  const collapsedPanel = byClass(collapsed, 'tri-act-more')[0]
  assert.ok(collapsedToggle)
  assert.ok(collapsedPanel)
  assert.equal(text(collapsedToggle), '+ see more')
  assert.equal(collapsedToggle.properties.ariaExpanded, 'false')
  assert.deepEqual(collapsedToggle.properties.ariaControls, ['tri-act-more-42'])
  assert.equal(collapsedPanel.properties.id, 'tri-act-more-42')

  const expanded = buildActivity(factory, detail({ id: 42 }), true)
  const expandedToggle = byClass(expanded, 'tri-act-toggle')[0]
  assert.ok(expandedToggle)
  assert.equal(text(expandedToggle), '− see less')
  assert.equal(expandedToggle.properties.ariaExpanded, 'true')
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

test('prefers active swim pace and adds stroke rate and count to the main stats', () => {
  const rendered = buildActivity(
    factory,
    detail({
      sport: 'swim',
      distanceKm: 1,
      movingTimeS: 1_200,
      route: [],
      bestEfforts: null,
      swimPaceSPer100m: 95.4,
      strokeRateSpm: 31.5,
      strokeCount: 876,
    }),
  )
  const stats = byClass(rendered, 'tri-act-stats')[0]
  assert.ok(stats)
  assert.deepEqual(bodyRows(stats), [
    ['distance', '1,000 m'],
    ['time', "20'"],
    ['pace', '1:35 /100m'],
    ['stroke rate', '31.5 str/min'],
    ['strokes', '876'],
    ['avg hr', '148 bpm'],
  ])
})

const swimTrendDetail = (overrides: Partial<StravaActivityDetail> = {}): StravaActivityDetail =>
  detail({
    id: 5,
    sport: 'swim',
    name: 'Pool swim',
    date: '2026-07-05',
    start: '2026-07-05T12:00:00Z',
    distanceKm: 1.5,
    movingTimeS: 1_500,
    route: [],
    bestEfforts: null,
    swimPaceSPer100m: 100,
    strokeRateSpm: 28,
    strokeCount: 700,
    ...overrides,
  })

const swimTrendPoints: SwimTrendPoint[] = [
  {
    id: 1,
    date: '2026-07-01',
    start: '2026-07-01T12:00:00Z',
    paceSPer100m: 112,
    strokeRateSpm: 20,
  },
  {
    id: 2,
    date: '2026-07-02',
    start: '2026-07-02T12:00:00Z',
    paceSPer100m: 110,
    strokeRateSpm: 22,
  },
  {
    id: 3,
    date: '2026-07-03',
    start: '2026-07-03T12:00:00Z',
    paceSPer100m: 108,
    strokeRateSpm: 24,
  },
  {
    id: 4,
    date: '2026-07-04',
    start: '2026-07-04T12:00:00Z',
    paceSPer100m: 106,
    strokeRateSpm: 26,
  },
  {
    id: 5,
    date: '2026-07-05',
    start: '2026-07-05T12:00:00Z',
    paceSPer100m: 100,
    strokeRateSpm: 28,
  },
  { id: 7, date: '2026-07-05', start: '2026-07-05T18:00:00Z', paceSPer100m: 90, strokeRateSpm: 40 },
  { id: 6, date: '2026-07-06', start: '2026-07-06T12:00:00Z', paceSPer100m: 90, strokeRateSpm: 40 },
]

test('renders aligned swim trends with the selected value and prior-four delta', () => {
  const rendered = buildSwimTrends(factory, swimTrendDetail(), swimTrendPoints)
  assert.ok(rendered)
  assert.equal(rendered.tagName, 'section')
  assert.equal(rendered.properties.ariaLabel, 'Swim trends')
  assert.deepEqual(byClass(rendered, 'tri-swim-trend-title').map(text), [
    'pace /100m',
    'stroke rate str/min',
  ])
  assert.deepEqual(byClass(rendered, 'tri-swim-trend-value').map(text), ['1:40', '28'])
  assert.deepEqual(byClass(rendered, 'tri-swim-trend-delta').map(text), [
    '9s faster vs prior 4',
    '+5 str/min vs prior 4',
  ])

  const pace = byClass(rendered, 'tri-swim-trend--pace')[0]
  const stroke = byClass(rendered, 'tri-swim-trend--stroke')[0]
  assert.ok(pace)
  assert.ok(stroke)
  assert.equal(byClass(rendered, 'tri-zone-duo').length, 1)
  assert.ok(classNames(pace).includes('tri-zone'))
  assert.ok(classNames(stroke).includes('tri-zone'))
  assert.deepEqual(byClass(pace, 'tri-cax-yt').map(text), ['1:40', '1:45', '1:50', '1:55'])
  assert.deepEqual(byClass(stroke, 'tri-cax-yt').map(text), ['20', '22', '24', '26', '28'])
  assert.deepEqual(byClass(pace, 'tri-cax-xt').map(text), ['Jul 1', 'Jul 3', 'Jul 5'])
  assert.deepEqual(
    byClass(stroke, 'tri-cax-xt').map(tick => [text(tick), tick.properties.style]),
    byClass(pace, 'tri-cax-xt').map(tick => [text(tick), tick.properties.style]),
  )
  const paceSvg = byClass(pace, 'tri-swim-trend-svg')[0]
  assert.ok(paceSvg)
  assert.equal(paceSvg.properties.role, 'slider')
  assert.equal(paceSvg.properties.tabIndex, 0)
  assert.equal(paceSvg.properties.ariaOrientation, 'horizontal')
  assert.equal(paceSvg.properties.ariaValueMin, 0)
  assert.equal(paceSvg.properties.ariaValueMax, 4)
  assert.equal(paceSvg.properties.ariaValueNow, 4)
  assert.match(
    String(paceSvg.properties.ariaValueText),
    /Jul 5, 2026, at 8:00 AM, swim pace 1:40 per 100 metres\. 9 seconds faster than prior 4\./,
  )
  assert.equal(paceSvg.properties.dataSwimKind, 'pace')
  assert.equal(paceSvg.properties.dataSwimIndex, 4)
  const paceSeries = JSON.parse(String(paceSvg.properties.dataSwimSeries)) as SwimTrendChartPoint[]
  assert.deepEqual(paceSeries[0], {
    activityId: 1,
    date: '2026-07-01',
    start: '2026-07-01T12:00:00Z',
    value: 112,
    xPct: 0,
    yPct: 80,
  })
  assert.deepEqual(paceSeries.at(-1), {
    activityId: 5,
    date: '2026-07-05',
    start: '2026-07-05T12:00:00Z',
    value: 100,
    xPct: 100,
    yPct: 0,
  })
  const pacePath = byClass(paceSvg, 'tri-swim-trend-line')[0]
  assert.ok(pacePath)
  assert.match(String(pacePath.properties.d), /^M 0\.00 24\.00 .* L 100\.00 0\.00$/)
  assert.deepEqual(
    byClass(rendered, 'tri-swim-trend-current').map(marker => [
      marker.properties.dataActivityId,
      marker.properties.style,
    ]),
    [
      ['5', 'left:100.00%;top:0.00%'],
      ['5', 'left:100.00%;top:0.00%'],
    ],
  )
  assert.deepEqual(
    byClass(rendered, 'tri-swim-trend-hover').map(point => point.properties.hidden),
    [true, true],
  )
  assert.equal(byClass(rendered, 'tri-chart-cursor').length, 2)
  assert.deepEqual(byClass(pace, 'tri-swim-trend-readout').map(text), ['Jul 5 · 8:00 AM1:40 /100m'])
})

test('keeps same-date swim activities distinct and equally spaced', () => {
  const rendered = buildSwimTrends(
    factory,
    swimTrendDetail({
      id: 7,
      start: '2026-07-05T18:00:00Z',
      swimPaceSPer100m: 90,
      strokeRateSpm: 40,
    }),
    swimTrendPoints,
  )
  assert.ok(rendered)
  const paceSvg = byClass(rendered, 'tri-swim-trend-svg--pace')[0]
  assert.ok(paceSvg)
  const series = JSON.parse(String(paceSvg.properties.dataSwimSeries)) as SwimTrendChartPoint[]

  assert.deepEqual(
    series.map(point => [point.activityId, point.date, point.start, point.xPct]),
    [
      [1, '2026-07-01', '2026-07-01T12:00:00Z', 0],
      [2, '2026-07-02', '2026-07-02T12:00:00Z', 20],
      [3, '2026-07-03', '2026-07-03T12:00:00Z', 40],
      [4, '2026-07-04', '2026-07-04T12:00:00Z', 60],
      [5, '2026-07-05', '2026-07-05T12:00:00Z', 80],
      [7, '2026-07-05', '2026-07-05T18:00:00Z', 100],
    ],
  )
  assert.match(String(paceSvg.properties.ariaValueText), /Jul 5, 2026, at 2:00 PM/)
  assert.deepEqual(byClass(rendered, 'tri-swim-trend-readout-date').map(text), [
    'Jul 5 · 2:00 PM',
    'Jul 5 · 2:00 PM',
  ])
})

test('keeps sparse stroke observations aligned and falls back to pace alone below four', () => {
  const sparse = swimTrendPoints
    .slice(0, 5)
    .map((point, index) => ({ ...point, strokeRateSpm: index === 1 ? null : point.strokeRateSpm }))
  const rendered = buildSwimTrends(factory, swimTrendDetail(), sparse)
  assert.ok(rendered)
  assert.equal(byClass(rendered, 'tri-swim-trend').length, 2)
  const paceSvg = byClass(rendered, 'tri-swim-trend-svg--pace')[0]
  const strokeSvg = byClass(rendered, 'tri-swim-trend-svg--stroke')[0]
  const strokePath = byClass(
    byClass(rendered, 'tri-swim-trend--stroke')[0],
    'tri-swim-trend-line',
  )[0]
  assert.ok(paceSvg)
  assert.ok(strokeSvg)
  assert.ok(strokePath)
  assert.equal(String(strokePath.properties.d).match(/[ML]/g)?.length, 4)
  assert.match(String(strokePath.properties.d), /^M 0\.00 .* M 50\.00 .* L 75\.00 .* L 100\.00/)
  const paceSeries = JSON.parse(String(paceSvg.properties.dataSwimSeries)) as SwimTrendChartPoint[]
  const strokeSeries = JSON.parse(
    String(strokeSvg.properties.dataSwimSeries),
  ) as SwimTrendChartPoint[]
  assert.deepEqual(
    paceSeries.map(point => point.xPct),
    [0, 25, 50, 75, 100],
  )
  assert.deepEqual(
    strokeSeries.map(point => [point.activityId, point.xPct]),
    [
      [1, 0],
      [3, 50],
      [4, 75],
      [5, 100],
    ],
  )

  const paceOnly = buildSwimTrends(
    factory,
    swimTrendDetail(),
    sparse.map((point, index) => ({
      ...point,
      strokeRateSpm: index < 2 ? null : point.strokeRateSpm,
    })),
  )
  assert.ok(paceOnly)
  assert.equal(byClass(paceOnly, 'tri-swim-trend--pace').length, 1)
  assert.equal(byClass(paceOnly, 'tri-swim-trend--stroke').length, 0)
  assert.equal(byClass(paceOnly, 'tri-zone-duo').length, 0)

  assert.equal(buildSwimTrends(factory, swimTrendDetail(), swimTrendPoints.slice(0, 2)), null)
})

test('renders a stroke-rate trend independently when pace is unavailable', () => {
  const points = swimTrendPoints.slice(0, 5).map(point => ({ ...point, paceSPer100m: null }))
  const rendered = buildSwimTrends(factory, swimTrendDetail({ swimPaceSPer100m: null }), points)

  assert.ok(rendered)
  assert.equal(byClass(rendered, 'tri-swim-trend--pace').length, 0)
  assert.equal(byClass(rendered, 'tri-swim-trend--stroke').length, 1)
})

test('selects one latest-sixteen activity window for both swim metrics', () => {
  const points = Array.from({ length: 20 }, (_, index): SwimTrendPoint => {
    const day = (index + 1).toString().padStart(2, '0')
    return {
      id: index + 1,
      date: `2026-07-${day}`,
      start: `2026-07-${day}T12:00:00Z`,
      paceSPer100m: 120 - index,
      strokeRateSpm: index < 3 ? 20 + index : null,
    }
  })
  const rendered = buildSwimTrends(
    factory,
    swimTrendDetail({
      id: 20,
      date: '2026-07-20',
      start: '2026-07-20T12:00:00Z',
      swimPaceSPer100m: 101,
      strokeRateSpm: 28,
    }),
    points,
  )

  assert.ok(rendered)
  assert.equal(byClass(rendered, 'tri-swim-trend--pace').length, 1)
  assert.equal(byClass(rendered, 'tri-swim-trend--stroke').length, 0)
})

test('includes swim trends in the default server-rendered day card', () => {
  const current = swimTrendDetail()
  const rendered = buildDayCard(factory, current.date, {
    details: { [current.id]: current },
    swimTrend: swimTrendPoints,
    health: {},
  })

  assert.equal(byClass(rendered, 'tri-swim-trends').length, 1)
  assert.equal(byClass(rendered, 'tri-act-toggle').length, 1)
  assert.equal(byClass(rendered, 'tri-act-more').length, 1)
})

test('day-card date renders as a month link only when extras provide an href', () => {
  const current = detail({ id: 7, date: '2026-07-09' })
  const payload = { details: { 7: current }, health: {} }
  const linked = buildDayCard(factory, '2026-07-09', payload, {
    dateHref: '../../../triathlon/on/2026/07',
  })
  const anchor = byClass(linked, 'tri-pop-date')[0]
  assert.equal(anchor.tagName, 'a')
  assert.equal(anchor.properties.href, '../../../triathlon/on/2026/07')
  const plain = buildDayCard(factory, '2026-07-09', payload)
  assert.equal(byClass(plain, 'tri-pop-date')[0].tagName, 'span')
})

test('expanded day-card extras render every activity pre-expanded', () => {
  const first = detail({ id: 1, date: '2026-07-09' })
  const second = detail({ id: 2, date: '2026-07-09', sport: 'run' })
  const rendered = buildDayCard(
    factory,
    '2026-07-09',
    { details: { 1: first, 2: second }, health: {} },
    { expanded: true },
  )
  assert.equal(byClass(rendered, 'tri-act--expanded').length, 2)
  const toggles = byClass(rendered, 'tri-act-toggle')
  assert.ok(toggles.length >= 1)
  for (const toggle of toggles) {
    assert.equal(text(toggle), '− see less')
    assert.equal(toggle.properties.ariaExpanded, 'true')
  }
})

test('limits swim trends to the latest sixteen sessions', () => {
  const points = Array.from({ length: 20 }, (_, index): SwimTrendPoint => {
    const day = (index + 1).toString().padStart(2, '0')
    return {
      id: index + 1,
      date: `2026-07-${day}`,
      start: index === 4 ? '' : `2026-07-${day}T12:00:00Z`,
      paceSPer100m: 120 - index,
      strokeRateSpm: null,
    }
  })
  const rendered = buildSwimTrends(
    factory,
    swimTrendDetail({
      id: 20,
      date: '2026-07-20',
      start: '2026-07-20T12:00:00Z',
      swimPaceSPer100m: 101,
      strokeRateSpm: null,
    }),
    points,
  )
  assert.ok(rendered)
  const path = byClass(rendered, 'tri-swim-trend-line')[0]
  assert.ok(path)
  assert.equal(String(path.properties.d).match(/[ML]/g)?.length, 16)
  assert.deepEqual(byClass(rendered, 'tri-cax-xt').map(text), ['Jul 5', 'Jul 12', 'Jul 20'])
  const svg = byClass(rendered, 'tri-swim-trend-svg')[0]
  assert.ok(svg)
  const series = JSON.parse(String(svg.properties.dataSwimSeries)) as SwimTrendChartPoint[]
  assert.deepEqual(
    series.map(point => point.activityId),
    Array.from({ length: 16 }, (_, i) => i + 5),
  )
  assert.deepEqual(
    series.map(point => Number(point.xPct.toFixed(2))),
    [0, 6.67, 13.33, 20, 26.67, 33.33, 40, 46.67, 53.33, 60, 66.67, 73.33, 80, 86.67, 93.33, 100],
  )
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
  curveYearRef: [],
  curveYear: null,
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
    '600w',
  ])
})

test('labels a three-hour power curve through its endpoint', () => {
  const curve = buildPowerCurve(
    factory,
    detail({
      powerCurve: [
        { s: 1, w: 565 },
        { s: 10_800, w: 180 },
      ],
    }),
    ctx(),
  )
  assert.ok(curve)
  assert.deepEqual(byClass(curve, 'tri-cax-xt').map(text), [
    '1s',
    '5s',
    '30s',
    '1m',
    '5m',
    '20m',
    '1h',
    '3h',
  ])
  const lastTick = byClass(curve, 'tri-cax-xt').at(-1)
  assert.ok(lastTick)
  assert.equal(classNames(lastTick).includes('tri-cax-xt--last'), true)
})

test('keeps every hover value while bounding a dense power curve path', () => {
  const powerCurve = Array.from({ length: 10_800 }, (_, index) => ({
    s: index + 1,
    w: 700 - Math.floor(index / 20),
  }))
  const curve = buildPowerCurve(factory, detail({ powerCurve }), ctx())
  assert.ok(curve)
  const svg = byClass(curve, 'tri-curve-svg')[0]
  const path = byClass(curve, 'tri-curve-line')[0]
  assert.ok(svg)
  assert.ok(path)
  const encoded = String(svg.properties.dataCurve)
  const decoded = decodePowerCurve(encoded)
  assert.equal(decoded.length, powerCurve.length)
  for (const seconds of [61, 3_601, 7_200, 10_800]) {
    assert.deepEqual(decoded[seconds - 1], powerCurve[seconds - 1])
    assert.equal(
      powerCurveHoverAt(
        decoded,
        [],
        powerCurveFraction(seconds, decoded[0].s, decoded[decoded.length - 1].s),
      )?.durationS,
      seconds,
    )
  }
  assert.equal((String(path.properties.d).match(/[ML]/g) ?? []).length <= 1_024, true)
  assert.equal(encoded.length < JSON.stringify(powerCurve).length / 2, true)
})

test('scales the power curve axis above the six-week peak and renders selected points', () => {
  const curve = buildPowerCurve(
    factory,
    zonedDetail(),
    ctx({
      curveRef: [
        { s: 1, w: 1_060 },
        { s: 5, w: 1_020 },
        { s: 60, w: 400 },
        { s: 300, w: 300 },
        { s: 1_200, w: 240 },
        { s: 3_600, w: 210 },
      ],
    }),
  )
  assert.ok(curve)
  assert.deepEqual(byClass(curve, 'tri-cax-yt').map(text), [
    '0',
    '200w',
    '400w',
    '600w',
    '800w',
    '1,000w',
    '1,200w',
  ])
  const svg = byClass(curve, 'tri-curve-svg')[0]
  assert.ok(svg)
  assert.equal(svg.properties.dataCurveDomainMax, 1_200)
  const ridePoint = byClass(curve, 'tri-curve-point--ride')[0]
  const referencePoint = byClass(curve, 'tri-curve-point--ref')[0]
  assert.ok(ridePoint)
  assert.ok(referencePoint)
  assert.equal(ridePoint.properties.ariaHidden, 'true')
  assert.equal(referencePoint.properties.ariaHidden, 'true')
})

test('renders six-week and calendar-year comparison ranges on one watt domain', () => {
  const curve = buildPowerCurve(
    factory,
    zonedDetail(),
    ctx({
      curveRef: [
        { s: 1, w: 700 },
        { s: 60, w: 400 },
        { s: 3_600, w: 220 },
      ],
      curveYearRef: [
        { s: 1, w: 1_060 },
        { s: 60, w: 440 },
        { s: 3_600, w: 240 },
      ],
      curveYear: 2026,
    }),
  )
  assert.ok(curve)
  assert.deepEqual(byClass(curve, 'tri-cax-yt').map(text), [
    '0',
    '200w',
    '400w',
    '600w',
    '800w',
    '1,000w',
    '1,200w',
  ])
  const ranges = byClass(curve, 'tri-curve-range')
  assert.deepEqual(ranges.map(text), ['6 weeks', 'all of 2026'])
  assert.deepEqual(
    ranges.map(button => button.properties.ariaPressed),
    ['true', 'false'],
  )
  const paths = byClass(curve, 'tri-curve-ref')
  assert.equal(paths.length, 2)
  assert.equal('hidden' in paths[0].properties, false)
  assert.equal('hidden' in paths[1].properties, true)
  const svg = byClass(curve, 'tri-curve-svg')[0]
  assert.ok(svg)
  assert.equal(svg.properties.dataCurveRange, 'six-weeks')
  assert.equal(svg.properties.dataCurveYear, 2026)
  assert.equal(decodePowerCurve(String(svg.properties.dataCurveRefSixWeeks))[0].w, 700)
  assert.equal(decodePowerCurve(String(svg.properties.dataCurveRefYear))[0].w, 1_060)
  const stage = byClass(curve, 'tri-cax-stage')[0]
  assert.ok(stage)
  assert.equal(byClass(stage, 'tri-curve-readout').length, 1)
})

test('uses the calendar year when no six-week power reference exists', () => {
  const curve = buildPowerCurve(
    factory,
    zonedDetail(),
    ctx({
      curveYearRef: [
        { s: 1, w: 900 },
        { s: 60, w: 380 },
        { s: 3_600, w: 215 },
      ],
      curveYear: 2026,
    }),
  )
  assert.ok(curve)
  const ranges = byClass(curve, 'tri-curve-range')
  assert.equal(ranges[0].properties.disabled, true)
  assert.deepEqual(
    ranges.map(button => button.properties.ariaPressed),
    ['false', 'true'],
  )
  const svg = byClass(curve, 'tri-curve-svg')[0]
  assert.ok(svg)
  assert.equal(svg.properties.dataCurveRange, 'year')
  assert.deepEqual(byClass(curve, 'tri-curve-readout-label').map(text), ['this ride', '2026 best'])
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
  assert.equal(svg.properties.ariaReadonly, undefined)
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
