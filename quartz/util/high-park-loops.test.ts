import assert from 'node:assert/strict'
import test from 'node:test'
import { buildHighParkLoops, type HighParkActivity } from './high-park-loops'

const circuit = [
  [43.645655, -79.46473],
  [43.647451, -79.464921],
  [43.649286, -79.464933],
  [43.651173, -79.464803],
  [43.652093, -79.465654],
  [43.650527, -79.467289],
  [43.648459, -79.467779],
  [43.646277, -79.467284],
  [43.645228, -79.465449],
  [43.645655, -79.46473],
]
function activity(sport: 'bike' | 'run', laps = 2, reverse = false): HighParkActivity {
  const path = reverse ? [...circuit].reverse() : circuit
  const samples = path.flatMap((p, i) =>
    i === path.length - 1
      ? [p]
      : Array.from({ length: 10 }, (_, j) => [
          p[0] + ((path[i + 1][0] - p[0]) * j) / 10,
          p[1] + ((path[i + 1][1] - p[1]) * j) / 10,
        ]),
  )
  const points = Array.from({ length: laps }, () => samples.slice(0, -1))
    .flat()
    .concat([samples[0]])
  return {
    id: 123,
    date: '2026-11-07',
    start: '2026-11-07T13:00:00Z',
    name: 'park',
    sport,
    route: points.map(([lat, lng], i) => ({ lat, lng, d: (i * 1.85) / 90, elapsedS: i * 3 })),
    analysisRanges: [],
  }
}
const effort = {
  id: 'one',
  kind: 'segment',
  label: 'High Park - Main Loop - Centre Road',
  startElapsedS: 0,
  endElapsedS: 200,
  durationS: 200,
  distanceKm: 1.8261,
} satisfies HighParkActivity['analysisRanges'][number]

test('native circuit efforts count once and exclude overlapping, half and multi-lap segments', () => {
  const a = activity('bike')
  a.analysisRanges = [
    effort,
    effort,
    { ...effort, id: 'other', label: 'High Park Once Around from the Top' },
    { ...effort, id: 'half', label: 'High Park Loop - Southbound' },
    { ...effort, id: 'many', label: '#TBM - 6 Laps at High Park' },
  ]
  const result = buildHighParkLoops([a])
  assert.equal(result.cycling.laps, 1)
  assert.equal(result.cycling.activities[0].source, 'Strava segment')
  assert.equal(result.cycling.activities[0].href, '/triathlon/on/2026/11/07#tri-activity-123')
})

test('GPS circuits count separately for cycling and running in either direction', () => {
  const run = { ...activity('run', 2, true), id: 456 }
  const latlng: [number, number][] = run.route.map(p => [p.lat, p.lng])
  const result = buildHighParkLoops([activity('bike'), { ...run, route: [] }], {
    '456': {
      latlng,
      time: run.route.map(p => p.elapsedS),
      distance: run.route.map(p => p.d * 1000),
    },
  })
  assert.equal(result.cycling.laps, 2)
  assert.equal(result.running.laps, 2)
  assert.equal(result.running.activities[0].source, 'GPS estimate')
})

test('partial visits, backtracking, absent routes and virtual rides stay out of the ledger', () => {
  const a = activity('bike')
  const half = a.route.slice(0, 44)
  const result = buildHighParkLoops([
    { ...a, route: half },
    { ...a, route: half.concat([...half].reverse()) },
    { ...a, route: [] },
    { ...a, virtual: true },
  ])
  assert.equal(result.cycling.activities.length, 0)
  assert.equal(result.running.activities.length, 0)
  assert.equal(result.running.fastest, null)
})

test('invalid links and non-finite or broken telemetry cannot produce a lap', () => {
  const a = activity('bike', 1)
  assert.equal(buildHighParkLoops([{ ...a, date: 'invalid' }]).cycling.laps, 0)
  assert.equal(
    buildHighParkLoops([{ ...a, route: a.route.map(p => ({ ...p, d: NaN })) }]).cycling.laps,
    0,
  )
  assert.equal(
    buildHighParkLoops([{ ...a, route: a.route.filter((_, i) => i % 30 === 0) }]).cycling.laps,
    0,
  )
})

test('the fastest lap is drawn from the map route inside the park, with a frame that holds it', () => {
  const a = activity('bike', 2)
  a.mapRoute = [a.route.map(({ lat, lng, d }) => ({ lat, lng, d }))]
  a.analysisRanges = [
    { ...effort, durationS: 210, startDistanceKm: 0 },
    {
      ...effort,
      id: 'two',
      startElapsedS: 300,
      endElapsedS: 495,
      durationS: 195,
      startDistanceKm: 1.85,
    },
  ]
  const result = buildHighParkLoops([a])
  const ride = result.cycling.activities[0]
  assert.deepEqual(ride.lapsS, [210, 195])
  assert.match(ride.trace, /^M\d+ \d+l/)
  assert.ok(result.frame && result.frame.width > 200 && result.frame.height > 600)
  assert.equal(buildHighParkLoops([activity('bike')]).frame, null)
})

test('the ride glyph keeps the whole route and inks only the in-park stretch', () => {
  const a = activity('bike', 1)
  const outside = { lat: 43.66, lng: -79.4, d: 0, elapsedS: 0 }
  a.route = [outside, ...a.route].map((p, i, all) => ({ ...p, x: i / (all.length - 1), y: 0.5 }))
  a.analysisRanges = [effort]
  const glyph = buildHighParkLoops([a]).cycling.activities[0].glyph
  assert.ok(glyph)
  assert.match(glyph.route, /^M0 20l/)
  assert.doesNotMatch(glyph.park, /^M0 /)
})
