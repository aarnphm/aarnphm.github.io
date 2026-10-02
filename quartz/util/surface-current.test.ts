import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildSurfaceCurrentEstimate,
  findSurfaceCurrentElement,
  loofsRequestsForHour,
  parseLoofsFieldAscii,
  parseLoofsMeshAscii,
  parsePublicSurfaceCurrentEstimate,
  parseSurfaceCurrentEstimate,
  type SurfaceCurrentField,
  type SurfaceCurrentMesh,
} from './surface-current'

// Live model data cannot exercise malformed cache fields, dry cells, reversed vectors,
// land coordinates, GPS pauses, or a route that stops before the activity clock does.
const mesh: SurfaceCurrentMesh = {
  latitudes: [43, 44, 43],
  longitudes: [-80, -80, -79],
  triangles: [[0, 1, 2]],
}
const start = '2026-09-27T18:00:00.000Z'
const sourceUrl =
  'https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/2026/09/27/loofs.t18z.20260927.fields.n006.nc.ascii'
const field = (hour: number, uMps: number, vMps: number, wet = true): SurfaceCurrentField => ({
  validTime: new Date(Date.parse(start) + hour * 3_600_000).toISOString(),
  cycleTime: '2026-09-28T00:00:00.000Z',
  sourceUrl,
  elements: new Map([[0, { uMps, vMps, wet }]]),
})
const estimate = (fields = [field(0, 0.1, 0), field(1, 0.1, 0)]) =>
  buildSurfaceCurrentEstimate({
    activityId: 101,
    routeFingerprint: 'route',
    start,
    end: '2026-09-27T18:02:00.000Z',
    computedAt: Date.parse('2026-10-02T12:00:00.000Z'),
    timeS: [0, 30, 60, 90, 120],
    latlng: [
      [43.2, -79.8],
      [43.2, -79.8],
      [43.2, -79.8],
      [43.2, -79.8],
      [43.2, -79.8],
    ],
    mesh,
    fields,
  })

test('containing triangle selection rejects a nearby coordinate outside the lake mesh', () => {
  assert.equal(findSurfaceCurrentElement(mesh, 43.2, -79.8), 0)
  assert.equal(findSurfaceCurrentElement(mesh, 43.8, -79.2), null)
  assert.equal(findSurfaceCurrentElement(mesh, Number.NaN, -79.8), null)
})

test('parses provider ASCII mesh connectivity as one-based nodes and wraps longitude', () => {
  const value = parseLoofsMeshAscii(
    `Dataset {}\n---------------------------------------------\nlon[3]\n280, 280, 281\n\nlat[3]\n43, 44, 43\n\nnv[3][1]\n[0], 1\n[1], 2\n[2], 3\n`,
  )
  assert.deepEqual(value, mesh)
  assert.equal(parseLoofsMeshAscii('Dataset {}\nlon[3]\n280, 280, NaN'), null)
})

test('parses a bounded provider field and rejects wrong valid time or missing velocity', () => {
  const text = `Dataset {}\n---------------------------------------------\nTimes[1]\n"2026-09-27T18:00:00.000000"\n\nu[1][1][2]\n[0][0], 0.1, NaN\n\nv[1][1][2]\n[0][0], 0.2, 0.3\n\nwet_cells[1][2]\n[0], 1, 1\n`
  const value = parseLoofsFieldAscii(text, 17, start, start, sourceUrl)
  assert.deepEqual(value?.elements.get(17), { uMps: 0.1, vMps: 0.2, wet: true })
  assert.equal(value?.elements.has(18), false)
  assert.equal(parseLoofsFieldAscii(text, 17, start, '2026-09-27T19:00:00Z', sourceUrl), null)
})

test('malformed provider timestamps return a missing field without throwing', () => {
  const text = `Dataset {}\n---------------------------------------------\nTimes[1]\n"unavailable"\n\nu[1][1][1]\n[0][0], 0.1\n\nv[1][1][1]\n[0][0], 0.2\n\nwet_cells[1][1]\n[0], 1\n`
  assert.equal(parseLoofsFieldAscii(text, 0, start, start, sourceUrl), null)
})

test('historical requests select the retrospective six-hour cycle across midnight', () => {
  const value = loofsRequestsForHour('2026-09-27T19:00:00.000Z')
  assert.equal(value?.cycleTime, '2026-09-28T00:00:00.000Z')
  assert.ok(value?.recentUrl.endsWith('/2026/09/28/loofs.t00z.20260928.fields.n001.nc.ascii'))
  assert.ok(value?.archiveUrl.endsWith('/2026/09/loofs.t00z.20260928.fields.n001.nc.ascii'))
})

test('integrates vector speed over the route and records direction toward', () => {
  const value = estimate()
  assert.equal(value.summary.averageSpeedMps, 0.1)
  assert.equal(value.summary.averageDirectionDeg, 90)
  assert.equal(value.summary.coveredDurationS, 120)
  assert.equal(value.summary.coveragePct, 100)
  assert.equal(parseSurfaceCurrentEstimate(value)?.activityId, 101)
})

test('interpolates opposite velocity vectors before deriving speed', () => {
  const value = estimate([field(0, 1, 0), field(1, -1, 0)])
  assert.equal(value.samples.at(-1)?.uMps, 1 - (2 * 120) / 3_600)
  assert.ok(value.summary.averageSpeedMps != null && value.summary.averageSpeedMps < 1)
})

test('dry cells and absent hourly coverage remain null', () => {
  const dry = estimate([field(0, 0.1, 0, false), field(1, 0.1, 0, false)])
  assert.equal(dry.summary.coveragePct, 0)
  assert.equal(dry.summary.averageSpeedMps, null)
  assert.ok(dry.samples.every(sample => sample.speedMps === null))
  const missing = estimate([field(0, 0.1, 0)])
  assert.equal(missing.summary.coveragePct, 0)
})

test('GPS pauses and an uncovered end of the route reduce elapsed coverage', () => {
  const value = buildSurfaceCurrentEstimate({
    activityId: 101,
    routeFingerprint: 'route',
    start,
    end: '2026-09-27T18:04:00.000Z',
    computedAt: Date.now(),
    timeS: [0, 30, 150, 180],
    latlng: [
      [43.2, -79.8],
      [43.2, -79.8],
      [43.2, -79.8],
      [43.2, -79.8],
    ],
    mesh,
    fields: [field(0, 0.1, 0), field(1, 0.1, 0)],
  })
  assert.equal(value.summary.coveredDurationS, 60)
  assert.equal(value.summary.coveragePct, 25)
  assert.equal(value.samples.at(-1)?.speedMps, null)
  assert.ok(
    value.samples.some(
      sample => sample.elapsedS > 30 && sample.elapsedS < 150 && sample.speedMps === null,
    ),
  )
})

test('serialized samples stay bounded and preserve pauses while summary uses every interval', () => {
  const timeS = Array.from({ length: 1_201 }, (_, i) => (i < 600 ? i : i + 120))
  const value = buildSurfaceCurrentEstimate({
    activityId: 101,
    routeFingerprint: 'route',
    start,
    end: new Date(Date.parse(start) + 1_320_000).toISOString(),
    computedAt: Date.now(),
    timeS,
    latlng: timeS.map(() => [43.2, -79.8]),
    mesh,
    fields: [field(0, 0.1, 0), field(1, 0.1, 0)],
  })
  assert.ok(value.samples.length <= 512)
  assert.equal(value.summary.coveredDurationS, 1_199)
  assert.ok(
    value.samples.some(
      sample => sample.speedMps === null && sample.elapsedS > 599 && sample.elapsedS < 720,
    ),
  )
})

test('strict cache parsing rejects impossible coverage, timelines, vectors and private public fields', () => {
  const value = estimate()
  assert.equal(
    parseSurfaceCurrentEstimate({ ...value, summary: { ...value.summary, coveragePct: 110 } }),
    null,
  )
  assert.equal(
    parseSurfaceCurrentEstimate({ ...value, samples: [...value.samples].reverse() }),
    null,
  )
  const first = value.samples[0]
  assert.ok(first)
  assert.equal(
    parseSurfaceCurrentEstimate({
      ...value,
      samples: [{ ...first, speedMps: 20 }, ...value.samples.slice(1)],
    }),
    null,
  )
  assert.equal(parsePublicSurfaceCurrentEstimate(value), null)
  const { routeFingerprint: _fingerprint, ...publicValue } = value
  assert.equal(parsePublicSurfaceCurrentEstimate(publicValue)?.activityId, 101)
  assert.equal(parsePublicSurfaceCurrentEstimate({ ...publicValue, latitude: 43.2 }), null)
})
