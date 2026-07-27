import { XMLParser } from 'fast-xml-parser'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { isRecord } from '../util/type-guards'
import {
  exportAppleMultisportTcx,
  parseAppleMultisportTcxArgs,
} from './export-apple-multisport-tcx'

const workoutId = '11111111-2222-3333-4444-555555555555'

interface FixtureOptions {
  activities?: Record<string, unknown>[]
  routeDisciplines?: readonly ('swim' | 'bike' | 'run' | 'transition')[]
  boundaryRoutePoints?: boolean
  extraHeartRate?: Record<string, unknown>[]
  heartRate?: Record<string, unknown>[]
}

function segment(
  id: string,
  activity: string,
  start: string,
  end: string,
  values: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    id,
    activity,
    start,
    end,
    durationS: (Date.parse(end) - Date.parse(start)) / 1000,
    elapsedTimeS: (Date.parse(end) - Date.parse(start)) / 1000,
    distanceM: null,
    activeEnergyKcal: null,
    averageHeartRateBpm: null,
    averagePowerW: null,
    averageCadencePerMinute: null,
    lapCount: null,
    ...values,
  }
}

function fixtureActivities(): Record<string, unknown>[] {
  return [
    segment(
      'aaaaaaaa-0000-0000-0000-000000000001',
      'swimming',
      '2026-07-26T08:00:00Z',
      '2026-07-26T08:01:00Z',
      {
        durationS: 55,
        distanceM: 400,
        activeEnergyKcal: 120,
        averageHeartRateBpm: 123,
        lapCount: 1,
      },
    ),
    segment(
      'aaaaaaaa-0000-0000-0000-000000000002',
      'transition',
      '2026-07-26T08:01:00Z',
      '2026-07-26T08:01:30Z',
    ),
    segment(
      'aaaaaaaa-0000-0000-0000-000000000003',
      'cycling',
      '2026-07-26T08:01:30Z',
      '2026-07-26T08:02:30Z',
      {
        durationS: 58,
        distanceM: 24_300,
        activeEnergyKcal: 800,
        averageHeartRateBpm: 151,
        averagePowerW: 225,
        averageCadencePerMinute: 91,
        lapCount: 1,
      },
    ),
    segment(
      'aaaaaaaa-0000-0000-0000-000000000004',
      'transition',
      '2026-07-26T08:02:30Z',
      '2026-07-26T08:03:00Z',
    ),
    segment(
      'aaaaaaaa-0000-0000-0000-000000000005',
      'running',
      '2026-07-26T08:03:00Z',
      '2026-07-26T08:04:00Z',
      {
        durationS: 57,
        distanceM: 9_500,
        activeEnergyKcal: 650,
        averageHeartRateBpm: 171,
        averagePowerW: 265,
        averageCadencePerMinute: 87,
        lapCount: 1,
      },
    ),
  ]
}

function routePoint(
  time: string,
  latitude: number,
  longitude: number,
  heartRate: number,
  cadence: number | null,
  speed: number,
  power: number | null,
): string {
  const extensions = [
    '    <extensions>',
    ...(power == null ? [] : [`     <power>${power}</power>`]),
    '     <gpxtpx:TrackPointExtension>',
    `      <gpxtpx:hr>${heartRate}</gpxtpx:hr>`,
    ...(cadence == null ? [] : [`      <gpxtpx:cad>${cadence}</gpxtpx:cad>`]),
    `      <gpxtpx:speed>${speed}</gpxtpx:speed>`,
    '     </gpxtpx:TrackPointExtension>',
    '    </extensions>',
  ]
  return [
    `   <trkpt lat="${latitude}" lon="${longitude}">`,
    '    <ele>76.5</ele>',
    `    <time>${time}</time>`,
    ...extensions,
    '   </trkpt>',
  ].join('\n')
}

function fixtureRoute(
  disciplines: readonly ('swim' | 'bike' | 'run' | 'transition')[],
  boundaryRoutePoints = false,
): string {
  const allPoints: {
    discipline: 'swim' | 'bike' | 'run' | 'transition'
    xml: string
    boundary?: boolean
  }[] = [
    {
      discipline: 'swim',
      xml: routePoint('2026-07-26T08:00:10Z', 43.64, -79.38, 120, null, 1.2, null),
    },
    {
      discipline: 'swim',
      xml: routePoint('2026-07-26T08:00:50Z', 43.6401, -79.3801, 126, null, 1.3, null),
    },
    {
      discipline: 'transition',
      boundary: true,
      xml: routePoint('2026-07-26T08:01:00Z', 43.64015, -79.38015, 130, null, 1.5, null),
    },
    {
      discipline: 'transition',
      xml: routePoint('2026-07-26T08:01:15Z', 43.6402, -79.3802, 135, null, 2, null),
    },
    {
      discipline: 'bike',
      xml: routePoint('2026-07-26T08:01:40Z', 43.641, -79.381, 148, 92, 9.1, 220),
    },
    {
      discipline: 'bike',
      xml: routePoint('2026-07-26T08:02:20Z', 43.642, -79.382, 154, 94, 9.3, 232),
    },
    {
      discipline: 'transition',
      boundary: true,
      xml: routePoint('2026-07-26T08:02:30Z', 43.64205, -79.38205, 156, null, 2, null),
    },
    {
      discipline: 'transition',
      xml: routePoint('2026-07-26T08:02:45Z', 43.6421, -79.3821, 158, null, 2.1, null),
    },
    {
      discipline: 'run',
      xml: routePoint('2026-07-26T08:03:10Z', 43.643, -79.383, 168, 86, 3.4, 258),
    },
    {
      discipline: 'run',
      xml: routePoint('2026-07-26T08:03:50Z', 43.644, -79.384, 174, 88, 3.5, 272),
    },
    {
      discipline: 'run',
      boundary: true,
      xml: routePoint('2026-07-26T08:04:00Z', 43.6442, -79.3842, 176, 90, 3.6, 275),
    },
  ]
  const points = allPoints.filter(
    point => disciplines.includes(point.discipline) && (!point.boundary || boundaryRoutePoints),
  )
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1" xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">',
    ' <trk>',
    '  <trkseg>',
    ...points.map(point => point.xml),
    '  </trkseg>',
    ' </trk>',
    '</gpx>',
    '',
  ].join('\n')
}

function fixtureCourseGpx(): string {
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">',
    ' <rte>',
    '  <name>Swim course</name>',
    '  <rtept lat="43.6300000" lon="-79.4230000"><ele>75.0</ele></rtept>',
    '  <rtept lat="43.6300000" lon="-79.4240000"><ele>75.5</ele></rtept>',
    '  <rtept lat="43.6310000" lon="-79.4240000"><ele>76.0</ele></rtept>',
    ' </rte>',
    '</gpx>',
    '',
  ].join('\n')
}

function fixtureCourseTcx(): string {
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">',
    ' <Courses>',
    '  <Course>',
    '   <Name>T1 course</Name>',
    '   <Track>',
    '    <Trackpoint>',
    '     <Position><LatitudeDegrees>43.6310000</LatitudeDegrees><LongitudeDegrees>-79.4240000</LongitudeDegrees></Position>',
    '     <AltitudeMeters>76.0</AltitudeMeters>',
    '    </Trackpoint>',
    '    <Trackpoint>',
    '     <Position><LatitudeDegrees>43.6320000</LatitudeDegrees><LongitudeDegrees>-79.4245000</LongitudeDegrees></Position>',
    '     <AltitudeMeters>82.0</AltitudeMeters>',
    '    </Trackpoint>',
    '   </Track>',
    '  </Course>',
    ' </Courses>',
    '</TrainingCenterDatabase>',
    '',
  ].join('\n')
}

async function writeFixture(root: string, options: FixtureOptions = {}): Promise<string> {
  const inputPath = join(root, 'apple-health-import.json')
  await fs.mkdir(join(root, 'GPX'), { recursive: true })
  await fs.writeFile(
    inputPath,
    JSON.stringify({
      version: 10,
      workouts: [
        {
          id: workoutId,
          activity: 'swimBikeRun',
          start: '2026-07-26T08:00:00Z',
          end: '2026-07-26T08:04:00Z',
          durationS: 220,
          elapsedTimeS: 240,
          distanceM: 34_200,
          activeEnergyKcal: 1_570,
          averageHeartRateBpm: 150,
          gpxFile: `GPX/${workoutId}.gpx`,
          heartRate: options.heartRate ?? [
            { time: '2026-07-26T08:00:10Z', bpm: 111 },
            { time: '2026-07-26T08:00:50Z', bpm: 112 },
            { time: '2026-07-26T08:01:40Z', bpm: 145 },
            { time: '2026-07-26T08:02:20Z', bpm: 150 },
            { time: '2026-07-26T08:03:10Z', bpm: 165 },
            { time: '2026-07-26T08:03:50Z', bpm: 170 },
            ...(options.extraHeartRate ?? []),
          ],
          activities: options.activities ?? fixtureActivities(),
        },
      ],
    }),
  )
  await fs.writeFile(
    join(root, 'GPX', `${workoutId}.gpx`),
    fixtureRoute(
      options.routeDisciplines ?? ['swim', 'transition', 'bike', 'run'],
      options.boundaryRoutePoints,
    ),
  )
  return inputPath
}

function requiredRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) assert.fail(`${label} must be an object`)
  return value
}

function recordValue(record: Record<string, unknown>, key: string): Record<string, unknown> {
  return requiredRecord(record[key], key)
}

function arrayValue(record: Record<string, unknown>, key: string): unknown[] {
  const value = record[key]
  return Array.isArray(value) ? value : value == null ? [] : [value]
}

async function parsedActivity(
  path: string,
): Promise<{
  xml: string
  activity: Record<string, unknown>
  lap: Record<string, unknown>
  trackPoints: Record<string, unknown>[]
}> {
  const xml = await fs.readFile(path, 'utf8')
  const parser = new XMLParser({ ignoreAttributes: false })
  const parsed: unknown = parser.parse(xml)
  const database = requiredRecord(parsed, 'document')
  const activities = recordValue(recordValue(database, 'TrainingCenterDatabase'), 'Activities')
  const activity = recordValue(activities, 'Activity')
  const lap = recordValue(activity, 'Lap')
  const track = recordValue(lap, 'Track')
  const trackPoints = arrayValue(track, 'Trackpoint').map((value, index) =>
    requiredRecord(value, `Trackpoint ${index}`),
  )
  return { xml, activity, lap, trackPoints }
}

test('exports three standalone TCX sports with route telemetry and no transition points', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const inputPath = await writeFixture(root)
  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
  })

  assert.deepEqual(
    result.files.map(file => [file.discipline, file.filename]),
    [
      ['swim', 'swim.tcx'],
      ['bike', 'bike.tcx'],
      ['run', 'run.tcx'],
    ],
  )
  const swim = await parsedActivity(join(root, 'out', 'swim.tcx'))
  const bike = await parsedActivity(join(root, 'out', 'bike.tcx'))
  const run = await parsedActivity(join(root, 'out', 'run.tcx'))

  assert.equal(swim.activity['@_Sport'], 'Swimming')
  assert.equal(bike.activity['@_Sport'], 'Biking')
  assert.equal(run.activity['@_Sport'], 'Running')
  assert.equal(swim.xml.includes('2026-07-26T08:01:15'), false)
  assert.equal(bike.xml.includes('2026-07-26T08:01:15'), false)
  assert.equal(bike.xml.includes('2026-07-26T08:02:45'), false)
  assert.equal(run.xml.includes('2026-07-26T08:02:45'), false)

  const swimHeartRate = recordValue(swim.trackPoints[0] ?? {}, 'HeartRateBpm')
  assert.equal(swimHeartRate.Value, 111)
  assert.equal(bike.trackPoints[0]?.Cadence, 92)
  const bikeTpx = recordValue(recordValue(bike.trackPoints[0] ?? {}, 'Extensions'), 'ns3:TPX')
  assert.equal(bikeTpx['ns3:Speed'], 9.1)
  assert.equal(bikeTpx['ns3:Watts'], 220)
  assert.equal(bikeTpx['ns3:RunCadence'], undefined)
  assert.equal(run.trackPoints[0]?.Cadence, undefined)
  const runTpx = recordValue(recordValue(run.trackPoints[0] ?? {}, 'Extensions'), 'ns3:TPX')
  assert.equal(runTpx['ns3:RunCadence'], 86)
  assert.equal(runTpx['ns3:Speed'], 3.4)
  assert.equal(runTpx['ns3:Watts'], 258)
  assert.equal(bike.trackPoints.at(-1)?.DistanceMeters, 24_300)
  assert.equal(run.trackPoints.at(-1)?.DistanceMeters, 9_500)
  assert.match(String(run.activity.Notes), new RegExp(workoutId))
  assert.match(String(run.activity.Notes), /aaaaaaaa-0000-0000-0000-000000000005/)
})

test('uses parent heart rate fallback and writes only selected swim and run files', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-fallback-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const inputPath = await writeFixture(root, { routeDisciplines: ['transition', 'bike', 'run'] })
  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
    sports: ['swim', 'run'],
  })

  assert.deepEqual(
    result.files.map(file => file.discipline),
    ['swim', 'run'],
  )
  assert.equal(result.files[0]?.usedHeartRateFallback, true)
  await assert.rejects(fs.access(join(root, 'out', 'bike.tcx')))
  const swim = await parsedActivity(join(root, 'out', 'swim.tcx'))
  assert.equal(swim.trackPoints.length, 2)
  assert.equal(swim.trackPoints[0]?.Position, undefined)
  assert.equal(recordValue(swim.trackPoints[0] ?? {}, 'HeartRateBpm').Value, 111)
  assert.equal(recordValue(swim.trackPoints[1] ?? {}, 'HeartRateBpm').Value, 112)
  assert.equal(swim.trackPoints[0]?.DistanceMeters, 0)
  assert.equal(swim.trackPoints[1]?.DistanceMeters, 400)

  const manifestRaw: unknown = JSON.parse(await fs.readFile(result.manifestPath, 'utf8'))
  const manifest = requiredRecord(manifestRaw, 'manifest')
  assert.deepEqual(manifest.selectedSports, ['swim', 'run'])
  const transitions = arrayValue(manifest, 'transitions').map((value, index) =>
    requiredRecord(value, `transition ${index}`),
  )
  assert.deepEqual(
    transitions.map(transition => transition.id),
    ['aaaaaaaa-0000-0000-0000-000000000002', 'aaaaaaaa-0000-0000-0000-000000000004'],
  )
})

test('merges every distinct parent HR timestamp into a route-rich leg', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-hr-merge-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const inputPath = await writeFixture(root, {
    extraHeartRate: [{ time: '2026-07-26T08:00:30Z', bpm: 119 }],
  })
  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
    sports: ['swim'],
  })

  assert.equal(result.files[0]?.routePointCount, 2)
  assert.equal(result.files[0]?.usedHeartRateFallback, false)
  const swim = await parsedActivity(join(root, 'out', 'swim.tcx'))
  assert.deepEqual(
    swim.trackPoints.map(point => point.Time),
    ['2026-07-26T08:00:10.000Z', '2026-07-26T08:00:30.000Z', '2026-07-26T08:00:50.000Z'],
  )
  assert.equal(swim.trackPoints.filter(point => point.Position != null).length, 2)
  assert.equal(recordValue(swim.trackPoints[0] ?? {}, 'HeartRateBpm').Value, 111)
  const mergedHeartRate = swim.trackPoints[1]
  if (!mergedHeartRate) assert.fail('merged HR trackpoint is missing')
  assert.equal(mergedHeartRate.Position, undefined)
  assert.equal(recordValue(mergedHeartRate, 'HeartRateBpm').Value, 119)
  assert.equal(recordValue(swim.trackPoints[2] ?? {}, 'HeartRateBpm').Value, 112)
  assert.deepEqual(
    swim.trackPoints.map(point => point.DistanceMeters),
    [0, 0, 400],
  )
})

test('disambiguates duplicate legacy HR timestamps without losing either bpm', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-duplicate-hr-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const inputPath = await writeFixture(root, {
    heartRate: [
      { time: '2026-07-26T08:00:20Z', bpm: 110 },
      { time: '2026-07-26T08:00:20Z', bpm: 111 },
      { time: '2026-07-26T08:00:50Z', bpm: 112 },
    ],
  })
  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
    sports: ['swim'],
  })

  const swim = await parsedActivity(join(root, 'out', 'swim.tcx'))
  const times = swim.trackPoints.map(point => String(point.Time))
  assert.equal(
    times.every(
      (time, index) => index === 0 || Date.parse(time) > Date.parse(times[index - 1] ?? ''),
    ),
    true,
  )
  assert.equal(times.includes('2026-07-26T08:00:20.000Z'), true)
  assert.equal(times.includes('2026-07-26T08:00:20.001Z'), true)
  const heartRates = swim.trackPoints.flatMap(point =>
    isRecord(point.HeartRateBpm) ? [point.HeartRateBpm.Value] : [],
  )
  assert.deepEqual(heartRates, [110, 111, 112])

  const manifestRaw: unknown = JSON.parse(await fs.readFile(result.manifestPath, 'utf8'))
  const repairs = arrayValue(
    requiredRecord(manifestRaw, 'manifest'),
    'heartRateTimestampRepairs',
  ).map((value, index) => requiredRecord(value, `HR repair ${index}`))
  assert.deepEqual(repairs, [
    {
      sourceTimestamp: '2026-07-26T08:00:20.000Z',
      outputTimestamp: '2026-07-26T08:00:20.001Z',
      bpm: 111,
      reason: 'duplicate legacy parent heart-rate timestamp',
      provenance: 'converter:stable-millisecond-disambiguation',
    },
  ])
})

test('does not repair already-distinct fractional HR timestamps', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-fractional-hr-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const inputPath = await writeFixture(root, {
    heartRate: [
      { time: '2026-07-26T08:00:20.123Z', bpm: 110 },
      { time: '2026-07-26T08:00:20.124Z', bpm: 111 },
    ],
  })
  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
    sports: ['swim'],
  })

  const swim = await parsedActivity(join(root, 'out', 'swim.tcx'))
  assert.equal(
    swim.trackPoints.some(point => point.Time === '2026-07-26T08:00:20.123Z'),
    true,
  )
  assert.equal(
    swim.trackPoints.some(point => point.Time === '2026-07-26T08:00:20.124Z'),
    true,
  )
  const manifestRaw: unknown = JSON.parse(await fs.readFile(result.manifestPath, 'utf8'))
  assert.deepEqual(requiredRecord(manifestRaw, 'manifest').heartRateTimestampRepairs, [])
})

test('retains GPX heart rate when a route-rich leg has no parent HR samples', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-gpx-hr-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const inputPath = await writeFixture(root, { heartRate: [] })
  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
    sports: ['swim'],
  })

  assert.equal(result.files[0]?.routePointCount, 2)
  assert.equal(result.files[0]?.usedHeartRateFallback, false)
  const swim = await parsedActivity(join(root, 'out', 'swim.tcx'))
  assert.deepEqual(
    swim.trackPoints.map(point => recordValue(point, 'HeartRateBpm').Value),
    [120, 126],
  )
})

test('uses half-open swim and bike windows while retaining the final run endpoint', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-boundary-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const inputPath = await writeFixture(root, {
    boundaryRoutePoints: true,
    extraHeartRate: [
      { time: '2026-07-26T08:01:00Z', bpm: 131 },
      { time: '2026-07-26T08:02:30Z', bpm: 157 },
      { time: '2026-07-26T08:04:00Z', bpm: 175 },
    ],
  })
  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
  })

  assert.deepEqual(
    result.files.map(file => file.routePointCount),
    [2, 2, 3],
  )
  const swim = await parsedActivity(join(root, 'out', 'swim.tcx'))
  const bike = await parsedActivity(join(root, 'out', 'bike.tcx'))
  const run = await parsedActivity(join(root, 'out', 'run.tcx'))
  assert.equal(
    swim.trackPoints.some(point => point.Time === '2026-07-26T08:01:00.000Z'),
    false,
  )
  assert.equal(
    bike.trackPoints.some(point => point.Time === '2026-07-26T08:02:30.000Z'),
    false,
  )
  const runEnd = run.trackPoints.find(point => point.Time === '2026-07-26T08:04:00.000Z')
  if (!runEnd) assert.fail('final run endpoint is missing')
  assert.ok(runEnd.Position)
  assert.equal(recordValue(runEnd, 'HeartRateBpm').Value, 175)
  assert.equal(runEnd.DistanceMeters, 9_500)
})

test('overrides generated swim distance without mutating source distance provenance', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-swim-distance-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const activities = fixtureActivities()
  const swimActivity = activities[0]
  if (!swimActivity) assert.fail('fixture swim is missing')
  activities[0] = { ...swimActivity, distanceM: 438.896_423_339_843_75 }
  const inputPath = await writeFixture(root, { activities })
  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
    sports: ['swim', 'run'],
    swimDistanceM: 1_500,
  })

  assert.deepEqual(
    result.files.map(file => file.filename),
    ['swim.tcx', 'run.tcx'],
  )
  await assert.rejects(fs.access(join(root, 'out', 'bike.tcx')))
  const swim = await parsedActivity(join(root, 'out', 'swim.tcx'))
  assert.equal(swim.lap.DistanceMeters, 1_500)
  assert.equal(swim.trackPoints.at(-1)?.DistanceMeters, 1_500)
  assert.equal(swim.trackPoints[0]?.Time, '2026-07-26T08:00:10.000Z')
  assert.equal(swim.trackPoints[0]?.AltitudeMeters, 76.5)
  assert.ok(swim.trackPoints[0]?.Position)
  assert.equal(recordValue(swim.trackPoints[0] ?? {}, 'HeartRateBpm').Value, 111)

  const manifestRaw: unknown = JSON.parse(await fs.readFile(result.manifestPath, 'utf8'))
  const manifest = requiredRecord(manifestRaw, 'manifest')
  const sourceActivities = arrayValue(manifest, 'activities').map((value, index) =>
    requiredRecord(value, `activity ${index}`),
  )
  const sourceSwim = sourceActivities.find(activity => activity.discipline === 'swim')
  if (!sourceSwim) assert.fail('source swim manifest entry is missing')
  assert.equal(sourceSwim.distanceM, 438.896_423_339_843_75)
  const overrides = arrayValue(manifest, 'distanceOverrides').map((value, index) =>
    requiredRecord(value, `distance override ${index}`),
  )
  assert.deepEqual(overrides, [
    {
      discipline: 'swim',
      sourceActivityId: 'aaaaaaaa-0000-0000-0000-000000000001',
      sourceDistanceM: 438.896_423_339_843_75,
      outputDistanceM: 1_500,
      provenance: 'cli:--swim-distance-m',
    },
  ])
})

test('retimes swim and T1 onto supplied GPX and TCX courses', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-course-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const inputPath = await writeFixture(root, {
    boundaryRoutePoints: true,
    extraHeartRate: [{ time: '2026-07-26T08:01:20Z', bpm: 140 }],
  })
  const swimRoutePath = join(root, 'swim-course.gpx')
  const transition1RoutePath = join(root, 'transition-1-course.tcx')
  await Promise.all([
    fs.writeFile(swimRoutePath, fixtureCourseGpx()),
    fs.writeFile(transition1RoutePath, fixtureCourseTcx()),
  ])

  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
    sports: ['swim'],
    swimDistanceM: 1_500,
    swimDurationS: 50,
    swimElapsedTimeS: 60,
    swimRoutePath,
    transition1DurationS: 40,
    transition1RoutePath,
    includeTransitions: true,
  })

  const swim = await parsedActivity(join(root, 'out', 'swim.tcx'))
  assert.equal(swim.lap.TotalTimeSeconds, 50)
  assert.equal(swim.lap.DistanceMeters, 1_500)
  assert.equal(swim.trackPoints[0]?.Time, '2026-07-26T08:00:00.000Z')
  assert.equal(swim.trackPoints.at(-1)?.Time, '2026-07-26T08:01:00.000Z')
  assert.equal(swim.trackPoints.at(-1)?.DistanceMeters, 1_500)
  assert.equal(
    swim.trackPoints.every(point => point.Position != null),
    true,
  )
  assert.equal(
    swim.trackPoints.every(point => point.AltitudeMeters != null),
    true,
  )
  const swimMovingEnd = swim.trackPoints.find(point => point.Time === '2026-07-26T08:00:50.000Z')
  if (!swimMovingEnd) assert.fail('retimed swim moving endpoint is missing')
  assert.equal(recordValue(swimMovingEnd, 'HeartRateBpm').Value, 112)

  const transition = await parsedActivity(join(root, 'out', 'transition-1.tcx'))
  assert.equal(transition.lap.TotalTimeSeconds, 40)
  assert.equal(transition.trackPoints[0]?.Time, '2026-07-26T08:01:00.000Z')
  assert.equal(transition.trackPoints.at(-1)?.Time, '2026-07-26T08:01:40.000Z')
  assert.equal(
    transition.trackPoints.every(point => point.Position != null),
    true,
  )
  assert.equal(
    transition.trackPoints.every(point => point.AltitudeMeters != null),
    true,
  )
  assert.ok(Number(transition.lap.DistanceMeters) > 0)
  assert.equal(
    recordValue(
      transition.trackPoints.find(point => point.Time === '2026-07-26T08:01:20.000Z') ?? {},
      'HeartRateBpm',
    ).Value,
    140,
  )

  const manifestRaw: unknown = JSON.parse(await fs.readFile(result.manifestPath, 'utf8'))
  const manifest = requiredRecord(manifestRaw, 'manifest')
  const timingOverrides = arrayValue(manifest, 'timingOverrides').map((value, index) =>
    requiredRecord(value, `timing override ${index}`),
  )
  assert.deepEqual(
    timingOverrides.map(override => [override.discipline, override.outputDurationS]),
    [
      ['swim', 50],
      ['transition', 40],
    ],
  )
  const routeOverrides = arrayValue(manifest, 'routeOverrides').map((value, index) =>
    requiredRecord(value, `route override ${index}`),
  )
  assert.deepEqual(
    routeOverrides.map(override => [override.discipline, override.sourcePointCount]),
    [
      ['swim', 3],
      ['transition', 2],
    ],
  )
})

test('exports explicit transitions as half-open Other activities with raw route distance', async t => {
  const root = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-transitions-'))
  t.after(() => fs.rm(root, { recursive: true, force: true }))
  const activities = fixtureActivities().map(activity =>
    activity.activity === 'transition'
      ? {
          ...activity,
          activeEnergyKcal: 0,
          averageHeartRateBpm: 0,
          averagePowerW: 0,
          averageCadencePerMinute: 0,
          lapCount: 0,
        }
      : activity,
  )
  const inputPath = await writeFixture(root, {
    activities,
    boundaryRoutePoints: true,
    extraHeartRate: [
      { time: '2026-07-26T08:01:00Z', bpm: 132 },
      { time: '2026-07-26T08:01:05Z', bpm: 133 },
      { time: '2026-07-26T08:01:30Z', bpm: 140 },
      { time: '2026-07-26T08:02:30Z', bpm: 160 },
      { time: '2026-07-26T08:02:35Z', bpm: 161 },
      { time: '2026-07-26T08:03:00Z', bpm: 165 },
    ],
  })
  const result = await exportAppleMultisportTcx({
    inputPath,
    workoutId,
    outputDir: join(root, 'out'),
    sports: ['swim', 'run'],
    includeTransitions: true,
  })

  assert.deepEqual(
    result.files.map(file => file.filename),
    ['swim.tcx', 'run.tcx', 'transition-1.tcx', 'transition-2.tcx'],
  )
  await assert.rejects(fs.access(join(root, 'out', 'bike.tcx')))
  const first = await parsedActivity(join(root, 'out', 'transition-1.tcx'))
  const second = await parsedActivity(join(root, 'out', 'transition-2.tcx'))
  assert.equal(first.activity['@_Sport'], 'Other')
  assert.equal(second.activity['@_Sport'], 'Other')
  assert.equal(first.lap.TotalTimeSeconds, 30)
  assert.equal(first.lap.Calories, 0)
  assert.equal(first.lap.AverageHeartRateBpm, undefined)
  assert.equal(
    first.trackPoints.some(point => point.Time === '2026-07-26T08:01:30.000Z'),
    false,
  )
  assert.equal(
    second.trackPoints.some(point => point.Time === '2026-07-26T08:03:00.000Z'),
    false,
  )
  assert.equal(first.trackPoints.filter(point => point.Position != null).length, 2)
  assert.equal(second.trackPoints.filter(point => point.Position != null).length, 2)
  const firstExactHeartRate = first.trackPoints.find(
    point => point.Time === '2026-07-26T08:01:05.000Z',
  )
  if (!firstExactHeartRate) assert.fail('first transition exact HR sample is missing')
  assert.equal(firstExactHeartRate.Position, undefined)
  assert.equal(recordValue(firstExactHeartRate, 'HeartRateBpm').Value, 133)
  const firstDistance = first.lap.DistanceMeters
  const secondDistance = second.lap.DistanceMeters
  assert.equal(typeof firstDistance, 'number')
  assert.equal(typeof secondDistance, 'number')
  assert.ok(Number(firstDistance) > 0)
  assert.ok(Number(secondDistance) > 0)
  assert.equal(first.trackPoints.at(-1)?.DistanceMeters, firstDistance)
  assert.equal(second.trackPoints.at(-1)?.DistanceMeters, secondDistance)
  assert.match(String(first.activity.Notes), /aaaaaaaa-0000-0000-0000-000000000002/)
  assert.match(String(second.activity.Notes), /aaaaaaaa-0000-0000-0000-000000000004/)
})

test('accepts zero-valued v9 transition summaries while sport averages stay positive', async t => {
  const transitionRoot = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-v9-'))
  const sportRoot = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-sport-zero-'))
  t.after(() =>
    Promise.all([
      fs.rm(transitionRoot, { recursive: true, force: true }),
      fs.rm(sportRoot, { recursive: true, force: true }),
    ]),
  )
  const activities = fixtureActivities().map(activity =>
    activity.activity === 'transition'
      ? {
          ...activity,
          activeEnergyKcal: 0,
          averageHeartRateBpm: 0,
          averagePowerW: 0,
          averageCadencePerMinute: 0,
          lapCount: 0,
        }
      : activity,
  )
  const transitionInput = await writeFixture(transitionRoot, { activities })
  const result = await exportAppleMultisportTcx({
    inputPath: transitionInput,
    workoutId,
    outputDir: join(transitionRoot, 'out'),
    sports: ['swim', 'run'],
  })
  const manifestRaw: unknown = JSON.parse(await fs.readFile(result.manifestPath, 'utf8'))
  const transitions = arrayValue(requiredRecord(manifestRaw, 'manifest'), 'transitions').map(
    (value, index) => requiredRecord(value, `transition ${index}`),
  )
  assert.equal(transitions.length, 2)
  for (const transition of transitions) {
    assert.equal(transition.activeEnergyKcal, 0)
    assert.equal(transition.averageHeartRateBpm, 0)
    assert.equal(transition.averagePowerW, 0)
    assert.equal(transition.averageCadencePerMinute, 0)
    assert.equal(transition.lapCount, 0)
  }

  const invalidSport = fixtureActivities()
  const swim = invalidSport[0]
  if (!swim) assert.fail('fixture swim is missing')
  invalidSport[0] = { ...swim, averageHeartRateBpm: 0 }
  const sportInput = await writeFixture(sportRoot, { activities: invalidSport })
  await assert.rejects(
    exportAppleMultisportTcx({
      inputPath: sportInput,
      workoutId,
      outputDir: join(sportRoot, 'out'),
      sports: ['swim', 'run'],
    }),
    /averageHeartRateBpm must be positive/,
  )
})

test('rejects overlapping segments and a missing run leg', async t => {
  const overlapRoot = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-overlap-'))
  const missingRoot = await fs.mkdtemp(join(tmpdir(), 'apple-multisport-tcx-missing-'))
  t.after(() =>
    Promise.all([
      fs.rm(overlapRoot, { recursive: true, force: true }),
      fs.rm(missingRoot, { recursive: true, force: true }),
    ]),
  )
  const overlapping = fixtureActivities()
  const firstTransition = overlapping[1]
  if (!firstTransition) assert.fail('fixture transition is missing')
  overlapping[1] = { ...firstTransition, end: '2026-07-26T08:01:40Z' }
  const overlapInput = await writeFixture(overlapRoot, { activities: overlapping })
  await assert.rejects(
    exportAppleMultisportTcx({
      inputPath: overlapInput,
      workoutId,
      outputDir: join(overlapRoot, 'out'),
    }),
    /overlap/,
  )

  const missingRun = fixtureActivities().slice(0, 4)
  const missingInput = await writeFixture(missingRoot, { activities: missingRun })
  await assert.rejects(
    exportAppleMultisportTcx({
      inputPath: missingInput,
      workoutId,
      outputDir: join(missingRoot, 'out'),
    }),
    /exactly one swim, bike, and run/,
  )
})

test('parses a swim and run CLI selection', () => {
  assert.deepEqual(
    parseAppleMultisportTcxArgs([
      '--input',
      'apple.json',
      '--id',
      workoutId,
      '--output',
      'tcx',
      '--sports',
      'swim,run',
      '--swim-distance-m',
      '1500',
      '--swim-duration-s',
      '2468',
      '--swim-elapsed-s',
      '2476',
      '--swim-route',
      'swim.gpx',
      '--transition-1-duration-s',
      '522',
      '--transition-1-route',
      'transition-1.tcx',
      '--include-transitions',
    ]),
    {
      inputPath: 'apple.json',
      workoutId,
      outputDir: 'tcx',
      sports: ['swim', 'run'],
      swimDistanceM: 1_500,
      swimDurationS: 2_468,
      swimElapsedTimeS: 2_476,
      swimRoutePath: 'swim.gpx',
      transition1DurationS: 522,
      transition1RoutePath: 'transition-1.tcx',
      includeTransitions: true,
    },
  )
})
