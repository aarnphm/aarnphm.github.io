import { createHash } from 'node:crypto'
import { z } from 'zod'

const activityId = z.string().regex(/^[1-9]\d*$/)
const timestamp = z.iso.datetime()
const nullableNumber = z.number().nullable()
const chartPair = z.tuple([
  z.number().nonnegative(),
  z.union([nullableNumber, z.tuple([nullableNumber, nullableNumber])]),
])
const rawValue = z.union([nullableNumber, z.tuple([z.string(), nullableNumber])])

const chart = z.strictObject({
  type: z.string(),
  title: z.string().optional(),
  series: z.array(
    z.strictObject({
      label: z.string().nullable(),
      values: z.array(chartPair),
      scale: z.string().nullable(),
    }),
  ),
})

const nonFiniteValue = z.strictObject({
  path: z.string().startsWith('/runtime/'),
  kind: z.enum(['NaN', 'Infinity', '-Infinity']),
  start: z.number().int().nonnegative().nullable(),
  end: z.number().int().nonnegative().nullable(),
  component: z.literal(1).optional(),
})

const browserCapture = z.strictObject({
  adapterVersion: z.literal('mywindsock-classic-browser-v1'),
  transport: z.literal('browser-runtime'),
  valueKind: z.literal('provider-analysis'),
  conditionKind: z.enum(['historical', 'forecast', 'unknown']),
  capturedAt: timestamp,
  pageUrl: z.url(),
  stravaId: activityId,
  viewOnStravaId: activityId,
  providerCourseId: activityId,
  providerStravaId: z.number().int().nonnegative(),
  rideTimestamp: z.number().int().nonnegative(),
  pageType: z.literal('activity'),
  pageMode: z.literal('analyst'),
  runtime: z.strictObject({
    averages: z.record(z.string(), nullableNumber),
    series: z.record(z.string(), z.array(rawValue)),
    course: z.array(z.tuple([z.number(), z.number()])),
    distancesM: z.array(z.number().nonnegative()),
    elevationM: z.array(z.number()),
  }),
  weather: z.array(
    z.strictObject({
      latitude: z.number(),
      longitude: z.number(),
      timezone: z.string(),
      offset: z.number(),
      side_of_road: z.string(),
      hourly: z.json(),
      daily: z.json(),
      currently: z.json(),
    }),
  ),
  modelProfileCandidates: z.array(
    z.strictObject({
      typeID: z.union([z.string(), z.number()]),
      weight: z.union([z.string(), nullableNumber]),
      rr: z.union([z.string(), nullableNumber]),
      dtl: z.union([z.string(), nullableNumber]),
      cda: z.union([z.string(), nullableNumber]),
      cda_up: z.union([z.string(), nullableNumber]),
      cda_down: z.union([z.string(), nullableNumber]),
      watts: z.union([z.string(), nullableNumber]),
      watts_up: z.union([z.string(), nullableNumber]),
      watts_down: z.union([z.string(), nullableNumber]),
      height: z.union([z.string(), nullableNumber]),
      ftp: z.union([z.string(), nullableNumber]),
      critical_power: z.union([z.string(), nullableNumber]),
    }),
  ),
  charts: z.strictObject({
    cda: z.array(chart),
    gradient: z.array(chart),
    feelsLike: z.array(chart).optional(),
  }),
  observations: z
    .strictObject({ summaryText: z.string(), weatherText: z.string(), navigationText: z.string() })
    .optional(),
  excludedFields: z.array(z.string()),
  nonFiniteValues: z.array(nonFiniteValue),
})

const descriptionReport = z.strictObject({
  source: z.literal('provider-native'),
  transport: z.literal('strava-description'),
  schemaVersion: z.literal(1),
  activityId: z.number().int().positive(),
  retrievedAt: z.number().nonnegative(),
  provider: z.literal('mywindsock'),
  weatherImpactPct: nullableNumber,
  cdaM2: nullableNumber,
  feelsLikeElevationM: nullableNumber,
  headwindPct: nullableNumber,
  headwindMinKph: nullableNumber,
  headwindMaxKph: nullableNumber,
  longestHeadwindS: nullableNumber,
  airSpeedKph: nullableNumber,
  averageTemperatureC: nullableNumber,
  precipitationProbabilityPct: nullableNumber,
  precipitationRateMmPerHour: nullableNumber,
})

export const myWindsockGraphCaptureSchema = z.strictObject({
  adapterVersion: z.literal('mywindsock-graphs-browser-v1'),
  capturedAt: timestamp,
  pageUrl: z.url(),
  stravaId: activityId,
  viewOnStravaId: activityId,
  providerCourseId: activityId,
  rideTimestamp: z.number().int().nonnegative(),
  menu: z.array(z.strictObject({ key: z.string().min(1), label: z.string().min(1) })).min(1),
  graphs: z.record(
    z.string(),
    z.strictObject({
      label: z.string(),
      source: z.enum(['menu', 'inline']),
      state: z.enum(['captured', 'empty', 'unavailable', 'failed']),
      capturedAt: timestamp,
      configuration: z.json().nullable(),
      nonFiniteValues: z.array(
        z.strictObject({
          path: z.string().startsWith('/'),
          kind: z.enum(['NaN', 'Infinity', '-Infinity']),
        }),
      ),
      note: z.string().nullable(),
    }),
  ),
})
export type MyWindsockGraphCapture = z.infer<typeof myWindsockGraphCaptureSchema>
export const myWindsockGraphCaptureSha256 = (capture: MyWindsockGraphCapture) =>
  createHash('sha256').update(JSON.stringify(capture)).digest('hex')

export const myWindsockArchiveSchema = z.strictObject({
  schemaVersion: z.literal(1),
  provider: z.literal('mywindsock'),
  activity: z.strictObject({
    stravaId: activityId,
    sport: z.enum(['run', 'bike']),
    stravaSportType: z.string(),
    startedAt: timestamp,
    localDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
    stravaBaseline: z.strictObject({
      distanceM: z.number().nonnegative(),
      movingTimeS: z.number().nonnegative(),
      elapsedTimeS: z.number().nonnegative(),
      elevationGainM: nullableNumber,
    }),
  }),
  entrypoints: z.strictObject({
    stravaUrl: z.url(),
    myWindsockUrl: z.url(),
    providerCourseId: activityId.nullable(),
  }),
  ingestion: z.strictObject({
    state: z.enum(['summary-only', 'captured', 'ready']),
    pending: z.array(z.string()),
    sections: z.record(z.string(), z.enum(['captured', 'not-captured', 'unavailable'])),
    attempts: z.array(
      z.strictObject({
        at: timestamp,
        status: z.enum(['success', 'partial', 'unavailable', 'failed']),
        note: z.string(),
      }),
    ),
    events: z.array(
      z.strictObject({
        at: timestamp,
        type: z.enum([
          'description-import',
          'browser-import',
          'graph-import',
          'manual-edit',
          'verify',
        ]),
        note: z.string(),
        captureSha256: z
          .string()
          .regex(/^[a-f0-9]{64}$/)
          .nullable(),
      }),
    ),
    captureSha256: z
      .string()
      .regex(/^[a-f0-9]{64}$/)
      .nullable(),
    graphsSha256: z
      .string()
      .regex(/^[a-f0-9]{64}$/)
      .optional(),
  }),
  description: z.strictObject({ rawReport: z.string(), parsed: descriptionReport }).nullable(),
  browserCapture: browserCapture.nullable(),
  graphCapture: myWindsockGraphCaptureSchema.optional(),
  mappings: z.array(
    z.strictObject({
      path: z.string(),
      meaning: z.string(),
      unit: z.string().nullable(),
      multiplier: nullableNumber,
      status: z.enum(['verified', 'unresolved']),
      evidence: z.string(),
    }),
  ),
  normalized: z.json().nullable(),
  context: z.strictObject({
    notes: z.array(z.string()),
    equipment: z.string().nullable(),
    position: z.string().nullable(),
  }),
})

export type MyWindsockArchive = z.infer<typeof myWindsockArchiveSchema>

export const myWindsockCaptureSha256 = (
  capture: NonNullable<MyWindsockArchive['browserCapture']>,
) => createHash('sha256').update(JSON.stringify(capture)).digest('hex')

const privateFields = new Set([
  'auth',
  'cookie',
  'authorization',
  'password',
  'apikey',
  'accesstoken',
  'refreshtoken',
  'memberid',
  'coachid',
  'coachmemberid',
])

function containsPrivateField(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(containsPrivateField)
  if (value === null || typeof value !== 'object') return false
  return Object.entries(value).some(
    ([key, item]) =>
      privateFields.has(key.toLowerCase().replaceAll('_', '')) || containsPrivateField(item),
  )
}

export function validateMyWindsockArchive(value: unknown, filenameId: string): MyWindsockArchive {
  const record = myWindsockArchiveSchema.parse(value)
  const id = record.activity.stravaId
  const fail = (message: string): never => {
    throw new Error(`${filenameId}: ${message}`)
  }
  if (containsPrivateField(record)) fail('capture contains an authentication or account field')
  if (id !== filenameId) fail('filename and Strava ID disagree')
  if (record.entrypoints.stravaUrl !== `https://www.strava.com/activities/${id}`)
    fail('Strava entrypoint does not match the activity')
  if (record.entrypoints.myWindsockUrl !== `https://mywindsock.com/activity/${id}/`)
    fail('myWindsock entrypoint does not match the activity')
  if (record.description && String(record.description.parsed.activityId) !== id)
    fail('description identity does not match the activity')
  if (!record.description && !record.browserCapture) fail('no myWindsock evidence')
  if (record.ingestion.state === 'ready') fail('ready projection is not implemented in schema v1')
  if (record.ingestion.state === 'captured' && !record.browserCapture)
    fail('captured stage requires a browser capture')
  const graphCapture = record.graphCapture
  if (graphCapture) {
    if (
      graphCapture.stravaId !== id ||
      graphCapture.viewOnStravaId !== id ||
      graphCapture.pageUrl !== record.entrypoints.myWindsockUrl ||
      graphCapture.providerCourseId !== record.entrypoints.providerCourseId ||
      graphCapture.rideTimestamp * 1000 !== Date.parse(record.activity.startedAt)
    )
      fail('graph capture identity does not match the activity')
    if (record.ingestion.graphsSha256 !== myWindsockGraphCaptureSha256(graphCapture))
      fail('graph capture checksum changed')
    const keys = graphCapture.menu.map(item => item.key)
    if (
      new Set(keys).size !== keys.length ||
      keys.some(key => !graphCapture.graphs[key] || graphCapture.graphs[key].source !== 'menu')
    )
      fail('graph inventory is incomplete or duplicated')
    for (const graph of Object.values(graphCapture.graphs)) {
      if ((graph.state === 'captured' || graph.state === 'empty') && graph.configuration === null)
        fail('captured graph has no configuration')
      for (const evidence of graph.nonFiniteValues) {
        let value: unknown = graph.configuration
        for (const segment of evidence.path
          .slice(1)
          .split('/')
          .map(s => s.replaceAll('~1', '/').replaceAll('~0', '~'))) {
          if (Array.isArray(value)) value = value[Number(segment)]
          else if (value !== null && typeof value === 'object' && segment in value)
            value = Object.entries(value).find(([key]) => key === segment)?.[1]
          else fail('graph non-finite evidence has an invalid path')
        }
        if (value !== null) fail('graph non-finite evidence does not point to a null')
      }
    }
    if (
      record.ingestion.sections.allGraphs === 'captured' &&
      Object.values(graphCapture.graphs).some(graph => graph.state === 'failed')
    )
      fail('failed graph marked as complete')
  } else if (record.ingestion.graphsSha256 !== undefined)
    fail('graph checksum without a graph capture')
  const capture = record.browserCapture
  if (!capture) {
    if (record.ingestion.captureSha256 !== null) fail('checksum without a capture')
    return record
  }
  if (record.ingestion.state === 'summary-only') fail('browser capture requires captured stage')
  if (
    capture.stravaId !== id ||
    capture.viewOnStravaId !== id ||
    capture.pageUrl !== record.entrypoints.myWindsockUrl ||
    capture.providerCourseId !== record.entrypoints.providerCourseId
  )
    fail('browser identity does not match the activity')
  if (capture.rideTimestamp * 1_000 !== Date.parse(record.activity.startedAt))
    fail('analysis start does not match the recording start')
  if (record.ingestion.captureSha256 !== myWindsockCaptureSha256(capture))
    fail('capture checksum changed')
  for (const groups of Object.values(capture.charts)) {
    for (const group of groups) {
      for (const series of group.series) {
        if (series.values.some(([x], i, pairs) => i > 0 && x < pairs[i - 1][0]))
          fail('chart coordinates are not monotonic')
      }
    }
  }
  for (const evidence of capture.nonFiniteValues) {
    if (evidence.start === null && evidence.end === null && evidence.component === undefined) {
      const key = evidence.path.replace('/runtime/averages/', '')
      if (!evidence.path.startsWith('/runtime/averages/') || capture.runtime.averages[key] !== null)
        fail('non-finite aggregate evidence does not point to a null')
    } else {
      const key = evidence.path.replace('/runtime/series/', '')
      const series = capture.runtime.series[key]
      if (
        evidence.start === null ||
        evidence.end === null ||
        evidence.end < evidence.start ||
        !evidence.path.startsWith('/runtime/series/') ||
        !series ||
        evidence.end >= series.length ||
        series
          .slice(evidence.start, evidence.end + 1)
          .some(v =>
            evidence.component === undefined
              ? v !== null
              : !Array.isArray(v) || v[evidence.component] !== null,
          )
      )
        fail('non-finite series evidence does not point to a null range')
    }
  }
  return record
}
