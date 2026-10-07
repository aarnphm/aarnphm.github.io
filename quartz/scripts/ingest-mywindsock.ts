import { randomUUID } from 'node:crypto'
import { mkdir, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { z } from 'zod'
import { normalizeSport, type RawStravaActivity } from '../plugins/stores/strava'
import { parseMyWindsockReport } from '../util/activity-provider-reports'
import {
  myWindsockArchiveSchema,
  myWindsockCaptureSha256,
  myWindsockGraphCaptureSchema,
  myWindsockGraphCaptureSha256,
  validateMyWindsockArchive,
  type MyWindsockArchive,
} from '../util/mywindsock-archive'
import { verifyMyWindsockRouteMappings } from '../util/mywindsock-route'
import { readStravaCacheFile } from '../util/strava-cache-file'

const cutoff = '2026-10-03'
const today = new Intl.DateTimeFormat('en-CA', {
  timeZone: 'America/Toronto',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
}).format(new Date())
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    since: { type: 'string', default: cutoff },
    until: { type: 'string', default: today },
    cache: { type: 'string', default: 'quartz/.quartz-cache/strava.json' },
    directory: { type: 'string', default: 'content/triathlon/wind' },
    state: { type: 'string', default: 'quartz/.quartz-cache/mywindsock-ingestion.json' },
    replace: { type: 'boolean', default: false },
  },
})
const command = z
  .enum(['queue', 'seed', 'import', 'graphs', 'verify', 'attempt'])
  .parse(positionals[0])
const since = date(values.since ?? cutoff)
const until = date(values.until ?? today)
if (since < cutoff || since > until)
  throw new Error(`Use a window on or after ${cutoff}, with since <= until`)
const directory = values.directory ?? 'content/triathlon/wind'
const statePath = values.state ?? 'quartz/.quartz-cache/mywindsock-ingestion.json'
const cache = await readStravaCacheFile(values.cache ?? 'quartz/.quartz-cache/strava.json')
if (!cache) throw new Error('Strava cache missing; refresh it before ingestion')
const attemptSchema = z.strictObject({
  at: z.iso.datetime(),
  localDay: z.string(),
  status: z.enum(['failed', 'unavailable']),
  note: z.string(),
})
const ledgerSchema = z.record(z.string().regex(/^[1-9]\d*$/), z.array(attemptSchema))
type Archive = MyWindsockArchive

function date(value: string): string {
  if (
    !/^\d{4}-\d{2}-\d{2}$/.test(value) ||
    !Number.isFinite(Date.parse(`${value}T00:00:00Z`)) ||
    new Date(`${value}T00:00:00Z`).toISOString().slice(0, 10) !== value
  )
    throw new Error(`Invalid date: ${value}`)
  return value
}
async function source(filename: string): Promise<string | null> {
  try {
    return await readFile(filename, 'utf8')
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return null
    throw error
  }
}
async function atomicWrite(filename: string, body: string, previous: string | null): Promise<void> {
  await mkdir(path.dirname(filename), { recursive: true })
  const temporary = `${filename}.tmp-${randomUUID()}`
  try {
    await writeFile(temporary, body, { flag: 'wx' })
    if ((await source(filename)) !== previous)
      throw new Error(`Concurrent edit to ${filename}; preserving that edit`)
    await rename(temporary, filename)
  } finally {
    await rm(temporary, { force: true })
  }
}
function outdoorSport(activity: RawStravaActivity): 'run' | 'bike' | null {
  const sport = normalizeSport(activity.sportType)
  return !activity.trainer &&
    !activity.sportType.startsWith('Virtual') &&
    (sport === 'run' || sport === 'bike')
    ? sport
    : null
}
const activities = Object.values(cache.activities)
  .filter(
    a =>
      outdoorSport(a) !== null &&
      a.startDateLocal.slice(0, 10) >= since &&
      a.startDateLocal.slice(0, 10) <= until,
  )
  .sort((a, b) => b.startDate.localeCompare(a.startDate))
function activityFor(id: string): RawStravaActivity {
  z.string()
    .regex(/^[1-9]\d*$/)
    .parse(id)
  const activity = activities.find(a => String(a.id) === id)
  if (!activity) throw new Error(`${id}: activity is outside the outdoor run/bike window`)
  return activity
}
async function readArchive(id: string) {
  const filename = path.join(directory, `${id}.json`)
  const previous = await source(filename)
  return {
    filename,
    previous,
    record: previous === null ? null : validateMyWindsockArchive(JSON.parse(previous), id),
  }
}
function description(activity: RawStravaActivity): Archive['description'] {
  const detail = cache?.activityDetails?.[String(activity.id)]
  if (!detail?.description || detail.fetchedAt === undefined) return null
  const blocks = [
    ...detail.description.matchAll(/--\s*myWindsock Report\s*--[\s\S]*?--\s*END\s*--/gi),
  ]
  for (const match of blocks.toReversed()) {
    const parsed = parseMyWindsockReport(match[0], activity.id, detail.fetchedAt)
    if (parsed) return { rawReport: match[0], parsed }
  }
  return null
}
function base(activity: RawStravaActivity): Archive {
  const sport = outdoorSport(activity)
  if (sport === null) throw new Error('Unsupported activity')
  const id = String(activity.id)
  return {
    schemaVersion: 1,
    provider: 'mywindsock',
    activity: {
      stravaId: id,
      sport,
      stravaSportType: activity.sportType,
      startedAt: activity.startDate,
      localDate: activity.startDateLocal.slice(0, 10),
      stravaBaseline: {
        distanceM: activity.distance,
        movingTimeS: activity.movingTime,
        elapsedTimeS: activity.elapsedTime,
        elevationGainM: activity.totalElevationGain ?? null,
      },
    },
    entrypoints: {
      stravaUrl: `https://www.strava.com/activities/${id}`,
      myWindsockUrl: `https://mywindsock.com/activity/${id}/`,
      providerCourseId: null,
    },
    ingestion: {
      state: 'summary-only',
      pending: [
        'Capture the available browser analysis.',
        'Verify provider units and model settings before graph normalization.',
      ],
      sections: {
        runtime: 'not-captured',
        weatherPayload: 'not-captured',
        gradientChart: 'not-captured',
        cdaChart: 'not-captured',
      },
      attempts: [],
      events: [],
      captureSha256: null,
    },
    description: null,
    browserCapture: null,
    mappings: [],
    normalized: null,
    context: { notes: [], equipment: null, position: null },
  }
}
function serialize(record: Archive): string {
  return (
    '{\n' +
    Object.entries(record)
      .map(
        ([key, value]) =>
          '  ' +
          JSON.stringify(key) +
          ': ' +
          (key === 'browserCapture' || key === 'graphCapture'
            ? JSON.stringify(value)
            : JSON.stringify(value, null, 2).replaceAll('\n', '\n  ')),
      )
      .join(',\n') +
    '\n}\n'
  )
}
async function save(record: Archive, previous: string | null): Promise<void> {
  const id = record.activity.stravaId
  validateMyWindsockArchive(record, id)
  await atomicWrite(path.join(directory, `${id}.json`), serialize(record), previous)
}
function addDescription(record: Archive, report: NonNullable<Archive['description']>): boolean {
  if (record.description?.rawReport === report.rawReport) return false
  if (record.description && report.parsed.retrievedAt < record.description.parsed.retrievedAt)
    return false
  record.description = report
  record.ingestion.events.push({
    at: new Date(report.parsed.retrievedAt).toISOString(),
    type: 'description-import',
    note: 'Delimited myWindsock report imported from the local Strava cache.',
    captureSha256: null,
  })
  return true
}

if (command === 'queue') {
  if (positionals.length !== 1) throw new Error('queue takes no positional arguments')
  const stateSource = await source(statePath)
  const ledger = ledgerSchema.parse(stateSource ? JSON.parse(stateSource) : {})
  const rows = []
  for (const activity of activities) {
    const id = String(activity.id)
    const { record } = await readArchive(id)
    const lastAttempt = ledger[id]?.at(-1) ?? null
    rows.push({
      id,
      name: activity.name,
      sport: outdoorSport(activity),
      localDate: activity.startDateLocal.slice(0, 10),
      reportInDescription: description(activity) !== null,
      state: record?.ingestion.state ?? 'missing',
      due:
        (!record?.browserCapture || record.ingestion.sections.allGraphs !== 'captured') &&
        lastAttempt?.localDay !== today,
      needs: !record?.browserCapture
        ? 'runtime-and-graphs'
        : record.ingestion.sections.allGraphs !== 'captured'
          ? 'graphs'
          : null,
      lastAttempt,
      url: `https://mywindsock.com/activity/${id}/`,
      pending: record?.ingestion.pending ?? [],
    })
  }
  console.log(
    JSON.stringify(
      {
        since,
        until,
        timezone: 'America/Toronto',
        total: rows.length,
        due: rows.filter(a => a.due).length,
        activities: rows,
      },
      null,
      2,
    ),
  )
} else {
  await mkdir(path.dirname(statePath), { recursive: true })
  const lock = `${statePath}.lock`
  try {
    await mkdir(lock)
  } catch (error) {
    throw new Error(
      `Ingestion lock exists at ${lock}; check the other run before removing a stale lock`,
      { cause: error },
    )
  }
  try {
    if (command === 'seed') {
      if (positionals.length !== 1) throw new Error('seed takes no positional arguments')
      let changed = 0
      for (const activity of activities) {
        const report = description(activity)
        if (!report) continue
        const { record: existing, previous } = await readArchive(String(activity.id))
        const record = existing ?? base(activity)
        if (!addDescription(record, report)) continue
        await save(record, previous)
        changed += 1
      }
      console.log(JSON.stringify({ changed, total: activities.length }))
    } else if (command === 'import') {
      if (positionals.length !== 3) throw new Error('import requires <strava_id> <capture.json>')
      const id = positionals[1]
      const activity = activityFor(id)
      const capture = myWindsockArchiveSchema.shape.browserCapture
        .unwrap()
        .parse(JSON.parse(await readFile(positionals[2], 'utf8')))
      if (!capture.runtime.series.time?.length || !capture.runtime.course.length)
        throw new Error(`${id}: missing analysis intervals or route`)
      const { record: existing, previous } = await readArchive(id)
      const record = existing ?? base(activity)
      if (record.activity.startedAt !== activity.startDate)
        throw new Error(`${id}: cached recording start changed; review identity`)
      const hash = myWindsockCaptureSha256(capture)
      if (record.ingestion.captureSha256 === hash) {
        console.log(JSON.stringify({ id, changed: false, captureSha256: hash }))
      } else {
        if (record.browserCapture && !values.replace)
          throw new Error(`${id}: a capture already exists; explicit --replace requires review`)
        const report = description(activity)
        if (report) addDescription(record, report)
        record.browserCapture = capture
        // Mappings were verified against the previous capture.
        record.mappings = []
        record.entrypoints.providerCourseId = capture.providerCourseId
        record.ingestion.captureSha256 = hash
        record.ingestion.state = 'captured'
        record.ingestion.pending = [
          'Verify provider units, direction conventions, selected model settings, and remaining sections before normalization.',
        ]
        record.ingestion.sections = {
          ...record.ingestion.sections,
          runtime: 'captured',
          weatherPayload: capture.weather.length ? 'captured' : 'unavailable',
          cdaChart: capture.charts.cda.length
            ? 'captured'
            : activity.sportType.includes('Run')
              ? 'unavailable'
              : 'not-captured',
          gradientChart: capture.charts.gradient.length ? 'captured' : 'not-captured',
          feelsLikeChart: capture.charts.feelsLike?.length ? 'captured' : 'not-captured',
          uiTotals: capture.observations?.summaryText ? 'captured' : 'not-captured',
          weatherUi: capture.observations?.weatherText ? 'captured' : 'not-captured',
          navigationLabels: capture.observations?.navigationText ? 'captured' : 'not-captured',
          selectedModelSettings: 'not-captured',
          extrema: 'not-captured',
        }
        record.ingestion.attempts.push({
          at: capture.capturedAt,
          status: 'partial',
          note: 'Available runtime, weather, and chart data captured. Semantic review remains pending.',
        })
        record.ingestion.events.push({
          at: capture.capturedAt,
          type: 'browser-import',
          note: 'Validated browser capture saved from the matching activity page.',
          captureSha256: hash,
        })
        await save(record, previous)
        console.log(
          JSON.stringify({
            id,
            changed: true,
            captureSha256: hash,
            bytes: Buffer.byteLength(serialize(record)),
          }),
        )
      }
    } else if (command === 'graphs') {
      if (positionals.length !== 3) throw new Error('graphs requires <strava_id> <graphs.json>')
      const id = positionals[1]
      activityFor(id)
      const graphCapture = myWindsockGraphCaptureSchema.parse(
        JSON.parse(await readFile(positionals[2], 'utf8')),
      )
      const { record, previous } = await readArchive(id)
      if (!record?.browserCapture)
        throw new Error(`${id}: import the runtime capture before its graphs`)
      // A later incomplete graph bundle cannot discard an earlier successful graph.
      if (record.graphCapture) {
        for (const [key, oldGraph] of Object.entries(record.graphCapture.graphs)) {
          const next = graphCapture.graphs[key]
          if (
            !next ||
            ((next.state === 'failed' || next.state === 'unavailable') &&
              (oldGraph.state === 'captured' || oldGraph.state === 'empty'))
          )
            graphCapture.graphs[key] = oldGraph
        }
      }
      const hash = myWindsockGraphCaptureSha256(graphCapture)
      if (record.ingestion.graphsSha256 === hash)
        console.log(JSON.stringify({ id, changed: false, graphsSha256: hash }))
      else {
        record.graphCapture = graphCapture
        record.ingestion.graphsSha256 = myWindsockGraphCaptureSha256(graphCapture)
        record.ingestion.sections.allGraphs = Object.values(graphCapture.graphs).some(
          graph => graph.state === 'failed',
        )
          ? 'not-captured'
          : 'captured'
        record.ingestion.events.push({
          at: graphCapture.capturedAt,
          type: 'graph-import',
          note: `Graph inventory saved: ${graphCapture.menu.length} menu entries, ${Object.keys(graphCapture.graphs).length} total graph/view records.`,
          captureSha256: record.ingestion.graphsSha256,
        })
        await save(record, previous)
        console.log(
          JSON.stringify({
            id,
            changed: true,
            graphs: Object.keys(graphCapture.graphs).length,
            graphsSha256: record.ingestion.graphsSha256,
          }),
        )
      }
    } else if (command === 'verify') {
      if (positionals.length > 2) throw new Error('verify takes an optional <strava_id>')
      const ids = positionals[1]
        ? [positionals[1]]
        : (await readdir(directory))
            .filter(name => /^[1-9]\d*\.json$/.test(name))
            .map(name => name.slice(0, -5))
      const results = []
      for (const id of ids) {
        const { record, previous } = await readArchive(id)
        if (!record?.browserCapture) continue
        const computed = verifyMyWindsockRouteMappings(record)
        // A verified record describes the current capture and keeps its evidence; a replaced capture
        // starts with no mappings. Unresolved records are recomputed.
        const changed = computed.filter(item => {
          const existing = record.mappings.find(entry => entry.path === item.path)
          return (
            existing?.status !== 'verified' &&
            (existing?.status !== item.status || existing.evidence !== item.evidence)
          )
        })
        if (changed.length > 0) {
          record.mappings = [
            ...record.mappings.filter(entry => !changed.some(item => item.path === entry.path)),
            ...changed,
          ]
          record.ingestion.events.push({
            at: new Date().toISOString(),
            type: 'verify',
            note: `Route identities recomputed for ${changed.length} fields: ${changed.filter(item => item.status === 'verified').length} verified, ${changed.filter(item => item.status === 'unresolved').length} unresolved.`,
            captureSha256: record.ingestion.captureSha256,
          })
          await save(record, previous)
        }
        results.push({
          id,
          changed: changed.length,
          mappings: computed.map(item => ({
            path: item.path.replace('/runtime/series/', ''),
            status: item.status,
            evidence: item.evidence,
          })),
        })
      }
      console.log(JSON.stringify(results, null, 2))
    } else {
      if (positionals.length !== 4)
        throw new Error('attempt requires <strava_id> <failed|unavailable> <note>')
      const id = positionals[1]
      activityFor(id)
      const status = z.enum(['failed', 'unavailable']).parse(positionals[2])
      const note = z.string().min(1).max(1000).parse(positionals[3])
      const at = new Date().toISOString()
      const stateSource = await source(statePath)
      const ledger = ledgerSchema.parse(stateSource ? JSON.parse(stateSource) : {})
      const attempts = ledger[id] ?? []
      const duplicate = attempts.at(-1)
      if (duplicate?.localDay !== today || duplicate.status !== status || duplicate.note !== note) {
        const { record, previous } = await readArchive(id)
        if (record) {
          record.ingestion.attempts.push({ at, status, note })
          await save(record, previous)
        }
        ledger[id] = [...attempts, { at, localDay: today, status, note }]
        await atomicWrite(statePath, JSON.stringify(ledger, null, 2) + '\n', stateSource)
      }
      console.log(JSON.stringify({ id, status, retry: 'next local day' }))
    }
  } finally {
    await rm(lock, { recursive: true, force: true })
  }
}
