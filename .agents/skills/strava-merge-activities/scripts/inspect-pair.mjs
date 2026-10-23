import fs from 'node:fs/promises'
import path from 'node:path'

const argv = process.argv.slice(2)

function option(name, required = false) {
  const index = argv.indexOf(`--${name}`)
  const value = index >= 0 ? argv[index + 1] : undefined
  if (required && (!value || value.startsWith('--'))) throw new Error(`missing --${name}`)
  return value
}

function activityId(value, name) {
  const match = value?.match(/(?:activities\/)?(\d{8,})/)
  if (!match) throw new Error(`invalid --${name}`)
  return match[1]
}

const TELEMETRY = ['latlng', 'altitude', 'distance', 'watts', 'heartrate', 'cadence']

const CACHE_METADATA = [
  'name',
  'sportType',
  'startDate',
  'startDateLocal',
  'elapsedTime',
  'movingTime',
  'distance',
  'totalElevationGain',
  'calories',
]

const BROWSER_ONLY_METADATA = [
  'description',
  'visibility',
  'hidden detail toggles',
  'tags',
  'trainer state',
  'mute state',
  'private note',
  'gear',
  'media',
  'structured exercise or set data',
]

function stream(cache, id) {
  const value = cache.streams?.[id]
  if (!value || !Array.isArray(value.time) || value.time.length < 2) return undefined
  return value
}

function field(source, name) {
  const values = source?.[name]
  if (!Array.isArray(values) || values.length < 2) return undefined
  if (values.length !== source.time.length) throw new Error(`${name} length does not match time`)
  return values
}

function span(source, name) {
  if (!field(source, name)) return undefined
  return [source.time[0], source.time.at(-1)]
}

function coverage(destinationSpan, sourceSpan, duration, offsetSeconds) {
  let measured = 0
  for (let elapsed = 0; elapsed <= duration; elapsed++) {
    const inDestination =
      destinationSpan && elapsed >= destinationSpan[0] && elapsed <= destinationSpan[1]
    const sourceElapsed = elapsed - offsetSeconds
    const inSource = sourceSpan && sourceElapsed >= sourceSpan[0] && sourceElapsed <= sourceSpan[1]
    if (inDestination || inSource) measured++
  }
  return measured
}

const workspace = path.resolve(option('workspace', true))
const cachePath = path.resolve(
  option('cache') ?? path.join(workspace, 'quartz/.quartz-cache/strava.json'),
)
const sourceId = activityId(option('source', true), 'source')
const destinationId = activityId(option('destination', true), 'destination')
if (sourceId === destinationId) throw new Error('source and destination must differ')

const cache = JSON.parse(await fs.readFile(cachePath, 'utf8'))
const source = cache.activities?.[sourceId]
const destination = cache.activities?.[destinationId]
if (!source) throw new Error(`source activity ${sourceId} is missing from the cache`)
if (!destination) throw new Error(`destination activity ${destinationId} is missing from the cache`)

const sourceStream = stream(cache, sourceId)
const destinationStream = stream(cache, destinationId)
if (!sourceStream) throw new Error(`source activity ${sourceId} has no usable stream in the cache`)

const destinationStart = new Date(destination.startDate)
const sourceStart = new Date(source.startDate)
const duration = Math.round(destination.elapsedTime)
const sourceDuration = Math.round(source.elapsedTime)
const offsetSeconds = (sourceStart.getTime() - destinationStart.getTime()) / 1_000
const overlapSeconds = Math.max(
  0,
  Math.min(duration, offsetSeconds + sourceDuration) - Math.max(0, offsetSeconds),
)

const telemetry = {}
for (const name of TELEMETRY) {
  const sourceSpan = span(sourceStream, name)
  const destinationSpan = span(destinationStream, name)
  if (!sourceSpan && !destinationSpan) continue
  const owner =
    name === 'heartrate'
      ? sourceSpan
        ? sourceId
        : destinationId
      : destinationSpan
        ? destinationId
        : sourceId
  const measured = coverage(destinationSpan, sourceSpan, duration, offsetSeconds)
  telemetry[name] = {
    owner,
    inSource: sourceSpan != null,
    inDestination: destinationSpan != null,
    measuredSeconds: measured,
    boundaryFillSeconds: duration + 1 - measured,
    measuredFraction: Number((measured / (duration + 1)).toFixed(4)),
  }
}

const sourceOnlyFields = Object.entries(telemetry)
  .filter(([, value]) => value.inSource && !value.inDestination)
  .map(([name]) => name)

const report = {
  cachePath,
  cacheVersion: cache.version,
  lastSync: new Date(cache.lastSync).toISOString(),
  lastSyncAgeHours: Number(((Date.now() - cache.lastSync) / 3_600_000).toFixed(2)),
  source: { id: sourceId, ...Object.fromEntries(CACHE_METADATA.map(k => [k, source[k]])) },
  destination: {
    id: destinationId,
    ...Object.fromEntries(CACHE_METADATA.map(k => [k, destination[k]])),
  },
  geometry: {
    offsetSeconds,
    destinationDuration: duration,
    sourceDuration,
    overlapSeconds,
    builderOverlapRatio: Number((overlapSeconds / Math.min(duration, sourceDuration)).toFixed(4)),
    destinationOverlapRatio: Number((overlapSeconds / duration).toFixed(4)),
  },
  telemetry,
  sourceOnlyFields,
  destinationHasStream: destinationStream != null,
  destinationHasDetail: cache.activityDetails?.[destinationId] != null,
  mustReadFromBrowser: BROWSER_ONLY_METADATA,
}

const warnings = []
if (report.geometry.destinationOverlapRatio < 0.8)
  warnings.push(
    `only ${(report.geometry.destinationOverlapRatio * 100).toFixed(1)}% of the destination window is covered by the source; the rest is boundary hold`,
  )
if (report.geometry.builderOverlapRatio < 0.5)
  warnings.push('builder will reject this pair: overlap below the 0.5 threshold')
if (sourceOnlyFields.length === 0)
  warnings.push('source contributes no field the destination lacks; confirm the merge is wanted')
if (!destinationStream)
  warnings.push('destination has no stream; every non-heart-rate field comes from the source')
if (report.lastSyncAgeHours > 24)
  warnings.push(`cache is ${report.lastSyncAgeHours}h old; run pnpm strava:sync before trusting it`)
report.warnings = warnings

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`)
