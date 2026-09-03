import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import type { RawStravaActivity, StravaRawCache, StravaStreams } from '../plugins/stores/strava'
import {
  planTrainingPeaksBackfill,
  type ActivityBridgeInputs,
  type StravaTrainingPeaksBackfillPlan,
  type TrainingPeaksBackfillPlan,
  type TrainingPeaksBackfillSource,
  type WahooTrainingPeaksBackfillPlan,
} from '../plugins/stores/activity-bridge'
import {
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  readGarminConnectSession,
} from '../util/garmin-session'
import { joinSegments, QUARTZ } from '../util/path'
import { readStravaCacheFile } from '../util/strava-cache-file'
import { stravaActivityTcx, timedStravaStreamsFromCache } from '../util/strava-tcx'
import { isRecord } from '../util/type-guards'
import { wahooCloudClientFromEnv } from '../util/wahoo-cloud'
import {
  fetchGarminActivityFile,
  fetchWahooActivityFit,
  parseActivityBridgeStravaActivities,
  readActivityBridgeGarminActivities,
  readActivityBridgeLedger,
  readActivityBridgeWahooActivities,
  type ActivityBridgeFile,
} from './sync-activity-providers'

const STRAVA_CACHE_FILE = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
export const TRAININGPEAKS_BACKFILL_DIR = joinSegments(
  QUARTZ,
  '.quartz-cache',
  'trainingpeaks-backfill',
)
const TRAININGPEAKS_MANIFEST = 'manifest.json'
const DEFAULT_REMOTE_DELAY_MS = 1000
export const TRAININGPEAKS_CALENDAR_URL = 'https://app.trainingpeaks.com/#calendar'
const execFileAsync = promisify(execFile)

export interface TrainingPeaksBackfillArgs {
  source: TrainingPeaksBackfillSource
  write: boolean
  openCalendar: boolean
  since: string | null
  until: string | null
  limit: number | null
  ids: readonly string[]
  outputDir: string
  delayMs: number
}

type TrainingPeaksFileProvenance = 'garmin-original' | 'strava-generated-tcx' | 'wahoo-original-fit'

export interface TrainingPeaksFile {
  bytes: Uint8Array
  sha256: string
  kind: 'fit' | 'tcx'
  provenance: TrainingPeaksFileProvenance
}

interface TrainingPeaksBackfillManifestFile {
  title: string
  sport: string
  localDate: string
  sourceProvider: TrainingPeaksBackfillSource
  sourceActivityId: string
  artifactProvenance: TrainingPeaksFileProvenance
  artifactSha256: string
  fileKind: 'fit' | 'tcx'
  filename: string
  byteLength: number
}

interface TrainingPeaksBackfillManifest {
  version: 2
  sourceProvider: TrainingPeaksBackfillSource
  generatedAt: number
  files: TrainingPeaksBackfillManifestFile[]
}

function argumentValue(argv: readonly string[], index: number, argument: string): string {
  const value = argv[index]
  if (!value || value.startsWith('--')) throw new Error(`${argument} requires a value`)
  return value
}

function positiveInteger(value: string, argument: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed <= 0)
    throw new Error(`${argument} requires a positive integer`)
  return parsed
}

function nonnegativeInteger(value: string, argument: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0)
    throw new Error(`${argument} requires a nonnegative integer`)
  return parsed
}

function isoDay(value: string, argument: string): string {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error(`${argument} requires YYYY-MM-DD`)
  const parsed = new Date(`${value}T00:00:00.000Z`)
  if (!Number.isFinite(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== value)
    throw new Error(`${argument} requires a valid calendar day`)
  return value
}

function source(value: string): TrainingPeaksBackfillSource {
  if (value === 'garmin' || value === 'strava' || value === 'wahoo') return value
  throw new Error('--source requires garmin, strava, or wahoo')
}

function activityId(value: string): string {
  const normalized = value.replace(/^(?:connect|wahoo):/, '')
  if (!/^[1-9]\d*$/.test(normalized)) throw new Error('--id requires a positive activity id')
  return normalized
}

export function trainingPeaksBackfillUsage(): string {
  return [
    'usage: pnpm trainingpeaks:backfill -- --source garmin|strava|wahoo [--write] [--open-calendar] [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--limit N] [--id ACTIVITY_ID] [--output DIRECTORY] [--delay-ms N]',
    '       pnpm trainingpeaks:backfill-garmin -- [options]',
    '       pnpm trainingpeaks:backfill-strava -- [options]',
    '       pnpm trainingpeaks:backfill-wahoo -- [options]',
    'defaults to dry-run. Garmin and Wahoo preserve downloaded device files. Strava generates TCX from cached streams.',
    '--write materializes private files locally. --open-calendar then opens the output folder and TrainingPeaks calendar for drag-and-drop upload.',
  ].join('\n')
}

export function parseTrainingPeaksBackfillArgs(argv: readonly string[]): TrainingPeaksBackfillArgs {
  let selectedSource: TrainingPeaksBackfillSource | null = null
  let write = false
  let openCalendar = false
  let since: string | null = null
  let until: string | null = null
  let limit: number | null = null
  let outputDir: string | null = null
  let delayMs: number | null = null
  const ids: string[] = []
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index]
    if (argument === '--') continue
    if (argument === '--source') {
      const parsed = source(argumentValue(argv, ++index, argument))
      if (selectedSource != null && selectedSource !== parsed)
        throw new Error('--source may select only one provider')
      selectedSource = parsed
    } else if (argument === '--write') write = true
    else if (argument === '--dry-run') write = false
    else if (argument === '--open-calendar') openCalendar = true
    else if (argument === '--since')
      since = isoDay(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--until')
      until = isoDay(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--limit')
      limit = positiveInteger(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--id') ids.push(activityId(argumentValue(argv, ++index, argument)))
    else if (argument === '--output') outputDir = argumentValue(argv, ++index, argument).trim()
    else if (argument === '--delay-ms')
      delayMs = nonnegativeInteger(argumentValue(argv, ++index, argument), argument)
    else throw new Error(`unknown TrainingPeaks backfill argument: ${argument}`)
  }
  if (selectedSource == null) throw new Error('--source requires garmin, strava, or wahoo')
  if (openCalendar && !write) throw new Error('--open-calendar requires --write')
  if (since != null && until != null && since > until)
    throw new Error('--since must be on or before --until')
  if (outputDir === '') throw new Error('--output requires a nonempty directory')
  return {
    source: selectedSource,
    write,
    openCalendar,
    since,
    until,
    limit,
    ids: [...new Set(ids)].sort(),
    outputDir: outputDir ?? joinSegments(TRAININGPEAKS_BACKFILL_DIR, selectedSource),
    delayMs: delayMs ?? (selectedSource === 'strava' ? 0 : DEFAULT_REMOTE_DELAY_MS),
  }
}

function sourceId(plan: TrainingPeaksBackfillPlan): string {
  return activityId(plan.source.id)
}

export function selectTrainingPeaksBackfillPlans(
  plans: readonly TrainingPeaksBackfillPlan[],
  args: TrainingPeaksBackfillArgs,
): TrainingPeaksBackfillPlan[] {
  const ids = new Set(args.ids)
  const selected = plans.filter(
    plan =>
      plan.sourceProvider === args.source &&
      (args.since == null || plan.localDate >= args.since) &&
      (args.until == null || plan.localDate <= args.until) &&
      (ids.size === 0 || ids.has(sourceId(plan))),
  )
  return args.limit == null ? selected : selected.slice(0, args.limit)
}

export function trainingPeaksBackfillFilename(
  plan: TrainingPeaksBackfillPlan,
  kind: 'fit' | 'tcx' = 'tcx',
): string {
  return `${plan.localDate}-${plan.sourceProvider}-${sourceId(plan)}.${kind}`
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex')
}

export function stravaTrainingPeaksFile(
  activity: RawStravaActivity,
  streams: StravaStreams,
  sport: StravaTrainingPeaksBackfillPlan['sport'],
): TrainingPeaksFile {
  const bytes = new TextEncoder().encode(
    stravaActivityTcx(activity, timedStravaStreamsFromCache(streams), sport),
  )
  return { bytes, sha256: sha256(bytes), kind: 'tcx', provenance: 'strava-generated-tcx' }
}

async function existingFileMatches(path: string, expectedSha256: string): Promise<boolean> {
  try {
    return sha256(await fs.readFile(path)) === expectedSha256
  } catch (error) {
    if (isRecord(error) && error.code === 'ENOENT') return false
    throw error
  }
}

export async function writeTrainingPeaksFile(
  path: string,
  bytes: Uint8Array,
): Promise<'created' | 'existing'> {
  const expectedSha256 = sha256(bytes)
  if (await existingFileMatches(path, expectedSha256)) return 'existing'
  try {
    await fs.access(path)
    throw new Error(`refusing to replace a different TrainingPeaks backfill file: ${path}`)
  } catch (error) {
    if (!(isRecord(error) && error.code === 'ENOENT')) throw error
  }
  await fs.mkdir(dirname(path), { recursive: true, mode: 0o700 })
  const temporary = `${path}.tmp-${process.pid}-${Date.now()}`
  try {
    await fs.writeFile(temporary, bytes, { flag: 'wx', mode: 0o600 })
    try {
      await fs.link(temporary, path)
    } catch (error) {
      if (!(isRecord(error) && error.code === 'EEXIST')) throw error
      if (!(await existingFileMatches(path, expectedSha256)))
        throw new Error(`refusing to replace a different TrainingPeaks backfill file: ${path}`)
      return 'existing'
    }
    return 'created'
  } finally {
    await fs.rm(temporary, { force: true })
  }
}

async function writeManifest(path: string, manifest: TrainingPeaksBackfillManifest): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}-${Date.now()}`
  try {
    await fs.writeFile(temporary, JSON.stringify(manifest, null, 2), { flag: 'wx', mode: 0o600 })
    await fs.rename(temporary, path)
  } finally {
    await fs.rm(temporary, { force: true })
  }
}

function planSummary(plan: TrainingPeaksBackfillPlan): string {
  return `provider=${plan.sourceProvider} source=${plan.source.id} day=${plan.localDate} sport=${plan.sport} title=${JSON.stringify(plan.title)}`
}

function manifestFile(
  plan: TrainingPeaksBackfillPlan,
  file: TrainingPeaksFile,
  filename: string,
): TrainingPeaksBackfillManifestFile {
  return {
    title: plan.title,
    sport: plan.sport,
    localDate: plan.localDate,
    sourceProvider: plan.sourceProvider,
    sourceActivityId: plan.source.id,
    artifactProvenance: file.provenance,
    artifactSha256: file.sha256,
    fileKind: file.kind,
    filename,
    byteLength: file.bytes.byteLength,
  }
}

function stravaCacheActivity(
  cache: StravaRawCache,
  plan: StravaTrainingPeaksBackfillPlan,
): RawStravaActivity {
  const activity = cache.activities[sourceId(plan)]
  if (!activity) throw new Error(`Strava activity ${sourceId(plan)} is absent from the cache`)
  return activity
}

function stravaCacheStreams(
  cache: StravaRawCache,
  plan: StravaTrainingPeaksBackfillPlan,
): StravaStreams | null {
  const streams = cache.streams?.[sourceId(plan)]
  return streams?.time && streams.time.length >= 2 ? streams : null
}

function stravaPlan(plan: TrainingPeaksBackfillPlan): plan is StravaTrainingPeaksBackfillPlan {
  return plan.sourceProvider === 'strava'
}

function wahooPlan(plan: TrainingPeaksBackfillPlan): plan is WahooTrainingPeaksBackfillPlan {
  return plan.sourceProvider === 'wahoo'
}

function garminTrainingPeaksFile(file: ActivityBridgeFile): TrainingPeaksFile {
  return { bytes: file.bytes, sha256: file.sha256, kind: file.kind, provenance: 'garmin-original' }
}

function wahooTrainingPeaksFile(file: ActivityBridgeFile): TrainingPeaksFile {
  return { bytes: file.bytes, sha256: file.sha256, kind: 'fit', provenance: 'wahoo-original-fit' }
}

async function openTrainingPeaksUpload(outputDir: string): Promise<void> {
  if (process.platform !== 'darwin')
    throw new Error('--open-calendar currently requires macOS and /usr/bin/open')
  await execFileAsync('/usr/bin/open', [resolve(outputDir), TRAININGPEAKS_CALENDAR_URL])
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2)
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log(trainingPeaksBackfillUsage())
    return
  }
  const args = parseTrainingPeaksBackfillArgs(argv)
  let stravaCache: StravaRawCache | null = null
  let inputs: ActivityBridgeInputs
  if (args.source === 'strava') {
    stravaCache = await readStravaCacheFile(STRAVA_CACHE_FILE)
    if (stravaCache == null) throw new Error(`Strava cache is missing at ${STRAVA_CACHE_FILE}`)
    inputs = { strava: parseActivityBridgeStravaActivities(stravaCache), garmin: [], wahoo: [] }
  } else if (args.source === 'garmin') {
    inputs = { strava: [], garmin: await readActivityBridgeGarminActivities(), wahoo: [] }
  } else {
    inputs = { strava: [], garmin: [], wahoo: await readActivityBridgeWahooActivities() }
  }
  const ledger = await readActivityBridgeLedger()
  const selected = selectTrainingPeaksBackfillPlans(
    planTrainingPeaksBackfill(inputs, ledger, args.source),
    args,
  )
  const unavailable =
    stravaCache == null
      ? []
      : selected.filter(plan => stravaPlan(plan) && stravaCacheStreams(stravaCache, plan) == null)
  const unavailableIds = new Set(unavailable.map(sourceId))
  const plans = selected.filter(plan => !unavailableIds.has(sourceId(plan)))
  for (const plan of plans) console.log(`[trainingpeaks-backfill] candidate ${planSummary(plan)}`)
  for (const plan of unavailable)
    console.log(`[trainingpeaks-backfill] unavailable ${planSummary(plan)} reason=no-cached-stream`)
  if (!args.write) {
    console.log(
      `[trainingpeaks-backfill] dry-run source=${args.source} ready=${plans.length} unavailable=${unavailable.length}`,
    )
    return
  }
  if (unavailable.length > 0)
    throw new Error(
      `refusing an incomplete Strava export; ${unavailable.length} selected activities have no cached timed stream`,
    )
  if (plans.length === 0) {
    console.log('[trainingpeaks-backfill] no workout files selected')
    return
  }
  const garminSession = args.source === 'garmin' ? await readGarminConnectSession() : null
  const wahooClient = args.source === 'wahoo' ? await wahooCloudClientFromEnv() : null
  const garminBase = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_BASE_URL?.trim() || DEFAULT_GARMIN_CONNECT_BASE,
  )
  const outputDir = resolve(args.outputDir)
  await fs.mkdir(outputDir, { recursive: true, mode: 0o700 })
  const files: TrainingPeaksBackfillManifestFile[] = []
  let created = 0
  let existing = 0
  for (let index = 0; index < plans.length; index++) {
    const plan = plans[index]
    let file: TrainingPeaksFile
    if (stravaPlan(plan)) {
      if (stravaCache == null) throw new Error('Strava cache is unavailable')
      const streams = stravaCacheStreams(stravaCache, plan)
      if (streams == null) throw new Error(`Strava activity ${sourceId(plan)} has no cached stream`)
      file = stravaTrainingPeaksFile(stravaCacheActivity(stravaCache, plan), streams, plan.sport)
    } else if (wahooPlan(plan)) {
      if (wahooClient == null) throw new Error('Wahoo client is unavailable')
      file = wahooTrainingPeaksFile(await fetchWahooActivityFit(plan.source, wahooClient))
    } else {
      if (garminSession == null) throw new Error('Garmin session is unavailable')
      file = garminTrainingPeaksFile(
        await fetchGarminActivityFile(plan.source, garminSession, garminBase),
      )
    }
    if (sha256(file.bytes) !== file.sha256)
      throw new Error(`artifact SHA-256 mismatch for ${plan.source.id}`)
    const filename = trainingPeaksBackfillFilename(plan, file.kind)
    const status = await writeTrainingPeaksFile(join(outputDir, filename), file.bytes)
    if (status === 'created') created++
    else existing++
    files.push(manifestFile(plan, file, filename))
    console.log(`[trainingpeaks-backfill] ${status} ${filename} sha256=${file.sha256}`)
    if (index + 1 < plans.length && args.delayMs > 0) await sleep(args.delayMs)
  }
  const manifest: TrainingPeaksBackfillManifest = {
    version: 2,
    sourceProvider: args.source,
    generatedAt: Date.now(),
    files,
  }
  await writeManifest(join(outputDir, TRAININGPEAKS_MANIFEST), manifest)
  console.log(
    `[trainingpeaks-backfill] source=${args.source} prepared=${files.length} created=${created} existing=${existing} unavailable=${unavailable.length} output=${outputDir}`,
  )
  console.log(
    '[trainingpeaks-backfill] drag the .fit and .tcx files onto the TrainingPeaks athlete calendar to upload them',
  )
  if (args.openCalendar) {
    await openTrainingPeaksUpload(outputDir)
    console.log(
      `[trainingpeaks-backfill] opened output=${outputDir} calendar=${TRAININGPEAKS_CALENDAR_URL}`,
    )
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => {
    console.error(
      `[trainingpeaks-backfill] failed: ${error instanceof Error ? error.message : error}`,
    )
    process.exit(1)
  })
}
