import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import {
  activityBridgeReceiptKey,
  emptyActivityBridgeLedger,
  isTerminalActivityBridgeReceipt,
  planActivityBridge,
  upsertActivityBridgeReceipt,
  type ActivityBridgeDirection,
  type ActivityBridgeGarminActivity,
  type ActivityBridgeInputs,
  type ActivityBridgeLedger,
  type ActivityBridgePlan,
  type ActivityBridgeProvider,
  type ActivityBridgeReceipt,
  type ActivityBridgeStravaActivity,
  type ActivityBridgeUploadStatus,
  type ActivityBridgeWahooActivity,
} from '../plugins/stores/activity-bridge'
import { parseWahooCache } from '../plugins/stores/wahoo'
import { garminActivityFileFromArchive } from '../util/garmin-fit'
import {
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  fetchGarminBytes,
  readGarminConnectSession,
  type GarminConnectSession,
} from '../util/garmin-session'
import { encodeGarminTcxActivityFit } from '../util/garmin-tcx-fit'
import { updateGarminActivityTitle } from '../util/garmin-title-update'
import { joinSegments, QUARTZ } from '../util/path'
import { isRecord, type UnknownRecord } from '../util/type-guards'
import {
  WahooCloudClient,
  wahooCloudClientFromEnv,
  type WahooWorkoutFileUpload,
} from '../util/wahoo-cloud'
import { wahooFitSha256 } from '../util/wahoo-fit'
import { uploadGarminFit } from './garmin-fit-upload'

const STRAVA_CACHE_FILE = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
const GARMIN_CACHE_FILE = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')
const WAHOO_CACHE_FILE = joinSegments(QUARTZ, '.quartz-cache', 'wahoo.json')
export const ACTIVITY_BRIDGE_LEDGER_FILE = joinSegments(
  QUARTZ,
  '.quartz-cache',
  'activity-bridge.json',
)

export interface ActivityBridgeArgs {
  write: boolean
  limit: number | null
}

export interface ActivityBridgeFile {
  bytes: Uint8Array
  filename: string
  sha256: string
  kind: 'fit' | 'tcx'
  sourceKind: 'fit' | 'tcx'
}

interface ActivityBridgeExecution {
  ledger: ActivityBridgeLedger
  uploadStatus: ActivityBridgeUploadStatus
}

export function parseActivityBridgeArgs(argv: readonly string[]): ActivityBridgeArgs {
  let write = false
  let limit: number | null = null
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]
    if (arg === '--write') write = true
    else if (arg === '--limit') {
      const value = argv[++index]
      const parsed = Number(value)
      if (!value || !Number.isInteger(parsed) || parsed <= 0)
        throw new Error('--limit requires a positive integer')
      limit = parsed
    } else throw new Error(`unknown activity bridge argument: ${arg}`)
  }
  return { write, limit }
}

function requiredRecord(value: unknown, label: string): UnknownRecord {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return value
}

function requiredString(record: UnknownRecord, key: string, label: string): string {
  const value = record[key]
  if (typeof value !== 'string' || !value.trim())
    throw new Error(`${label}.${key} must be a string`)
  return value
}

function optionalString(record: UnknownRecord, key: string, label: string): string | null {
  const value = record[key]
  if (value == null) return null
  if (typeof value !== 'string') throw new Error(`${label}.${key} must be a string or null`)
  return value
}

function requiredNumber(record: UnknownRecord, key: string, label: string): number {
  const value = record[key]
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error(`${label}.${key} must be finite`)
  return value
}

function optionalNumber(record: UnknownRecord, key: string, label: string): number | null {
  const value = record[key]
  if (value == null) return null
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error(`${label}.${key} must be finite or null`)
  return value
}

function activityRecords(value: unknown, label: string): [string, UnknownRecord][] {
  const root = requiredRecord(value, label)
  const activities = requiredRecord(root.activities, `${label}.activities`)
  return Object.entries(activities).map(([id, activity]) => [
    id,
    requiredRecord(activity, `${label}.activities.${id}`),
  ])
}

export function parseActivityBridgeStravaActivities(
  value: unknown,
): ActivityBridgeStravaActivity[] {
  return activityRecords(value, 'Strava cache').map(([key, record]) => {
    const id = requiredNumber(record, 'id', `Strava activity ${key}`)
    return {
      id: String(id),
      name: requiredString(record, 'name', `Strava activity ${key}`),
      sportType: requiredString(record, 'sportType', `Strava activity ${key}`),
      startDate: requiredString(record, 'startDate', `Strava activity ${key}`),
      startDateLocal: requiredString(record, 'startDateLocal', `Strava activity ${key}`),
      distanceM: requiredNumber(record, 'distance', `Strava activity ${key}`),
      movingTimeS: requiredNumber(record, 'movingTime', `Strava activity ${key}`),
      elapsedTimeS: requiredNumber(record, 'elapsedTime', `Strava activity ${key}`),
    }
  })
}

function providerSport(record: UnknownRecord, label: string): 'bike' | 'run' | 'swim' | null {
  const value = optionalString(record, 'sport', label)
  if (value == null || value === 'bike' || value === 'run' || value === 'swim') return value
  throw new Error(`${label}.sport is invalid`)
}

export function parseActivityBridgeGarminActivities(
  value: unknown,
): ActivityBridgeGarminActivity[] {
  return activityRecords(value, 'Garmin cache').map(([key, record]) => {
    const label = `Garmin activity ${key}`
    return {
      id: requiredString(record, 'id', label),
      name: requiredString(record, 'name', label),
      sport: providerSport(record, label),
      startDate: requiredString(record, 'startDate', label),
      startDateLocal: requiredString(record, 'startDateLocal', label),
      distanceM: optionalNumber(record, 'distanceM', label),
      movingTimeS: optionalNumber(record, 'movingTimeS', label),
      elapsedTimeS: optionalNumber(record, 'elapsedTimeS', label),
    }
  })
}

export function parseActivityBridgeInputs(
  stravaValue: unknown,
  garminValue: unknown,
  wahooValue: unknown,
): ActivityBridgeInputs {
  return {
    strava: parseActivityBridgeStravaActivities(stravaValue),
    garmin: parseActivityBridgeGarminActivities(garminValue),
    wahoo: parseActivityBridgeWahooActivities(wahooValue),
  }
}

export function parseActivityBridgeWahooActivities(value: unknown): ActivityBridgeWahooActivity[] {
  return Object.values(parseWahooCache(value).activities).map(activity => ({
    id: activity.id,
    name:
      activity.name?.trim() ||
      activity.summary.name?.trim() ||
      `Wahoo workout ${activity.workoutId}`,
    workoutId: activity.workoutId,
    sport: activity.sport,
    startDate: activity.startDate,
    startDateLocal: activity.startDateLocal,
    distanceM: activity.distanceM,
    movingTimeS: activity.movingTimeS,
    elapsedTimeS: activity.elapsedTimeS,
    fitUrl: activity.sourceFile.url,
    fitSha256: activity.sourceFile.sha256,
  }))
}

export async function readActivityBridgeGarminActivities(): Promise<
  ActivityBridgeGarminActivity[]
> {
  return parseActivityBridgeGarminActivities(await readJson(GARMIN_CACHE_FILE))
}

export async function readActivityBridgeWahooActivities(): Promise<ActivityBridgeWahooActivity[]> {
  return parseActivityBridgeWahooActivities(await readJson(WAHOO_CACHE_FILE))
}

export async function readActivityBridgeInputs(): Promise<ActivityBridgeInputs> {
  const [stravaValue, garminValue, wahooValue] = await Promise.all([
    readJson(STRAVA_CACHE_FILE),
    readJson(GARMIN_CACHE_FILE),
    readJson(WAHOO_CACHE_FILE),
  ])
  return parseActivityBridgeInputs(stravaValue, garminValue, wahooValue)
}

function direction(value: unknown, label: string): ActivityBridgeDirection {
  if (value === 'garmin-to-wahoo' || value === 'wahoo-to-garmin') return value
  throw new Error(`${label}.direction is invalid`)
}

function provider(value: unknown, label: string): ActivityBridgeProvider {
  if (value === 'garmin' || value === 'wahoo') return value
  throw new Error(`${label} provider is invalid`)
}

function uploadStatus(value: unknown, label: string): ActivityBridgeUploadStatus {
  if (
    value === 'pending' ||
    value === 'in_progress' ||
    value === 'complete' ||
    value === 'duplicate' ||
    value === 'error'
  )
    return value
  throw new Error(`${label}.uploadStatus is invalid`)
}

function parseReceipt(value: unknown, label: string): ActivityBridgeReceipt {
  const record = requiredRecord(value, label)
  const sourceProvider = provider(record.sourceProvider, `${label}.sourceProvider`)
  const destinationProvider = provider(record.destinationProvider, `${label}.destinationProvider`)
  const sourceFitSha256 = requiredString(record, 'sourceFitSha256', label).toLowerCase()
  const receipt: ActivityBridgeReceipt = {
    direction: direction(record.direction, label),
    sourceProvider,
    sourceActivityId: requiredString(record, 'sourceActivityId', label),
    sourceFitSha256,
    destinationProvider,
    destinationActivityId: optionalString(record, 'destinationActivityId', label),
    stravaActivityId: requiredString(record, 'stravaActivityId', label),
    uploadToken: optionalString(record, 'uploadToken', label),
    uploadStatus: uploadStatus(record.uploadStatus, label),
    createdAt: requiredNumber(record, 'createdAt', label),
    updatedAt: requiredNumber(record, 'updatedAt', label),
  }
  activityBridgeReceiptKey(
    receipt.sourceProvider,
    receipt.sourceActivityId,
    receipt.sourceFitSha256,
    receipt.destinationProvider,
  )
  return receipt
}

export function parseActivityBridgeLedger(value: unknown): ActivityBridgeLedger {
  const record = requiredRecord(value, 'Activity bridge ledger')
  if (record.version !== 1) throw new Error('Activity bridge ledger version must be 1')
  const receiptsValue = requiredRecord(record.receipts, 'Activity bridge ledger.receipts')
  const receipts: Record<string, ActivityBridgeReceipt> = {}
  for (const [key, value] of Object.entries(receiptsValue)) {
    const receipt = parseReceipt(value, `Activity bridge receipt ${key}`)
    const expected = activityBridgeReceiptKey(
      receipt.sourceProvider,
      receipt.sourceActivityId,
      receipt.sourceFitSha256,
      receipt.destinationProvider,
    )
    if (key !== expected)
      throw new Error(`Activity bridge receipt key ${key} does not match payload`)
    receipts[key] = receipt
  }
  return {
    version: 1,
    updatedAt: requiredNumber(record, 'updatedAt', 'Activity bridge ledger'),
    receipts,
  }
}

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await fs.readFile(path, 'utf8'))
}

export async function readActivityBridgeLedger(
  path = ACTIVITY_BRIDGE_LEDGER_FILE,
): Promise<ActivityBridgeLedger> {
  try {
    return parseActivityBridgeLedger(await readJson(path))
  } catch (error) {
    if (isRecord(error) && error.code === 'ENOENT') return emptyActivityBridgeLedger()
    throw error
  }
}

export async function writeActivityBridgeLedgerAtomic(
  ledger: ActivityBridgeLedger,
  path = ACTIVITY_BRIDGE_LEDGER_FILE,
): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}-${Date.now()}`
  await fs.mkdir(dirname(path), { recursive: true })
  try {
    await fs.writeFile(temporary, JSON.stringify(ledger, null, 2))
    await fs.rename(temporary, path)
  } finally {
    await fs.rm(temporary, { force: true })
  }
}

function garminActivityId(value: string): string {
  const id = value.startsWith('connect:') ? value.slice('connect:'.length) : value
  if (!id || !/^\d+$/.test(id)) throw new Error(`Garmin activity id is invalid: ${value}`)
  return id
}

function fileSha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex')
}

export async function fetchWahooActivityFit(
  activity: ActivityBridgeWahooActivity,
  client: WahooCloudClient,
): Promise<ActivityBridgeFile> {
  const bytes = await client.downloadFit(activity.fitUrl)
  const sha256 = wahooFitSha256(bytes)
  if (sha256 !== activity.fitSha256.toLowerCase())
    throw new Error(`Wahoo FIT SHA-256 changed for ${activity.id}`)
  return {
    bytes,
    filename: `wahoo-${activity.workoutId}.fit`,
    sha256,
    kind: 'fit',
    sourceKind: 'fit',
  }
}

export async function fetchGarminActivityFile(
  activity: ActivityBridgeGarminActivity,
  session: GarminConnectSession,
  base: string,
): Promise<ActivityBridgeFile> {
  const sourceId = garminActivityId(activity.id)
  const archive = await fetchGarminBytes(
    session,
    base,
    `/download-service/files/activity/${encodeURIComponent(sourceId)}`,
  )
  const sourceFile = garminActivityFileFromArchive(archive)
  return {
    bytes: sourceFile.bytes,
    filename: `garmin-${sourceId}.${sourceFile.kind}`,
    sha256: fileSha256(sourceFile.bytes),
    kind: sourceFile.kind,
    sourceKind: sourceFile.kind,
  }
}

export async function fetchGarminActivityFit(
  activity: ActivityBridgeGarminActivity,
  session: GarminConnectSession,
  base: string,
): Promise<ActivityBridgeFile> {
  const source = await fetchGarminActivityFile(activity, session, base)
  if (source.kind === 'fit') return source
  const sourceId = garminActivityId(activity.id)
  const bytes = encodeGarminTcxActivityFit(source.bytes, sourceId).bytes
  return {
    bytes,
    filename: `garmin-${sourceId}.fit`,
    sha256: fileSha256(bytes),
    kind: 'fit',
    sourceKind: source.kind,
  }
}

function destinationWahooActivityId(upload: WahooWorkoutFileUpload): string | null {
  return upload.workoutId == null ? null : `wahoo:${upload.workoutId}`
}

function receiptFor(
  plan: ActivityBridgePlan,
  sourceFitSha256: string,
  upload: WahooWorkoutFileUpload | null,
  destinationActivityId: string | null,
  createdAt: number,
): ActivityBridgeReceipt {
  const wahooDirection = plan.direction === 'garmin-to-wahoo'
  return {
    direction: plan.direction,
    sourceProvider: wahooDirection ? 'garmin' : 'wahoo',
    sourceActivityId: plan.source.id,
    sourceFitSha256,
    destinationProvider: wahooDirection ? 'wahoo' : 'garmin',
    destinationActivityId,
    stravaActivityId: plan.stravaActivityId,
    uploadToken: upload?.token ?? null,
    uploadStatus: upload?.status ?? 'complete',
    createdAt,
    updatedAt: Date.now(),
  }
}

async function bridgeWahooToGarmin(
  plan: Extract<ActivityBridgePlan, { direction: 'wahoo-to-garmin' }>,
  wahooClient: WahooCloudClient,
  garminSession: GarminConnectSession,
  garminBase: string,
  ledger: ActivityBridgeLedger,
): Promise<ActivityBridgeExecution> {
  const file = await fetchWahooActivityFit(plan.source, wahooClient)
  const key = activityBridgeReceiptKey('wahoo', plan.source.id, file.sha256, 'garmin')
  const existing = ledger.receipts[key]
  if (existing && isTerminalActivityBridgeReceipt(existing))
    return { ledger, uploadStatus: existing.uploadStatus }
  const destinationId = await uploadGarminFit(garminSession, garminBase, file.filename, file.bytes)
  const receipt = receiptFor(plan, file.sha256, null, `connect:${destinationId}`, Date.now())
  const next = upsertActivityBridgeReceipt(ledger, receipt)
  await writeActivityBridgeLedgerAtomic(next)
  await updateGarminActivityTitle(garminSession, garminBase, destinationId, plan.title)
  return { ledger: next, uploadStatus: 'complete' }
}

async function bridgeGarminToWahoo(
  plan: Extract<ActivityBridgePlan, { direction: 'garmin-to-wahoo' }>,
  wahooClient: WahooCloudClient,
  garminSession: GarminConnectSession,
  garminBase: string,
  ledger: ActivityBridgeLedger,
): Promise<ActivityBridgeExecution> {
  const file = await fetchGarminActivityFit(plan.source, garminSession, garminBase)
  if (file.sourceKind === 'tcx')
    console.log(`[activity-bridge] converted Garmin TCX source=${plan.source.id} to FIT`)
  const key = activityBridgeReceiptKey('garmin', plan.source.id, file.sha256, 'wahoo')
  const existing = ledger.receipts[key]
  if (existing && isTerminalActivityBridgeReceipt(existing))
    return { ledger, uploadStatus: existing.uploadStatus }
  let upload: WahooWorkoutFileUpload
  const createdAt = existing?.createdAt ?? Date.now()
  if (existing?.uploadToken) {
    upload = await wahooClient.getWorkoutFileUpload(existing.uploadToken)
  } else {
    upload = await wahooClient.createWorkoutFileUpload({
      bytes: file.bytes,
      filename: file.filename,
      workoutName: plan.title,
    })
  }
  if (upload.status === 'error')
    throw new Error(`Wahoo workout file upload failed: ${upload.error ?? 'unknown error'}`)
  const destinationId = destinationWahooActivityId(upload)
  if (!destinationId && upload.status === 'complete')
    throw new Error(`Wahoo upload ${upload.token} completed without a destination workout id`)
  const receipt = receiptFor(plan, file.sha256, upload, destinationId, createdAt)
  const next = upsertActivityBridgeReceipt(ledger, receipt)
  await writeActivityBridgeLedgerAtomic(next)
  return { ledger: next, uploadStatus: upload.status }
}

function planSummary(plan: ActivityBridgePlan): string {
  return `${plan.direction} strava=${plan.stravaActivityId} source=${plan.source.id} title=${JSON.stringify(plan.title)}`
}

async function main(): Promise<void> {
  const args = parseActivityBridgeArgs(process.argv.slice(2))
  const [inputs, initialLedger] = await Promise.all([
    readActivityBridgeInputs(),
    readActivityBridgeLedger(),
  ])
  const planned = planActivityBridge(inputs, initialLedger)
  const plans = args.limit == null ? planned : planned.slice(0, args.limit)
  for (const plan of plans) console.log(`[activity-bridge] ${planSummary(plan)}`)
  if (!args.write) {
    console.log(`[activity-bridge] dry run: ${plans.length} upload${plans.length === 1 ? '' : 's'}`)
    return
  }
  if (plans.length === 0) {
    console.log('[activity-bridge] no uploads needed')
    return
  }
  const wahooClient = await wahooCloudClientFromEnv()
  const garminSession = await readGarminConnectSession()
  const garminBase = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_BASE_URL?.trim() || DEFAULT_GARMIN_CONNECT_BASE,
  )
  let ledger = initialLedger
  for (const plan of plans) {
    const execution =
      plan.direction === 'wahoo-to-garmin'
        ? await bridgeWahooToGarmin(plan, wahooClient, garminSession, garminBase, ledger)
        : await bridgeGarminToWahoo(plan, wahooClient, garminSession, garminBase, ledger)
    ledger = execution.ledger
    console.log(
      `[activity-bridge] ${execution.uploadStatus === 'complete' || execution.uploadStatus === 'duplicate' ? 'completed' : `queued status=${execution.uploadStatus}`} ${planSummary(plan)}`,
    )
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => {
    console.error(
      `[activity-bridge] sync failed: ${error instanceof Error ? error.message : error}`,
    )
    process.exit(1)
  })
}
