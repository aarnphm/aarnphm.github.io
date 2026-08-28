import fs from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import type { RawStravaActivity } from '../plugins/stores/strava'
import {
  parseWahooCache,
  selectWahooTitleUpdates,
  type WahooTitleUpdate,
  type WahooTitleStravaCache,
} from '../plugins/stores/wahoo'
import { joinSegments, QUARTZ } from '../util/path'
import { isRecord, type UnknownRecord } from '../util/type-guards'
import { wahooCloudClientFromEnv } from '../util/wahoo-cloud'
import { WAHOO_CACHE_FILE, writeWahooCache } from './sync-wahoo'

const STRAVA_CACHE_FILE = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
const DEFAULT_DELAY_MS = 1200

interface Args {
  write: boolean
  since: string | null
  limit: number
  ids: Set<string>
  delayMs: number
  includeEdited: boolean
}

function usage(): string {
  return [
    'usage: pnpm wahoo:sync-titles -- [--write] [--since YYYY-MM-DD] [--limit N] [--id STRAVA_ID] [--include-edited]',
    '',
    'defaults to a dry-run of cycling titles matched by start, distance, and duration.',
  ].join('\n')
}

function nonnegativeInteger(value: string, label: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0) throw new Error(`${label} must be nonnegative`)
  return parsed
}

function positiveInteger(value: string, label: string): number {
  const parsed = nonnegativeInteger(value, label)
  if (parsed === 0) throw new Error(`${label} must be positive`)
  return parsed
}

function argValue(argv: string[], index: number, flag: string): string {
  const value = argv[index]
  if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`)
  return value
}

export function parseWahooTitleArgs(argv: string[]): Args {
  const args: Args = {
    write: false,
    since: null,
    limit: 0,
    ids: new Set(),
    delayMs: process.env.WAHOO_TITLE_SYNC_DELAY_MS
      ? nonnegativeInteger(process.env.WAHOO_TITLE_SYNC_DELAY_MS, 'WAHOO_TITLE_SYNC_DELAY_MS')
      : DEFAULT_DELAY_MS,
    includeEdited: false,
  }
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index]
    if (arg === '--') continue
    if (arg === '--write') args.write = true
    else if (arg === '--dry-run') args.write = false
    else if (arg === '--include-edited') args.includeEdited = true
    else if (arg === '--since') args.since = argValue(argv, ++index, arg)
    else if (arg === '--limit') args.limit = positiveInteger(argValue(argv, ++index, arg), arg)
    else if (arg === '--delay-ms')
      args.delayMs = nonnegativeInteger(argValue(argv, ++index, arg), arg)
    else if (arg === '--id') args.ids.add(argValue(argv, ++index, arg))
    else if (arg === '--ids') {
      for (const id of argValue(argv, ++index, arg).split(','))
        if (id.trim()) args.ids.add(id.trim())
    } else if (arg === '--help' || arg === '-h') {
      console.log(usage())
      return args
    } else throw new Error(`unknown argument ${arg}\n${usage()}`)
  }
  if (args.since && !/^\d{4}-\d{2}-\d{2}$/.test(args.since))
    throw new Error(`--since must be YYYY-MM-DD, got ${args.since}`)
  return args
}

function requiredNumber(record: UnknownRecord, key: string, label: string): number {
  const value = record[key]
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error(`${label}.${key} must be finite`)
  return value
}

function optionalNumber(record: UnknownRecord, key: string, label: string): number | undefined {
  if (record[key] == null) return undefined
  return requiredNumber(record, key, label)
}

function requiredString(record: UnknownRecord, key: string, label: string): string {
  const value = record[key]
  if (typeof value !== 'string') throw new Error(`${label}.${key} must be a string`)
  return value
}

function optionalBoolean(record: UnknownRecord, key: string, label: string): boolean | undefined {
  const value = record[key]
  if (value == null) return undefined
  if (typeof value !== 'boolean') throw new Error(`${label}.${key} must be a boolean`)
  return value
}

function parseStravaActivity(value: unknown, label: string): RawStravaActivity {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return {
    id: requiredNumber(value, 'id', label),
    name: requiredString(value, 'name', label),
    sportType: requiredString(value, 'sportType', label),
    distance: requiredNumber(value, 'distance', label),
    movingTime: requiredNumber(value, 'movingTime', label),
    elapsedTime: requiredNumber(value, 'elapsedTime', label),
    totalElevationGain: requiredNumber(value, 'totalElevationGain', label),
    startDate: requiredString(value, 'startDate', label),
    startDateLocal: requiredString(value, 'startDateLocal', label),
    averageSpeed: requiredNumber(value, 'averageSpeed', label),
    averageHeartrate: optionalNumber(value, 'averageHeartrate', label),
    maxHeartrate: optionalNumber(value, 'maxHeartrate', label),
    averageWatts: optionalNumber(value, 'averageWatts', label),
    weightedAverageWatts: optionalNumber(value, 'weightedAverageWatts', label),
    maxWatts: optionalNumber(value, 'maxWatts', label),
    kilojoules: optionalNumber(value, 'kilojoules', label),
    deviceWatts: optionalBoolean(value, 'deviceWatts', label),
    averageCadence: optionalNumber(value, 'averageCadence', label),
    sufferScore: optionalNumber(value, 'sufferScore', label),
    averageTemp: optionalNumber(value, 'averageTemp', label),
    calories: optionalNumber(value, 'calories', label),
  }
}

export function parseWahooTitleStravaCache(value: unknown): WahooTitleStravaCache {
  if (!isRecord(value) || !isRecord(value.activities))
    throw new Error('Strava cache must contain an activities object')
  const activities: Record<string, RawStravaActivity> = {}
  for (const [id, activity] of Object.entries(value.activities))
    activities[id] = parseStravaActivity(activity, `Strava cache.activities.${id}`)
  return { activities }
}

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await fs.readFile(path, 'utf8'))
}

function describe(update: WahooTitleUpdate): string {
  return `${update.startDateLocal || update.startDate} | ${update.from || '(untitled)'} -> ${update.to} | strava ${update.stravaId} | wahoo ${update.wahooWorkoutId}`
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolveSleep => setTimeout(resolveSleep, ms))
}

async function main(): Promise<void> {
  const args = parseWahooTitleArgs(process.argv.slice(2))
  const strava = parseWahooTitleStravaCache(await readJson(STRAVA_CACHE_FILE))
  const wahoo = parseWahooCache(await readJson(WAHOO_CACHE_FILE))
  const updates = selectWahooTitleUpdates(strava, wahoo, {
    since: args.since,
    limit: args.limit,
    ids: args.ids,
    includeEdited: args.includeEdited,
  })
  console.log(
    `[wahoo-title] ${args.write ? 'write' : 'dry-run'} ${updates.length} cycling title candidates${args.since ? ` since ${args.since}` : ''}`,
  )
  for (const update of updates) console.log(`[wahoo-title] candidate ${describe(update)}`)
  if (!args.write || updates.length === 0) return

  const client = await wahooCloudClientFromEnv()
  for (const [index, update] of updates.entries()) {
    const result = await client.updateWorkoutName(update.wahooWorkoutId, update.to)
    if (result.name !== update.to)
      throw new Error(
        `Wahoo title update ${update.wahooWorkoutId} returned ${result.name ?? 'no name'}`,
      )
    const activity = wahoo.activities[update.wahooId]
    if (!activity) throw new Error(`Wahoo cache lost activity ${update.wahooId}`)
    activity.name = update.to
    console.log(`[wahoo-title] updated ${update.wahooWorkoutId} -> ${update.to}`)
    if (args.delayMs > 0 && index + 1 < updates.length) await sleep(args.delayMs)
  }
  await writeWahooCache(wahoo)
  console.log(`[wahoo-title] done updated=${updates.length}`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => {
    console.error(`[wahoo-title] failed: ${error instanceof Error ? error.message : error}`)
    process.exit(1)
  })
}
