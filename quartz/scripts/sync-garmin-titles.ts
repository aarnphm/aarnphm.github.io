import { randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import type { GarminCache } from '../plugins/stores/garmin'
import {
  ACTIVITY_KINDS,
  isActivityKind,
  type ActivityKind,
  type StravaRawCache,
} from '../plugins/stores/strava'
import {
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  readGarminConnectSession,
} from '../util/garmin-session'
import {
  selectGarminActivityTypeUpdates,
  selectGarminTitleUpdates,
  type GarminActivityTypeUpdate,
  type GarminTitleUpdate,
} from '../util/garmin-title-sync'
import {
  GARMIN_POOL_SWIM_ACTIVITY_TYPE,
  updateGarminActivityTitle,
  updateGarminActivityType,
} from '../util/garmin-title-update'
import { joinSegments, QUARTZ } from '../util/path'

const CACHE_DIR = joinSegments(QUARTZ, '.quartz-cache')
const STRAVA_CACHE = joinSegments(CACHE_DIR, 'strava.json')
const GARMIN_CACHE = joinSegments(CACHE_DIR, 'garmin.json')
const DEFAULT_DELAY_MS = 1200

interface Args {
  write: boolean
  kind: ActivityKind | undefined
  type: ActivityTypeTarget | null
  since: string | null
  limit: number
  ids: Set<string>
  delayMs: number
}

type ActivityTypeTarget = 'pool-swim'

function usage(): string {
  return [
    `usage: pnpm garmin:sync-titles -- [--write] [--kind all|${ACTIVITY_KINDS.join('|')}|cardio] [--type pool-swim] [--since YYYY-MM-DD] [--limit N] [--id STRAVA_ID]`,
    '',
    'defaults to a dry-run of all activity titles. --write renames matched Garmin activities to their Strava names or updates a requested activity type. Cardio uses the strength/workout group.',
  ].join('\n')
}

export function parseGarminTitleArgs(argv: string[]): Args {
  const args: Args = {
    write: false,
    kind: undefined,
    type: null,
    since: null,
    limit: 0,
    ids: new Set(),
    delayMs: envNumber('GARMIN_TITLE_SYNC_DELAY_MS', DEFAULT_DELAY_MS),
  }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--') continue
    if (arg === '--write') args.write = true
    else if (arg === '--dry-run') args.write = false
    else if (arg === '--kind' || arg === '--sport')
      args.kind = parseKind(readArgValue(argv, ++i, arg))
    else if (arg === '--type') args.type = parseType(readArgValue(argv, ++i, arg))
    else if (arg === '--since') args.since = readArgValue(argv, ++i, arg)
    else if (arg === '--limit') args.limit = positiveInteger(readArgValue(argv, ++i, arg), arg)
    else if (arg === '--id') args.ids.add(readArgValue(argv, ++i, arg))
    else if (arg === '--ids') {
      for (const id of readArgValue(argv, ++i, arg).split(','))
        if (id.trim()) args.ids.add(id.trim())
    } else if (arg === '--delay-ms')
      args.delayMs = nonnegativeInteger(readArgValue(argv, ++i, arg), arg)
    else if (arg === '--help' || arg === '-h') {
      console.log(usage())
      process.exit(0)
    } else throw new Error(`unknown argument ${arg}\n${usage()}`)
  }
  if (args.since && !/^\d{4}-\d{2}-\d{2}$/.test(args.since))
    throw new Error(`--since must be YYYY-MM-DD, got ${args.since}`)
  return args
}

function parseType(value: string): ActivityTypeTarget {
  if (value === 'pool-swim') return value
  throw new Error(`--type must be pool-swim, got ${value}`)
}

function parseKind(value: string): ActivityKind | undefined {
  if (value === 'all') return undefined
  if (value === 'cardio') return 'strength'
  if (isActivityKind(value)) return value
  throw new Error(`--kind must be all, ${ACTIVITY_KINDS.join(', ')}, or cardio, got ${value}`)
}

function readArgValue(argv: string[], index: number, flag: string): string {
  const value = argv[index]
  if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`)
  return value
}

function envNumber(name: string, fallback: number): number {
  const value = process.env[name]
  if (!value?.trim()) return fallback
  return nonnegativeInteger(value, name)
}

function nonnegativeInteger(value: string, name: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0)
    throw new Error(`${name} must be a nonnegative integer`)
  return parsed
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed <= 0)
    throw new Error(`${name} must be a positive integer`)
  return parsed
}

async function readJsonFile<T>(path: string): Promise<T> {
  return JSON.parse(await fs.readFile(path, 'utf8')) as T
}

function describe(update: GarminTitleUpdate): string {
  return `${update.startDateLocal || update.startDate} | ${update.from} -> ${update.to} | strava ${update.stravaId} | garmin ${update.garminActivityId}`
}

function describeType(update: GarminActivityTypeUpdate): string {
  return `${update.startDateLocal || update.startDate} | ${update.from ?? 'unknown'} -> ${update.to} | ${update.title} | strava ${update.stravaId} | garmin ${update.garminActivityId}`
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export async function persistGarminTitleUpdates(
  updates: readonly GarminTitleUpdate[],
  path = GARMIN_CACHE,
): Promise<void> {
  if (updates.length === 0) return
  const cache = await readJsonFile<GarminCache>(path)
  const titles = new Map(updates.map(update => [update.garminId, update.to]))
  for (const activity of Object.values(cache.activities)) {
    const title = titles.get(activity.id)
    if (title != null) activity.name = title
  }
  const temporary = `${path}.${randomUUID()}.tmp`
  try {
    await fs.writeFile(temporary, JSON.stringify(cache, null, 2))
    await fs.rename(temporary, path)
  } finally {
    await fs.rm(temporary, { force: true })
  }
}

async function syncPoolSwimTypes(args: Args): Promise<void> {
  const strava = await readJsonFile<StravaRawCache>(STRAVA_CACHE)
  const garmin = await readJsonFile<GarminCache>(GARMIN_CACHE)
  const updates = selectGarminActivityTypeUpdates(strava, garmin, {
    kind: args.kind,
    since: args.since,
    limit: args.limit,
    ids: args.ids,
  })
  console.log(
    `[garmin-type] ${args.write ? 'write' : 'dry-run'} ${updates.length} candidate ${args.kind ?? 'swim'} ${args.type} types${args.since ? ` since ${args.since}` : ''}`,
  )
  for (const update of updates) console.log(`[garmin-type] candidate ${describeType(update)}`)
  if (!args.write || updates.length === 0) return

  const session = await readGarminConnectSession()
  const base = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_TITLE_BASE_URL?.trim() || DEFAULT_GARMIN_CONNECT_BASE,
  )

  let updated = 0
  for (const update of updates) {
    await updateGarminActivityType(
      session,
      base,
      update.garminActivityId,
      update.title,
      GARMIN_POOL_SWIM_ACTIVITY_TYPE,
    )
    updated++
    console.log(`[garmin-type] updated ${update.garminActivityId} -> ${update.to}`)
    if (args.delayMs > 0) await sleep(args.delayMs)
  }
  console.log(`[garmin-type] done updated=${updated}`)
}

async function main(): Promise<void> {
  const args = parseGarminTitleArgs(process.argv.slice(2))
  if (args.type) {
    await syncPoolSwimTypes(args)
    return
  }
  const strava = await readJsonFile<StravaRawCache>(STRAVA_CACHE)
  const garmin = await readJsonFile<GarminCache>(GARMIN_CACHE)
  const updates = selectGarminTitleUpdates(strava, garmin, {
    kind: args.kind,
    since: args.since,
    limit: args.limit,
    ids: args.ids,
  })
  console.log(
    `[garmin-title] ${args.write ? 'write' : 'dry-run'} ${updates.length} candidate ${args.kind ?? 'all-activity'} titles${args.since ? ` since ${args.since}` : ''}`,
  )
  for (const update of updates) console.log(`[garmin-title] candidate ${describe(update)}`)
  if (!args.write || updates.length === 0) return

  const session = await readGarminConnectSession()
  const base = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_TITLE_BASE_URL?.trim() || DEFAULT_GARMIN_CONNECT_BASE,
  )

  const updated: GarminTitleUpdate[] = []
  try {
    for (const update of updates) {
      await updateGarminActivityTitle(session, base, update.garminActivityId, update.to)
      updated.push(update)
      console.log(`[garmin-title] updated ${update.garminActivityId} -> ${update.to}`)
      if (args.delayMs > 0) await sleep(args.delayMs)
    }
  } finally {
    await persistGarminTitleUpdates(updated)
  }
  console.log(`[garmin-title] done updated=${updated.length}`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[garmin-title] failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
