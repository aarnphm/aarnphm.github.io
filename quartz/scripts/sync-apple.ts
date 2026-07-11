import { createReadStream } from 'node:fs'
import fs from 'node:fs/promises'
import { homedir } from 'node:os'
import { resolve } from 'node:path'
import readline from 'node:readline'
import { pathToFileURL } from 'node:url'
import {
  aggregateAppleRecords,
  aggregateSwimLaps,
  AppleSwimDistanceRecord,
  AppleSwimLap,
  AppleSwimStrokeRecord,
  AppleSwimWorkoutWindow,
  AppleCache,
  AppleDaily,
  AppleWorkout,
  AppleRecord,
  AppleSwim,
  latestAppleDate,
  matchAppleRecord,
  matchStrokeStyle,
  matchSwimDistance,
  matchSwimStrokeOpen,
  matchSwimWorkout,
  mergeAppleDay,
  parseAppleJson,
} from '../plugins/stores/apple'
import { joinSegments, QUARTZ } from '../util/path'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'

const CACHE_VERSION = 8
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'apple-health.json')
const healthExporterICloudContainer = 'iCloud~xyz~aarnphm~healthexporter'
const healthExporterImportFile = 'apple-health-import.json'
const healthExporterICloudDriveFolder = 'HealthExporter'

function unquotePath(path: string): string {
  const trimmed = path.trim()
  if (trimmed.length < 2) return trimmed
  const first = trimmed[0]
  const last = trimmed[trimmed.length - 1]
  return (first === '"' && last === '"') || (first === "'" && last === "'")
    ? trimmed.slice(1, -1)
    : trimmed
}

export function healthExporterICloudPath(home: string): string {
  return joinSegments(
    home,
    'Library',
    'Mobile Documents',
    healthExporterICloudContainer,
    'Documents',
    healthExporterImportFile,
  )
}

function healthExporterCloudDocsPath(home: string): string {
  return joinSegments(
    home,
    'Library',
    'Mobile Documents',
    'com~apple~CloudDocs',
    healthExporterICloudDriveFolder,
    healthExporterImportFile,
  )
}

export function expandAppleHealthPath(path: string, home = homedir()): string {
  const normalized = unquotePath(path)
  if (normalized === '~' || normalized === '$HOME' || normalized === '${HOME}') return home
  if (
    normalized ===
    joinSegments('iCloud Drive', healthExporterICloudDriveFolder, healthExporterImportFile)
  )
    return healthExporterICloudPath(home)
  if (normalized.startsWith('~/')) return joinSegments(home, normalized.slice(2))
  if (normalized.startsWith('$HOME/')) return joinSegments(home, normalized.slice(6))
  if (normalized.startsWith('${HOME}/')) return joinSegments(home, normalized.slice(8))
  return normalized
}

export function appleImportCandidates(envFile: string | undefined, home = homedir()): string[] {
  if (envFile?.trim()) return [expandAppleHealthPath(envFile, home)]
  return [
    healthExporterICloudPath(home),
    healthExporterCloudDocsPath(home),
    joinSegments(QUARTZ, '.quartz-cache', healthExporterImportFile),
    joinSegments(QUARTZ, '.quartz-cache', 'apple-health-import.xml'),
  ]
}

async function readCache(): Promise<AppleCache | null> {
  try {
    return JSON.parse(await fs.readFile(cacheFile, 'utf8')) as AppleCache
  } catch {
    return null
  }
}

interface AppleEntries {
  days: AppleDaily[]
  swims: AppleSwim[]
  workouts: AppleWorkout[]
}

async function parseXmlFile(path: string): Promise<AppleEntries> {
  const records: AppleRecord[] = []
  const distanceLaps: AppleSwimDistanceRecord[] = []
  const strokeLaps: AppleSwimLap[] = []
  const swimWorkouts: AppleSwimWorkoutWindow[] = []
  let pendingStroke: AppleSwimStrokeRecord | null = null
  const rl = readline.createInterface({ input: createReadStream(path), crlfDelay: Infinity })
  for await (const line of rl) {
    const r = matchAppleRecord(line)
    if (r) records.push(r)
    const dist = matchSwimDistance(line)
    if (dist) distanceLaps.push(dist)
    const workout = matchSwimWorkout(line)
    if (workout) swimWorkouts.push(workout)
    if (pendingStroke) {
      const stroke = matchStrokeStyle(line)
      if (stroke) {
        strokeLaps.push({ ...pendingStroke, stroke })
        pendingStroke = null
      } else if (line.includes('</Record>')) {
        strokeLaps.push({ ...pendingStroke, stroke: null })
        pendingStroke = null
      }
    } else {
      const open = matchSwimStrokeOpen(line)
      if (open) {
        if (line.trimEnd().endsWith('/>')) strokeLaps.push({ ...open, stroke: null })
        else pendingStroke = open
      }
    }
  }
  return {
    days: aggregateAppleRecords(records),
    swims: aggregateSwimLaps(strokeLaps, distanceLaps, swimWorkouts),
    workouts: [],
  }
}

async function loadEntries(path: string): Promise<AppleEntries> {
  if (path.endsWith('.json')) return parseAppleJson(JSON.parse(await fs.readFile(path, 'utf8')))
  return parseXmlFile(path)
}

export function mergeAppleSwims(
  current: Record<string, AppleSwim>,
  incoming: AppleSwim[],
): Record<string, AppleSwim> {
  const merged = { ...current }
  const stableDates = new Set(incoming.filter(swim => swim.id).map(swim => swim.date))
  if (stableDates.size > 0) {
    for (const [key, swim] of Object.entries(merged)) {
      if (!swim.id && stableDates.has(swim.date)) delete merged[key]
    }
  }
  for (const swim of incoming) {
    if (
      !swim.id &&
      Object.values(merged).some(existing => existing.id && existing.date === swim.date)
    ) {
      continue
    }
    merged[swim.id ?? swim.date] = swim
  }
  return merged
}

async function main(): Promise<void> {
  const prev = await readCache()
  const days: Record<string, AppleDaily> = { ...prev?.days }
  let swims: Record<string, AppleSwim> = { ...prev?.swims }
  const workouts: Record<string, AppleWorkout> = { ...prev?.workouts }

  const candidates = appleImportCandidates(process.env.APPLE_HEALTH_FILE)

  let touched = 0
  let read = 0
  for (const path of candidates) {
    try {
      await fs.stat(path)
    } catch {
      continue
    }
    read += 1
    const entries = await loadEntries(path)
    for (const e of entries.days) {
      days[e.date] = mergeAppleDay(days[e.date], e)
      touched += 1
    }
    swims = mergeAppleSwims(swims, entries.swims)
    for (const workout of entries.workouts) workouts[workout.id] = workout
    console.log(
      `[apple] read ${entries.days.length} days, ${entries.swims.length} swims, ${entries.workouts.length} workouts from ${path}`,
    )
  }

  if (read === 0) {
    console.log(
      `[apple] no import found. export HealthExporter to iCloud Drive/${healthExporterICloudDriveFolder}/${healthExporterImportFile} (${healthExporterICloudPath(homedir())}), set APPLE_HEALTH_FILE=<export.xml|day.json>, or drop one at quartz/.quartz-cache/apple-health-import.{json,xml}`,
    )
    if (!prev) return
    const latest = latestAppleDate(prev.days)
    const lastSync = new Date(prev.lastSync).toISOString()
    console.log(`[apple] keeping previous cache (${lastSync}, latest day ${latest ?? 'none'})`)
    return
  }

  const cache: AppleCache = { version: CACHE_VERSION, lastSync: Date.now(), days, swims, workouts }
  await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  await refreshTriathlonRouteSource()
  console.log(
    `[apple] merged ${touched} day-entries → ${Object.keys(days).length} days, ${Object.keys(workouts).length} workouts → ${cacheFile}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[apple] sync failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
