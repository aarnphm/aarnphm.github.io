import fs from 'node:fs/promises'
import type {
  GarminActivity,
  GarminCache,
  GarminRunningLactateThreshold,
} from '../../plugins/stores/garmin'
import { localIsoDayOffset, shiftIsoDay } from '../../util/local-date'
import { joinSegments, QUARTZ } from '../../util/path'
import { isRecord, type UnknownRecord } from '../../util/type-guards'

export const CACHE_VERSION = 14
export const SWIM_CACHE_VERSION = 13
const DEFAULT_PAGE_SIZE = 100
const DEFAULT_DELAY_MS = 1200
const TRIATHLON_PAGE = joinSegments(QUARTZ, '..', 'content', 'triathlon.md')
export const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')

export interface GarminFetchOutcome<T> {
  ok: boolean
  value?: T
}

export function resolveGarminFetch<T>(
  outcome: GarminFetchOutcome<T>,
  previous: T | undefined,
): T | undefined {
  return outcome.ok ? outcome.value : previous
}

export function garminActivityDay(activity: GarminActivity): string | null {
  const day = (activity.startDateLocal || activity.startDate).slice(0, 10)
  return /^\d{4}-\d{2}-\d{2}$/.test(day) ? day : null
}

function garminActivityDateBounds(activities: Readonly<Record<string, GarminActivity>>): {
  earliest: string | null
  latest: string | null
} {
  let earliest: string | null = null
  let latest: string | null = null
  for (const activity of Object.values(activities)) {
    const day = garminActivityDay(activity)
    if (!day) continue
    if (earliest == null || day < earliest) earliest = day
    if (latest == null || day > latest) latest = day
  }
  return { earliest, latest }
}

export function garminRefreshStart(
  previous: GarminCache | null,
  configuredStart: string,
  refreshWindowDays: number,
): string {
  const { earliest, latest } = garminActivityDateBounds(previous?.activities ?? {})
  if (!previous || (previous.version ?? 0) < CACHE_VERSION)
    return earliest != null && earliest < configuredStart ? earliest : configuredStart
  if (latest == null) return configuredStart
  const overlapStart = shiftIsoDay(latest, -refreshWindowDays)
  return overlapStart > configuredStart ? overlapStart : configuredStart
}

function sortedIsoDays(record: Readonly<Record<string, unknown>> | undefined): string[] {
  return Object.keys(record ?? {})
    .filter(day => /^\d{4}-\d{2}-\d{2}$/.test(day))
    .sort()
}

function hasGarminHealthError(day: UnknownRecord | undefined): boolean {
  if (!day) return true
  for (const key of [
    'bodyBattery',
    'trainingReadiness',
    'trainingStatus',
    'enduranceScore',
    'hillScore',
  ]) {
    const reading = day[key]
    if (isRecord(reading) && reading.status === 'error') return true
  }
  return false
}

export function garminHealthRefreshStart(
  previous: GarminCache | null,
  configuredStart: string,
  end: string,
  refreshWindowDays: number,
): string {
  const days = sortedIsoDays(previous?.health)
  if (!previous || (previous.version ?? 0) < CACHE_VERSION)
    return days[0] != null && days[0] < configuredStart ? days[0] : configuredStart
  if (days.length === 0) return configuredStart
  const earliest = days[0] > configuredStart ? days[0] : configuredStart
  const latest = days[days.length - 1]
  const windowStart = shiftIsoDay(end, -refreshWindowDays)
  const overlapStart = shiftIsoDay(latest, -refreshWindowDays)
  let start = overlapStart < windowStart ? overlapStart : windowStart
  for (let day = earliest; day <= latest; day = shiftIsoDay(day, 1)) {
    const entry = previous.health?.[day]
    if (!entry || hasGarminHealthError(entry as unknown as UnknownRecord)) {
      if (day < start) start = day
      break
    }
  }
  const bounded = start > configuredStart ? start : configuredStart
  return bounded <= end ? bounded : end
}

export function garminSleepRefreshStart(
  previous: GarminCache | null,
  configuredStart: string,
  end: string,
  refreshWindowDays: number,
): string {
  const days = sortedIsoDays(previous?.sleep)
  if (!previous || (previous.version ?? 0) < CACHE_VERSION)
    return days[0] != null && days[0] < configuredStart ? days[0] : configuredStart
  if (days.length === 0 && !previous.sleepLastSync) return configuredStart
  const windowStart = shiftIsoDay(end, -refreshWindowDays)
  const anchorDay =
    days[days.length - 1] ??
    (previous.sleepLastSync ? localIsoDayOffset(0, previous.sleepLastSync) : end)
  const overlapStart = shiftIsoDay(anchorDay, -refreshWindowDays)
  let start = overlapStart < windowStart ? overlapStart : windowStart
  if (sortedIsoDays(previous.health).length > 0) {
    const healthGapStart = garminHealthRefreshStart(
      previous,
      configuredStart,
      end,
      refreshWindowDays,
    )
    if (healthGapStart < start) start = healthGapStart
  }
  const bounded = start > configuredStart ? start : configuredStart
  return bounded <= end ? bounded : end
}

export function garminLactateThresholdRefreshStart(
  previous: GarminRunningLactateThreshold | null | undefined,
  configuredStart: string,
  end: string,
  refreshWindowDays: number,
  stale = false,
): string {
  if (stale) return configuredStart
  const latestSpeed = previous?.history?.speedMps.at(-1)?.date
  const latestHr = previous?.history?.heartRateBpm.at(-1)?.date
  if (!latestSpeed || !latestHr) return configuredStart
  const anchor = latestSpeed < latestHr ? latestSpeed : latestHr
  const windowStart = shiftIsoDay(end, -refreshWindowDays)
  const overlapStart = shiftIsoDay(anchor, -refreshWindowDays)
  const start = overlapStart < windowStart ? overlapStart : windowStart
  const bounded = start > configuredStart ? start : configuredStart
  return bounded <= end ? bounded : end
}

export async function readCache(): Promise<GarminCache | null> {
  try {
    return JSON.parse(await fs.readFile(cacheFile, 'utf8')) as GarminCache
  } catch {
    return null
  }
}

function envNumber(name: string, fallback: number): number {
  const value = process.env[name]
  if (!value?.trim()) return fallback
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`${name} must be nonnegative`)
  return parsed
}

function envFlag(name: string, fallback: boolean): boolean {
  const value = process.env[name]?.trim()
  if (!value) return fallback
  if (value === '0' || value.toLowerCase() === 'false') return false
  if (value === '1' || value.toLowerCase() === 'true') return true
  throw new Error(`${name} must be true/false or 1/0`)
}

export function cleanDay(value: string | undefined): string | null {
  if (!value?.trim()) return null
  const day = value.trim()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`${value} is not YYYY-MM-DD`)
  return day
}

export async function readTriathlonStart(): Promise<string | null> {
  try {
    const content = await fs.readFile(TRIATHLON_PAGE, 'utf8')
    const match = /^strava:\s*['"]?(\d{4}-\d{2}-\d{2})['"]?\s*$/m.exec(content)
    return match?.[1] ?? null
  } catch {
    return null
  }
}

export function endDate(): string {
  return cleanDay(process.env.GARMIN_CONNECT_END_DATE) ?? localIsoDayOffset(0)
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export async function prepareGarminRefreshWindows(
  previous: GarminCache | null,
  refreshWindowDays: number,
): Promise<{
  pageSize: number
  delayMs: number
  maxActivities: number
  fetchStreams: boolean
  start: string
  end: string
  healthStart: string
  sleepStart: string
}> {
  const pageSize = Math.max(1, envNumber('GARMIN_CONNECT_PAGE_SIZE', DEFAULT_PAGE_SIZE))
  const delayMs = envNumber('GARMIN_CONNECT_DELAY_MS', DEFAULT_DELAY_MS)
  const maxActivities = envNumber('GARMIN_CONNECT_MAX_ACTIVITIES', 0)
  const fetchStreams = envFlag('GARMIN_CONNECT_FETCH_STREAMS', true)
  const explicitStart =
    cleanDay(process.env.GARMIN_CONNECT_START_DATE) ?? cleanDay(process.env.GARMIN_CONNECT_SINCE)
  const configuredStart = (await readTriathlonStart()) ?? localIsoDayOffset(-90)
  const start = explicitStart ?? garminRefreshStart(previous, configuredStart, refreshWindowDays)
  const end = endDate()
  if (start > end) throw new Error(`Garmin sync start ${start} is after end ${end}`)
  const healthStart =
    explicitStart ?? garminHealthRefreshStart(previous, configuredStart, end, refreshWindowDays)
  const sleepStart =
    explicitStart ?? garminSleepRefreshStart(previous, configuredStart, end, refreshWindowDays)
  return { pageSize, delayMs, maxActivities, fetchStreams, start, end, healthStart, sleepStart }
}
