import fs from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import type {
  GarminActivity,
  GarminCache,
  GarminClimbSegment,
  GarminStreams,
  GarminVo2Day,
  GarminWeightSample,
} from '../plugins/stores/garmin'
import {
  garminConnectActivities,
  garminConnectActivity,
  garminConnectClimbSegments,
  garminConnectStreams,
  garminConnectVo2,
  garminConnectWeightSamples,
  type GarminConnectActivityListItem,
} from '../util/garmin-connect'
import {
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  fetchGarminJson,
  readGarminConnectSession,
  type GarminConnectSession,
} from '../util/garmin-session'
import { localDayEndUtcMs, localDayStartUtcMs, localIsoDayOffset } from '../util/local-date'
import { joinSegments, QUARTZ } from '../util/path'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord, type UnknownRecord } from '../util/type-guards'

const CACHE_VERSION = 4
const DEFAULT_PAGE_SIZE = 100
const DEFAULT_DELAY_MS = 1200
const TRIATHLON_PAGE = joinSegments(QUARTZ, '..', 'content', 'triathlon.md')
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'garmin.json')

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

export function resolveGarminWeightDay(
  day: string,
  outcome: GarminFetchOutcome<GarminWeightSample[]>,
  summary: GarminWeightSample,
  previous: GarminWeightSample[],
): GarminWeightSample[] {
  if (outcome.ok) return outcome.value?.length ? outcome.value : [summary]
  const prior = previous.filter(sample => sample.date === day)
  return prior.length ? prior : [summary]
}

export function initialGarminSyncRecords<T>(
  previous: Record<string, T> | undefined,
  partial: boolean,
): Record<string, T> {
  return partial ? { ...previous } : {}
}

async function readCache(): Promise<GarminCache | null> {
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

function cleanDay(value: string | undefined): string | null {
  if (!value?.trim()) return null
  const day = value.trim()
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`${value} is not YYYY-MM-DD`)
  return day
}

async function readTriathlonStart(): Promise<string | null> {
  try {
    const content = await fs.readFile(TRIATHLON_PAGE, 'utf8')
    const match = /^strava:\s*['"]?(\d{4}-\d{2}-\d{2})['"]?\s*$/m.exec(content)
    return match?.[1] ?? null
  } catch {
    return null
  }
}

async function startDate(): Promise<string> {
  return (
    cleanDay(process.env.GARMIN_CONNECT_START_DATE) ??
    cleanDay(process.env.GARMIN_CONNECT_SINCE) ??
    (await readTriathlonStart()) ??
    localIsoDayOffset(-90)
  )
}

function endDate(): string {
  return cleanDay(process.env.GARMIN_CONNECT_END_DATE) ?? localIsoDayOffset(0)
}

function activityStartMs(item: GarminConnectActivityListItem): number | null {
  const start = garminConnectActivity(null, item.record, 0)?.startDate
  if (!start) return null
  const ms = Date.parse(start)
  return Number.isFinite(ms) ? ms : null
}

async function fetchActivities(
  session: GarminConnectSession,
  base: string,
  start: string,
  end: string,
  pageSize: number,
  maxActivities: number,
): Promise<GarminConnectActivityListItem[]> {
  const out: GarminConnectActivityListItem[] = []
  const seen = new Set<string>()
  const startMs = localDayStartUtcMs(start)
  const endMs = localDayEndUtcMs(end)
  for (let offset = 0; ; offset += pageSize) {
    const raw = await fetchGarminJson(session, base, '/graphql-gateway/graphql', undefined, {
      method: 'POST',
      body: JSON.stringify({
        query: `query{searchActivitiesScalar(start:${offset}, limit:${pageSize}, excludedActivitySubTypes:["assistance"])}`,
      }),
    })
    const page = garminConnectActivities(raw)
    let oldestMs: number | null = null
    for (const item of page) {
      if (seen.has(item.id)) continue
      seen.add(item.id)
      const ms = activityStartMs(item)
      if (ms != null) {
        oldestMs = oldestMs == null ? ms : Math.min(oldestMs, ms)
        if (ms < startMs || ms > endMs) continue
      }
      out.push(item)
      if (maxActivities > 0 && out.length >= maxActivities) return out
    }
    if (page.length < pageSize) return out
    if (oldestMs != null && oldestMs < startMs) return out
  }
}

async function fetchActivityDetail(
  session: GarminConnectSession,
  base: string,
  id: string,
): Promise<UnknownRecord | null> {
  const raw = await fetchGarminJson(
    session,
    base,
    `/activity-service/activity/${encodeURIComponent(id)}`,
  )
  return isRecord(raw) ? raw : null
}

async function fetchActivityStreamDetail(
  session: GarminConnectSession,
  base: string,
  id: string,
): Promise<UnknownRecord | null> {
  const raw = await fetchGarminJson(
    session,
    base,
    `/activity-service/activity/${encodeURIComponent(id)}/details`,
  )
  return isRecord(raw) ? raw : null
}

async function fetchActivityClimbDetail(
  session: GarminConnectSession,
  base: string,
  id: string,
): Promise<UnknownRecord | null> {
  const raw = await fetchGarminJson(
    session,
    base,
    `/activity-service/activity/${encodeURIComponent(id)}/typedsplits`,
  )
  return isRecord(raw) ? raw : null
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function main(): Promise<void> {
  const previous = await readCache()
  const session = await readGarminConnectSession()
  const base = cleanGarminConnectBaseUrl(
    process.env.GARMIN_CONNECT_BASE_URL?.trim() || DEFAULT_GARMIN_CONNECT_BASE,
  )
  const pageSize = Math.max(1, envNumber('GARMIN_CONNECT_PAGE_SIZE', DEFAULT_PAGE_SIZE))
  const delayMs = envNumber('GARMIN_CONNECT_DELAY_MS', DEFAULT_DELAY_MS)
  const maxActivities = envNumber('GARMIN_CONNECT_MAX_ACTIVITIES', 0)
  const fetchStreams = envFlag('GARMIN_CONNECT_FETCH_STREAMS', true)
  const start = await startDate()
  const end = endDate()

  console.log(`[garmin] fetching Garmin Connect activities ${start} -> ${end}`)
  const list = await fetchActivities(session, base, start, end, pageSize, maxActivities)
  list.sort((a, b) => {
    const left = garminConnectActivity(null, a.record, 0)?.startDate ?? ''
    const right = garminConnectActivity(null, b.record, 0)?.startDate ?? ''
    return left.localeCompare(right)
  })
  console.log(`[garmin] found ${list.length} activities`)

  const partial = maxActivities > 0
  const activities = initialGarminSyncRecords(previous?.activities, partial)
  const streams = initialGarminSyncRecords(previous?.streams, partial)
  const climbs = initialGarminSyncRecords(previous?.climbs, partial)
  let details = 0
  let streamDetails = 0
  let climbDetails = 0
  let skipped = 0
  for (let i = 0; i < list.length; i++) {
    const item = list[i]
    const listedActivity = garminConnectActivity(null, item.record, i)
    const cacheId = listedActivity?.id ?? `connect:${item.id}`
    let activityOutcome: GarminFetchOutcome<GarminActivity> = { ok: false }
    try {
      const detail = await fetchActivityDetail(session, base, item.id)
      if (detail) {
        details++
        const activity = garminConnectActivity(detail, item.record, i) ?? listedActivity
        activityOutcome = activity ? { ok: true, value: activity } : { ok: true }
      }
    } catch (err) {
      console.warn(`[garmin] detail ${item.id} failed: ${err instanceof Error ? err.message : err}`)
    }
    const activity =
      resolveGarminFetch(activityOutcome, previous?.activities[cacheId]) ?? listedActivity
    if (activity) {
      activities[activity.id] = activity
      let streamOutcome: GarminFetchOutcome<GarminStreams> = { ok: false }
      if (fetchStreams) {
        try {
          const streamDetail = await fetchActivityStreamDetail(session, base, item.id)
          if (streamDetail) {
            streamDetails++
            const stream = garminConnectStreams(streamDetail)
            streamOutcome = stream ? { ok: true, value: stream } : { ok: true }
          }
        } catch (err) {
          console.warn(
            `[garmin] stream ${item.id} failed: ${err instanceof Error ? err.message : err}`,
          )
        }
      }
      const stream = resolveGarminFetch(
        streamOutcome,
        previous?.streams?.[activity.id] ?? previous?.streams?.[cacheId],
      )
      if (stream) streams[activity.id] = stream
      else delete streams[activity.id]
      if (activity.sport === 'bike') {
        let climbOutcome: GarminFetchOutcome<GarminClimbSegment[]> = { ok: false }
        try {
          const climbDetail = await fetchActivityClimbDetail(session, base, item.id)
          if (climbDetail) {
            climbDetails++
            const segments = garminConnectClimbSegments(climbDetail)
            climbOutcome = segments.length > 0 ? { ok: true, value: segments } : { ok: true }
          }
        } catch (err) {
          console.warn(
            `[garmin] climbs ${item.id} failed: ${err instanceof Error ? err.message : err}`,
          )
        }
        const segments = resolveGarminFetch(
          climbOutcome,
          previous?.climbs?.[activity.id] ?? previous?.climbs?.[cacheId],
        )
        if (segments) climbs[activity.id] = segments
        else delete climbs[activity.id]
      }
    } else skipped++
    if (delayMs > 0) await sleep(delayMs)
  }

  let vo2Outcome: GarminFetchOutcome<Record<string, GarminVo2Day>> = { ok: false }
  try {
    const raw = await fetchGarminJson(
      session,
      base,
      `/metrics-service/metrics/maxmet/daily/${encodeURIComponent(start)}/${encodeURIComponent(end)}`,
    )
    const fetched: Record<string, GarminVo2Day> = {}
    for (const day of garminConnectVo2(raw)) fetched[day.date] = day
    vo2Outcome = { ok: true, value: fetched }
    console.log(`[garmin] vo2max days: ${Object.keys(fetched).length}`)
  } catch (err) {
    console.warn(`[garmin] vo2max fetch failed: ${err instanceof Error ? err.message : err}`)
  }
  const vo2max = resolveGarminFetch(vo2Outcome, previous?.vo2max) ?? {}

  let weightOutcome: GarminFetchOutcome<GarminWeightSample[]> = { ok: false }
  try {
    const rangeRaw = await fetchGarminJson(
      session,
      base,
      '/weight-service/weight/dateRange',
      new URLSearchParams({ startDate: start, endDate: end }),
    )
    const byDay = new Map<string, GarminWeightSample>()
    for (const s of garminConnectWeightSamples(rangeRaw)) byDay.set(s.date, s)
    const collected: GarminWeightSample[] = []
    for (const day of [...byDay.keys()].sort()) {
      let dayOutcome: GarminFetchOutcome<GarminWeightSample[]> = { ok: false }
      try {
        const dv = await fetchGarminJson(
          session,
          base,
          `/weight-service/weight/dayview/${encodeURIComponent(day)}`,
          new URLSearchParams({ includeAll: 'true' }),
        )
        dayOutcome = { ok: true, value: garminConnectWeightSamples(dv) }
      } catch (err) {
        console.warn(
          `[garmin] weight dayview ${day} failed: ${err instanceof Error ? err.message : err}`,
        )
      }
      collected.push(
        ...resolveGarminWeightDay(day, dayOutcome, byDay.get(day)!, previous?.weight ?? []),
      )
      if (delayMs > 0) await sleep(delayMs)
    }
    const deduped = new Map<number, GarminWeightSample>()
    for (const s of collected) deduped.set(s.ts, s)
    const weight = [...deduped.values()].sort((a, b) => a.ts - b.ts)
    weightOutcome = { ok: true, value: weight }
    const days = new Set(weight.map(s => s.date)).size
    console.log(`[garmin] weight samples: ${weight.length} over ${days} days`)
  } catch (err) {
    console.warn(`[garmin] weight fetch failed: ${err instanceof Error ? err.message : err}`)
  }
  const weight = resolveGarminFetch(weightOutcome, previous?.weight) ?? []

  const sorted: Record<string, GarminActivity> = {}
  for (const activity of Object.values(activities).sort((a, b) =>
    a.startDate.localeCompare(b.startDate),
  )) {
    sorted[activity.id] = activity
  }

  const now = Date.now()
  const sortedStreams: Record<string, GarminStreams> = {}
  for (const id of Object.keys(sorted)) if (streams[id]) sortedStreams[id] = streams[id]
  const sortedClimbs: Record<string, GarminClimbSegment[]> = {}
  for (const id of Object.keys(sorted)) if (climbs[id]) sortedClimbs[id] = climbs[id]
  const cache: GarminCache = {
    version: CACHE_VERSION,
    lastSync: now,
    activities: sorted,
    streams: sortedStreams,
    climbs: sortedClimbs,
    vo2max,
    weight,
  }
  await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  await refreshTriathlonRouteSource()
  console.log(
    `[garmin] wrote ${Object.keys(sorted).length} activities (${details} detail responses, ${streamDetails} stream responses, ${climbDetails} climb responses, ${Object.values(sortedClimbs).reduce((sum, segments) => sum + segments.length, 0)} climbs, ${skipped} skipped) -> ${cacheFile}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[garmin] sync failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
