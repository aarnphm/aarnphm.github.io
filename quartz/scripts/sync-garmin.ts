import fs from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import type {
  GarminActivity,
  GarminCache,
  GarminClimbSegment,
  GarminCyclingDynamics,
  GarminFitTrainingEffect,
  GarminGearShift,
  GarminRunWalkData,
  GarminSwimData,
  GarminStreams,
  GarminVo2Day,
  GarminWeightSample,
} from '../plugins/stores/garmin'
import {
  garminConnectActivities,
  garminConnectActivity,
  garminConnectActivityStartDate,
  garminConnectClimbSegments,
  garminConnectRunWalk,
  garminConnectStreams,
  garminConnectVo2,
  garminConnectWeightSamples,
  type GarminConnectActivityListItem,
} from '../util/garmin-connect'
import {
  decodeGarminActivityFit,
  garminActivityFileFromArchive,
  type GarminActivityFitData,
} from '../util/garmin-fit'
import {
  cleanGarminConnectBaseUrl,
  DEFAULT_GARMIN_CONNECT_BASE,
  fetchGarminBytes,
  fetchGarminJson,
  readGarminConnectSession,
  type GarminConnectSession,
} from '../util/garmin-session'
import {
  localDayEndUtcMs,
  localDayStartUtcMs,
  localIsoDayOffset,
  shiftIsoDay,
} from '../util/local-date'
import { joinSegments, QUARTZ } from '../util/path'
import { syncRefreshDays } from '../util/sync-refresh-window'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord, type UnknownRecord } from '../util/type-guards'

const CACHE_VERSION = 14
const SWIM_CACHE_VERSION = 13
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

function garminActivityDay(activity: GarminActivity): string | null {
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

export function reconcileGarminActivities(
  previous: Readonly<Record<string, GarminActivity>> | undefined,
  start: string,
  end: string,
  preserveRefreshRange: boolean,
): Record<string, GarminActivity> {
  const activities: Record<string, GarminActivity> = {}
  for (const [id, activity] of Object.entries(previous ?? {})) {
    const day = garminActivityDay(activity)
    if (preserveRefreshRange || day == null || day < start || day > end) activities[id] = activity
  }
  return activities
}

export function mergeGarminFitTrainingEffect(
  activity: GarminActivity,
  fit: GarminFitTrainingEffect | undefined,
): GarminActivity {
  const aerobicTrainingEffect = activity.metrics.aerobicTrainingEffect ?? fit?.aerobic ?? null
  const anaerobicTrainingEffect = activity.metrics.anaerobicTrainingEffect ?? fit?.anaerobic ?? null
  if (
    aerobicTrainingEffect === activity.metrics.aerobicTrainingEffect &&
    anaerobicTrainingEffect === activity.metrics.anaerobicTrainingEffect
  )
    return activity
  return {
    ...activity,
    metrics: { ...activity.metrics, aerobicTrainingEffect, anaerobicTrainingEffect },
  }
}

export function needsGarminActivityFit(
  sport: GarminActivity['sport'],
  hasTrainingEffect: boolean,
  hasCyclingData: boolean,
  hasSwimData: boolean,
): boolean {
  if (!hasTrainingEffect) return true
  if (sport === 'bike') return !hasCyclingData
  return sport === 'swim' && !hasSwimData
}

export function mergeGarminVo2Range(
  previous: Record<string, GarminVo2Day>,
  fetched: Record<string, GarminVo2Day>,
  start: string,
  end: string,
): Record<string, GarminVo2Day> {
  const merged: Record<string, GarminVo2Day> = {}
  for (const [date, value] of Object.entries(previous))
    if (date < start || date > end) merged[date] = value
  for (const [date, value] of Object.entries(fetched)) merged[date] = value
  return Object.fromEntries(
    Object.entries(merged).sort(([left], [right]) => left.localeCompare(right)),
  )
}

export function mergeGarminWeightRange(
  previous: GarminWeightSample[],
  fetched: GarminWeightSample[],
  start: string,
  end: string,
): GarminWeightSample[] {
  const merged = new Map<number, GarminWeightSample>()
  for (const sample of previous)
    if (sample.date < start || sample.date > end) merged.set(sample.ts, sample)
  for (const sample of fetched) merged.set(sample.ts, sample)
  return [...merged.values()].sort((left, right) => left.ts - right.ts)
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

async function startDate(previous: GarminCache | null, refreshWindowDays: number): Promise<string> {
  const explicit =
    cleanDay(process.env.GARMIN_CONNECT_START_DATE) ?? cleanDay(process.env.GARMIN_CONNECT_SINCE)
  if (explicit) return explicit
  const configured = (await readTriathlonStart()) ?? localIsoDayOffset(-90)
  return garminRefreshStart(previous, configured, refreshWindowDays)
}

function endDate(): string {
  return cleanDay(process.env.GARMIN_CONNECT_END_DATE) ?? localIsoDayOffset(0)
}

function activityStartMs(item: GarminConnectActivityListItem): number | null {
  const start = garminConnectActivityStartDate(item.record)
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
    const raw = await fetchGarminJson(
      session,
      base,
      '/activitylist-service/activities/search/activities',
      new URLSearchParams({
        startDate: start,
        endDate: end,
        start: String(offset),
        limit: String(pageSize),
      }),
    )
    const page = garminConnectActivities(raw)
    let oldestMs: number | null = null
    for (const item of page) {
      if (seen.has(item.id)) continue
      seen.add(item.id)
      const ms = activityStartMs(item)
      if (ms == null) throw new Error(`Garmin activity ${item.id} has no parseable start date`)
      oldestMs = oldestMs == null ? ms : Math.min(oldestMs, ms)
      if (ms < startMs || ms > endMs) continue
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

async function fetchActivityTypedSplits(
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

async function fetchActivityFit(
  session: GarminConnectSession,
  base: string,
  id: string,
): Promise<GarminActivityFitData | null> {
  const archive = await fetchGarminBytes(
    session,
    base,
    `/download-service/files/activity/${encodeURIComponent(id)}`,
  )
  const file = garminActivityFileFromArchive(archive)
  return file.kind === 'fit' ? decodeGarminActivityFit(file.bytes) : null
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
  const refreshWindowDays = syncRefreshDays()
  const start = await startDate(previous, refreshWindowDays)
  const end = endDate()
  if (start > end) throw new Error(`Garmin sync start ${start} is after end ${end}`)

  console.log(`[garmin] fetching Garmin Connect activities ${start} -> ${end}`)
  const list = await fetchActivities(session, base, start, end, pageSize, maxActivities)
  list.sort((a, b) => {
    const left = garminConnectActivity(null, a.record, 0)?.startDate ?? ''
    const right = garminConnectActivity(null, b.record, 0)?.startDate ?? ''
    return left.localeCompare(right)
  })
  console.log(`[garmin] found ${list.length} activities`)

  const activities = reconcileGarminActivities(previous?.activities, start, end, maxActivities > 0)
  const streams: Record<string, GarminStreams> = { ...previous?.streams }
  const gearShifts: Record<string, GarminGearShift[]> = { ...previous?.gearShifts }
  const cyclingDynamics: Record<string, GarminCyclingDynamics> = { ...previous?.cyclingDynamics }
  const fitTrainingEffects: Record<string, GarminFitTrainingEffect> = {
    ...previous?.fitTrainingEffects,
  }
  const swims: Record<string, GarminSwimData> = { ...previous?.swims }
  const climbs: Record<string, GarminClimbSegment[]> = { ...previous?.climbs }
  const runWalks: Record<string, GarminRunWalkData> = { ...previous?.runWalks }
  let details = 0
  let streamDetails = 0
  let activityArchives = 0
  let climbDetails = 0
  let runWalkDetails = 0
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

      const previousTrainingEffect =
        previous?.fitTrainingEffects?.[activity.id] ?? previous?.fitTrainingEffects?.[cacheId]
      const hasPreviousTrainingEffect =
        Object.hasOwn(previous?.fitTrainingEffects ?? {}, activity.id) ||
        Object.hasOwn(previous?.fitTrainingEffects ?? {}, cacheId)
      const previousShifts = previous?.gearShifts?.[activity.id] ?? previous?.gearShifts?.[cacheId]
      const previousDynamics =
        previous?.cyclingDynamics?.[activity.id] ?? previous?.cyclingDynamics?.[cacheId]
      const hasPreviousDynamics =
        Object.hasOwn(previous?.cyclingDynamics ?? {}, activity.id) ||
        Object.hasOwn(previous?.cyclingDynamics ?? {}, cacheId)
      const previousSwim = previous?.swims?.[activity.id] ?? previous?.swims?.[cacheId]
      const hasPreviousSwim =
        (previous?.version ?? 0) >= SWIM_CACHE_VERSION &&
        (Object.hasOwn(previous?.swims ?? {}, activity.id) ||
          Object.hasOwn(previous?.swims ?? {}, cacheId))
      let fit: GarminActivityFitData | null | undefined
      if (
        needsGarminActivityFit(
          activity.sport,
          hasPreviousTrainingEffect,
          previousShifts != null && hasPreviousDynamics,
          hasPreviousSwim,
        )
      ) {
        try {
          fit = await fetchActivityFit(session, base, item.id)
          activityArchives++
        } catch (err) {
          console.warn(
            `[garmin] activity FIT ${item.id} failed: ${err instanceof Error ? err.message : err}`,
          )
        }
      }
      const trainingEffect =
        fit?.trainingEffect ??
        previousTrainingEffect ??
        (fit === null ? { aerobic: null, anaerobic: null } : undefined)
      if (trainingEffect) fitTrainingEffects[activity.id] = trainingEffect
      else delete fitTrainingEffects[activity.id]
      activities[activity.id] = mergeGarminFitTrainingEffect(activity, trainingEffect)

      if (activity.sport === 'bike') {
        const shifts = fit?.gearShifts ?? previousShifts
        if (shifts) gearShifts[activity.id] = shifts
        else delete gearShifts[activity.id]
        const dynamics = fit?.cyclingDynamics ?? previousDynamics
        if (dynamics) cyclingDynamics[activity.id] = dynamics
        else delete cyclingDynamics[activity.id]

        let climbOutcome: GarminFetchOutcome<GarminClimbSegment[]> = { ok: false }
        try {
          const climbDetail = await fetchActivityTypedSplits(session, base, item.id)
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
      } else if (activity.sport === 'swim') {
        const swim = fit ? (fit.swim ?? undefined) : previousSwim
        if (swim) swims[activity.id] = swim
        else delete swims[activity.id]
      } else if (activity.sport === 'run') {
        let runWalkOutcome: GarminFetchOutcome<GarminRunWalkData> = { ok: false }
        try {
          const typedSplits = await fetchActivityTypedSplits(session, base, item.id)
          if (typedSplits) {
            runWalkDetails++
            const runWalk = garminConnectRunWalk(typedSplits)
            runWalkOutcome = runWalk ? { ok: true, value: runWalk } : { ok: true }
          }
        } catch (err) {
          console.warn(
            `[garmin] run/walk ${item.id} failed: ${err instanceof Error ? err.message : err}`,
          )
        }
        const runWalk = resolveGarminFetch(
          runWalkOutcome,
          previous?.runWalks?.[activity.id] ?? previous?.runWalks?.[cacheId],
        )
        if (runWalk) runWalks[activity.id] = runWalk
        else delete runWalks[activity.id]
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
  const vo2max = vo2Outcome.ok
    ? mergeGarminVo2Range(previous?.vo2max ?? {}, vo2Outcome.value ?? {}, start, end)
    : (previous?.vo2max ?? {})

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
  const weight = weightOutcome.ok
    ? mergeGarminWeightRange(previous?.weight ?? [], weightOutcome.value ?? [], start, end)
    : (previous?.weight ?? [])

  const sorted: Record<string, GarminActivity> = {}
  for (const activity of Object.values(activities).sort((a, b) =>
    a.startDate.localeCompare(b.startDate),
  )) {
    sorted[activity.id] = activity
  }

  const now = Date.now()
  const sortedStreams: Record<string, GarminStreams> = {}
  for (const id of Object.keys(sorted)) if (streams[id]) sortedStreams[id] = streams[id]
  const sortedGearShifts: Record<string, GarminGearShift[]> = {}
  for (const id of Object.keys(sorted)) if (gearShifts[id]) sortedGearShifts[id] = gearShifts[id]
  const sortedCyclingDynamics: Record<string, GarminCyclingDynamics> = {}
  for (const id of Object.keys(sorted))
    if (cyclingDynamics[id]) sortedCyclingDynamics[id] = cyclingDynamics[id]
  const sortedFitTrainingEffects: Record<string, GarminFitTrainingEffect> = {}
  for (const id of Object.keys(sorted))
    if (fitTrainingEffects[id]) sortedFitTrainingEffects[id] = fitTrainingEffects[id]
  const sortedSwims: Record<string, GarminSwimData> = {}
  for (const id of Object.keys(sorted)) if (swims[id]) sortedSwims[id] = swims[id]
  const sortedClimbs: Record<string, GarminClimbSegment[]> = {}
  for (const id of Object.keys(sorted)) if (climbs[id]) sortedClimbs[id] = climbs[id]
  const sortedRunWalks: Record<string, GarminRunWalkData> = {}
  for (const id of Object.keys(sorted)) if (runWalks[id]) sortedRunWalks[id] = runWalks[id]
  const cache: GarminCache = {
    version: CACHE_VERSION,
    lastSync: now,
    activities: sorted,
    streams: sortedStreams,
    gearShifts: sortedGearShifts,
    cyclingDynamics: sortedCyclingDynamics,
    fitTrainingEffects: sortedFitTrainingEffects,
    swims: sortedSwims,
    climbs: sortedClimbs,
    runWalks: sortedRunWalks,
    vo2max,
    weight,
  }
  await fs.mkdir(joinSegments(QUARTZ, '.quartz-cache'), { recursive: true })
  await fs.writeFile(cacheFile, JSON.stringify(cache, null, 2))
  await refreshTriathlonRouteSource()
  console.log(
    `[garmin] wrote ${Object.keys(sorted).length} activities (${details} detail responses, ${streamDetails} stream responses, ${activityArchives} activity archives, ${Object.values(sortedGearShifts).reduce((sum, shifts) => sum + shifts.length, 0)} shift states, ${Object.values(sortedCyclingDynamics).reduce((sum, dynamics) => sum + dynamics.time.length, 0)} cycling dynamics samples, ${Object.values(sortedSwims).reduce((sum, swim) => sum + swim.lengths.length, 0)} swim lengths, ${climbDetails} climb responses, ${Object.values(sortedClimbs).reduce((sum, segments) => sum + segments.length, 0)} climbs, ${runWalkDetails} run/walk responses, ${Object.values(sortedRunWalks).reduce((sum, runWalk) => sum + runWalk.segments.length, 0)} run/walk segments, ${skipped} skipped) -> ${cacheFile}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[garmin] sync failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
