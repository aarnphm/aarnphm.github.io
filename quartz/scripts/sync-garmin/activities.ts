import type {
  GarminActivity,
  GarminCache,
  GarminClimbSegment,
  GarminCyclingDynamics,
  GarminFitTrainingEffect,
  GarminGearShift,
  GarminRunWalkData,
  GarminStreams,
  GarminSwimData,
} from '../../plugins/stores/garmin'
import {
  garminConnectActivities,
  garminConnectActivity,
  garminConnectActivityStartDate,
  garminConnectClimbSegments,
  garminConnectRunWalk,
  garminConnectStreams,
  type GarminConnectActivityListItem,
} from '../../util/garmin-connect'
import {
  decodeGarminActivityFit,
  garminActivityFileFromArchive,
  type GarminActivityFitData,
} from '../../util/garmin-fit'
import {
  fetchGarminBytes,
  fetchGarminJson,
  type GarminConnectSession,
} from '../../util/garmin-session'
import { localDayEndUtcMs, localDayStartUtcMs } from '../../util/local-date'
import { isRecord, type UnknownRecord } from '../../util/type-guards'
import {
  garminActivityDay,
  resolveGarminFetch,
  sleep,
  SWIM_CACHE_VERSION,
  type GarminFetchOutcome,
} from './refresh-window'

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

export async function syncGarminActivities(
  session: GarminConnectSession,
  base: string,
  previous: GarminCache | null,
  start: string,
  end: string,
  pageSize: number,
  maxActivities: number,
  fetchStreams: boolean,
  delayMs: number,
): Promise<{
  sorted: Record<string, GarminActivity>
  sortedStreams: Record<string, GarminStreams>
  sortedGearShifts: Record<string, GarminGearShift[]>
  sortedCyclingDynamics: Record<string, GarminCyclingDynamics>
  sortedFitTrainingEffects: Record<string, GarminFitTrainingEffect>
  sortedSwims: Record<string, GarminSwimData>
  sortedClimbs: Record<string, GarminClimbSegment[]>
  sortedRunWalks: Record<string, GarminRunWalkData>
  details: number
  streamDetails: number
  activityArchives: number
  climbDetails: number
  runWalkDetails: number
  skipped: number
}> {
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

  const sorted: Record<string, GarminActivity> = {}
  for (const activity of Object.values(activities).sort((a, b) =>
    a.startDate.localeCompare(b.startDate),
  )) {
    sorted[activity.id] = activity
  }

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

  return {
    sorted,
    sortedStreams,
    sortedGearShifts,
    sortedCyclingDynamics,
    sortedFitTrainingEffects,
    sortedSwims,
    sortedClimbs,
    sortedRunWalks,
    details,
    streamDetails,
    activityArchives,
    climbDetails,
    runWalkDetails,
    skipped,
  }
}
