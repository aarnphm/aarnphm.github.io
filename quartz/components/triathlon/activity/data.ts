import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { DetailCtx } from '../../../util/triathlon-card'
import {
  hydrateStravaMapPayload,
  isStravaDetailShardPath,
  STRAVA_DETAIL_INDEX_KIND,
  type StravaDetailIndex,
  type StravaDetailPayload,
  type StravaDetailShard,
} from '../../../util/strava-detail'
import { isTriathlonDailyAnalytics } from '../../../util/triathlon-day-analytics'
import { isRecord } from '../../../util/type-guards'
import { isActivityDetail } from './data/detail'
export { isActivityDetail } from './data/detail'

export type DetailPayload = StravaDetailPayload

const isDetailIndex = (value: unknown): value is StravaDetailIndex =>
  isRecord(value) &&
  value.kind === STRAVA_DETAIL_INDEX_KIND &&
  Array.isArray(value.shards) &&
  value.shards.every(isStravaDetailShardPath) &&
  isRecord(value.health) &&
  (value.dailyAnalytics === undefined || isTriathlonDailyAnalytics(value.dailyAnalytics))

const isDetailShard = (value: unknown): value is StravaDetailShard =>
  isRecord(value) &&
  isRecord(value.details) &&
  Object.entries(value.details).every(
    ([id, detail]) => /^\d+$/.test(id) && isActivityDetail(detail) && String(detail.id) === id,
  )

export async function readDetailPayload(
  response: Response,
  signal: AbortSignal,
): Promise<DetailPayload> {
  const value: unknown = await response.json()
  if (!isDetailIndex(value)) throw new Error('invalid Strava detail index')
  const shardDetails = await Promise.all(
    value.shards.map(async path => {
      const shardResponse = await fetch(new URL(path, response.url), { signal })
      if (!shardResponse.ok) throw new Error(`${path} returned ${shardResponse.status}`)
      const shard: unknown = await shardResponse.json()
      if (!isDetailShard(shard)) throw new Error(`${path} is not a valid Strava detail shard`)
      return shard.details
    }),
  )
  const details: Record<string, StravaActivityDetail> = {}
  for (const shard of shardDetails)
    for (const [id, detail] of Object.entries(shard)) {
      if (details[id]) throw new Error(`duplicate Strava detail ${id}`)
      details[id] = detail
    }
  return {
    details,
    swimTrend: value.swimTrend,
    health: value.health,
    dailyAnalytics: value.dailyAnalytics,
    zones: value.zones,
    powerCurveRef: value.powerCurveRef,
    powerCurveYearRef: value.powerCurveYearRef,
    runPowerCurveRef: value.runPowerCurveRef,
    runPowerCurveYearRef: value.runPowerCurveYearRef,
    powerCurveYear: value.powerCurveYear,
    criticalPower: value.criticalPower,
    criticalPowerYear: value.criticalPowerYear,
    ftp: value.ftp,
    goalFtp: value.goalFtp,
    vt1Hr: value.vt1Hr,
    matchedRuns: value.matchedRuns,
    matchedRides: value.matchedRides,
  }
}

export async function readMapPayload(response: Response): Promise<DetailPayload> {
  const value: unknown = await response.json()
  const payload = hydrateStravaMapPayload(value)
  if (!payload) throw new Error('invalid Strava map payload')
  return payload
}

export async function readDetailIndex(
  detailUrl: string,
  signal: AbortSignal,
): Promise<{ baseUrl: string; index: StravaDetailIndex } | null> {
  const response = await fetch(detailUrl, { signal })
  if (!response.ok) return null
  const value: unknown = await response.json()
  if (!isDetailIndex(value)) return null
  return { baseUrl: response.url || detailUrl, index: value }
}

export function parseActivityDetailFromShardMap(
  details: Record<string, unknown> | null | undefined,
  id: string,
): StravaActivityDetail | null {
  const candidate = details?.[id]
  if (!candidate || !isActivityDetail(candidate) || String(candidate.id) !== id) return null
  return candidate
}

export function warmActivityShard(
  shardUrl: string,
  signal: AbortSignal,
  shardCache: Map<string, Promise<Record<string, unknown> | null>>,
  resolvedShardCache?: Map<string, Record<string, unknown>>,
): Promise<Record<string, unknown> | null> {
  const resolved = resolvedShardCache?.get(shardUrl)
  if (resolved) return Promise.resolve(resolved)
  const cached = shardCache.get(shardUrl)
  if (cached) return cached
  const pending = fetch(shardUrl, { signal })
    .then(async response => {
      if (!response.ok) return null
      const shard: unknown = await response.json()
      const details = isRecord(shard) && isRecord(shard.details) ? shard.details : null
      if (details) resolvedShardCache?.set(shardUrl, details)
      return details
    })
    .catch(() => {
      shardCache.delete(shardUrl)
      return null
    })
  shardCache.set(shardUrl, pending)
  return pending
}

export async function readActivityDetailFromShard(
  shardUrl: string,
  id: string,
  signal: AbortSignal,
  shardCache: Map<string, Promise<Record<string, unknown> | null>>,
  resolvedShardCache?: Map<string, Record<string, unknown>>,
): Promise<StravaActivityDetail | null> {
  const details = await warmActivityShard(shardUrl, signal, shardCache, resolvedShardCache)
  return parseActivityDetailFromShardMap(details, id)
}

export const detailContextFromPayload = (payload?: DetailPayload | null): DetailCtx => ({
  zones: payload?.zones ?? null,
  curveRef: payload?.powerCurveRef ?? [],
  curveYearRef: payload?.powerCurveYearRef ?? [],
  runCurveRef: payload?.runPowerCurveRef ?? [],
  runCurveYearRef: payload?.runPowerCurveYearRef ?? [],
  curveYear: payload?.powerCurveYear ?? null,
  criticalPower: payload?.criticalPower ?? null,
  criticalPowerYear: payload?.criticalPowerYear ?? null,
  ftp: payload?.ftp ?? null,
  goalFtp: payload?.goalFtp ?? null,
  vt1: payload?.vt1Hr ?? null,
})
