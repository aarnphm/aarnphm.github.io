import type { DetailCtx } from '../../../util/triathlon-card'
import { isActivityKind, type StravaActivityDetail } from '../../../plugins/stores/strava'
import {
  isStravaDetailShardPath,
  STRAVA_DETAIL_INDEX_KIND,
  type StravaDetailIndex,
  type StravaDetailPayload,
  type StravaDetailShard,
} from '../../../util/strava-detail'
import { isRecord } from '../../../util/type-guards'

export type DetailPayload = StravaDetailPayload

const isDetailIndex = (value: unknown): value is StravaDetailIndex =>
  isRecord(value) &&
  value.kind === STRAVA_DETAIL_INDEX_KIND &&
  Array.isArray(value.shards) &&
  value.shards.every(isStravaDetailShardPath) &&
  isRecord(value.health)

const isActivityDetail = (value: unknown): value is StravaActivityDetail =>
  isRecord(value) &&
  typeof value.id === 'number' &&
  /^\d{4}-\d{2}-\d{2}$/.test(typeof value.date === 'string' ? value.date : '') &&
  isActivityKind(value.sport)

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
    zones: value.zones,
    powerCurveRef: value.powerCurveRef,
    powerCurveYearRef: value.powerCurveYearRef,
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

export const detailContextFromPayload = (payload?: DetailPayload | null): DetailCtx => ({
  zones: payload?.zones ?? null,
  curveRef: payload?.powerCurveRef ?? [],
  curveYearRef: payload?.powerCurveYearRef ?? [],
  curveYear: payload?.powerCurveYear ?? null,
  criticalPower: payload?.criticalPower ?? null,
  criticalPowerYear: payload?.criticalPowerYear ?? null,
  ftp: payload?.ftp ?? null,
  goalFtp: payload?.goalFtp ?? null,
  vt1: payload?.vt1Hr ?? null,
})
