import type { CriticalPowerEstimate } from '../plugins/stores/critical-power'
import type { MatchedRidesBlock } from '../plugins/stores/matched-rides'
import type { MatchedRunsBlock } from '../plugins/stores/matched-runs'
import type {
  ActivityHealth,
  PowerCurvePoint,
  StravaActivityDetail,
  StravaZones,
  SwimTrendPoint,
} from '../plugins/stores/strava'
import type { TriathlonDailyAnalytics } from './triathlon-day-analytics'

export const STRAVA_DETAIL_INDEX_KIND = 'strava-detail-index-v1'
export const STRAVA_DETAIL_SHARD_MAX_BYTES = 20 * 1024 * 1024

const STRAVA_DETAIL_SHARD_PATH = /^strava-detail\/\d{4}-\d{2}(?:-\d+)?\.json$/
const STRAVA_DETAIL_DATE = /^(\d{4}-\d{2})-\d{2}$/
const SHARD_PREFIX = '{"details":{'
const SHARD_SUFFIX = '}}'
const textEncoder = new TextEncoder()

export type StravaDetailPayload<TDetail extends { date: string } = StravaActivityDetail> = {
  details: Record<string, TDetail>
  swimTrend?: SwimTrendPoint[]
  health: Record<string, ActivityHealth>
  dailyAnalytics?: TriathlonDailyAnalytics
  zones?: StravaZones
  powerCurveRef?: PowerCurvePoint[]
  powerCurveYearRef?: PowerCurvePoint[]
  powerCurveYear?: number | null
  criticalPower?: CriticalPowerEstimate | null
  criticalPowerYear?: CriticalPowerEstimate | null
  ftp?: number | null
  goalFtp?: number | null
  vt1Hr?: number | null
  matchedRuns?: MatchedRunsBlock
  matchedRides?: MatchedRidesBlock
}

export type StravaDetailIndex = Omit<StravaDetailPayload, 'details'> & {
  kind: typeof STRAVA_DETAIL_INDEX_KIND
  shards: string[]
}

export type StravaDetailShard<TDetail extends { date: string } = StravaActivityDetail> = {
  details: Record<string, TDetail>
}

export type SerializedStravaDetailShard = {
  name: string
  path: string
  content: string
  bytes: number
}

export type SerializedStravaDetails = { manifest: string; shards: SerializedStravaDetailShard[] }

const jsonBytes = (value: string): number => textEncoder.encode(value).byteLength

export const isStravaDetailShardPath = (value: unknown): value is string =>
  typeof value === 'string' && STRAVA_DETAIL_SHARD_PATH.test(value)

export function serializeStravaDetails<TDetail extends { date: string }>(
  payload: StravaDetailPayload<TDetail>,
  maxShardBytes = STRAVA_DETAIL_SHARD_MAX_BYTES,
): SerializedStravaDetails {
  if (!Number.isInteger(maxShardBytes) || maxShardBytes <= jsonBytes(SHARD_PREFIX + SHARD_SUFFIX))
    throw new Error('Strava detail shard byte limit must fit a JSON object')

  const entriesByMonth = new Map<string, string[]>()
  for (const [id, detail] of Object.entries(payload.details)) {
    const month = STRAVA_DETAIL_DATE.exec(detail.date)?.[1]
    if (!month) throw new Error(`Strava detail ${id} has invalid date ${detail.date}`)
    const serializedDetail = JSON.stringify(detail)
    if (!serializedDetail) throw new Error(`Strava detail ${id} is not JSON serializable`)
    const entry = `${JSON.stringify(id)}:${serializedDetail}`
    if (jsonBytes(SHARD_PREFIX + entry + SHARD_SUFFIX) > maxShardBytes)
      throw new Error(`Strava detail ${id} exceeds the shard byte limit`)
    const entries = entriesByMonth.get(month) ?? []
    entries.push(entry)
    entriesByMonth.set(month, entries)
  }

  const shards: SerializedStravaDetailShard[] = []
  for (const [month, entries] of [...entriesByMonth].sort(([left], [right]) =>
    right.localeCompare(left),
  )) {
    let chunk: string[] = []
    let chunkBytes = jsonBytes(SHARD_PREFIX + SHARD_SUFFIX)
    const flush = (): void => {
      if (chunk.length === 0) return
      const index = shards.filter(
        shard => shard.name === month || shard.name.startsWith(`${month}-`),
      ).length
      const name = index === 0 ? month : `${month}-${index + 1}`
      const content = SHARD_PREFIX + chunk.join(',') + SHARD_SUFFIX
      shards.push({ name, path: `strava-detail/${name}.json`, content, bytes: jsonBytes(content) })
      chunk = []
      chunkBytes = jsonBytes(SHARD_PREFIX + SHARD_SUFFIX)
    }

    for (const entry of entries) {
      const entryBytes = jsonBytes(entry) + (chunk.length > 0 ? 1 : 0)
      if (chunkBytes + entryBytes > maxShardBytes) flush()
      chunk.push(entry)
      chunkBytes += jsonBytes(entry) + (chunk.length > 1 ? 1 : 0)
    }
    flush()
  }

  const manifest: StravaDetailIndex = {
    kind: STRAVA_DETAIL_INDEX_KIND,
    shards: shards.map(shard => shard.path),
    swimTrend: payload.swimTrend,
    health: payload.health,
    dailyAnalytics: payload.dailyAnalytics,
    zones: payload.zones,
    powerCurveRef: payload.powerCurveRef,
    powerCurveYearRef: payload.powerCurveYearRef,
    powerCurveYear: payload.powerCurveYear,
    criticalPower: payload.criticalPower,
    criticalPowerYear: payload.criticalPowerYear,
    ftp: payload.ftp,
    goalFtp: payload.goalFtp,
    vt1Hr: payload.vt1Hr,
    matchedRuns: payload.matchedRuns,
    matchedRides: payload.matchedRides,
  }
  return { manifest: JSON.stringify(manifest), shards }
}
