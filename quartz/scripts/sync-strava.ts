import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import {
  hasFetchedActivityDetail,
  normalizeKind,
  RawStravaActivity,
  RawStravaActivityDetail,
  StravaGear,
  StravaRawCache,
  StravaStreams,
  StravaZones,
} from '../plugins/stores/strava'
import { joinSegments, QUARTZ } from '../util/path'
import { readStravaCacheFile, updateStravaCacheFile } from '../util/strava-cache-file'
import { compareStravaIds, requireStravaId, stravaObjectId } from '../util/strava-id'
import { reconcileStravaActivities, stravaFetchAfter } from '../util/strava-sync-window'
import { stravaSyncRefreshDays } from '../util/sync-refresh-window'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord } from '../util/type-guards'
import { syncRouvyWahooPairs } from './sync-rouvy-wahoo'
import { mapActivity, parseActivityDetail, parseGear } from './sync-strava/records'
import {
  apiUrl,
  authHeaders,
  fetchCity,
  fetchStrava,
  logDeferredReads,
  mapPool,
  resolveToken,
} from './sync-strava/transport'

export { mapActivity, parseGear, parseRunSplits } from './sync-strava/records'

const PER_PAGE = 200

const CACHE_VERSION = 8

const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')

const CONCURRENCY = 5

const DETAIL_BACKFILL_BATCH = 50

async function readCache(): Promise<StravaRawCache | null> {
  return readStravaCacheFile(cacheFile)
}

async function fetchActivities(
  token: string,
  after: number,
): Promise<{ activities: RawStravaActivity[]; athleteId: string | null }> {
  const headers = authHeaders(token)
  const activities: RawStravaActivity[] = []
  let athleteId: string | null = null
  for (let page = 1; ; page++) {
    const url = apiUrl('/athlete/activities', { after, per_page: PER_PAGE, page })
    const res = await fetchStrava(url, { headers })
    if (!res) throw new Error(`activity fetch failed at page ${page}`)
    const batch = (await res.json()) as Record<string, unknown>[]
    if (!Array.isArray(batch)) throw new Error('Invalid Strava activities response')
    if (batch.length === 0) break
    for (const raw of batch) {
      if (!isRecord(raw) || !isRecord(raw.athlete)) throw new Error('Missing Strava athlete ID')
      const id = stravaObjectId(raw.athlete)
      if (athleteId !== null && athleteId !== id)
        throw new Error('Strava athlete ID changed between activities')
      athleteId = id
      activities.push(mapActivity(raw))
    }
    console.log(`[strava] page ${page}: ${batch.length} activities`)
    if (batch.length < PER_PAGE) break
  }
  return { activities, athleteId }
}

async function fetchStreams(token: string, id: string): Promise<StravaStreams | null> {
  const headers = authHeaders(token)
  const url = apiUrl(`/activities/${id}/streams`, {
    keys: 'time,latlng,altitude,distance,watts,heartrate,cadence',
    key_by_type: 'true',
  })
  const res = await fetchStrava(url, { headers })
  if (!res) return null
  const data = (await res.json()) as Record<string, { data?: unknown[] }>
  return {
    time: (data.time?.data as number[]) ?? [],
    latlng: (data.latlng?.data as [number, number][]) ?? [],
    altitude: (data.altitude?.data as number[]) ?? [],
    distance: (data.distance?.data as number[]) ?? [],
    watts: (data.watts?.data as number[]) ?? [],
    heartrate: (data.heartrate?.data as number[]) ?? [],
    cadence: (data.cadence?.data as number[]) ?? [],
  }
}

async function fetchGear(token: string, id: string): Promise<StravaGear | null> {
  const res = await fetchStrava(apiUrl(`/gear/${encodeURIComponent(id)}`), {
    headers: authHeaders(token),
  })
  if (!res || !res.ok) return null
  const gear = parseGear(await res.json())
  return gear?.id === id ? gear : null
}

async function fetchAthlete(
  token: string,
  gear: Record<string, StravaGear>,
): Promise<{ id: string; ftp: number | null } | null> {
  const res = await fetchStrava(apiUrl('/athlete'), { headers: authHeaders(token) })
  if (!res || !res.ok) return null
  const data: unknown = await res.json()
  if (!isRecord(data)) throw new Error('Invalid Strava athlete response')
  const id = stravaObjectId(data)
  for (const items of [data.bikes, data.shoes]) {
    if (!Array.isArray(items)) continue
    for (const item of items) {
      const summary = parseGear(item)
      if (summary)
        gear[summary.id] = {
          ...gear[summary.id],
          ...summary,
          brandName: gear[summary.id]?.brandName ?? summary.brandName,
          modelName: gear[summary.id]?.modelName ?? summary.modelName,
        }
    }
  }
  return { id, ftp: typeof data.ftp === 'number' && data.ftp > 0 ? Math.round(data.ftp) : null }
}

async function refreshGear(
  token: string,
  gear: Record<string, StravaGear>,
  activities: Record<string, RawStravaActivity>,
): Promise<string[]> {
  const usedIds = Object.values(activities).flatMap(activity =>
    activity.gearId ? [activity.gearId] : [],
  )
  const gearIds = [...new Set([...Object.keys(gear), ...usedIds])]
  const failed: string[] = []
  await mapPool(gearIds, CONCURRENCY, async id => {
    const fetched = await fetchGear(token, id)
    if (fetched)
      gear[id] = { ...gear[id], ...fetched, name: fetched.name ?? gear[id]?.name ?? null }
    else failed.push(id)
  })
  console.log(
    `[strava] equipment: ${gearIds.length - failed.length}/${gearIds.length} refreshed, ${failed.length} deferred or unavailable`,
  )
  return failed
}

async function fetchZones(token: string, ftp: number | null): Promise<StravaZones | null> {
  const res = await fetchStrava(apiUrl('/athlete/zones'), { headers: authHeaders(token) })
  if (!res || !res.ok) return ftp != null ? { hr: [], power: [], ftp } : null
  const data = (await res.json()) as {
    heart_rate?: { zones?: { min: number; max: number }[] }
    power?: { zones?: { min: number; max: number }[] }
  }
  const bounds = (zones: { max: number }[] | undefined): number[] =>
    (zones ?? []).map(z => z.max).filter(m => m > 0)
  const hr = bounds(data.heart_rate?.zones)
  const power = bounds(data.power?.zones)
  if (hr.length === 0 && power.length === 0 && ftp == null) return null
  return { hr, power, ftp }
}

async function fetchActivityDetail(
  token: string,
  id: string,
  athleteId: string | null,
): Promise<RawStravaActivityDetail | null> {
  const res = await fetchStrava(apiUrl(`/activities/${id}`, { include_all_efforts: 'true' }), {
    headers: authHeaders(token),
  })
  if (!res) return null
  const data: unknown = await res.json()
  return parseActivityDetail(data, id, athleteId)
}

function progress(label: string, done: number, total: number): void {
  const width = 22
  const filled = Math.round((done / total) * width)
  const bar = '█'.repeat(filled) + '░'.repeat(width - filled)
  process.stdout.write(`\r[strava] ${label.padEnd(8)} [${bar}] ${done}/${total}`)
  if (done >= total) process.stdout.write('\n')
}

async function main(): Promise<void> {
  const prev = await readCache()
  const gearOnly = process.argv.includes('--gear-only')
  const detailArgument = process.argv.indexOf('--activity-detail')
  const detailId = detailArgument < 0 ? null : requireStravaId(process.argv[detailArgument + 1])
  if (detailId != null && (!prev || !prev.activities[detailId]))
    throw new Error('--activity-detail requires one existing cached activity ID')
  if (gearOnly && detailId != null) throw new Error('choose --gear-only or --activity-detail')
  if (gearOnly && !prev) throw new Error('--gear-only needs an existing Strava cache')
  const { access, refreshToken } = await resolveToken(prev)
  if (detailId != null && prev) {
    const detail = await fetchActivityDetail(access, detailId, prev.athleteId)
    if (!detail) throw new Error(`activity detail ${detailId} unavailable; cache preserved`)
    await updateStravaCacheFile(cacheFile, current => {
      if (!current || current.athleteId !== prev.athleteId || !current.activities[detailId])
        throw new Error('Strava cache identity changed during detail refresh; cache preserved')
      return {
        ...current,
        auth: { refreshToken, obtainedAt: Date.now() },
        activityDetails: { ...current.activityDetails, [detailId]: detail },
      }
    })
    await refreshTriathlonRouteSource()
    console.log(`[strava] refreshed activity detail ${detailId} → ${cacheFile}`)
    return
  }
  if (gearOnly && prev) {
    const gear: Record<string, StravaGear> = { ...prev.gear }
    const athlete = await fetchAthlete(access, gear)
    if (!athlete || (prev.athleteId !== null && athlete.id !== prev.athleteId))
      throw new Error('Strava athlete ID changed; cache preserved')
    const failed = await refreshGear(access, gear, prev.activities)
    if (failed.length > 0)
      throw new Error(`equipment refresh incomplete for ${failed.join(', ')}; cache preserved`)
    await updateStravaCacheFile(cacheFile, current => {
      if (!current || current.athleteId !== prev.athleteId)
        throw new Error('Strava cache athlete changed during equipment refresh; cache preserved')
      return {
        ...current,
        athleteId: athlete.id,
        auth: { refreshToken, obtainedAt: Date.now() },
        gear: { ...current.gear, ...gear },
      }
    })
    await refreshTriathlonRouteSource()
    console.log(`[strava] wrote ${Object.keys(gear).length} equipment records → ${cacheFile}`)
    return
  }
  const stale = (prev?.version ?? 0) < CACHE_VERSION
  if (stale && prev)
    console.log('[strava] cache schema bumped → re-pulling all summaries to backfill')
  const refreshWindowDays = stravaSyncRefreshDays()
  const after = stravaFetchAfter(prev?.lastActivityStart, stale, refreshWindowDays)
  if (!stale && after < (prev?.lastActivityStart ?? 0))
    console.log(`[strava] refreshing ${refreshWindowDays}d recent activity overlap`)
  const { activities, athleteId } = await fetchActivities(access, after)

  const reconciled = reconcileStravaActivities(prev?.activities, activities, after)
  const merged = reconciled.activities

  let lastActivityStart = 0
  for (const a of Object.values(merged)) {
    const epoch = Math.floor(Date.parse(a.startDate) / 1000)
    if (Number.isFinite(epoch) && epoch > lastActivityStart) lastActivityStart = epoch
  }

  const streams: Record<string, StravaStreams> = { ...prev?.streams }
  const geo: Record<string, string> = { ...prev?.geo }
  const activityDetails: Record<string, RawStravaActivityDetail> = { ...prev?.activityDetails }
  const gear: Record<string, StravaGear> = { ...prev?.gear }
  for (const id of reconciled.removedIds) {
    delete streams[id]
    delete geo[id]
    delete activityDetails[id]
  }
  for (const id of Object.keys(activityDetails))
    if (merged[id] === undefined) delete activityDetails[id]
  let zones: StravaZones | undefined = prev?.zones
  const writeCache = async (): Promise<void> => {
    const cache: StravaRawCache = {
      version: CACHE_VERSION,
      athleteId: authenticatedAthleteId,
      auth: { refreshToken, obtainedAt: Date.now() },
      lastSync: Date.now(),
      lastActivityStart,
      activities: Object.fromEntries(
        Object.entries(merged).sort(([a], [b]) => compareStravaIds(a, b)),
      ),
      activityDetails,
      gear,
      streams,
      geo,
      zones,
    }
    await updateStravaCacheFile(cacheFile, current => {
      if (current?.athleteId != null && current.athleteId !== cache.athleteId)
        throw new Error('Strava cache athlete changed during sync; cache preserved')
      for (const [id, detail] of Object.entries(current?.activityDetails ?? {}))
        if (merged[id] && (detail.fetchedAt ?? 0) > (activityDetails[id]?.fetchedAt ?? 0))
          activityDetails[id] = detail
      return cache
    })
  }
  const athlete = await fetchAthlete(access, gear)
  if (
    !athlete ||
    (athleteId !== null && athleteId !== athlete.id) ||
    (prev?.athleteId != null && prev.athleteId !== athlete.id)
  )
    throw new Error('Strava athlete identity unavailable or changed; cache preserved')
  const authenticatedAthleteId = athlete.id
  const fetchedZones = await fetchZones(access, athlete.ftp)
  if (fetchedZones) zones = fetchedZones
  else if (!zones) console.log('[strava] no athlete zones (needs profile:read_all) — deriving')

  await refreshGear(access, gear, merged)

  const needStreams = Object.values(merged)
    .filter(a => {
      if (normalizeKind(a.sportType) === null) return false
      const s = streams[String(a.id)]
      return !s || s.heartrate === undefined || !s.time?.length
    })
    .sort((x, y) => y.startDate.localeCompare(x.startDate))
  let si = 0
  await mapPool(needStreams, CONCURRENCY, async a => {
    const s = await fetchStreams(access, a.id)
    if (s) streams[String(a.id)] = s
    progress('streams', ++si, needStreams.length)
  })

  const needGeo = Object.values(merged).filter(a => {
    const s = streams[String(a.id)]
    return (
      normalizeKind(a.sportType) !== null &&
      s !== undefined &&
      s.latlng.length >= 2 &&
      !geo[String(a.id)]
    )
  })
  let gi = 0
  for (const a of needGeo) {
    const s = streams[String(a.id)]!
    const city = await fetchCity(s.latlng[0][0], s.latlng[0][1])
    if (city) geo[String(a.id)] = city
    progress('geocode', ++gi, needGeo.length)
  }

  const detailCandidates = Object.values(merged)
    .filter(a => normalizeKind(a.sportType) !== null)
    .sort((left, right) => right.startDate.localeCompare(left.startDate))
  const recentDetailCutoff = Date.now() - refreshWindowDays * 86_400_000
  const recentDetails = detailCandidates
    .filter(activity => Date.parse(activity.startDate) >= recentDetailCutoff)
    .sort(
      (left, right) =>
        (activityDetails[String(left.id)]?.fetchedAt ?? 0) -
          (activityDetails[String(right.id)]?.fetchedAt ?? 0) ||
        right.startDate.localeCompare(left.startDate),
    )
  const recentIds = new Set(recentDetails.map(activity => activity.id))
  const historicalBackfill = detailCandidates
    .filter(
      activity =>
        !recentIds.has(activity.id) &&
        !hasFetchedActivityDetail(activityDetails[String(activity.id)]),
    )
    .slice(0, DETAIL_BACKFILL_BATCH)
  const needDetails = [...recentDetails, ...historicalBackfill]
  let di = 0
  await mapPool(needDetails, CONCURRENCY, async a => {
    const detail = await fetchActivityDetail(access, a.id, authenticatedAthleteId)
    if (detail) {
      activityDetails[String(a.id)] = detail
      if (detail.calories != null) merged[String(a.id)].calories = detail.calories
    }
    progress('details', ++di, needDetails.length)
  })

  await writeCache()
  await syncRouvyWahooPairs()
  await refreshTriathlonRouteSource()
  logDeferredReads()
  console.log(
    `[strava] wrote ${Object.keys(merged).length} activities (+${activities.length} new), ${Object.keys(streams).length} streams, ${Object.keys(activityDetails).length} details, ${Object.keys(geo).length} located → ${cacheFile}`,
  )
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[strava] sync failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
