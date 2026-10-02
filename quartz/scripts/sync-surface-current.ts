import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import { dirname, join } from 'node:path'
import type { RawStravaActivity, StravaStreams } from '../plugins/stores/strava'
import { readStravaCacheFile } from '../util/strava-cache-file'
import {
  buildSurfaceCurrentEstimate,
  findSurfaceCurrentElement,
  loofsRequestsForHour,
  parseLoofsFieldAscii,
  parseLoofsMeshAscii,
  parseSurfaceCurrentEstimate,
  type SurfaceCurrentEstimate,
  type SurfaceCurrentField,
  type SurfaceCurrentMesh,
} from '../util/surface-current'
import { refreshTriathlonRouteSource } from '../util/triathlon-cache'
import { isRecord } from '../util/type-guards'
import { routeWeatherFingerprint } from '../util/weather-route-hours'

const HOUR_MS = 3_600_000
const CACHE = join(process.cwd(), 'quartz', '.quartz-cache')
const WEATHER = join(CACHE, 'weather.json')
const RUNS = join(CACHE, 'surface-current')
const hash = (value: string): string => createHash('sha256').update(value).digest('hex')

interface RequestEvidence {
  url: string
  status: number | null
  bytes: number
  sha256: string | null
  error: string | null
}

function argumentsFromCli(): { id: number | null; force: boolean } {
  let id: number | null = null
  let force = false
  for (let index = 2; index < process.argv.length; index += 1) {
    const argument = process.argv[index]
    if (argument === '--force') force = true
    else if (argument === '--id') {
      const value = Number(process.argv[++index])
      if (!Number.isSafeInteger(value) || value <= 0)
        throw new Error('--id requires a positive Strava activity ID')
      id = value
    } else throw new Error(`Unknown argument ${argument}; use --id STRAVA_ID and/or --force`)
  }
  return { id, force }
}

async function readWeather(): Promise<Record<string, unknown>> {
  const value: unknown = JSON.parse(await fs.readFile(WEATHER, 'utf8'))
  if (!isRecord(value) || !isRecord(value.activities))
    throw new Error('weather.json has no activities')
  return value
}

function sourceRoute(
  activity: RawStravaActivity,
  stream: StravaStreams | undefined,
): { timeS: number[]; latlng: [number, number][]; distanceM: number[] } | null {
  if (
    activity.sportType !== 'Swim' ||
    activity.trainer === true ||
    !stream?.time ||
    stream.latlng.length < 2 ||
    !Number.isFinite(activity.elapsedTime) ||
    activity.elapsedTime <= 0 ||
    activity.elapsedTime > 43_200
  )
    return null
  const validPoints = stream.time.filter((seconds, index) => {
    const coordinate = stream.latlng[index]
    return (
      Number.isFinite(seconds) &&
      seconds >= 0 &&
      seconds <= activity.elapsedTime &&
      coordinate &&
      coordinate.every(Number.isFinite) &&
      Math.abs(coordinate[0]) <= 90 &&
      Math.abs(coordinate[1]) <= 180
    )
  })
  if (validPoints.length < 2) return null
  return { timeS: stream.time, latlng: stream.latlng, distanceM: stream.distance }
}

async function requestText(url: string, evidence: RequestEvidence[]): Promise<string | null> {
  const record: RequestEvidence = { url, status: null, bytes: 0, sha256: null, error: null }
  evidence.push(record)
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(30_000) })
    record.status = response.status
    if (!response.ok) {
      record.error = `HTTP ${response.status}`
      return null
    }
    const text = await response.text()
    record.bytes = Buffer.byteLength(text)
    if (record.bytes > 20_000_000) throw new Error('NOAA ASCII response exceeds 20 MB')
    record.sha256 = hash(text)
    return text
  } catch (error) {
    record.error = error instanceof Error ? error.message : String(error)
    return null
  }
}

async function loadMesh(
  validTime: string,
  evidence: RequestEvidence[],
): Promise<{
  mesh: SurfaceCurrentMesh
  fingerprint: string
  sourceUrl: string
  date: string
} | null> {
  const request = loofsRequestsForHour(validTime)
  if (!request) return null
  const date = request.cycleTime.slice(0, 10)
  const path = join(RUNS, `mesh-${date}.json`)
  try {
    const cached: unknown = JSON.parse(await fs.readFile(path, 'utf8'))
    if (
      isRecord(cached) &&
      typeof cached.ascii === 'string' &&
      typeof cached.sourceUrl === 'string' &&
      cached.date === date
    ) {
      const mesh = parseLoofsMeshAscii(cached.ascii)
      if (mesh)
        return { mesh, fingerprint: hash(JSON.stringify(mesh)), sourceUrl: cached.sourceUrl, date }
    }
  } catch (error) {
    if (!isRecord(error) || error.code !== 'ENOENT')
      console.warn(`[water-current] ignoring unreadable mesh cache ${path}`)
  }
  for (const sourceUrl of [request.recentUrl, request.archiveUrl]) {
    const ascii = await requestText(`${sourceUrl}?lon,lat,nv,siglay%5B0%5D`, evidence)
    if (!ascii) continue
    const mesh = parseLoofsMeshAscii(ascii)
    if (!mesh) continue
    await fs.mkdir(RUNS, { recursive: true })
    await fs.writeFile(path, JSON.stringify({ date, sourceUrl, fetchedAt: Date.now(), ascii }))
    return { mesh, fingerprint: hash(JSON.stringify(mesh)), sourceUrl, date }
  }
  return null
}

async function loadField(
  validTime: string,
  firstElement: number,
  lastElement: number,
  evidence: RequestEvidence[],
  extracts: { url: string; ascii: string }[],
): Promise<SurfaceCurrentField | null> {
  const request = loofsRequestsForHour(validTime)
  if (!request) return null
  const slice = `[${firstElement}:1:${lastElement}]`
  const query = `Times,u[0][0]${slice},v[0][0]${slice},wet_cells[0]${slice}`
  for (const sourceUrl of [request.recentUrl, request.archiveUrl]) {
    const url = `${sourceUrl}?${encodeURIComponent(query).replaceAll('%2C', ',')}`
    const ascii = await requestText(url, evidence)
    if (!ascii) continue
    const field = parseLoofsFieldAscii(ascii, firstElement, request.cycleTime, validTime, sourceUrl)
    if (!field) continue
    extracts.push({ url, ascii })
    return field
  }
  return null
}

async function mergeEstimate(estimate: SurfaceCurrentEstimate): Promise<boolean> {
  await fs.mkdir(dirname(WEATHER), { recursive: true })
  const path = `${WEATHER}.tmp-${process.pid}-${randomUUID()}`
  try {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const original = await fs.readFile(WEATHER, 'utf8')
      const latest: unknown = JSON.parse(original)
      if (!isRecord(latest) || !isRecord(latest.activities))
        throw new Error('Weather cache changed shape during sync')
      const activity = latest.activities[String(estimate.activityId)]
      if (
        !isRecord(activity) ||
        activity.activityId !== estimate.activityId ||
        activity.routeFingerprint !== estimate.routeFingerprint ||
        typeof activity.start !== 'string' ||
        typeof activity.end !== 'string' ||
        Date.parse(activity.start) !== Date.parse(estimate.start) ||
        Date.parse(activity.end) !== Date.parse(estimate.end)
      )
        return false
      const merged = {
        ...latest,
        activities: {
          ...latest.activities,
          [String(estimate.activityId)]: { ...activity, surfaceCurrent: estimate },
        },
      }
      await fs.writeFile(path, JSON.stringify(merged, null, 2))
      if ((await fs.readFile(WEATHER, 'utf8')) !== original) continue
      await fs.rename(path, WEATHER)
      return true
    }
    throw new Error('Weather cache changed repeatedly during surface current merge')
  } finally {
    await fs.rm(path, { force: true })
  }
}

async function main(): Promise<void> {
  const args = argumentsFromCli()
  const strava = await readStravaCacheFile(join(CACHE, 'strava.json'))
  if (!strava) throw new Error('No local Strava cache')
  const weather = await readWeather()
  if (!isRecord(weather.activities)) throw new Error('No weather activities')
  const weatherActivities = weather.activities
  const cutoff = Date.now() - 14 * 86_400_000
  const candidates = Object.values(strava.activities).filter(activity =>
    args.id === null
      ? Date.parse(activity.startDate) >= cutoff &&
        Object.hasOwn(weatherActivities, String(activity.id))
      : activity.id === args.id,
  )
  if (args.id !== null && candidates.length === 0)
    throw new Error(`Strava activity ${args.id} is absent from the cache`)
  let updated = false
  for (const activity of candidates) {
    const route = sourceRoute(activity, strava.streams?.[String(activity.id)])
    const cachedWeather = weatherActivities[String(activity.id)]
    if (!route || !isRecord(cachedWeather)) {
      if (args.id !== null)
        throw new Error('Surface currents require a GPS swim and its matching weather entry')
      continue
    }
    const start = new Date(Date.parse(activity.startDate)).toISOString()
    const end = new Date(Date.parse(start) + activity.elapsedTime * 1_000).toISOString()
    const fingerprint = routeWeatherFingerprint(activity.id, start, end, route)
    if (
      cachedWeather.routeFingerprint !== fingerprint ||
      typeof cachedWeather.start !== 'string' ||
      typeof cachedWeather.end !== 'string' ||
      Date.parse(cachedWeather.start) !== Date.parse(start) ||
      Date.parse(cachedWeather.end) !== Date.parse(end)
    ) {
      console.warn(`[water-current] ${activity.id}: weather route is stale, run weather sync first`)
      continue
    }
    const previous = parseSurfaceCurrentEstimate(cachedWeather.surfaceCurrent)
    if (
      !args.force &&
      previous?.activityId === activity.id &&
      previous.routeFingerprint === fingerprint &&
      previous.start === start &&
      previous.end === end
    ) {
      console.log(
        `[water-current] ${activity.id}: unchanged route, cached ${previous.summary.coveragePct}% coverage`,
      )
      continue
    }
    const evidence: RequestEvidence[] = [],
      extracts: { url: string; ascii: string }[] = []
    const firstHourMs = Math.floor(Date.parse(start) / HOUR_MS) * HOUR_MS
    const lastHourMs = Math.ceil(Date.parse(end) / HOUR_MS) * HOUR_MS
    const mesh = await loadMesh(new Date(firstHourMs).toISOString(), evidence)
    if (!mesh) {
      console.warn(`[water-current] ${activity.id}: NOAA grid unavailable, preserving cached data`)
      continue
    }
    const elements = route.latlng.flatMap(coordinate => {
      const element = findSurfaceCurrentElement(mesh.mesh, coordinate[0], coordinate[1])
      return element === null ? [] : [element]
    })
    if (elements.length === 0) {
      console.log(`[water-current] ${activity.id}: swim lies outside the Lake Ontario model`)
      continue
    }
    const firstElement = Math.min(...elements),
      lastElement = Math.max(...elements)
    const fields: SurfaceCurrentField[] = [],
      meshes = new Map([[mesh.date, mesh]])
    for (let hourMs = firstHourMs; hourMs <= lastHourMs; hourMs += HOUR_MS) {
      const validTime = new Date(hourMs).toISOString()
      const request = loofsRequestsForHour(validTime)
      if (!request) continue
      const date = request.cycleTime.slice(0, 10)
      let hourlyMesh = meshes.get(date)
      if (!hourlyMesh) {
        const loaded = await loadMesh(validTime, evidence)
        if (loaded) {
          meshes.set(date, loaded)
          hourlyMesh = loaded
        }
      }
      if (!hourlyMesh || hourlyMesh.fingerprint !== mesh.fingerprint) {
        console.warn(
          `[water-current] ${activity.id}: NOAA grid changed at ${validTime}, retaining a gap`,
        )
        continue
      }
      const field = await loadField(validTime, firstElement, lastElement, evidence, extracts)
      if (field) fields.push(field)
      else console.warn(`[water-current] ${activity.id}: no valid NOAA field at ${validTime}`)
    }
    const estimate = buildSurfaceCurrentEstimate({
      activityId: activity.id,
      routeFingerprint: fingerprint,
      start,
      end,
      timeS: route.timeS,
      latlng: route.latlng,
      mesh: mesh.mesh,
      fields,
      computedAt: Date.now(),
    })
    if (!parseSurfaceCurrentEstimate(estimate))
      throw new Error(`Generated surface current estimate failed validation for ${activity.id}`)
    const failedProvider = fields.length !== (lastHourMs - firstHourMs) / HOUR_MS + 1
    const preservePrevious = failedProvider && previous?.routeFingerprint === fingerprint
    const persisted = preservePrevious ? false : await mergeEstimate(estimate)
    await fs.mkdir(RUNS, { recursive: true })
    await fs.writeFile(
      join(RUNS, `activity-${activity.id}.json`),
      JSON.stringify(
        {
          command: `pnpm water-current:sync --id ${activity.id} --force`,
          inputs: {
            stravaCache: 'quartz/.quartz-cache/strava.json',
            activityId: activity.id,
            start,
            end,
            routeFingerprint: fingerprint,
            meshSources: [...meshes.values()].map(value => ({
              date: value.date,
              sourceUrl: value.sourceUrl,
              fingerprint: value.fingerprint,
            })),
          },
          requests: evidence,
          fieldExtracts: extracts,
          estimate,
          persisted,
          preservedPrevious: preservePrevious,
        },
        null,
        2,
      ),
    )
    console.log(
      `[water-current] ${activity.id}: ${estimate.summary.averageSpeedMps ?? 'unavailable'} m/s, toward ${estimate.summary.averageDirectionDeg ?? 'unavailable'} degrees, ${estimate.summary.coveragePct}% elapsed coverage (${estimate.summary.coveredDurationS}/${activity.elapsedTime}s); ${persisted ? 'saved' : 'preserved previous cache'}`,
    )
    updated ||= persisted
  }
  if (updated) await refreshTriathlonRouteSource()
}

main().catch(error => {
  console.error(
    `[water-current] sync failed: ${error instanceof Error ? error.message : String(error)}`,
  )
  process.exitCode = 1
})
