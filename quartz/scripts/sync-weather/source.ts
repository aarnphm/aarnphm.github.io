import type { RawStravaActivity } from '../../plugins/stores/strava'
import { normalizeKind } from '../../plugins/stores/strava'
import { type WeatherActivityCandidate } from '../../plugins/stores/weather'
import { joinSegments, QUARTZ } from '../../util/path'
import { readStravaCacheFile } from '../../util/strava-cache-file'
import { stravaId } from '../../util/strava-id'
import { isRecord, readNumber, readString } from '../../util/type-guards'
import {
  routeHourQueries,
  routeWeatherFingerprint,
  type RouteHourQuery,
  type RouteWeatherStream,
} from '../../util/weather-route-hours'

const stravaCacheFile = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')

interface StravaWeatherSource {
  activities: RawStravaActivity[]
  streams: Record<string, RouteWeatherStream>
  details: Record<string, { description: string | null; fetchedAt: number }>
}

export interface RouteWeatherCandidate extends WeatherActivityCandidate {
  routeFingerprint: string
  stream: RouteWeatherStream
  queries: RouteHourQuery[]
}

function readActivity(value: unknown): RawStravaActivity | null {
  if (!isRecord(value)) return null
  const id = stravaId(value.id)
  const name = readString(value, 'name')
  const sportType = readString(value, 'sportType')
  const distance = readNumber(value, 'distance')
  const movingTime = readNumber(value, 'movingTime')
  const elapsedTime = readNumber(value, 'elapsedTime')
  const totalElevationGain = readNumber(value, 'totalElevationGain')
  const startDate = readString(value, 'startDate')
  const startDateLocal = readString(value, 'startDateLocal')
  const averageSpeed = readNumber(value, 'averageSpeed')
  if (
    id == null ||
    name == null ||
    sportType == null ||
    distance == null ||
    movingTime == null ||
    elapsedTime == null ||
    totalElevationGain == null ||
    startDate == null ||
    startDateLocal == null ||
    averageSpeed == null
  )
    return null
  return {
    id,
    name,
    sportType,
    distance,
    movingTime,
    elapsedTime,
    totalElevationGain,
    startDate,
    startDateLocal,
    averageSpeed,
    averageHeartrate: readNumber(value, 'averageHeartrate'),
    maxHeartrate: readNumber(value, 'maxHeartrate'),
    averageWatts: readNumber(value, 'averageWatts'),
    weightedAverageWatts: readNumber(value, 'weightedAverageWatts'),
    maxWatts: readNumber(value, 'maxWatts'),
    kilojoules: readNumber(value, 'kilojoules'),
    deviceWatts: typeof value.deviceWatts === 'boolean' ? value.deviceWatts : undefined,
    averageCadence: readNumber(value, 'averageCadence'),
    sufferScore: readNumber(value, 'sufferScore'),
    averageTemp: readNumber(value, 'averageTemp'),
    calories: readNumber(value, 'calories'),
  }
}

function coordinate(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null
  const lat = value[0]
  const lng = value[1]
  return typeof lat === 'number' &&
    typeof lng === 'number' &&
    Number.isFinite(lat) &&
    Number.isFinite(lng)
    ? [lat, lng]
    : null
}

function finiteNumberArray(value: unknown): number[] | null {
  if (!Array.isArray(value)) return null
  const numbers = value.filter(item => typeof item === 'number' && Number.isFinite(item))
  return numbers.length === value.length ? numbers : null
}

function isMonotonic(values: readonly number[], strictly: boolean): boolean {
  return values.every(
    (value, index) =>
      value >= 0 &&
      (index === 0 || (strictly ? value > values[index - 1] : value >= values[index - 1])),
  )
}

function readStreams(value: unknown): Record<string, RouteWeatherStream> {
  if (!isRecord(value)) return {}
  const out: Record<string, RouteWeatherStream> = {}
  for (const [id, raw] of Object.entries(value)) {
    if (!isRecord(raw) || !Array.isArray(raw.latlng)) continue
    const coordinates = raw.latlng.map(coordinate)
    const timeS = finiteNumberArray(raw.time)
    const distanceM = finiteNumberArray(raw.distance)
    if (
      coordinates.some(point => point === null) ||
      !timeS ||
      !distanceM ||
      coordinates.length < 2 ||
      timeS.length !== coordinates.length ||
      distanceM.length !== coordinates.length ||
      !isMonotonic(timeS, true) ||
      !isMonotonic(distanceM, false)
    )
      continue
    const latlng = coordinates.filter(point => point !== null)
    out[id] = { timeS, distanceM, latlng }
  }
  return out
}

function readDetails(
  value: unknown,
): Record<string, { description: string | null; fetchedAt: number }> {
  if (!isRecord(value)) return {}
  const details: Record<string, { description: string | null; fetchedAt: number }> = {}
  for (const [id, raw] of Object.entries(value)) {
    if (!isRecord(raw) || (raw.description !== null && typeof raw.description !== 'string'))
      continue
    const fetchedAt = readNumber(raw, 'fetchedAt')
    if (fetchedAt == null || !Number.isFinite(fetchedAt)) continue
    details[id] = { description: raw.description, fetchedAt }
  }
  return details
}

export async function readStravaSource(): Promise<StravaWeatherSource | null> {
  try {
    const raw = await readStravaCacheFile(stravaCacheFile)
    if (!isRecord(raw) || !isRecord(raw.activities)) return null
    const activities = Object.values(raw.activities)
      .map(readActivity)
      .filter(activity => activity !== null)
    return {
      activities,
      streams: readStreams(raw.streams),
      details: readDetails(raw.activityDetails),
    }
  } catch {
    return null
  }
}

function routeCenter(latlng: readonly [number, number][]): { latitude: number; longitude: number } {
  const stride = Math.max(1, Math.floor(latlng.length / 200))
  let latitude = 0
  let longitude = 0
  let count = 0
  for (let i = 0; i < latlng.length; i += stride) {
    latitude += latlng[i][0]
    longitude += latlng[i][1]
    count += 1
  }
  if (count === 0) throw new Error('route must contain at least one coordinate')
  return { latitude: latitude / count, longitude: longitude / count }
}

export function candidate(
  activity: RawStravaActivity,
  stream: RouteWeatherStream,
): RouteWeatherCandidate | null {
  const sport = normalizeKind(activity.sportType)
  if (!sport || sport === 'strength') return null
  const startMs = Date.parse(activity.startDate)
  const durationS = activity.elapsedTime > 0 ? activity.elapsedTime : activity.movingTime
  if (!Number.isFinite(startMs) || durationS <= 0) return null
  const end = new Date(startMs + durationS * 1000).toISOString()
  const center = routeCenter(stream.latlng)
  const base: WeatherActivityCandidate = {
    activityId: activity.id,
    date: activity.startDateLocal.slice(0, 10),
    start: new Date(startMs).toISOString(),
    end,
    latitude: center.latitude,
    longitude: center.longitude,
    durationS,
    routeFingerprint: '',
  }
  const routeFingerprint = routeWeatherFingerprint(activity.id, base.start, base.end, stream)
  const withFingerprint = { ...base, routeFingerprint }
  return { ...withFingerprint, stream, queries: routeHourQueries(withFingerprint, stream) }
}
