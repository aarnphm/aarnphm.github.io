import { XMLParser } from 'fast-xml-parser'
import fs from 'node:fs/promises'
import { resolve } from 'node:path'
import { isRecord } from '../../util/type-guards'
import {
  haversineMeters,
  timedValue,
  type CourseRoute,
  type CourseRoutePoint,
  type Position,
  type RoutePoint,
} from './conversion-model'

function values(value: unknown): unknown[] {
  return Array.isArray(value) ? value : value == null ? [] : [value]
}

function scalar(value: unknown): string | number | null {
  if (typeof value === 'string' || typeof value === 'number') return value
  if (!isRecord(value)) return null
  const text = value['#text']
  return typeof text === 'string' || typeof text === 'number' ? text : null
}

function xmlNumber(value: unknown, key: string, required: boolean): number | null {
  const raw = scalar(value)
  if (raw == null) {
    if (required) throw new Error(`GPX ${key} is required`)
    return null
  }
  const parsed = typeof raw === 'number' ? raw : Number(raw)
  if (!Number.isFinite(parsed)) throw new Error(`GPX ${key} must be finite`)
  return parsed
}

function extensionRecord(extensions: Record<string, unknown>): Record<string, unknown> | null {
  for (const key of ['gpxtpx:TrackPointExtension', 'TrackPointExtension']) {
    const candidates = values(extensions[key])
    const first = candidates[0]
    if (isRecord(first)) return first
  }
  return null
}

function telemetryNumber(
  record: Record<string, unknown> | null,
  keys: readonly string[],
  label: string,
): number | null {
  if (!record) return null
  for (const key of keys) {
    if (record[key] != null) return xmlNumber(record[key], label, false)
  }
  return null
}

function parseRoutePoint(raw: unknown, index: number): RoutePoint {
  if (!isRecord(raw)) throw new Error(`GPX trackpoint ${index} must be an object`)
  const latitude = xmlNumber(raw['@_lat'], `trackpoint ${index} latitude`, true)
  const longitude = xmlNumber(raw['@_lon'], `trackpoint ${index} longitude`, true)
  if (
    latitude == null ||
    longitude == null ||
    latitude < -90 ||
    latitude > 90 ||
    longitude < -180 ||
    longitude > 180
  )
    throw new Error(`GPX trackpoint ${index} has invalid coordinates`)
  const rawTime = scalar(raw.time)
  if (rawTime == null) throw new Error(`GPX trackpoint ${index} has no time`)
  const timed = timedValue(String(rawTime), `GPX trackpoint ${index} time`)
  const altitudeM = xmlNumber(raw.ele, `trackpoint ${index} elevation`, false)
  const extensions = isRecord(raw.extensions) ? raw.extensions : null
  const trackPointExtensions = extensions ? extensionRecord(extensions) : null
  const heartRateBpm = telemetryNumber(
    trackPointExtensions,
    ['gpxtpx:hr', 'hr'],
    `trackpoint ${index} heart rate`,
  )
  const cadencePerMinute = telemetryNumber(
    trackPointExtensions,
    ['gpxtpx:cad', 'cad'],
    `trackpoint ${index} cadence`,
  )
  const speedMps = telemetryNumber(
    trackPointExtensions,
    ['gpxtpx:speed', 'speed'],
    `trackpoint ${index} speed`,
  )
  const powerW = telemetryNumber(
    extensions,
    ['power', 'gpxtpx:power', 'watts'],
    `trackpoint ${index} power`,
  )
  if (heartRateBpm != null && (heartRateBpm <= 0 || heartRateBpm > 300))
    throw new Error(`GPX trackpoint ${index} heart rate is invalid`)
  if (cadencePerMinute != null && cadencePerMinute < 0)
    throw new Error(`GPX trackpoint ${index} cadence is invalid`)
  if (speedMps != null && speedMps < 0) throw new Error(`GPX trackpoint ${index} speed is invalid`)
  if (powerW != null && powerW < 0) throw new Error(`GPX trackpoint ${index} power is invalid`)
  return {
    ...timed,
    position: { latitude, longitude },
    altitudeM,
    heartRateBpm: heartRateBpm == null ? null : Math.round(heartRateBpm),
    cadencePerMinute,
    speedMps,
    powerW,
  }
}

export function parseGpx(raw: string): RoutePoint[] {
  const parser = new XMLParser({ ignoreAttributes: false })
  const parsed: unknown = parser.parse(raw)
  if (!isRecord(parsed) || !isRecord(parsed.gpx)) throw new Error('GPX root element is missing')
  const points: RoutePoint[] = []
  for (const track of values(parsed.gpx.trk)) {
    if (!isRecord(track)) continue
    for (const segment of values(track.trkseg)) {
      if (!isRecord(segment)) continue
      for (const point of values(segment.trkpt)) points.push(parseRoutePoint(point, points.length))
    }
  }
  for (let index = 1; index < points.length; index++) {
    const previous = points[index - 1]
    const current = points[index]
    if (previous && current && current.timeMs <= previous.timeMs)
      throw new Error('GPX trackpoint times must be strictly increasing')
  }
  return points
}

function coursePosition(latitudeValue: unknown, longitudeValue: unknown, label: string): Position {
  const latitude = xmlNumber(latitudeValue, `${label} latitude`, true)
  const longitude = xmlNumber(longitudeValue, `${label} longitude`, true)
  if (
    latitude == null ||
    longitude == null ||
    latitude < -90 ||
    latitude > 90 ||
    longitude < -180 ||
    longitude > 180
  )
    throw new Error(`${label} has invalid coordinates`)
  return { latitude, longitude }
}

function courseFromPoints(
  path: string,
  points: { position: Position; altitudeM: number | null }[],
): CourseRoute {
  if (points.length < 2) throw new Error(`course ${path} has fewer than two route points`)
  let distanceM = 0
  const routed: CourseRoutePoint[] = points.map((point, index) => {
    const previous = points[index - 1]
    if (previous) distanceM += haversineMeters(previous.position, point.position)
    return { ...point, distanceM }
  })
  if (distanceM <= 0) throw new Error(`course ${path} has no distance`)
  return { path, points: routed, distanceM }
}

function parseGpxCourse(raw: string, path: string): CourseRoute {
  const parser = new XMLParser({ ignoreAttributes: false })
  const parsed: unknown = parser.parse(raw)
  if (!isRecord(parsed) || !isRecord(parsed.gpx)) throw new Error('GPX root element is missing')
  const points: { position: Position; altitudeM: number | null }[] = []
  for (const route of values(parsed.gpx.rte)) {
    if (!isRecord(route)) continue
    for (const [index, point] of values(route.rtept).entries()) {
      if (!isRecord(point)) throw new Error(`GPX route point ${index} must be an object`)
      points.push({
        position: coursePosition(
          point['@_lat'],
          point['@_lon'],
          `GPX route point ${points.length}`,
        ),
        altitudeM: xmlNumber(point.ele, `route point ${points.length} elevation`, false),
      })
    }
  }
  if (points.length === 0) {
    for (const track of values(parsed.gpx.trk)) {
      if (!isRecord(track)) continue
      for (const segment of values(track.trkseg)) {
        if (!isRecord(segment)) continue
        for (const [index, point] of values(segment.trkpt).entries()) {
          if (!isRecord(point)) throw new Error(`GPX trackpoint ${index} must be an object`)
          points.push({
            position: coursePosition(
              point['@_lat'],
              point['@_lon'],
              `GPX trackpoint ${points.length}`,
            ),
            altitudeM: xmlNumber(point.ele, `trackpoint ${points.length} elevation`, false),
          })
        }
      }
    }
  }
  return courseFromPoints(path, points)
}

function parseTcxCourse(raw: string, path: string): CourseRoute {
  const parser = new XMLParser({ ignoreAttributes: false })
  const parsed: unknown = parser.parse(raw)
  if (!isRecord(parsed) || !isRecord(parsed.TrainingCenterDatabase))
    throw new Error('TCX TrainingCenterDatabase root element is missing')
  const database = parsed.TrainingCenterDatabase
  if (!isRecord(database.Courses)) throw new Error('TCX Courses element is missing')
  const courses = values(database.Courses.Course)
  if (courses.length !== 1) throw new Error(`TCX course ${path} must contain exactly one course`)
  const course = courses[0]
  if (!isRecord(course)) throw new Error(`TCX course ${path} must be an object`)
  const points: { position: Position; altitudeM: number | null; distanceM: number | null }[] = []
  for (const track of values(course.Track)) {
    if (!isRecord(track)) continue
    for (const [index, point] of values(track.Trackpoint).entries()) {
      if (!isRecord(point)) throw new Error(`TCX trackpoint ${index} must be an object`)
      if (!isRecord(point.Position))
        throw new Error(`TCX trackpoint ${points.length} has no position`)
      points.push({
        position: coursePosition(
          point.Position.LatitudeDegrees,
          point.Position.LongitudeDegrees,
          `TCX trackpoint ${points.length}`,
        ),
        altitudeM: xmlNumber(
          point.AltitudeMeters,
          `TCX trackpoint ${points.length} elevation`,
          false,
        ),
        distanceM: xmlNumber(
          point.DistanceMeters,
          `TCX trackpoint ${points.length} distance`,
          false,
        ),
      })
    }
  }
  const lap = values(course.Lap)[0]
  const summaryDistanceM = isRecord(lap)
    ? xmlNumber(lap.DistanceMeters, 'TCX course lap distance', false)
    : null
  if (points.length >= 2 && points.every(point => point.distanceM != null)) {
    let previousDistanceM = Number.NEGATIVE_INFINITY
    for (const point of points) {
      if (point.distanceM == null || point.distanceM < previousDistanceM)
        throw new Error(`TCX course ${path} distance is not monotonic`)
      previousDistanceM = point.distanceM
    }
    const firstDistanceM = points[0]?.distanceM ?? 0
    const finalDistanceM = (points.at(-1)?.distanceM ?? 0) - firstDistanceM
    const distanceM =
      summaryDistanceM != null && summaryDistanceM > 0 ? summaryDistanceM : finalDistanceM
    if (finalDistanceM > 0 && distanceM > 0)
      return {
        path,
        distanceM,
        points: points.map(point => ({
          position: point.position,
          altitudeM: point.altitudeM,
          distanceM:
            (((point.distanceM ?? firstDistanceM) - firstDistanceM) / finalDistanceM) * distanceM,
        })),
      }
  }
  return courseFromPoints(path, points)
}

export async function parseCourseRoute(path: string): Promise<CourseRoute> {
  const resolvedPath = resolve(path)
  const raw = await fs.readFile(resolvedPath, 'utf8')
  const parsed: unknown = new XMLParser({ ignoreAttributes: false }).parse(raw)
  if (!isRecord(parsed)) throw new Error(`course ${resolvedPath} has no XML root`)
  if (isRecord(parsed.gpx)) return parseGpxCourse(raw, resolvedPath)
  if (isRecord(parsed.TrainingCenterDatabase)) return parseTcxCourse(raw, resolvedPath)
  throw new Error(`course ${resolvedPath} must be GPX or TCX`)
}
