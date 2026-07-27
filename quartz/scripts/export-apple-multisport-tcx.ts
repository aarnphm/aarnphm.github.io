import { XMLParser } from 'fast-xml-parser'
import fs from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { isRecord } from '../util/type-guards'

const MAX_FILE_BYTES = 25 * 1024 * 1024
const TCX_NAMESPACE = 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'
const ACTIVITY_EXTENSION_NAMESPACE = 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'

export type MultisportDiscipline = 'swim' | 'bike' | 'run'

type ActivityDiscipline = MultisportDiscipline | 'transition'
type TcxSport = 'Swimming' | 'Biking' | 'Running' | 'Other'

interface TimedValue {
  iso: string
  timeMs: number
}

interface HeartRateSample extends TimedValue {
  bpm: number
  sourceIso: string
  sourceTimeMs: number
}

interface HeartRateTimestampRepair {
  sourceTimestamp: string
  outputTimestamp: string
  bpm: number
  reason: string
  provenance: string
}

interface MultisportActivityBase {
  id: string
  activity: string
  start: TimedValue
  end: TimedValue
  durationS: number
  elapsedTimeS: number
  distanceM: number | null
  activeEnergyKcal: number | null
  averageHeartRateBpm: number | null
  averagePowerW: number | null
  averageCadencePerMinute: number | null
  lapCount: number | null
}

interface SportActivity extends MultisportActivityBase {
  kind: 'sport'
  discipline: MultisportDiscipline
}

interface TransitionActivity extends MultisportActivityBase {
  kind: 'transition'
  discipline: 'transition'
}

type MultisportActivity = SportActivity | TransitionActivity

interface MultisportWorkout {
  id: string
  activity: string
  start: TimedValue
  end: TimedValue
  gpxFile: string
  heartRate: HeartRateSample[]
  heartRateTimestampRepairs: HeartRateTimestampRepair[]
  activities: MultisportActivity[]
  legs: [SportActivity, SportActivity, SportActivity]
  transitions: TransitionActivity[]
}

interface Position {
  latitude: number
  longitude: number
}

interface RoutePoint extends TimedValue {
  position: Position
  altitudeM: number | null
  heartRateBpm: number | null
  cadencePerMinute: number | null
  speedMps: number | null
  powerW: number | null
}

interface CourseRoutePoint {
  position: Position
  altitudeM: number | null
  distanceM: number
}

interface CourseRoute {
  path: string
  points: CourseRoutePoint[]
  distanceM: number
}

interface TcxTrackPoint extends TimedValue {
  position: Position | null
  altitudeM: number | null
  distanceM: number
  heartRateBpm: number | null
  cadencePerMinute: number | null
  speedMps: number | null
  powerW: number | null
}

interface PreparedActivity {
  sourceActivity: MultisportActivity
  activity: MultisportActivity
  sport: TcxSport
  trackPoints: TcxTrackPoint[]
  routePointCount: number
  usedHeartRateFallback: boolean
  distanceM: number
  distanceOverrideM: number | null
  routeOverridePath: string | null
}

export interface ExportAppleMultisportTcxOptions {
  inputPath: string
  workoutId: string
  outputDir: string
  sports?: readonly MultisportDiscipline[]
  swimDistanceM?: number
  swimDurationS?: number
  swimElapsedTimeS?: number
  swimRoutePath?: string
  transition1DurationS?: number
  transition1RoutePath?: string
  includeTransitions?: boolean
}

export interface ExportedTcxFile {
  discipline: ActivityDiscipline
  sport: TcxSport
  filename: string
  path: string
  sourceActivityId: string
  trackPointCount: number
  routePointCount: number
  usedHeartRateFallback: boolean
  sourceDistanceM: number | null
  outputDistanceM: number
  distanceOverrideM: number | null
  sourceStart: string
  sourceEnd: string
  outputStart: string
  outputEnd: string
  outputDurationS: number
  outputElapsedTimeS: number
  routeOverridePath: string | null
}

export interface ExportAppleMultisportTcxResult {
  workoutId: string
  outputDir: string
  manifestPath: string
  files: ExportedTcxFile[]
}

interface CliArgs extends ExportAppleMultisportTcxOptions {
  sports: MultisportDiscipline[]
  swimDistanceM?: number
  swimDurationS?: number
  swimElapsedTimeS?: number
  swimRoutePath?: string
  transition1DurationS?: number
  transition1RoutePath?: string
  includeTransitions: boolean
}

function requiredString(record: Record<string, unknown>, key: string): string {
  const value = record[key]
  if (typeof value !== 'string' || value.trim().length === 0)
    throw new Error(`${key} must be a nonempty string`)
  return value.trim()
}

function optionalString(record: Record<string, unknown>, key: string): string | null {
  const value = record[key]
  if (value == null) return null
  if (typeof value !== 'string' || value.trim().length === 0)
    throw new Error(`${key} must be a nonempty string when present`)
  return value.trim()
}

function requiredNumber(record: Record<string, unknown>, key: string): number {
  const value = record[key]
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error(`${key} must be a finite number`)
  return value
}

function optionalNumber(record: Record<string, unknown>, key: string): number | null {
  const value = record[key]
  if (value == null) return null
  if (typeof value !== 'number' || !Number.isFinite(value))
    throw new Error(`${key} must be a finite number when present`)
  return value
}

function nonnegative(value: number, key: string): number {
  if (value < 0) throw new Error(`${key} must be nonnegative`)
  return value
}

function positiveOptional(value: number | null, key: string): number | null {
  if (value == null) return null
  if (value <= 0) throw new Error(`${key} must be positive when present`)
  return value
}

function nonnegativeOptional(value: number | null, key: string): number | null {
  if (value == null) return null
  return nonnegative(value, key)
}

function summaryOptional(
  value: number | null,
  key: string,
  discipline: MultisportDiscipline | 'transition',
): number | null {
  return discipline === 'transition'
    ? nonnegativeOptional(value, key)
    : positiveOptional(value, key)
}

function timedValue(value: string, key: string): TimedValue {
  const timeMs = Date.parse(value)
  if (!Number.isFinite(timeMs)) throw new Error(`${key} must be an ISO timestamp`)
  return { iso: new Date(timeMs).toISOString(), timeMs }
}

function disciplineForActivity(activity: string): MultisportDiscipline | 'transition' | null {
  switch (activity) {
    case 'swim':
    case 'swimming':
      return 'swim'
    case 'bike':
    case 'cycling':
      return 'bike'
    case 'run':
    case 'running':
      return 'run'
    case 'transition':
      return 'transition'
    default:
      return null
  }
}

function parseActivity(raw: unknown, index: number): MultisportActivity {
  if (!isRecord(raw)) throw new Error(`activities[${index}] must be an object`)
  const prefix = `activities[${index}]`
  const id = requiredString(raw, 'id')
  const activity = requiredString(raw, 'activity')
  const discipline = disciplineForActivity(activity)
  if (!discipline)
    throw new Error(`${prefix}.activity must be swimming, transition, cycling, or running`)
  const start = timedValue(requiredString(raw, 'start'), `${prefix}.start`)
  const end = timedValue(requiredString(raw, 'end'), `${prefix}.end`)
  if (end.timeMs <= start.timeMs) throw new Error(`${prefix}.end must be after start`)
  const durationS = nonnegative(requiredNumber(raw, 'durationS'), `${prefix}.durationS`)
  const elapsedTimeS = nonnegative(requiredNumber(raw, 'elapsedTimeS'), `${prefix}.elapsedTimeS`)
  const distanceM = nonnegativeOptional(optionalNumber(raw, 'distanceM'), `${prefix}.distanceM`)
  const activeEnergyKcal = nonnegativeOptional(
    optionalNumber(raw, 'activeEnergyKcal'),
    `${prefix}.activeEnergyKcal`,
  )
  const averageHeartRateBpm = summaryOptional(
    optionalNumber(raw, 'averageHeartRateBpm'),
    `${prefix}.averageHeartRateBpm`,
    discipline,
  )
  const averagePowerW = summaryOptional(
    optionalNumber(raw, 'averagePowerW'),
    `${prefix}.averagePowerW`,
    discipline,
  )
  const averageCadencePerMinute = summaryOptional(
    optionalNumber(raw, 'averageCadencePerMinute'),
    `${prefix}.averageCadencePerMinute`,
    discipline,
  )
  const lapCount = summaryOptional(
    optionalNumber(raw, 'lapCount'),
    `${prefix}.lapCount`,
    discipline,
  )
  const base: MultisportActivityBase = {
    id,
    activity,
    start,
    end,
    durationS,
    elapsedTimeS,
    distanceM,
    activeEnergyKcal,
    averageHeartRateBpm,
    averagePowerW,
    averageCadencePerMinute,
    lapCount,
  }
  return discipline === 'transition'
    ? { ...base, kind: 'transition', discipline }
    : { ...base, kind: 'sport', discipline }
}

function parseHeartRate(raw: unknown): {
  samples: HeartRateSample[]
  repairs: HeartRateTimestampRepair[]
} {
  if (raw == null) return { samples: [], repairs: [] }
  if (!Array.isArray(raw)) throw new Error('heartRate must be an array')
  const parsed: { timed: TimedValue; bpm: number; sourceIndex: number }[] = []
  for (const [index, value] of raw.entries()) {
    if (!isRecord(value)) throw new Error(`heartRate[${index}] must be an object`)
    const timed = timedValue(requiredString(value, 'time'), `heartRate[${index}].time`)
    const bpm = requiredNumber(value, 'bpm')
    if (bpm <= 0 || bpm > 300) throw new Error(`heartRate[${index}].bpm is invalid`)
    parsed.push({ timed, bpm: Math.round(bpm), sourceIndex: index })
  }
  parsed.sort(
    (left, right) => left.timed.timeMs - right.timed.timeMs || left.sourceIndex - right.sourceIndex,
  )
  const reserved = new Set(parsed.map(sample => sample.timed.timeMs))
  const used = new Set<number>()
  const samples: HeartRateSample[] = []
  const repairs: HeartRateTimestampRepair[] = []
  for (const sample of parsed) {
    let outputTimeMs = sample.timed.timeMs
    if (used.has(outputTimeMs)) {
      outputTimeMs += 1
      while (reserved.has(outputTimeMs) || used.has(outputTimeMs)) outputTimeMs += 1
    }
    used.add(outputTimeMs)
    const outputIso = new Date(outputTimeMs).toISOString()
    samples.push({
      iso: outputIso,
      timeMs: outputTimeMs,
      bpm: sample.bpm,
      sourceIso: sample.timed.iso,
      sourceTimeMs: sample.timed.timeMs,
    })
    if (outputTimeMs !== sample.timed.timeMs)
      repairs.push({
        sourceTimestamp: sample.timed.iso,
        outputTimestamp: outputIso,
        bpm: sample.bpm,
        reason: 'duplicate legacy parent heart-rate timestamp',
        provenance: 'converter:stable-millisecond-disambiguation',
      })
  }
  samples.sort((left, right) => left.timeMs - right.timeMs)
  return { samples, repairs }
}

function parseWorkout(raw: unknown): MultisportWorkout {
  if (!isRecord(raw)) throw new Error('workout must be an object')
  const id = requiredString(raw, 'id')
  const activity = requiredString(raw, 'activity')
  if (activity !== 'swimBikeRun') throw new Error(`workout ${id} is not swimBikeRun`)
  const start = timedValue(requiredString(raw, 'start'), 'workout.start')
  const end = timedValue(requiredString(raw, 'end'), 'workout.end')
  if (end.timeMs <= start.timeMs) throw new Error('workout.end must be after start')
  const gpxFile = optionalString(raw, 'gpxFile')
  if (!gpxFile) throw new Error(`workout ${id} has no gpxFile`)
  if (!Array.isArray(raw.activities)) throw new Error('activities must be an array')
  const activities = raw.activities.map(parseActivity)
  for (const [index, segment] of activities.entries()) {
    if (segment.start.timeMs < start.timeMs || segment.end.timeMs > end.timeMs)
      throw new Error(`${segment.discipline} activity is outside the parent workout`)
    const previous = activities[index - 1]
    if (previous && previous.end.timeMs > segment.start.timeMs)
      throw new Error(`${previous.discipline} and ${segment.discipline} activities overlap`)
  }
  const sports = activities.filter(isSportActivity)
  const first = sports[0]
  const second = sports[1]
  const third = sports[2]
  if (!first || !second || !third || sports.length !== 3)
    throw new Error('multisport workout must contain exactly one swim, bike, and run')
  const legs: [SportActivity, SportActivity, SportActivity] = [first, second, third]
  if (
    legs[0].discipline !== 'swim' ||
    legs[1].discipline !== 'bike' ||
    legs[2].discipline !== 'run'
  )
    throw new Error('sport activities must be ordered swimming, cycling, running')
  const transitions = activities.filter(isTransitionActivity)
  for (const transition of transitions) {
    const previousSport = [...sports]
      .reverse()
      .find(sport => sport.end.timeMs <= transition.start.timeMs)
    const nextSport = sports.find(sport => sport.start.timeMs >= transition.end.timeMs)
    if (!previousSport || !nextSport)
      throw new Error('transition activities must occur between sport activities')
  }
  for (let index = 0; index < sports.length - 1; index++) {
    const previous = sports[index]
    const next = sports[index + 1]
    if (!previous || !next) continue
    const between = transitions.filter(
      transition =>
        transition.start.timeMs >= previous.end.timeMs &&
        transition.end.timeMs <= next.start.timeMs,
    )
    if (between.length > 1)
      throw new Error(`${previous.discipline} and ${next.discipline} contain multiple transitions`)
  }
  const heartRate = parseHeartRate(raw.heartRate)
  return {
    id,
    activity,
    start,
    end,
    gpxFile,
    heartRate: heartRate.samples,
    heartRateTimestampRepairs: heartRate.repairs,
    activities,
    legs,
    transitions,
  }
}

function isSportActivity(activity: MultisportActivity): activity is SportActivity {
  return activity.kind === 'sport'
}

function isTransitionActivity(activity: MultisportActivity): activity is TransitionActivity {
  return activity.kind === 'transition'
}

function workoutRecord(document: unknown, workoutId: string): Record<string, unknown> {
  if (!isRecord(document) || !Array.isArray(document.workouts))
    throw new Error('input must contain a workouts array')
  const matches = document.workouts.filter(value => isRecord(value) && value.id === workoutId)
  if (matches.length === 0) throw new Error(`workout ${workoutId} was not found`)
  if (matches.length > 1) throw new Error(`workout ${workoutId} is duplicated`)
  const match = matches[0]
  if (!isRecord(match)) throw new Error(`workout ${workoutId} must be an object`)
  return match
}

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

function parseGpx(raw: string): RoutePoint[] {
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

async function parseCourseRoute(path: string): Promise<CourseRoute> {
  const resolvedPath = resolve(path)
  const raw = await fs.readFile(resolvedPath, 'utf8')
  const parsed: unknown = new XMLParser({ ignoreAttributes: false }).parse(raw)
  if (!isRecord(parsed)) throw new Error(`course ${resolvedPath} has no XML root`)
  if (isRecord(parsed.gpx)) return parseGpxCourse(raw, resolvedPath)
  if (isRecord(parsed.TrainingCenterDatabase)) return parseTcxCourse(raw, resolvedPath)
  throw new Error(`course ${resolvedPath} must be GPX or TCX`)
}

function inActivityWindow(timeMs: number, activity: MultisportActivity): boolean {
  return (
    timeMs >= activity.start.timeMs &&
    (activity.discipline === 'run' ? timeMs <= activity.end.timeMs : timeMs < activity.end.timeMs)
  )
}

function routeTrackPoint(point: RoutePoint): Omit<TcxTrackPoint, 'distanceM'> {
  return {
    iso: point.iso,
    timeMs: point.timeMs,
    position: point.position,
    altitudeM: point.altitudeM,
    heartRateBpm: point.heartRateBpm,
    cadencePerMinute: point.cadencePerMinute,
    speedMps: point.speedMps,
    powerW: point.powerW,
  }
}

function mergedTrackPoints(
  routePoints: RoutePoint[],
  heartRate: HeartRateSample[],
  activity: MultisportActivity,
): Omit<TcxTrackPoint, 'distanceM'>[] {
  const byTime = new Map<number, Omit<TcxTrackPoint, 'distanceM'>>()
  const parentHeartRate = heartRate.filter(sample =>
    inActivityWindow(sample.sourceTimeMs, activity),
  )
  for (const point of routePoints) {
    const trackPoint = routeTrackPoint(point)
    byTime.set(
      point.timeMs,
      parentHeartRate.length > 0 ? { ...trackPoint, heartRateBpm: null } : trackPoint,
    )
  }
  for (const sample of parentHeartRate) {
    const current = byTime.get(sample.timeMs)
    if (current) {
      if (current.heartRateBpm == null)
        byTime.set(sample.timeMs, { ...current, heartRateBpm: sample.bpm })
      continue
    }
    byTime.set(sample.timeMs, {
      iso: sample.iso,
      timeMs: sample.timeMs,
      position: null,
      altitudeM: null,
      heartRateBpm: sample.bpm,
      cadencePerMinute: null,
      speedMps: null,
      powerW: null,
    })
  }
  return [...byTime.values()].sort((left, right) => left.timeMs - right.timeMs)
}

function radians(value: number): number {
  return (value * Math.PI) / 180
}

function haversineMeters(left: Position, right: Position): number {
  const earthRadiusM = 6_371_008.8
  const latitudeDelta = radians(right.latitude - left.latitude)
  const longitudeDelta = radians(right.longitude - left.longitude)
  const leftLatitude = radians(left.latitude)
  const rightLatitude = radians(right.latitude)
  const chord =
    Math.sin(latitudeDelta / 2) ** 2 +
    Math.cos(leftLatitude) * Math.cos(rightLatitude) * Math.sin(longitudeDelta / 2) ** 2
  return earthRadiusM * 2 * Math.atan2(Math.sqrt(chord), Math.sqrt(Math.max(0, 1 - chord)))
}

function withDistance(
  points: Omit<TcxTrackPoint, 'distanceM'>[],
  activity: MultisportActivity,
  distanceOverrideM: number | null,
): TcxTrackPoint[] {
  const rawDistances: number[] = []
  let cumulativeM = 0
  let previousPosition: Position | null = null
  for (const point of points) {
    if (point.position && previousPosition)
      cumulativeM += haversineMeters(previousPosition, point.position)
    if (point.position) previousPosition = point.position
    rawDistances.push(cumulativeM)
  }
  const targetM = distanceOverrideM ?? activity.distanceM
  return points.map((point, index) => {
    const rawDistanceM = rawDistances[index] ?? 0
    let distanceM = rawDistanceM
    if (targetM != null) {
      if (cumulativeM > 0) distanceM = (rawDistanceM / cumulativeM) * targetM
      else {
        const firstTimeMs = points[0]?.timeMs ?? activity.start.timeMs
        const lastTimeMs = points[points.length - 1]?.timeMs ?? activity.end.timeMs
        const elapsedMs = lastTimeMs - firstTimeMs
        distanceM = elapsedMs > 0 ? ((point.timeMs - firstTimeMs) / elapsedMs) * targetM : 0
      }
    }
    return { ...point, distanceM: Math.max(0, distanceM) }
  })
}

function sportForActivity(activity: MultisportActivity): TcxSport {
  switch (activity.discipline) {
    case 'swim':
      return 'Swimming'
    case 'bike':
      return 'Biking'
    case 'run':
      return 'Running'
    case 'transition':
      return 'Other'
  }
}

function averageHeartRate(
  heartRate: HeartRateSample[],
  startTimeMs: number,
  endTimeMs: number,
): number | null {
  const samples = heartRate.filter(
    sample => sample.sourceTimeMs >= startTimeMs && sample.sourceTimeMs <= endTimeMs,
  )
  if (samples.length === 0) return null
  return samples.reduce((total, sample) => total + sample.bpm, 0) / samples.length
}

function retimedActivity(
  activity: MultisportActivity,
  startTimeMs: number,
  durationS: number,
  heartRate: HeartRateSample[],
  elapsedTimeS = durationS,
): MultisportActivity {
  if (elapsedTimeS < durationS)
    throw new Error(`${activity.discipline} elapsed time must be at least its duration`)
  const endTimeMs = startTimeMs + elapsedTimeS * 1000
  const start = timedValue(new Date(startTimeMs).toISOString(), `${activity.discipline}.start`)
  const end = timedValue(new Date(endTimeMs).toISOString(), `${activity.discipline}.end`)
  return {
    ...activity,
    start,
    end,
    durationS,
    elapsedTimeS,
    averageHeartRateBpm:
      averageHeartRate(heartRate, startTimeMs, endTimeMs) ?? activity.averageHeartRateBpm,
  }
}

function interpolateCourse(
  course: CourseRoute,
  distanceM: number,
): { position: Position; altitudeM: number | null } {
  const boundedDistanceM = Math.min(Math.max(0, distanceM), course.distanceM)
  let index = 1
  while (
    index < course.points.length &&
    (course.points[index]?.distanceM ?? course.distanceM) < boundedDistanceM
  )
    index += 1
  const right = course.points[Math.min(index, course.points.length - 1)]
  const left = course.points[Math.max(0, index - 1)]
  if (!left || !right) throw new Error(`course ${course.path} has invalid route points`)
  const segmentDistanceM = right.distanceM - left.distanceM
  const ratio = segmentDistanceM > 0 ? (boundedDistanceM - left.distanceM) / segmentDistanceM : 0
  const altitudeM =
    left.altitudeM != null && right.altitudeM != null
      ? left.altitudeM + (right.altitudeM - left.altitudeM) * ratio
      : (left.altitudeM ?? right.altitudeM)
  return {
    position: {
      latitude: left.position.latitude + (right.position.latitude - left.position.latitude) * ratio,
      longitude:
        left.position.longitude + (right.position.longitude - left.position.longitude) * ratio,
    },
    altitudeM,
  }
}

function courseTrackPoints(
  activity: MultisportActivity,
  course: CourseRoute,
  heartRate: HeartRateSample[],
  outputDistanceM: number,
): TcxTrackPoint[] {
  const byTime = new Map<number, TcxTrackPoint>()
  const pointAtTime = (timeMs: number, heartRateBpm: number | null): TcxTrackPoint => {
    const elapsedS = (timeMs - activity.start.timeMs) / 1000
    const ratio = Math.min(1, Math.max(0, elapsedS / activity.durationS))
    const coursePoint = interpolateCourse(course, ratio * course.distanceM)
    return {
      iso: new Date(timeMs).toISOString(),
      timeMs,
      position: coursePoint.position,
      altitudeM: coursePoint.altitudeM,
      distanceM: ratio * outputDistanceM,
      heartRateBpm,
      cadencePerMinute: null,
      speedMps: elapsedS <= activity.durationS ? outputDistanceM / activity.durationS : 0,
      powerW: null,
    }
  }
  const wholeSeconds = Math.floor(activity.elapsedTimeS)
  for (let elapsedS = 0; elapsedS <= wholeSeconds; elapsedS++) {
    const timeMs = activity.start.timeMs + elapsedS * 1000
    byTime.set(timeMs, pointAtTime(timeMs, null))
  }
  if (!Number.isInteger(activity.elapsedTimeS))
    byTime.set(activity.end.timeMs, pointAtTime(activity.end.timeMs, null))
  for (const sample of heartRate) {
    if (sample.sourceTimeMs < activity.start.timeMs || sample.sourceTimeMs > activity.end.timeMs)
      continue
    const current = byTime.get(sample.timeMs)
    byTime.set(
      sample.timeMs,
      current ? { ...current, heartRateBpm: sample.bpm } : pointAtTime(sample.timeMs, sample.bpm),
    )
  }
  return [...byTime.values()].sort((left, right) => left.timeMs - right.timeMs)
}

function prepareActivity(
  activity: MultisportActivity,
  route: RoutePoint[],
  heartRate: HeartRateSample[],
  distanceOverrideM: number | null = null,
): PreparedActivity {
  const routePoints = route.filter(point => inActivityWindow(point.timeMs, activity))
  const usedHeartRateFallback = routePoints.length < 2
  const basePoints = mergedTrackPoints(routePoints, heartRate, activity)
  const trackPoints = withDistance(basePoints, activity, distanceOverrideM)
  const distanceM =
    distanceOverrideM ?? activity.distanceM ?? trackPoints[trackPoints.length - 1]?.distanceM ?? 0
  return {
    sourceActivity: activity,
    activity,
    sport: sportForActivity(activity),
    trackPoints,
    routePointCount: routePoints.length,
    usedHeartRateFallback,
    distanceM,
    distanceOverrideM,
    routeOverridePath: null,
  }
}

function prepareCourseActivity(
  sourceActivity: MultisportActivity,
  activity: MultisportActivity,
  course: CourseRoute,
  heartRate: HeartRateSample[],
  distanceOverrideM: number | null = null,
): PreparedActivity {
  const distanceM = distanceOverrideM ?? course.distanceM
  return {
    sourceActivity,
    activity,
    sport: sportForActivity(activity),
    trackPoints: courseTrackPoints(activity, course, heartRate, distanceM),
    routePointCount: course.points.length,
    usedHeartRateFallback: false,
    distanceM,
    distanceOverrideM,
    routeOverridePath: course.path,
  }
}

function validatePrepared(prepared: PreparedActivity): void {
  const { activity, trackPoints } = prepared
  if (trackPoints.length < 2)
    throw new Error(`${activity.discipline} activity has fewer than two trackpoints`)
  let previousTime = Number.NEGATIVE_INFINITY
  let previousDistance = Number.NEGATIVE_INFINITY
  for (const point of trackPoints) {
    if (point.timeMs <= previousTime)
      throw new Error(`${activity.discipline} trackpoint times are not strictly increasing`)
    if (!Number.isFinite(point.distanceM) || point.distanceM < previousDistance)
      throw new Error(`${activity.discipline} trackpoint distance is not monotonic`)
    if (
      point.position &&
      (point.position.latitude < -90 ||
        point.position.latitude > 90 ||
        point.position.longitude < -180 ||
        point.position.longitude > 180)
    )
      throw new Error(`${activity.discipline} trackpoint coordinates are invalid`)
    previousTime = point.timeMs
    previousDistance = point.distanceM
  }
}

function escapeXml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
}

function decimal(value: number, places = 3): string {
  if (!Number.isFinite(value)) throw new Error('cannot serialize a nonfinite number')
  return value.toFixed(places)
}

function integer(value: number): string {
  if (!Number.isFinite(value)) throw new Error('cannot serialize a nonfinite number')
  return String(Math.round(value))
}

function heartRateXml(tag: string, bpm: number, indent: string): string[] {
  return [`${indent}<${tag}>`, `${indent} <Value>${integer(bpm)}</Value>`, `${indent}</${tag}>`]
}

function trackPointXml(point: TcxTrackPoint, discipline: ActivityDiscipline): string {
  const lines = ['      <Trackpoint>', `       <Time>${point.iso}</Time>`]
  if (point.position) {
    lines.push(
      '       <Position>',
      `        <LatitudeDegrees>${decimal(point.position.latitude, 7)}</LatitudeDegrees>`,
      `        <LongitudeDegrees>${decimal(point.position.longitude, 7)}</LongitudeDegrees>`,
      '       </Position>',
    )
  }
  if (point.altitudeM != null)
    lines.push(`       <AltitudeMeters>${decimal(point.altitudeM, 1)}</AltitudeMeters>`)
  lines.push(`       <DistanceMeters>${decimal(point.distanceM)}</DistanceMeters>`)
  if (point.heartRateBpm != null)
    lines.push(...heartRateXml('HeartRateBpm', point.heartRateBpm, '       '))
  if ((discipline === 'bike' || discipline === 'transition') && point.cadencePerMinute != null)
    lines.push(`       <Cadence>${integer(point.cadencePerMinute)}</Cadence>`)
  const extensions: string[] = []
  if (discipline === 'run' && point.cadencePerMinute != null)
    extensions.push(`         <ns3:RunCadence>${integer(point.cadencePerMinute)}</ns3:RunCadence>`)
  if (point.speedMps != null)
    extensions.push(`         <ns3:Speed>${decimal(point.speedMps)}</ns3:Speed>`)
  if (point.powerW != null)
    extensions.push(`         <ns3:Watts>${integer(point.powerW)}</ns3:Watts>`)
  if (extensions.length > 0)
    lines.push(
      '       <Extensions>',
      '        <ns3:TPX>',
      ...extensions,
      '        </ns3:TPX>',
      '       </Extensions>',
    )
  lines.push('      </Trackpoint>')
  return lines.join('\n')
}

function lapExtensionsXml(prepared: PreparedActivity): string[] {
  const { activity, distanceM } = prepared
  const values: string[] = []
  if (activity.durationS > 0)
    values.push(`       <ns3:AvgSpeed>${decimal(distanceM / activity.durationS)}</ns3:AvgSpeed>`)
  if (activity.discipline === 'run' && activity.averageCadencePerMinute != null)
    values.push(
      `       <ns3:AvgRunCadence>${integer(activity.averageCadencePerMinute)}</ns3:AvgRunCadence>`,
    )
  if (activity.averagePowerW != null && activity.averagePowerW > 0)
    values.push(`       <ns3:AvgWatts>${integer(activity.averagePowerW)}</ns3:AvgWatts>`)
  if (values.length === 0) return []
  return ['     <Extensions>', '      <ns3:LX>', ...values, '      </ns3:LX>', '     </Extensions>']
}

function tcxXml(prepared: PreparedActivity, parentWorkoutId: string): string {
  const { activity, sport, trackPoints, distanceM } = prepared
  const lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<TrainingCenterDatabase xmlns="${TCX_NAMESPACE}" xmlns:ns3="${ACTIVITY_EXTENSION_NAMESPACE}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="${TCX_NAMESPACE} http://www.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd ${ACTIVITY_EXTENSION_NAMESPACE} http://www.garmin.com/xmlschemas/ActivityExtensionv2.xsd">`,
    ' <Activities>',
    `  <Activity Sport="${sport}">`,
    `   <Id>${activity.start.iso}</Id>`,
    `   <Lap StartTime="${activity.start.iso}">`,
    `    <TotalTimeSeconds>${decimal(activity.durationS)}</TotalTimeSeconds>`,
    `    <DistanceMeters>${decimal(distanceM)}</DistanceMeters>`,
    `    <Calories>${integer(activity.activeEnergyKcal ?? 0)}</Calories>`,
  ]
  if (activity.averageHeartRateBpm != null && activity.averageHeartRateBpm > 0)
    lines.push(...heartRateXml('AverageHeartRateBpm', activity.averageHeartRateBpm, '    '))
  lines.push('    <Intensity>Active</Intensity>')
  if (
    (activity.discipline === 'bike' || activity.discipline === 'transition') &&
    activity.averageCadencePerMinute != null &&
    activity.averageCadencePerMinute > 0
  )
    lines.push(`    <Cadence>${integer(activity.averageCadencePerMinute)}</Cadence>`)
  lines.push(
    '    <TriggerMethod>Manual</TriggerMethod>',
    '    <Track>',
    ...trackPoints.map(point => trackPointXml(point, activity.discipline)),
    '    </Track>',
    ...lapExtensionsXml(prepared),
    '   </Lap>',
    `   <Notes>${escapeXml(`HealthExporter multisport ${parentWorkoutId}; source activity ${activity.id}`)}</Notes>`,
    '  </Activity>',
    ' </Activities>',
    '</TrainingCenterDatabase>',
    '',
  )
  const xml = lines.join('\n')
  if (/(?:NaN|Infinity)/.test(xml)) throw new Error(`${activity.discipline} TCX is nonfinite`)
  if (Buffer.byteLength(xml) >= MAX_FILE_BYTES)
    throw new Error(`${activity.discipline} TCX is 25MB or larger`)
  return xml
}

function normalizedSports(
  sports: readonly MultisportDiscipline[] | undefined,
): MultisportDiscipline[] {
  const selected = sports ?? ['swim', 'bike', 'run']
  const unique: MultisportDiscipline[] = []
  for (const sport of selected) {
    if (sport !== 'swim' && sport !== 'bike' && sport !== 'run')
      throw new Error(`invalid sport ${String(sport)}`)
    if (!unique.includes(sport)) unique.push(sport)
  }
  if (unique.length === 0) throw new Error('at least one sport must be selected')
  return unique
}

function normalizedSwimDistance(value: number | undefined): number | null {
  if (value == null) return null
  if (!Number.isFinite(value) || value <= 0)
    throw new Error('swim distance override must be a positive finite number')
  return value
}

function normalizedDuration(value: number | undefined, label: string): number | null {
  if (value == null) return null
  if (!Number.isFinite(value) || value <= 0)
    throw new Error(`${label} must be a positive finite number`)
  return value
}

export async function exportAppleMultisportTcx(
  options: ExportAppleMultisportTcxOptions,
): Promise<ExportAppleMultisportTcxResult> {
  const inputPath = resolve(options.inputPath)
  const outputDir = resolve(options.outputDir)
  const sports = normalizedSports(options.sports)
  const swimDistanceM = normalizedSwimDistance(options.swimDistanceM)
  const swimDurationS = normalizedDuration(options.swimDurationS, 'swim duration override')
  const swimElapsedTimeS = normalizedDuration(
    options.swimElapsedTimeS,
    'swim elapsed time override',
  )
  if (swimDurationS != null && swimElapsedTimeS != null && swimElapsedTimeS < swimDurationS)
    throw new Error('swim elapsed time override must be at least the swim duration override')
  const transition1DurationS = normalizedDuration(
    options.transition1DurationS,
    'transition 1 duration override',
  )
  const includeTransitions = options.includeTransitions ?? false
  const document: unknown = JSON.parse(await fs.readFile(inputPath, 'utf8'))
  const workout = parseWorkout(workoutRecord(document, options.workoutId))
  const gpxPath = resolve(dirname(inputPath), workout.gpxFile)
  const route = parseGpx(await fs.readFile(gpxPath, 'utf8'))
  const [swimCourse, transition1Course] = await Promise.all([
    options.swimRoutePath ? parseCourseRoute(options.swimRoutePath) : null,
    options.transition1RoutePath ? parseCourseRoute(options.transition1RoutePath) : null,
  ])
  const sourceSwim = workout.legs[0]
  const outputSwim =
    swimDurationS == null && swimElapsedTimeS == null
      ? sourceSwim
      : retimedActivity(
          sourceSwim,
          sourceSwim.start.timeMs,
          swimDurationS ?? sourceSwim.durationS,
          workout.heartRate,
          swimElapsedTimeS ?? swimDurationS ?? sourceSwim.elapsedTimeS,
        )
  const preparedLegs = workout.legs.map(activity =>
    activity.discipline === 'swim' && swimCourse
      ? prepareCourseActivity(activity, outputSwim, swimCourse, workout.heartRate, swimDistanceM)
      : prepareActivity(
          activity.discipline === 'swim' ? outputSwim : activity,
          route,
          workout.heartRate,
          activity.discipline === 'swim' ? swimDistanceM : null,
        ),
  )
  if (!swimCourse && (swimDurationS != null || swimElapsedTimeS != null)) {
    const preparedSwim = preparedLegs[0]
    if (!preparedSwim) throw new Error('prepared swim is missing')
    preparedSwim.sourceActivity = sourceSwim
  }
  for (const activity of preparedLegs) validatePrepared(activity)
  const selected = preparedLegs
    .filter(
      activity =>
        activity.activity.kind === 'sport' && sports.includes(activity.activity.discipline),
    )
    .map(activity => ({ activity, filename: `${activity.activity.discipline}.tcx` }))
  if (includeTransitions) {
    for (const [index, transition] of workout.transitions.entries()) {
      const outputTransition =
        index === 0 && transition1DurationS != null
          ? retimedActivity(
              transition,
              outputSwim.end.timeMs,
              transition1DurationS,
              workout.heartRate,
            )
          : transition
      const activity =
        index === 0 && transition1Course
          ? prepareCourseActivity(
              transition,
              outputTransition,
              transition1Course,
              workout.heartRate,
            )
          : prepareActivity(outputTransition, route, workout.heartRate)
      activity.sourceActivity = transition
      validatePrepared(activity)
      selected.push({ activity, filename: `transition-${index + 1}.tcx` })
    }
  }
  const staged = selected.map(({ activity, filename }) => {
    return {
      activity,
      filename,
      path: join(outputDir, filename),
      xml: tcxXml(activity, workout.id),
    }
  })
  const files: ExportedTcxFile[] = staged.map(({ activity, filename, path }) => ({
    discipline: activity.activity.discipline,
    sport: activity.sport,
    filename,
    path,
    sourceActivityId: activity.sourceActivity.id,
    trackPointCount: activity.trackPoints.length,
    routePointCount: activity.routePointCount,
    usedHeartRateFallback: activity.usedHeartRateFallback,
    sourceDistanceM: activity.sourceActivity.distanceM,
    outputDistanceM: activity.distanceM,
    distanceOverrideM: activity.distanceOverrideM,
    sourceStart: activity.sourceActivity.start.iso,
    sourceEnd: activity.sourceActivity.end.iso,
    outputStart: activity.activity.start.iso,
    outputEnd: activity.activity.end.iso,
    outputDurationS: activity.activity.durationS,
    outputElapsedTimeS: activity.activity.elapsedTimeS,
    routeOverridePath: activity.routeOverridePath,
  }))
  const swim = sourceSwim
  const manifest = {
    version: 1,
    source: {
      input: inputPath,
      workoutId: workout.id,
      activity: workout.activity,
      start: workout.start.iso,
      end: workout.end.iso,
      gpxFile: workout.gpxFile,
    },
    selectedSports: sports,
    includeTransitions,
    heartRateTimestampRepairs: workout.heartRateTimestampRepairs,
    timingOverrides: [
      ...(swimDurationS == null && swimElapsedTimeS == null
        ? []
        : [
            {
              discipline: 'swim',
              sourceActivityId: swim.id,
              sourceStart: swim.start.iso,
              sourceEnd: swim.end.iso,
              sourceDurationS: swim.durationS,
              outputStart: outputSwim.start.iso,
              outputEnd: outputSwim.end.iso,
              outputDurationS: outputSwim.durationS,
              outputElapsedTimeS: outputSwim.elapsedTimeS,
              provenance: 'cli:--swim-duration-s/--swim-elapsed-s',
            },
          ]),
      ...(transition1DurationS == null || !workout.transitions[0]
        ? []
        : [
            {
              discipline: 'transition',
              ordinal: 1,
              sourceActivityId: workout.transitions[0].id,
              sourceStart: workout.transitions[0].start.iso,
              sourceEnd: workout.transitions[0].end.iso,
              sourceDurationS: workout.transitions[0].durationS,
              outputStart: outputSwim.end.iso,
              outputEnd: new Date(
                outputSwim.end.timeMs + transition1DurationS * 1000,
              ).toISOString(),
              outputDurationS: transition1DurationS,
              provenance: 'cli:--transition-1-duration-s',
            },
          ]),
    ],
    routeOverrides: [
      ...(swimCourse
        ? [
            {
              discipline: 'swim',
              sourceActivityId: swim.id,
              path: swimCourse.path,
              sourcePointCount: swimCourse.points.length,
              sourceDistanceM: swimCourse.distanceM,
              provenance: 'cli:--swim-route',
            },
          ]
        : []),
      ...(transition1Course && workout.transitions[0]
        ? [
            {
              discipline: 'transition',
              ordinal: 1,
              sourceActivityId: workout.transitions[0].id,
              path: transition1Course.path,
              sourcePointCount: transition1Course.points.length,
              sourceDistanceM: transition1Course.distanceM,
              provenance: 'cli:--transition-1-route',
            },
          ]
        : []),
    ],
    distanceOverrides:
      swimDistanceM == null
        ? []
        : [
            {
              discipline: 'swim',
              sourceActivityId: swim.id,
              sourceDistanceM: swim.distanceM,
              outputDistanceM: swimDistanceM,
              provenance: 'cli:--swim-distance-m',
            },
          ],
    activities: workout.activities.map(activity => ({
      id: activity.id,
      kind: activity.kind,
      discipline: activity.discipline,
      activity: activity.activity,
      start: activity.start.iso,
      end: activity.end.iso,
      durationS: activity.durationS,
      elapsedTimeS: activity.elapsedTimeS,
      distanceM: activity.distanceM,
      activeEnergyKcal: activity.activeEnergyKcal,
      averageHeartRateBpm: activity.averageHeartRateBpm,
      averagePowerW: activity.averagePowerW,
      averageCadencePerMinute: activity.averageCadencePerMinute,
      lapCount: activity.lapCount,
    })),
    transitions: workout.transitions.map(transition => ({
      id: transition.id,
      activity: transition.activity,
      start: transition.start.iso,
      end: transition.end.iso,
      durationS: transition.durationS,
      elapsedTimeS: transition.elapsedTimeS,
      distanceM: transition.distanceM,
      activeEnergyKcal: transition.activeEnergyKcal,
      averageHeartRateBpm: transition.averageHeartRateBpm,
      averagePowerW: transition.averagePowerW,
      averageCadencePerMinute: transition.averageCadencePerMinute,
      lapCount: transition.lapCount,
    })),
    files: files.map(file => ({
      discipline: file.discipline,
      sport: file.sport,
      filename: file.filename,
      sourceActivityId: file.sourceActivityId,
      trackPointCount: file.trackPointCount,
      routePointCount: file.routePointCount,
      usedHeartRateFallback: file.usedHeartRateFallback,
      sourceDistanceM: file.sourceDistanceM,
      outputDistanceM: file.outputDistanceM,
      distanceOverrideM: file.distanceOverrideM,
      sourceStart: file.sourceStart,
      sourceEnd: file.sourceEnd,
      outputStart: file.outputStart,
      outputEnd: file.outputEnd,
      outputDurationS: file.outputDurationS,
      outputElapsedTimeS: file.outputElapsedTimeS,
      routeOverridePath: file.routeOverridePath,
    })),
  }
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`
  if (Buffer.byteLength(manifestText) >= MAX_FILE_BYTES)
    throw new Error('manifest is 25MB or larger')
  const manifestPath = join(outputDir, 'manifest.json')
  await fs.mkdir(outputDir, { recursive: true })
  await Promise.all([
    ...staged.map(file => fs.writeFile(file.path, file.xml, 'utf8')),
    fs.writeFile(manifestPath, manifestText, 'utf8'),
  ])
  return { workoutId: workout.id, outputDir, manifestPath, files }
}

function usage(): string {
  return [
    'usage: tsx quartz/scripts/export-apple-multisport-tcx.ts --input FILE --id WORKOUT_UUID --output DIRECTORY [--sports swim,bike,run] [--swim-distance-m METERS] [--swim-duration-s SECONDS] [--swim-elapsed-s SECONDS] [--swim-route FILE] [--include-transitions] [--transition-1-duration-s SECONDS] [--transition-1-route FILE]',
    '',
    'writes standalone Strava-targeted TCX files and manifest.json.',
  ].join('\n')
}

function argumentValue(argv: string[], index: number, flag: string): string {
  const value = argv[index]
  if (!value || value.startsWith('--')) throw new Error(`${flag} requires a value`)
  return value
}

function parseSports(value: string): MultisportDiscipline[] {
  const sports: MultisportDiscipline[] = []
  for (const raw of value.split(',')) {
    const sport = raw.trim()
    if (sport !== 'swim' && sport !== 'bike' && sport !== 'run')
      throw new Error(`--sports accepts swim,bike,run, got ${value}`)
    if (!sports.includes(sport)) sports.push(sport)
  }
  if (sports.length === 0) throw new Error('--sports requires at least one sport')
  return sports
}

function positiveNumber(value: string, flag: string): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0)
    throw new Error(`${flag} requires a positive finite number`)
  return parsed
}

export function parseAppleMultisportTcxArgs(argv: string[]): CliArgs {
  let inputPath: string | null = null
  let workoutId: string | null = null
  let outputDir: string | null = null
  let sports: MultisportDiscipline[] = ['swim', 'bike', 'run']
  let swimDistanceM: number | undefined
  let swimDurationS: number | undefined
  let swimElapsedTimeS: number | undefined
  let swimRoutePath: string | undefined
  let transition1DurationS: number | undefined
  let transition1RoutePath: string | undefined
  let includeTransitions = false
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index]
    if (argument === '--') continue
    if (argument === '--input') inputPath = argumentValue(argv, ++index, argument)
    else if (argument === '--id') workoutId = argumentValue(argv, ++index, argument)
    else if (argument === '--output') outputDir = argumentValue(argv, ++index, argument)
    else if (argument === '--sports') sports = parseSports(argumentValue(argv, ++index, argument))
    else if (argument === '--swim-distance-m')
      swimDistanceM = positiveNumber(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--swim-duration-s')
      swimDurationS = positiveNumber(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--swim-elapsed-s')
      swimElapsedTimeS = positiveNumber(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--swim-route') swimRoutePath = argumentValue(argv, ++index, argument)
    else if (argument === '--transition-1-duration-s')
      transition1DurationS = positiveNumber(argumentValue(argv, ++index, argument), argument)
    else if (argument === '--transition-1-route')
      transition1RoutePath = argumentValue(argv, ++index, argument)
    else if (argument === '--include-transitions') includeTransitions = true
    else if (argument === '--help' || argument === '-h') throw new Error(usage())
    else throw new Error(`unknown argument ${argument}\n${usage()}`)
  }
  if (!inputPath || !workoutId || !outputDir) throw new Error(usage())
  return {
    inputPath,
    workoutId,
    outputDir,
    sports,
    ...(swimDistanceM == null ? {} : { swimDistanceM }),
    ...(swimDurationS == null ? {} : { swimDurationS }),
    ...(swimElapsedTimeS == null ? {} : { swimElapsedTimeS }),
    ...(swimRoutePath == null ? {} : { swimRoutePath }),
    ...(transition1DurationS == null ? {} : { transition1DurationS }),
    ...(transition1RoutePath == null ? {} : { transition1RoutePath }),
    includeTransitions,
  }
}

async function main(): Promise<void> {
  const args = parseAppleMultisportTcxArgs(process.argv.slice(2))
  const result = await exportAppleMultisportTcx(args)
  for (const file of result.files)
    console.log(
      `[apple-tcx] ${file.discipline} ${file.trackPointCount} trackpoints -> ${file.path}`,
    )
  console.log(`[apple-tcx] manifest -> ${result.manifestPath}`)
}

const entry = process.argv[1]
if (entry && import.meta.url === pathToFileURL(resolve(entry)).href) {
  main().catch(error => {
    console.error(`[apple-tcx] failed: ${error instanceof Error ? error.message : error}`)
    process.exitCode = 1
  })
}
