import type { StravaActivityDetail, StravaStreams } from '../plugins/stores/strava'
import {
  lapExtras,
  rideAltitude,
  type LapExtras,
  type LapMetricSource,
  type LapSample,
} from './high-park-lap-metrics'
import { triathlonActivityHref } from './triathlon-date-route'

type Point = Pick<StravaActivityDetail['route'][number], 'lat' | 'lng' | 'd' | 'elapsedS'> &
  Partial<Pick<StravaActivityDetail['route'][number], 'x' | 'y'>>
type MapPoint = StravaActivityDetail['mapRoute'][number][number]
type Range = Pick<
  StravaActivityDetail['analysisRanges'][number],
  'id' | 'kind' | 'label' | 'startElapsedS' | 'endElapsedS' | 'durationS' | 'distanceKm'
> &
  Partial<
    Pick<
      StravaActivityDetail['analysisRanges'][number],
      'startDistanceKm' | 'averageWatts' | 'averageHeartRate' | 'averageCadence'
    >
  >
export type HighParkActivity = Pick<
  StravaActivityDetail,
  'id' | 'date' | 'start' | 'name' | 'sport' | 'virtual' | 'electric'
> &
  Partial<Pick<LapMetricSource, 'cyclingDynamics' | 'cyclingTorque' | 'gearShifts'>> & {
    route: readonly (Point & LapSample)[]
    mapRoute?: readonly (readonly MapPoint[])[]
    analysisRanges: readonly Range[]
  }
export const HIGH_PARK_SEGMENT = 'High Park - Main Loop - Centre Road'
export const HIGH_PARK_METHOD =
  'Strava full-circuit efforts, with GPS estimates where unavailable. GPS requires four ordered road checkpoints and a 1.65–2.1 km return circuit. Partial visits and overlapping segment names are excluded.'

// Road checkpoints from the recorded main-circuit effort on 2026-08-01, activity 19561993083.
const gates = [
  { lat: 43.645655, lng: -79.46473 },
  { lat: 43.649286, lng: -79.464933 },
  { lat: 43.652093, lng: -79.465654 },
  { lat: 43.648459, lng: -79.467779 },
]
// The park box the GPS matcher accepts; traces outside it are clipped from the drawings.
const PARK = { south: 43.644, north: 43.654, west: -79.47, east: -79.463 }
const LAT0 = (PARK.south + PARK.north) / 2
const M_PER_LAT = 111_195
const M_PER_LNG = M_PER_LAT * Math.cos((LAT0 * Math.PI) / 180)
const GLYPH = 40

type Lap = {
  seconds: number
  distanceKm: number
  startKm: number | null
  endKm: number | null
  watts?: number | null
  heartRate?: number | null
  cadence?: number | null
}
/** One lap as drawn and hovered: duration, distance and the segment effort's averages. */
export interface HighParkLap extends LapExtras {
  s: number
  km: number
  w: number | null
  hr: number | null
  cad: number | null
}
export interface HighParkVisit {
  id: number
  date: string
  start: string
  name: string
  href: string
  source: 'Strava segment' | 'GPS estimate'
  laps: number
  distanceKm: number
  fastestS: number
  /** Every matched lap, in ride order. */
  lapsS: number[]
  lapDetail: HighParkLap[]
  /** Mean altitude through the lap (16 bins, metres above the lowest), for the profile charts. */
  altitude: number[] | null
  /** The fastest lap's GPS line in metres from the park box's north-west corner, or ''. */
  trace: string
  /** The whole (privacy-trimmed) ride in a GLYPH box, and its in-park stretch. */
  glyph: { route: string; park: string } | null
}
export interface HighParkSport {
  activities: HighParkVisit[]
  laps: number
  distanceKm: number
  estimatedLaps: number
  fastest: HighParkVisit | null
  mostLaps: HighParkVisit | null
}
export interface HighParkLoops {
  cycling: HighParkSport
  running: HighParkSport
  /** viewBox (metres) that holds every lap trace, padded; null when no ride has one. */
  frame: { x: number; y: number; width: number; height: number } | null
}

function metres(a: { lat: number; lng: number }, b: { lat: number; lng: number }): number {
  return Math.hypot((a.lat - b.lat) * M_PER_LAT, (a.lng - b.lng) * M_PER_LNG)
}

const inPark = (p: { lat: number; lng: number }): boolean =>
  p.lat > PARK.south && p.lat < PARK.north && p.lng > PARK.west && p.lng < PARK.east

type GpsStream = Pick<StravaStreams, 'latlng' | 'time' | 'distance'>

function* streamPoints(stream: GpsStream): Iterable<Point> {
  const time = stream.time ?? []
  for (let i = 0; i < time.length; i++) {
    yield {
      lat: stream.latlng[i]?.[0] ?? NaN,
      lng: stream.latlng[i]?.[1] ?? NaN,
      d: stream.distance[i] / 1000,
      elapsedS: time[i],
    }
  }
}

function gpsLaps(route: Iterable<Point>): Lap[] {
  const laps: Lap[] = []
  let visits: { gate: number; point: Point }[] = []
  let previous: Point | undefined
  for (const point of route) {
    const valid =
      [point.lat, point.lng, point.d, point.elapsedS].every(Number.isFinite) && inPark(point)
    const broken =
      previous &&
      (metres(previous, point) > 250 ||
        point.d < previous.d ||
        point.elapsedS <= previous.elapsedS ||
        point.elapsedS - previous.elapsedS > 120)
    if (!valid || broken) visits = []
    previous = valid ? point : undefined
    if (!valid) continue
    const gate = gates.findIndex(g => metres(g, point) <= 55)
    if (gate < 0 || visits.at(-1)?.gate === gate) continue
    visits.push({ gate, point })
    if (visits.length < 5) continue
    const window = visits.slice(-5)
    const first = window[0]
    const direction = (window[1].gate - first.gate + 4) % 4
    const ordered =
      (direction === 1 || direction === 3) &&
      window.every((v, i) => v.gate === (first.gate + direction * i) % 4)
    const distanceKm = point.d - first.point.d
    const seconds = point.elapsedS - first.point.elapsedS
    if (ordered && distanceKm >= 1.65 && distanceKm <= 2.1 && seconds > 0) {
      laps.push({ seconds, distanceKm, startKm: first.point.d, endKm: point.d })
      visits = [{ gate, point }]
    } else visits = window.slice(1)
  }
  return laps
}

function nativeLaps(ranges: readonly Range[]): Lap[] {
  const seen = new Set<string>()
  const laps: Lap[] = []
  let end = -Infinity
  for (const range of [...ranges].sort((a, b) => a.startElapsedS - b.startElapsedS)) {
    if (
      range.kind !== 'segment' ||
      range.label !== HIGH_PARK_SEGMENT ||
      seen.has(range.id) ||
      ![range.startElapsedS, range.endElapsedS, range.durationS, range.distanceKm].every(
        Number.isFinite,
      ) ||
      range.startElapsedS < end ||
      range.endElapsedS <= range.startElapsedS ||
      range.durationS <= 0 ||
      range.distanceKm < 1.65 ||
      range.distanceKm > 2.1
    )
      continue
    seen.add(range.id)
    end = range.endElapsedS
    const startKm = Number.isFinite(range.startDistanceKm) ? range.startDistanceKm! : null
    laps.push({
      seconds: range.durationS,
      distanceKm: range.distanceKm,
      startKm,
      endKm: startKm === null ? null : startKm + range.distanceKm,
      watts: range.averageWatts,
      heartRate: range.averageHeartRate,
      cadence: range.averageCadence,
    })
  }
  return laps
}

const finiteRounded = (value: number | null | undefined): number | null =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.round(value) : null

type XY = [number, number]

// Ramer–Douglas–Peucker on projected points; keeps the drawn line within `tolerance` of the GPS.
function simplify(points: XY[], tolerance: number): XY[] {
  if (points.length < 3) return points
  const keep = new Uint8Array(points.length)
  keep[0] = keep[points.length - 1] = 1
  const stack: [number, number][] = [[0, points.length - 1]]
  while (stack.length) {
    const [a, b] = stack.pop()!
    const [ax, ay] = points[a]
    const [bx, by] = points[b]
    const length = Math.hypot(bx - ax, by - ay) || 1
    let worst = -1
    let index = -1
    for (let i = a + 1; i < b; i++) {
      const [px, py] = points[i]
      const distance = Math.abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / length
      if (distance > worst) {
        worst = distance
        index = i
      }
    }
    if (worst > tolerance) {
      keep[index] = 1
      stack.push([a, index], [index, b])
    }
  }
  return points.filter((_, i) => keep[i])
}

// Absolute first point, then relative steps: about five bytes per vertex.
function pathData(runs: XY[][], tolerance: number, digits: number): string {
  const scale = 10 ** digits
  const q = (value: number) => Math.round(value * scale) / scale
  return runs
    .map(run => simplify(run, tolerance))
    .filter(run => run.length >= 2)
    .map(run => {
      let [px, py] = [q(run[0][0]), q(run[0][1])]
      let out = `M${px} ${py}l`
      for (const [x, y] of run.slice(1)) {
        const [qx, qy] = [q(x), q(y)]
        const dx = q(qx - px)
        const dy = q(qy - py)
        if (!dx && !dy) continue
        out += `${dx}${dy < 0 ? '' : ' '}${dy} `
        ;[px, py] = [qx, qy]
      }
      return out.trimEnd()
    })
    .join('')
}

const frameXY = (p: { lat: number; lng: number }): XY => [
  (p.lng - PARK.west) * M_PER_LNG,
  (PARK.north - p.lat) * M_PER_LAT,
]

// Splits a point list wherever it leaves the park box, so clipped stretches never join up.
function parkRuns<P extends { lat: number; lng: number }>(points: Iterable<P>): P[][] {
  const runs: P[][] = []
  let run: P[] = []
  for (const point of points) {
    if (Number.isFinite(point.lat) && Number.isFinite(point.lng) && inPark(point)) {
      run.push(point)
      continue
    }
    if (run.length >= 2) runs.push(run)
    run = []
  }
  if (run.length >= 2) runs.push(run)
  return runs
}

type Box = [number, number, number, number]

function lapTrace(activity: HighParkActivity, laps: Lap[]): { d: string; box: Box } | null {
  const fastest = laps.reduce((best, lap) => (lap.seconds < best.seconds ? lap : best), laps[0])
  if (fastest.startKm === null || fastest.endKm === null) return null
  const { startKm, endKm } = fastest
  const points = (activity.mapRoute ?? []).flat().filter(p => p.d >= startKm && p.d <= endKm)
  const runs = parkRuns(points).map(run => run.map(frameXY))
  if (runs.reduce((n, run) => n + run.length, 0) < 8) return null
  const box: Box = [Infinity, Infinity, -Infinity, -Infinity]
  for (const [x, y] of runs.flat()) {
    box[0] = Math.min(box[0], x)
    box[1] = Math.min(box[1], y)
    box[2] = Math.max(box[2], x)
    box[3] = Math.max(box[3], y)
  }
  return { d: pathData(runs, 1.5, 0), box }
}

function rideGlyph(route: readonly Point[]): HighParkVisit['glyph'] {
  const drawn = route.filter(p => Number.isFinite(p.x) && Number.isFinite(p.y))
  if (drawn.length < 2) return null
  // Route x/y are already normalised to a square with north up; flip y for SVG.
  const xy = (p: Point): XY => [p.x! * GLYPH, (1 - p.y!) * GLYPH]
  return {
    route: pathData([drawn.map(xy)], 0.35, 1),
    park: pathData(
      parkRuns(drawn).map(run => run.map(xy)),
      0.35,
      1,
    ),
  }
}

function summarize(activities: HighParkVisit[]): HighParkSport {
  activities.sort((a, b) => b.date.localeCompare(a.date) || b.start.localeCompare(a.start))
  return {
    activities,
    laps: activities.reduce((sum, a) => sum + a.laps, 0),
    distanceKm: activities.reduce((sum, a) => sum + a.distanceKm, 0),
    estimatedLaps: activities
      .filter(a => a.source === 'GPS estimate')
      .reduce((sum, a) => sum + a.laps, 0),
    // GPS gates and native segment endpoints differ; records use only the native circuit.
    fastest: activities
      .filter(a => a.source === 'Strava segment')
      .reduce<HighParkVisit | null>(
        (best, a) => (!best || a.fastestS < best.fastestS ? a : best),
        null,
      ),
    mostLaps: activities.reduce<HighParkVisit | null>(
      (best, a) => (!best || a.laps > best.laps ? a : best),
      null,
    ),
  }
}

export function buildHighParkLoops(
  activities: readonly HighParkActivity[],
  streams: Readonly<Record<string, GpsStream>> = {},
): HighParkLoops {
  const cycling: HighParkVisit[] = []
  const running: HighParkVisit[] = []
  const seen = new Set<number>()
  const box: Box = [Infinity, Infinity, -Infinity, -Infinity]
  for (const activity of activities) {
    if (
      seen.has(activity.id) ||
      activity.virtual ||
      activity.electric ||
      (activity.sport !== 'bike' && activity.sport !== 'run')
    )
      continue
    const href = triathlonActivityHref(activity.date, activity.id)
    if (!href) continue
    seen.add(activity.id)
    const native = nativeLaps(activity.analysisRanges)
    const stream = streams[String(activity.id)]
    const laps = native.length ? native : gpsLaps(stream ? streamPoints(stream) : activity.route)
    if (!laps.length) continue
    const trace = lapTrace(activity, laps)
    const extras = laps.map(lap => lapExtras(activity, lap.startKm, lap.endKm))
    if (trace) {
      box[0] = Math.min(box[0], trace.box[0])
      box[1] = Math.min(box[1], trace.box[1])
      box[2] = Math.max(box[2], trace.box[2])
      box[3] = Math.max(box[3], trace.box[3])
    }
    const visit: HighParkVisit = {
      id: activity.id,
      date: activity.date,
      start: activity.start,
      name: activity.name,
      href,
      source: native.length ? 'Strava segment' : 'GPS estimate',
      laps: laps.length,
      distanceKm: laps.reduce((sum, lap) => sum + lap.distanceKm, 0),
      fastestS: Math.min(...laps.map(lap => lap.seconds)),
      lapsS: laps.map(lap => Math.round(lap.seconds)),
      lapDetail: laps.map((lap, i) => {
        const { altitude: _, ...extra } = extras[i]
        const profile = extra.profile
        const routeMean = (values: (number | null)[] | undefined) => {
          const known = (values ?? []).filter((v): v is number => v !== null && v > 0)
          return known.length ? Math.round(known.reduce((a, b) => a + b, 0) / known.length) : null
        }
        return {
          s: Math.round(lap.seconds),
          km: Math.round(lap.distanceKm * 1000) / 1000,
          // The segment effort's averages win; GPS-estimated laps fall back to the route samples.
          w: finiteRounded(lap.watts) ?? routeMean(profile?.w),
          hr: finiteRounded(lap.heartRate) ?? routeMean(profile?.hr),
          cad: finiteRounded(lap.cadence) ?? routeMean(profile?.cad),
          ...extra,
        }
      }),
      altitude: rideAltitude(extras.map(extra => extra.altitude)),
      trace: trace?.d ?? '',
      glyph: rideGlyph(activity.route),
    }
    ;(activity.sport === 'bike' ? cycling : running).push(visit)
  }
  const pad = 24
  const frame = Number.isFinite(box[0])
    ? {
        x: Math.floor(box[0] - pad),
        y: Math.floor(box[1] - pad),
        width: Math.ceil(box[2] - box[0] + 2 * pad),
        height: Math.ceil(box[3] - box[1] + 2 * pad),
      }
    : null
  return { cycling: summarize(cycling), running: summarize(running), frame }
}
