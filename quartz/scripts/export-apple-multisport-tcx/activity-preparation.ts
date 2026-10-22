import {
  haversineMeters,
  timedValue,
  type CourseRoute,
  type HeartRateSample,
  type MultisportActivity,
  type Position,
  type PreparedActivity,
  type RoutePoint,
  type TcxSport,
  type TcxTrackPoint,
} from './conversion-model'

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

export function retimedActivity(
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

export function prepareActivity(
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

export function prepareCourseActivity(
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

export function validatePrepared(prepared: PreparedActivity): void {
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
