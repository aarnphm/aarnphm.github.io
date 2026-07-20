export interface MapRoutePoint {
  lat: number
  lng: number
  d: number
}

const EARTH_RADIUS_M = 6_371_000

export function mapDistanceMeters(a: MapRoutePoint, b: MapRoutePoint): number {
  const lat1 = (a.lat * Math.PI) / 180
  const lat2 = (b.lat * Math.PI) / 180
  const deltaLat = lat2 - lat1
  const deltaLng = ((b.lng - a.lng) * Math.PI) / 180
  const sinLat = Math.sin(deltaLat / 2)
  const sinLng = Math.sin(deltaLng / 2)
  const root = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLng * sinLng
  return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(root)))
}

function pointSegmentDistanceMeters(
  point: MapRoutePoint,
  start: MapRoutePoint,
  end: MapRoutePoint,
): number {
  const meanLat = ((start.lat + end.lat + point.lat) * Math.PI) / 540
  const scaleX = Math.cos(meanLat) * 111_320
  const ax = start.lng * scaleX
  const ay = start.lat * 111_320
  const bx = end.lng * scaleX
  const by = end.lat * 111_320
  const px = point.lng * scaleX
  const py = point.lat * 111_320
  const dx = bx - ax
  const dy = by - ay
  const lengthSquared = dx * dx + dy * dy
  if (lengthSquared === 0) return Math.hypot(px - ax, py - ay)
  const fraction = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSquared))
  return Math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy))
}

export function simplifyMapRoute(
  points: readonly MapRoutePoint[],
  toleranceM: number,
): MapRoutePoint[] {
  if (points.length <= 2 || toleranceM <= 0) return points.map(point => ({ ...point }))
  const keep = new Uint8Array(points.length)
  keep[0] = 1
  keep[points.length - 1] = 1
  const stack: [number, number][] = [[0, points.length - 1]]
  while (stack.length > 0) {
    const pair = stack.pop()
    if (!pair) break
    const [start, end] = pair
    let farthest = -1
    let distance = toleranceM
    for (let index = start + 1; index < end; index++) {
      const candidate = pointSegmentDistanceMeters(points[index], points[start], points[end])
      if (candidate <= distance) continue
      distance = candidate
      farthest = index
    }
    if (farthest < 0) continue
    keep[farthest] = 1
    stack.push([start, farthest], [farthest, end])
  }
  return points.filter((_, index) => keep[index] === 1).map(point => ({ ...point }))
}

function simplifyWithinLimit(
  points: readonly MapRoutePoint[],
  toleranceM: number,
  maxPoints: number,
): MapRoutePoint[] {
  let simplified = simplifyMapRoute(points, toleranceM)
  if (simplified.length <= maxPoints) return simplified
  let low = toleranceM
  let high = Math.max(4, toleranceM * 2)
  while (simplifyMapRoute(points, high).length > maxPoints && high < 256) high *= 2
  for (let iteration = 0; iteration < 12; iteration++) {
    const middle = (low + high) / 2
    const candidate = simplifyMapRoute(points, middle)
    if (candidate.length > maxPoints) low = middle
    else {
      high = middle
      simplified = candidate
    }
  }
  return simplified
}

function roundPoint(point: MapRoutePoint): MapRoutePoint {
  return {
    lat: Math.round(point.lat * 1_000_000) / 1_000_000,
    lng: Math.round(point.lng * 1_000_000) / 1_000_000,
    d: Math.round(point.d * 10_000) / 10_000,
  }
}

export function rawMapRouteSegments(
  latlng: readonly [number, number][],
  distanceM: readonly number[],
  elapsedS: readonly number[],
  lo: number,
  hi: number,
  toleranceM = 2,
  maxPoints = 4_000,
): MapRoutePoint[][] {
  const last = Math.min(hi, latlng.length - 1, distanceM.length - 1, elapsedS.length - 1)
  if (last - lo < 1) return []
  const segments: MapRoutePoint[][] = []
  let segment: MapRoutePoint[] = []
  let previousDistanceM = Math.max(0, distanceM[lo] ?? 0)
  let previousPoint: MapRoutePoint | null = null
  const flush = (): void => {
    if (segment.length >= 2) segments.push(segment)
    segment = []
  }
  for (let index = lo; index <= last; index++) {
    const coordinates = latlng[index]
    const time = elapsedS[index]
    if (
      !Number.isFinite(coordinates[0]) ||
      !Number.isFinite(coordinates[1]) ||
      !Number.isFinite(time)
    ) {
      flush()
      previousPoint = null
      continue
    }
    const currentDistanceM = Math.max(
      previousDistanceM,
      Number.isFinite(distanceM[index]) ? distanceM[index] : previousDistanceM,
    )
    const point = { lat: coordinates[0], lng: coordinates[1], d: currentDistanceM / 1000 }
    if (previousPoint) {
      const activityGapM = currentDistanceM - previousDistanceM
      const geographicGapM = mapDistanceMeters(previousPoint, point)
      if (geographicGapM > Math.max(200, activityGapM * 4 + 50)) flush()
    }
    if (
      segment.length === 0 ||
      point.lat !== segment[segment.length - 1].lat ||
      point.lng !== segment[segment.length - 1].lng
    )
      segment.push(point)
    previousDistanceM = currentDistanceM
    previousPoint = point
  }
  flush()
  const totalPoints = segments.reduce((sum, current) => sum + current.length, 0)
  return segments.map(current =>
    simplifyWithinLimit(
      current,
      toleranceM,
      Math.max(2, Math.round((current.length / totalPoints) * maxPoints)),
    ).map(roundPoint),
  )
}

function interpolatePoint(a: MapRoutePoint, b: MapRoutePoint, distanceKm: number): MapRoutePoint {
  const span = b.d - a.d
  const fraction = span > 0 ? Math.max(0, Math.min(1, (distanceKm - a.d) / span)) : 0
  return roundPoint({
    lat: a.lat + (b.lat - a.lat) * fraction,
    lng: a.lng + (b.lng - a.lng) * fraction,
    d: distanceKm,
  })
}

function appendPoint(points: MapRoutePoint[], point: MapRoutePoint): void {
  const previous = points[points.length - 1]
  if (
    previous &&
    previous.d === point.d &&
    previous.lat === point.lat &&
    previous.lng === point.lng
  )
    return
  points.push(point)
}

export function clipMapRoute(
  segments: readonly (readonly MapRoutePoint[])[],
  startDistanceKm: number,
  endDistanceKm: number,
): MapRoutePoint[][] {
  const start = Math.min(startDistanceKm, endDistanceKm)
  const end = Math.max(startDistanceKm, endDistanceKm)
  const clipped: MapRoutePoint[][] = []
  for (const segment of segments) {
    const points: MapRoutePoint[] = []
    for (let index = 1; index < segment.length; index++) {
      const a = segment[index - 1]
      const b = segment[index]
      if (b.d < start || a.d > end || b.d < a.d) continue
      const edgeStart = Math.max(start, a.d)
      const edgeEnd = Math.min(end, b.d)
      if (edgeEnd < edgeStart) continue
      appendPoint(points, edgeStart === a.d ? { ...a } : interpolatePoint(a, b, edgeStart))
      appendPoint(points, edgeEnd === b.d ? { ...b } : interpolatePoint(a, b, edgeEnd))
    }
    if (points.length >= 2) clipped.push(points)
  }
  return clipped
}

export function mapRoutePointAtDistance(
  segments: readonly (readonly MapRoutePoint[])[],
  distanceKm: number,
): MapRoutePoint | null {
  let nearest: MapRoutePoint | null = null
  let nearestDelta = Infinity
  for (const segment of segments) {
    for (let index = 1; index < segment.length; index++) {
      const a = segment[index - 1]
      const b = segment[index]
      if (distanceKm >= a.d && distanceKm <= b.d && b.d >= a.d)
        return interpolatePoint(a, b, distanceKm)
    }
    for (const point of [segment[0], segment[segment.length - 1]]) {
      if (!point) continue
      const delta = Math.abs(point.d - distanceKm)
      if (delta >= nearestDelta) continue
      nearest = point
      nearestDelta = delta
    }
  }
  return nearest ? { ...nearest } : null
}
