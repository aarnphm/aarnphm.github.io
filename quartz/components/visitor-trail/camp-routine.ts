import type { JourneyRoute } from './hiker-journey'
import { circleClearance, type CollisionActor } from './actor-spacing'
import { followTerrain, trailGroundScale, type Landscape, type Point } from './landscape'
import { detourTrail } from './obstacle-route'
import { trailRandom } from './random'

export type CampPhase = 'staying' | 'sleeping' | 'packing' | 'extinguishing' | 'departing' | 'empty'
export type CampRoutine = {
  phase: CampPhase
  since: number
  leaveAt: number
  sleepAt: number
  stay: number
  absence: number
  variation: 'overnight' | 'passing'
}

export function campRoutine(seed: number, site: number, now: number): CampRoutine {
  const random = trailRandom(seed ^ Math.imul(site + 1, 0x6d2b79f5))
  const variation = (site + (seed & 1)) % 2 ? 'overnight' : 'passing'
  const stay = Math.round(((variation === 'overnight' ? 8 : 2) + random() * 2) * 60 * 60 * 1000)
  return {
    phase: 'staying',
    since: now,
    leaveAt: now + stay,
    sleepAt: now + Math.round((20 + random() * 20) * 60 * 1000),
    stay,
    absence: Math.round((30 + random() * 30) * 60 * 1000),
    variation,
  }
}

export function campMayDepart(routine: CampRoutine, now: number, night: boolean) {
  return now >= routine.leaveAt && (routine.variation === 'passing' || !night)
}

export function campMaySleep(routine: CampRoutine, now: number, night: boolean) {
  const hour = new Date(now).getHours()
  return (
    routine.variation === 'overnight' && now >= routine.sleepAt && night && (hour >= 22 || hour < 7)
  )
}

function clearTrailPoint(
  point: Point,
  landscape: Landscape,
  solids: readonly CollisionActor[],
  radius: number,
) {
  // Steering can pass an obstacle, but cannot finish a waypoint inside its body.
  return (
    point.x >= 18 &&
    point.x <= landscape.width - 18 &&
    point.y >= landscape.treeline + 8 &&
    point.y <= landscape.height - 18 &&
    solids.every(
      solid => circleClearance(point, radius * trailGroundScale(point.y, landscape), solid) >= 3,
    )
  )
}

function clearTrailSegments(
  from: Point,
  points: Point[],
  landscape: Landscape,
  solids: readonly CollisionActor[],
  radius: number,
) {
  const planned = detourTrail([from, ...points], radius, solids, {
    left: 18,
    right: landscape.width - 18,
    top: landscape.treeline + 8,
    bottom: landscape.height - 18,
  })
  return planned?.points.slice(1) ?? points
}

export function campDeparture(
  from: Point,
  routes: JourneyRoute[],
  landscape: Landscape,
  solids: readonly CollisionActor[],
): Point[] {
  const clear = (point: Point) => clearTrailPoint(point, landscape, solids, 12)
  let nearest: { route: JourneyRoute; distance: number } | null = null
  let gap = Infinity
  for (const route of routes) {
    for (let index = 0; index <= 48; index++) {
      const distance = (route.length * index) / 48
      const point = route.point(distance)
      if (!clear(point)) continue
      const current = Math.hypot(point.x - from.x, point.y - from.y)
      if (current < gap) {
        gap = current
        nearest = { route, distance }
      }
    }
  }
  if (!nearest) return [from]
  const join = followTerrain([from, nearest.route.point(nearest.distance)], landscape).slice(1)
  const steps = Math.max(1, Math.ceil(nearest.distance / 16))
  const points = [
    ...join,
    ...Array.from({ length: steps }, (_, index) =>
      nearest.route.point(nearest.distance * (1 - (index + 1) / steps)),
    ),
  ].filter(clear)
  return clearTrailSegments(from, points, landscape, solids, 12)
}

export function campPatrolRoute(
  from: Point,
  to: Point,
  routes: JourneyRoute[],
  landscape: Landscape,
  solids: readonly CollisionActor[],
): Point[] {
  const clear = (point: Point) => clearTrailPoint(point, landscape, solids, 18)
  let best: { route: JourneyRoute; start: number; end: number } | null = null
  let cost = Infinity
  for (const route of routes) {
    let start = 0
    let end = 0
    let fromGap = Infinity
    let toGap = Infinity
    for (let index = 0; index <= 48; index++) {
      const distance = (route.length * index) / 48
      const point = route.point(distance)
      if (!clear(point)) continue
      const a = Math.hypot(point.x - from.x, point.y - from.y)
      const b = Math.hypot(point.x - to.x, point.y - to.y)
      if (a < fromGap) {
        fromGap = a
        start = distance
      }
      if (b < toGap) {
        toGap = b
        end = distance
      }
    }
    const total = fromGap + toGap + Math.abs(end - start)
    if (total < cost) {
      cost = total
      best = { route, start, end }
    }
  }
  if (!best)
    return clearTrailSegments(
      from,
      followTerrain([from, to], landscape).slice(1).filter(clear),
      landscape,
      solids,
      18,
    )
  const steps = Math.max(1, Math.ceil(Math.abs(best.end - best.start) / 16))
  const road = best
  const points = [
    ...followTerrain([from, road.route.point(road.start)], landscape).slice(1),
    ...Array.from({ length: steps }, (_, index) =>
      road.route.point(road.start + ((road.end - road.start) * (index + 1)) / steps),
    ),
    ...followTerrain([road.route.point(road.end), to], landscape).slice(1),
  ].filter(clear)
  return clearTrailSegments(from, points, landscape, solids, 18)
}
