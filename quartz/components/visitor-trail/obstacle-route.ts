import type { JourneyRoute } from './hiker-journey'
import type { Landscape, Point } from './landscape'
import {
  canMove,
  circleClearance,
  queuedAnimal,
  type CollisionActor,
  type SpacingBounds,
} from './actor-spacing'

export type PlannedTrail = { points: Point[]; detours: number }
export type NavigableTrail = JourneyRoute & { d: string; detours: number }
export type TrailSearch = Generator<void, Point[] | null, undefined>
export const detourNeighbourhood = 160

function finishSearch<T>(search: Generator<void, T, undefined>): T {
  let step = search.next()
  while (!step.done) step = search.next()
  return step.value
}

function segmentDistance(point: Point, from: Point, to: Point) {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const fraction = Math.max(
    0,
    Math.min(1, ((point.x - from.x) * dx + (point.y - from.y) * dy) / (dx * dx + dy * dy || 1)),
  )
  return Math.hypot(point.x - from.x - fraction * dx, point.y - from.y - fraction * dy)
}

function freePoint(
  point: Point,
  radius: number,
  obstacles: readonly CollisionActor[],
  bounds: SpacingBounds,
) {
  return (
    point.x >= bounds.left &&
    point.x <= bounds.right &&
    point.y >= bounds.top &&
    point.y <= bounds.bottom &&
    obstacles.every(obstacle => {
      if (obstacle.bounds) return circleClearance(point, radius, obstacle) >= 0.1
      const dx = point.x - obstacle.x
      const dy = point.y - obstacle.y
      const reach = radius + obstacle.radius + 2.1
      return dx * dx + dy * dy >= reach * reach
    })
  )
}

function freeEndpoint(
  point: Point,
  radius: number,
  obstacles: readonly CollisionActor[],
  bounds: SpacingBounds,
): Point | null {
  const center = {
    x: Math.max(bounds.left, Math.min(bounds.right, point.x)),
    y: Math.max(bounds.top, Math.min(bounds.bottom, point.y)),
  }
  if (freePoint(center, radius, obstacles, bounds)) return center
  for (let distance = 4; distance <= 96; distance += 4)
    for (let index = 0; index < 16; index++) {
      const angle = (index * Math.PI) / 8
      const candidate = {
        x: center.x + Math.cos(angle) * distance,
        y: center.y + Math.sin(angle) * distance,
      }
      if (freePoint(candidate, radius, obstacles, bounds)) return candidate
    }
  return null
}

function* segmentDetour(
  from: Point,
  to: Point,
  radius: number,
  obstacles: readonly CollisionActor[],
  bounds: SpacingBounds,
  detail: 'coarse' | 'fine',
  nearbyGoal: Point | null = null,
): TrailSearch {
  const nodes = [from, to]
  const neighbourhood = detail === 'coarse' ? detourNeighbourhood / 2 : detourNeighbourhood
  const sides = detail === 'coarse' ? 16 : 32
  const padding = detail === 'coarse' ? 6 : 2.25
  for (const obstacle of obstacles) {
    yield
    if (segmentDistance(obstacle, from, to) > radius + obstacle.radius + neighbourhood) continue
    // The polygon's chords stay outside the expanded collision circle.
    const reach = (radius + obstacle.radius + padding) / Math.cos(Math.PI / sides)
    for (let index = 0; index < sides; index++) {
      const angle = (index * Math.PI * 2) / sides
      const candidate = {
        x: obstacle.x + Math.cos(angle) * reach,
        y: obstacle.y + Math.sin(angle) * reach,
      }
      if (freePoint(candidate, radius, obstacles, bounds)) nodes.push(candidate)
    }
  }
  const left = Math.min(...nodes.map(point => point.x)) - radius - 2
  const right = Math.max(...nodes.map(point => point.x)) + radius + 2
  const top = Math.min(...nodes.map(point => point.y)) - radius - 2
  const bottom = Math.max(...nodes.map(point => point.y)) + radius + 2
  // Every candidate edge stays within this box. Bodies outside it cannot block the search.
  const blockers = obstacles.filter(other =>
    other.bounds
      ? other.bounds.right >= left &&
        other.bounds.left <= right &&
        other.bounds.bottom >= top &&
        other.bounds.top <= bottom
      : other.x + other.radius >= left &&
        other.x - other.radius <= right &&
        other.y + other.radius >= top &&
        other.y - other.radius <= bottom,
  )
  const cost = new Float64Array(nodes.length).fill(Infinity)
  const heuristic = nodes.map(point => Math.hypot(point.x - to.x, point.y - to.y))
  const previous = new Int32Array(nodes.length).fill(-1)
  const visited = new Uint8Array(nodes.length)
  cost[0] = 0
  let arrival = -1
  let nearest = 96
  const trace = (end: number) => {
    const result: Point[] = []
    for (let index = end; index > 0; index = previous[index]) result.unshift(nodes[index])
    return result
  }
  for (let iteration = 0; iteration < nodes.length; iteration++) {
    let best = Infinity
    let current = -1
    for (let index = 0; index < nodes.length; index++) {
      if (visited[index]) continue
      const estimate = cost[index] + heuristic[index]
      if (estimate < best) {
        best = estimate
        current = index
      }
    }
    if (current < 0) break
    if (current === 1) return trace(1)
    if (nearbyGoal && current > 1) {
      const distance = Math.hypot(nodes[current].x - nearbyGoal.x, nodes[current].y - nearbyGoal.y)
      if (distance < nearest) {
        nearest = distance
        arrival = current
      }
    }
    visited[current] = 1
    for (let index = 0; index < nodes.length; index++) {
      if (index % 32 === 0) yield
      if (visited[index]) continue
      const remaining = cost[index] - cost[current]
      if (remaining <= 0) continue
      const dx = nodes[index].x - nodes[current].x
      const dy = nodes[index].y - nodes[current].y
      const squared = dx * dx + dy * dy
      if (squared >= remaining * remaining) continue
      const distance = Math.sqrt(squared)
      if (
        cost[current] + distance < cost[index] &&
        canMove(nodes[current], nodes[index], radius, blockers)
      ) {
        cost[index] = cost[current] + distance
        previous[index] = current
      }
    }
  }
  return arrival >= 0 ? trace(arrival) : null
}

export function detourTrail(
  road: readonly Point[],
  radius: number,
  obstacles: readonly CollisionActor[],
  bounds: SpacingBounds,
  allowNearbyArrival = true,
): PlannedTrail | null {
  return finishSearch(detourTrailSteps(road, radius, obstacles, bounds, allowNearbyArrival))
}

export function* detourTrailSteps(
  road: readonly Point[],
  radius: number,
  obstacles: readonly CollisionActor[],
  bounds: SpacingBounds,
  allowNearbyArrival = true,
): Generator<void, PlannedTrail | null, undefined> {
  if (!road.length) return null
  const first = freeEndpoint(road[0], radius, obstacles, bounds)
  const last = freeEndpoint(road[road.length - 1], radius, obstacles, bounds)
  if (!first || !last) return null
  const points = [first]
  let detours = 0
  for (let index = 1; index < road.length; index++) {
    yield
    const goal = index === road.length - 1 ? last : road[index]
    if (!freePoint(goal, radius, obstacles, bounds)) continue
    const from = points[points.length - 1]
    if (Math.hypot(goal.x - from.x, goal.y - from.y) < 0.01) continue
    if (canMove(from, goal, radius, obstacles)) points.push(goal)
    else {
      // The finer search includes neighbouring bodies and the narrow gaps a coarse polygon
      // can close. An enclosed camp ends at its nearest reachable edge within the arrival radius.
      const detour =
        (yield* segmentDetour(from, goal, radius, obstacles, bounds, 'coarse')) ??
        (yield* segmentDetour(
          from,
          goal,
          radius,
          obstacles,
          bounds,
          'fine',
          allowNearbyArrival && index === road.length - 1 ? road[index] : null,
        ))
      if (!detour) {
        if (index === road.length - 1) return null
        continue
      }
      points.push(...detour)
      detours++
    }
  }
  return { points, detours }
}

export function replanOccupiedWaypoints(
  id: string,
  from: Point,
  waypoints: readonly Point[],
  radius: number,
  obstacles: readonly CollisionActor[],
  bounds: SpacingBounds,
  waitingForAnimal?: string | null,
): Point[] | null {
  return finishSearch(
    replanOccupiedWaypointsSteps(id, from, waypoints, radius, obstacles, bounds, waitingForAnimal),
  )
}

export function* replanOccupiedWaypointsSteps(
  id: string,
  from: Point,
  waypoints: readonly Point[],
  radius: number,
  obstacles: readonly CollisionActor[],
  bounds: SpacingBounds,
  waitingForAnimal?: string | null,
): TrailSearch {
  const goal = waypoints[0]
  if (!goal) return null
  const others = obstacles.filter(other => other.id !== id)
  const restingAnimal = others.find(
    other => other.id === waitingForAnimal && other.kind === 'animal' && other.resting,
  )
  if (waitingForAnimal && !restingAnimal) return null
  const blockers = others.filter(
    other => circleClearance(goal, radius, other) < 0 || !canMove(from, goal, radius, [other]),
  )
  if (
    !restingAnimal &&
    (!blockers.length ||
      blockers.some(
        other =>
          other.kind === 'animal' || (other.kind === 'traveler' && queuedAnimal(id, other, others)),
      ))
  )
    return null
  const bodies = restingAnimal
    ? others.map(other =>
        other === restingAnimal ? { ...other, radius: other.radius + radius * 1.5 + 14 } : other,
      )
    : others.filter(other => other.kind !== 'animal')
  let rejoin = 0
  // Rejoin beyond the occupied run of waypoints. Preserve the rest of the patrol, including
  // distant legs that may become clear before the walker reaches them.
  while (
    rejoin < waypoints.length - 1 &&
    (!freePoint(waypoints[rejoin], radius, bodies, bounds) ||
      !canMove(waypoints[rejoin], waypoints[rejoin + 1], radius, bodies))
  )
    rejoin++
  const planned = yield* detourTrailSteps(
    [from, waypoints[rejoin]],
    radius,
    bodies,
    bounds,
    rejoin === waypoints.length - 1,
  )
  if (!planned) return null
  const start = planned.points[0]
  const repair =
    Math.hypot(start.x - from.x, start.y - from.y) > 0.1 ? planned.points : planned.points.slice(1)
  return [...repair, ...waypoints.slice(rejoin + 1)]
}

export function trailNavigationInput(
  route: JourneyRoute,
  landscape: Landscape,
  obstacles: readonly CollisionActor[],
) {
  const steps = Math.max(1, Math.ceil(route.length / 16))
  const road = Array.from({ length: steps + 1 }, (_, index) =>
    route.point((route.length * index) / steps),
  )
  return {
    road,
    radius: 18,
    obstacles: obstacles.map(({ id, kind, x, y, radius, bounds }) => ({
      id,
      kind,
      x,
      y,
      radius,
      bounds,
    })),
    bounds: {
      left: 18,
      right: landscape.width - 18,
      top: Math.max(landscape.treeline + 8, landscape.alpine.base - 18),
      bottom: landscape.height - 18,
    },
  }
}

export function navigableTrail(
  route: JourneyRoute,
  landscape: Landscape,
  obstacles: readonly CollisionActor[],
): NavigableTrail | null {
  const input = trailNavigationInput(route, landscape, obstacles)
  return plannedTrail(route, detourTrail(input.road, input.radius, input.obstacles, input.bounds))
}

export function plannedTrail(
  route: JourneyRoute,
  planned: PlannedTrail | null,
): NavigableTrail | null {
  if (!planned) return null
  let length = 0
  const distances = planned.points.map((point, index) => {
    if (index)
      length += Math.hypot(
        point.x - planned.points[index - 1].x,
        point.y - planned.points[index - 1].y,
      )
    return length
  })
  const point = (distance: number): Point => {
    let left = 0
    let right = distances.length - 1
    const position = Math.max(0, Math.min(length, distance))
    while (left < right) {
      const middle = Math.floor((left + right) / 2)
      if (distances[middle] < position) left = middle + 1
      else right = middle
    }
    if (!left) return planned.points[0]
    const from = planned.points[left - 1]
    const to = planned.points[left]
    const fraction = (position - distances[left - 1]) / (distances[left] - distances[left - 1] || 1)
    return { x: from.x + (to.x - from.x) * fraction, y: from.y + (to.y - from.y) * fraction }
  }
  return {
    ...route,
    length,
    point,
    detours: planned.detours,
    d: planned.points
      .map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
      .join(''),
  }
}
