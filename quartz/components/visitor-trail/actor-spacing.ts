import type { Point } from './landscape'

export type CollisionActor = Point & {
  id: string
  kind: 'animal' | 'traveler' | 'trunk'
  radius: number
  element?: SVGGElement
}
type SpacingStatus = 'clear' | 'passing' | 'yielding' | 'escaping'
export type ActorSpacing = {
  offset: Point
  lane: Point
  blocker: string | null
  status: SpacingStatus
}
export type SpacingBounds = { left: number; right: number; top: number; bottom: number }

export function readActors(scene: ParentNode): CollisionActor[] {
  const actors: CollisionActor[] = []
  for (const element of scene.querySelectorAll<SVGGElement>('[data-trail-collision]')) {
    if (element.dataset.collisionActive === 'false' || !element.transform.baseVal.numberOfItems)
      continue
    const kind = element.dataset.trailCollision
    const radius = Number(element.dataset.collisionRadius)
    if ((kind !== 'animal' && kind !== 'traveler') || !Number.isFinite(radius) || radius <= 0)
      continue
    const matrix = element.transform.baseVal.getItem(0).matrix
    const id =
      element.dataset.collisionId ??
      element.dataset.trailWander ??
      element.dataset.reader ??
      element.dataset.alpineTraveler
    if (!id) continue
    actors.push({ id, kind, x: matrix.e, y: matrix.f, radius, element })
  }
  return actors
}

export function readTrunks(scene: ParentNode): CollisionActor[] {
  return Array.from(scene.querySelectorAll<SVGGElement>('.trail-tree')).flatMap(
    (element, index) => {
      if (!element.transform.baseVal.numberOfItems) return []
      const matrix = element.transform.baseVal.getItem(0).matrix
      const radius = Number(element.dataset.trunkRadius ?? 3 * Number(element.dataset.treeScale))
      if (!Number.isFinite(radius)) return []
      const trunk: CollisionActor = {
        id: `trunk-${index}`,
        kind: 'trunk',
        x: matrix.e,
        y: matrix.f,
        radius,
        element,
      }
      return [trunk]
    },
  )
}

export function circleClearance(point: Point, radius: number, other: CollisionActor) {
  return Math.hypot(point.x - other.x, point.y - other.y) - radius - other.radius - 2
}

export function canMove(
  from: Point,
  to: Point,
  radius: number,
  obstacles: readonly CollisionActor[],
) {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const squared = dx * dx + dy * dy
  for (const other of obstacles) {
    const x = from.x - other.x
    const y = from.y - other.y
    const initial = Math.hypot(x, y)
    const limit = radius + other.radius + 2
    if (initial < limit - 0.01) {
      // A resize can tighten spacing. Escape must increase separation from its first step.
      if (x * dx + y * dy < -0.00001) return false
      if (Math.hypot(to.x - other.x, to.y - other.y) <= initial + 0.00001) return false
      continue
    }
    const fraction = squared ? Math.max(0, Math.min(1, -(x * dx + y * dy) / squared)) : 0
    if (Math.hypot(x + dx * fraction, y + dy * fraction) < limit - 0.01) return false
  }
  return true
}

export function actorSpacing(): ActorSpacing {
  return { offset: { x: 0, y: 0 }, lane: { x: 0, y: 0 }, blocker: null, status: 'clear' }
}

export function spacedPosition(base: Point, spacing: ActorSpacing): Point {
  return { x: base.x + spacing.offset.x, y: base.y + spacing.offset.y }
}

function towards(from: Point, to: Point, distance: number): Point {
  const length = Math.hypot(to.x - from.x, to.y - from.y)
  const fraction = length ? Math.min(1, distance / length) : 1
  return { x: from.x + (to.x - from.x) * fraction, y: from.y + (to.y - from.y) * fraction }
}

export function spacedStep(
  id: string,
  base: Point,
  next: Point,
  radius: number,
  budget: number,
  spacing: ActorSpacing,
  obstacles: readonly CollisionActor[],
  bounds?: SpacingBounds,
) {
  if (budget <= 0) return false
  const others = obstacles.filter(other => other.id !== id)
  const from = spacedPosition(base, spacing)
  const intended = spacedPosition(next, spacing)
  const distance = Math.hypot(next.x - base.x, next.y - base.y)
  const outside = (point: Point) =>
    bounds
      ? Math.hypot(
          point.x - Math.max(bounds.left, Math.min(bounds.right, point.x)),
          point.y - Math.max(bounds.top, Math.min(bounds.bottom, point.y)),
        )
      : 0
  const clears = (point: Point) => {
    const remaining = outside(point)
    return (
      (remaining === 0 || remaining < outside(from) - 0.00001) &&
      Math.hypot(point.x - from.x, point.y - from.y) <= budget + 0.01 &&
      canMove(from, point, radius, others)
    )
  }
  if (bounds && outside(from) > 0) {
    const edge = {
      x: Math.max(bounds.left, Math.min(bounds.right, from.x)),
      y: Math.max(bounds.top, Math.min(bounds.bottom, from.y)),
    }
    const escape = towards(from, edge, budget)
    if (clears(escape)) {
      spacing.offset = { x: escape.x - base.x, y: escape.y - base.y }
      spacing.status = 'escaping'
      return false
    }
  }
  const heading = distance
    ? { x: (next.x - base.x) / distance, y: (next.y - base.y) / distance }
    : { x: 1, y: 0 }
  const probe = {
    x: intended.x + heading.x * (radius * 1.5 + 6),
    y: intended.y + heading.y * (radius * 1.5 + 6),
  }
  const blocker = others
    .filter(other => !canMove(from, probe, radius, [other]))
    .sort(
      (left, right) => circleClearance(from, radius, left) - circleClearance(from, radius, right),
    )[0]
  if (!blocker) {
    spacing.blocker = null
    const recovered = towards(spacing.offset, { x: 0, y: 0 }, Math.max(0, budget - distance))
    const point = { x: next.x + recovered.x, y: next.y + recovered.y }
    if (clears(point)) {
      spacing.offset = recovered
      spacing.status = Math.hypot(recovered.x, recovered.y) > 0.1 ? 'passing' : 'clear'
      return true
    }
    if (clears(intended)) {
      spacing.status = 'passing'
      return true
    }
    spacing.status = 'yielding'
    return false
  }

  const intrusion = circleClearance(from, radius, blocker) < 0
  if (blocker.kind === 'traveler' && id < blocker.id && !intrusion) {
    spacing.status = 'yielding'
    return clears(intended)
  }
  if (spacing.blocker !== blocker.id) {
    spacing.blocker = blocker.id
    const normal = { x: -heading.y, y: heading.x }
    const sign = normal.y < -0.01 ? -1 : normal.y > 0.01 ? 1 : id.length % 2 ? -1 : 1
    const clearance = radius + blocker.radius + 5
    spacing.lane = {
      x: spacing.offset.x + normal.x * sign * clearance,
      y: spacing.offset.y + normal.y * sign * clearance,
    }
  }
  if (intrusion) {
    const away = { x: from.x - blocker.x, y: from.y - blocker.y }
    const length = Math.hypot(away.x, away.y)
    spacing.lane = {
      x: spacing.offset.x + (length ? away.x / length : 0) * (radius + blocker.radius + 5),
      y: spacing.offset.y + (length ? away.y / length : 1) * (radius + blocker.radius + 5),
    }
  }
  if (Math.hypot(spacing.lane.x - spacing.offset.x, spacing.lane.y - spacing.offset.y) < 0.05) {
    if (clears(intended)) {
      spacing.status = 'passing'
      return true
    }
    const normal = { x: -heading.y, y: heading.x }
    const sign = normal.y < -0.01 ? -1 : normal.y > 0.01 ? 1 : id.length % 2 ? -1 : 1
    spacing.lane = {
      x: spacing.offset.x + normal.x * sign * (radius + blocker.radius + 5),
      y: spacing.offset.y + normal.y * sign * (radius + blocker.radius + 5),
    }
  }
  const offset = towards(spacing.offset, spacing.lane, budget)
  const sidestep = { x: base.x + offset.x, y: base.y + offset.y }
  if (clears(sidestep)) {
    spacing.offset = offset
    spacing.status = intrusion ? 'escaping' : 'passing'
    return false
  }
  const away = { x: from.x - blocker.x, y: from.y - blocker.y }
  const length = Math.hypot(away.x, away.y)
  if (length) {
    const escape = {
      x: from.x + (away.x * budget) / length,
      y: from.y + (away.y * budget) / length,
    }
    if (clears(escape)) {
      spacing.offset = { x: escape.x - base.x, y: escape.y - base.y }
      spacing.status = 'escaping'
      return false
    }
  }
  if (clears(intended)) {
    spacing.status = 'passing'
    return true
  }
  spacing.status = 'yielding'
  return false
}

export function publishActor(
  element: SVGGElement,
  id: string,
  point: Point,
  scale: number,
  spacing: ActorSpacing,
  active = true,
) {
  const x = Number(point.x.toFixed(2))
  const y = Number(point.y.toFixed(2))
  element.setAttribute('transform', `translate(${x} ${y}) scale(${scale.toFixed(4)})`)
  element.dataset.trailCollision = 'traveler'
  element.dataset.collisionId = id
  element.dataset.collisionActive = String(active)
  element.dataset.collisionRadius = (12 * scale).toFixed(4)
  element.dataset.groundX = String(x)
  element.dataset.groundY = String(y)
  element.dataset.scale = scale.toFixed(4)
  element.dataset.spacing = spacing.status
  element.dataset.pathOffset = Math.hypot(spacing.offset.x, spacing.offset.y).toFixed(2)
}
