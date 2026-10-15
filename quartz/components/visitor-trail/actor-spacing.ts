import type { Point } from './landscape'
import { readClockObstacles } from './clock-clearance'
import { parallaxDepth, parallaxMovement } from './perspective'

export type CollisionActor = Point & {
  id: string
  kind: 'animal' | 'traveler' | 'pet' | 'trunk'
  radius: number
  bounds?: SpacingBounds
  resting?: boolean
  element?: SVGGElement
}
type SpacingStatus = 'clear' | 'passing' | 'yielding' | 'escaping'
export type ActorSpacing = {
  offset: Point
  lane: Point
  blocker: string | null
  side: -1 | 1
  status: SpacingStatus
  yieldingTo: string | null
  yieldingThrough: string | null
  yieldEpisode: number
  // The animal this walker has been yielding to and since when; cleared once it is out of the way.
  waited: { id: string; since: number } | null
  placement: { bounds: SpacingBounds; grounded: (point: Point, radius: number) => boolean } | null
}
export type SpacingBounds = { left: number; right: number; top: number; bottom: number }

const depthCache = new WeakMap<SVGGElement, { parent: Element | null; value: number }>()
const trunkFrames = new WeakMap<
  ParentNode,
  { time: DocumentTimeline['currentTime']; trunks: CollisionActor[] }
>()

function elementDepth(element?: SVGGElement) {
  if (!element) return 0
  const parent = element.parentElement
  const cached = depthCache.get(element)
  if (cached && cached.parent === parent) return cached.value
  const layer = element.closest<SVGGElement>('[data-trail-depth]')
  const value = layer
    ? layer.dataset.trailDepth === 'alpine'
      ? parallaxDepth.alpine
      : parallaxDepth.foothill
    : 0
  depthCache.set(element, { parent, value })
  return value
}

function cameraClearance(element: SVGGElement, relativeTo?: SVGGElement) {
  // Reserve all camera positions so parallax cannot move a background body into a valley walker.
  return Math.abs(elementDepth(element) - elementDepth(relativeTo)) * parallaxMovement.scroll
}

export function readActors(
  scene: ParentNode,
  includeWaiting = false,
  relativeTo?: SVGGElement,
): CollisionActor[] {
  const actors: CollisionActor[] = []
  for (const element of scene.querySelectorAll<SVGGElement>('[data-trail-collision]')) {
    if (
      element.dataset.enabled === 'false' ||
      (element.dataset.collisionActive === 'false' &&
        !(
          includeWaiting &&
          element.dataset.phase === 'waiting' &&
          element.dataset.placement !== 'blocked'
        )) ||
      !element.transform.baseVal.numberOfItems
    )
      continue
    const kind = element.dataset.trailCollision
    const radius = Number(element.dataset.collisionRadius)
    if (
      (kind !== 'animal' && kind !== 'traveler' && kind !== 'pet') ||
      !Number.isFinite(radius) ||
      radius <= 0
    )
      continue
    const matrix = element.transform.baseVal.getItem(0).matrix
    const id =
      element.dataset.collisionId ??
      element.dataset.trailWander ??
      element.dataset.reader ??
      element.dataset.alpineTraveler
    if (!id) continue
    actors.push({
      id,
      kind,
      x: matrix.e,
      y: matrix.f,
      radius: radius + cameraClearance(element, relativeTo),
      resting: kind === 'animal' && element.dataset.phase === 'resting',
      element,
    })
  }
  return actors
}

function walkingActor(actor: CollisionActor) {
  return ['walking', 'arriving', 'departing', 'climbing', 'descending', 'patrolling'].includes(
    actor.element?.dataset.phase ?? '',
  )
}

export function readStationaryActors(scene: ParentNode, except: string): CollisionActor[] {
  return readActors(scene).filter(
    actor => actor.id !== except && actor.kind === 'traveler' && !walkingActor(actor),
  )
}

function readBaseTrunks(scene: ParentNode): CollisionActor[] {
  const doc = 'ownerDocument' in scene ? scene.ownerDocument : null
  const time = doc?.timeline?.currentTime
  if (typeof time === 'number' && time > 0) {
    const known = trunkFrames.get(scene)
    if (known?.time === time && known.trunks.every(trunk => trunk.element?.isConnected))
      return known.trunks
  }
  const trunks: CollisionActor[] = []
  for (const element of scene.querySelectorAll<SVGGElement>('[data-camp-solid]')) {
    trunks.push({
      id: element.dataset.campSolid ?? '',
      kind: 'trunk',
      x: Number(element.dataset.groundX),
      y: Number(element.dataset.groundY),
      radius: Number(element.dataset.solidRadius),
      element,
    })
  }
  let index = 0
  for (const element of scene.querySelectorAll<SVGGElement>('.trail-tree, .trail-woodland-tree')) {
    const currentIndex = index++
    if (!element.transform.baseVal.numberOfItems) continue
    const matrix = element.transform.baseVal.getItem(0).matrix
    const radius = Number(element.dataset.trunkRadius ?? 3 * Number(element.dataset.treeScale))
    if (!Number.isFinite(radius)) continue
    trunks.push({
      id: `trunk-${element.dataset.treeId ?? element.dataset.woodlandId ?? currentIndex}`,
      kind: 'trunk',
      x: matrix.e,
      y: matrix.f,
      radius,
      element,
    })
  }
  if (typeof time === 'number' && time > 0) trunkFrames.set(scene, { time, trunks })
  return trunks
}

export function readTrunks(
  scene: ParentNode,
  relativeTo?: SVGGElement,
  reserveCamera = true,
): CollisionActor[] {
  const base = readBaseTrunks(scene)
  const clocks = readClockObstacles(scene, relativeTo)
  if (!reserveCamera) return [...clocks, ...base]
  const relativeDepth = elementDepth(relativeTo)
  return [
    ...clocks,
    ...base.map(trunk => {
      const clearance = trunk.element
        ? Math.abs(elementDepth(trunk.element) - relativeDepth) * parallaxMovement.scroll
        : 0
      return clearance ? { ...trunk, radius: trunk.radius + clearance } : trunk
    }),
  ]
}

export function circleClearance(point: Point, radius: number, other: CollisionActor) {
  if (other.bounds)
    return (
      Math.max(
        other.bounds.left - point.x,
        point.x - other.bounds.right,
        other.bounds.top - point.y,
        point.y - other.bounds.bottom,
      ) -
      radius -
      2
    )
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
  const left = Math.min(from.x, to.x)
  const right = Math.max(from.x, to.x)
  const top = Math.min(from.y, to.y)
  const bottom = Math.max(from.y, to.y)
  for (const other of obstacles) {
    if (other.bounds) {
      const initial = circleClearance(from, radius, other)
      if (initial < -0.01) {
        const fraction = Math.min(1, 1 / (Math.hypot(dx, dy) || 1))
        const probe = { x: from.x + dx * fraction, y: from.y + dy * fraction }
        if (
          !squared ||
          circleClearance(probe, radius, other) < initial - 0.00001 ||
          circleClearance(to, radius, other) < initial - 0.00001
        )
          return false
        continue
      }
      let enter = 0
      let exit = 1
      const padding = radius + 2 - 0.01
      for (const [position, delta, low, high] of [
        [from.x, dx, other.bounds.left - padding, other.bounds.right + padding],
        [from.y, dy, other.bounds.top - padding, other.bounds.bottom + padding],
      ]) {
        if (!delta) {
          if (position < low || position > high) exit = -1
          continue
        }
        const first = (low - position) / delta
        const last = (high - position) / delta
        enter = Math.max(enter, Math.min(first, last))
        exit = Math.min(exit, Math.max(first, last))
      }
      if (enter <= exit) return false
      continue
    }
    const limit = radius + other.radius + 2
    if (
      other.x < left - limit ||
      other.x > right + limit ||
      other.y < top - limit ||
      other.y > bottom + limit
    )
      continue
    const x = from.x - other.x
    const y = from.y - other.y
    const clearance = (limit - 0.01) ** 2
    const initialSquared = x * x + y * y
    if (initialSquared < clearance) {
      // A resize can tighten spacing. Escape must increase separation from its first step.
      if (x * dx + y * dy < -0.00001) return false
      if (Math.hypot(to.x - other.x, to.y - other.y) <= Math.sqrt(initialSquared) + 0.00001)
        return false
      continue
    }
    const fraction = squared ? Math.max(0, Math.min(1, -(x * dx + y * dy) / squared)) : 0
    const nearX = x + dx * fraction
    const nearY = y + dy * fraction
    if (nearX * nearX + nearY * nearY < clearance) return false
  }
  return true
}

export function actorSpacing(): ActorSpacing {
  return {
    offset: { x: 0, y: 0 },
    lane: { x: 0, y: 0 },
    blocker: null,
    side: 1,
    status: 'clear',
    yieldingTo: null,
    yieldingThrough: null,
    yieldEpisode: 0,
    waited: null,
    placement: null,
  }
}

export function spacedPosition(base: Point, spacing: ActorSpacing): Point {
  return { x: base.x + spacing.offset.x, y: base.y + spacing.offset.y }
}

export function placeOnReflow(
  element: SVGGElement,
  id: string,
  base: Point,
  radiusAt: (point: Point) => number,
  spacing: ActorSpacing,
  obstacles: CollisionActor[],
  bounds: SpacingBounds,
  grounded: (point: Point, radius: number) => boolean = () => true,
) {
  const others = obstacles.filter(other => other.id !== id)
  const free = (point: Point) => {
    const radius = radiusAt(point)
    return (
      point.x >= bounds.left &&
      point.x <= bounds.right &&
      point.y >= bounds.top &&
      point.y <= bounds.bottom &&
      grounded(point, radius) &&
      others.every(other => circleClearance(point, radius, other) >= 0)
    )
  }
  const desired = spacedPosition(base, spacing)
  let placed: Point | undefined = free(desired) ? desired : undefined
  const center = {
    x: Math.max(bounds.left, Math.min(bounds.right, desired.x)),
    y: Math.max(bounds.top, Math.min(bounds.bottom, desired.y)),
  }
  if (!placed && free(center)) placed = center
  for (let distance = 4; distance <= 96 && !placed; distance += 4)
    for (let index = 0; index < 16 && !placed; index++) {
      const angle = Math.PI / 2 + (index * Math.PI * 2) / 16
      const candidate = {
        x: center.x + Math.cos(angle) * distance,
        y: center.y + Math.sin(angle) * distance,
      }
      if (free(candidate)) placed = candidate
    }
  for (let row = 0; row < 25 && !placed; row++)
    for (let column = 0; column < 25 && !placed; column++) {
      const candidate = {
        x: bounds.left + ((column + 0.5) * (bounds.right - bounds.left)) / 25,
        y: bounds.top + ((row + 0.5) * (bounds.bottom - bounds.top)) / 25,
      }
      if (free(candidate)) placed = candidate
    }
  element.dataset.placement = placed ? 'ready' : 'blocked'
  spacing.placement = placed ? null : { bounds, grounded }
  if (!placed) return
  if (placed !== desired) {
    spacing.offset = { x: placed.x - base.x, y: placed.y - base.y }
    spacing.lane = { ...spacing.offset }
    spacing.blocker = null
    spacing.status = Math.hypot(spacing.offset.x, spacing.offset.y) > 0.1 ? 'passing' : 'clear'
  }
  const reservation = obstacles.find(other => other.id === id)
  const circle = {
    ...placed,
    id,
    kind: 'traveler',
    radius: radiusAt(placed),
  } satisfies CollisionActor
  if (reservation) Object.assign(reservation, circle)
  else obstacles.push(circle)
}

function towards(from: Point, to: Point, distance: number): Point {
  const length = Math.hypot(to.x - from.x, to.y - from.y)
  const fraction = length ? Math.min(1, distance / length) : 1
  return { x: from.x + (to.x - from.x) * fraction, y: from.y + (to.y - from.y) * fraction }
}

export function animalInWay(
  from: Point,
  next: Point,
  radius: number,
  obstacles: readonly CollisionActor[],
  waitingFor?: string | null,
) {
  const distance = Math.hypot(next.x - from.x, next.y - from.y)
  return obstacles.find(other => {
    if (other.kind !== 'animal' || circleClearance(from, radius, other) < -1) return false
    if (
      other.id === waitingFor &&
      distance &&
      (other.x - from.x) * (next.x - from.x) + (other.y - from.y) * (next.y - from.y) >= 0 &&
      circleClearance(from, radius, other) < radius * 1.5 + 14
    )
      return true
    const buffer = other.id === waitingFor ? 8 : 0
    const ahead = radius * 1.5 + 6 + buffer
    const probe = distance
      ? {
          x: next.x + ((next.x - from.x) * ahead) / distance,
          y: next.y + ((next.y - from.y) * ahead) / distance,
        }
      : next
    return !canMove(from, probe, radius + buffer / 2, [other])
  })
}

export function queuedAnimal(
  id: string,
  blocker: CollisionActor,
  actors: readonly CollisionActor[],
) {
  const seen = new Set([id])
  let lead: CollisionActor | undefined = blocker
  while (lead) {
    if (seen.has(lead.id)) return null
    seen.add(lead.id)
    const reason = lead.element?.dataset.yieldingTo
    if (!reason) return null
    const through: string | undefined = lead.element?.dataset.yieldingThrough
    if (!through) {
      const animal = actors.find(actor => actor.kind === 'animal' && actor.id === reason)
      return animal && circleClearance(lead, lead.radius, animal) < lead.radius * 1.5 + 14
        ? animal.id
        : null
    }
    lead = actors.find(actor => actor.id === through && actor.kind === 'traveler')
  }
  return null
}

// A walker grants an animal the way for this long. An animal that rests across the path would
// otherwise hold the walker for its whole nap, so after this it is passed like a trunk.
export const yieldPatience = 5000
const clock = () => (typeof performance === 'undefined' ? Date.now() : performance.now())

export function spacedStep(
  id: string,
  base: Point,
  next: Point,
  radius: number,
  budget: number,
  spacing: ActorSpacing,
  obstacles: readonly CollisionActor[],
  bounds?: SpacingBounds,
  yieldToTravelers = true,
  destination?: Point,
  planned = false,
) {
  if (budget <= 0) return false
  const waitingFor = spacing.yieldingTo
  spacing.yieldingTo = null
  spacing.yieldingThrough = null
  const others = obstacles.filter(other => other.id !== id)
  const from = spacedPosition(base, spacing)
  const distance = Math.hypot(next.x - base.x, next.y - base.y)
  const intended = distance ? spacedPosition(next, spacing) : towards(from, next, budget)
  const outside = (point: Point) =>
    bounds
      ? Math.hypot(
          point.x - Math.max(bounds.left, Math.min(bounds.right, point.x)),
          point.y - Math.max(bounds.top, Math.min(bounds.bottom, point.y)),
        )
      : 0
  const advance = (status: SpacingStatus) => {
    spacing.status = status
    return true
  }
  const clears = (point: Point) => {
    const remaining = outside(point)
    return (
      (remaining === 0 || remaining < outside(from) - 0.00001) &&
      Math.hypot(point.x - from.x, point.y - from.y) <= budget + 0.00001 &&
      canMove(from, point, radius, others)
    )
  }
  const now = clock()
  let animal = animalInWay(from, intended, radius, others, waitingFor)
  if (!animal) spacing.waited = null
  else if (spacing.waited?.id === animal.id && now - spacing.waited.since > yieldPatience)
    animal = undefined
  if (animal) {
    if (waitingFor !== animal.id) spacing.yieldEpisode++
    if (spacing.waited?.id !== animal.id) spacing.waited = { id: animal.id, since: now }
    spacing.blocker = animal.id
    spacing.yieldingTo = animal.id
    spacing.status = 'yielding'
    return false
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
  const headingDelta = distance
    ? { x: next.x - base.x, y: next.y - base.y }
    : { x: intended.x - from.x, y: intended.y - from.y }
  const headingLength = Math.hypot(headingDelta.x, headingDelta.y)
  const heading = headingLength
    ? { x: headingDelta.x / headingLength, y: headingDelta.y / headingLength }
    : { x: 1, y: 0 }
  const reach = destination
    ? Math.min(radius * 1.5 + 6, Math.hypot(destination.x - next.x, destination.y - next.y))
    : radius * 1.5 + 6
  const probe = { x: intended.x + heading.x * reach, y: intended.y + heading.y * reach }
  // Planned corners already clear static bodies; anticipation follows moving actors only.
  const blocker = others
    .filter(
      other =>
        !canMove(
          from,
          planned && (other.kind === 'trunk' || (other.kind === 'traveler' && !walkingActor(other)))
            ? intended
            : probe,
          radius,
          [other],
        ),
    )
    .sort(
      (left, right) => circleClearance(from, radius, left) - circleClearance(from, radius, right),
    )[0]
  if (!blocker) {
    spacing.blocker = null
    const recovered = towards(spacing.offset, { x: 0, y: 0 }, Math.max(0, budget - distance))
    const point = { x: next.x + recovered.x, y: next.y + recovered.y }
    const recovery = Math.hypot(recovered.x - spacing.offset.x, recovered.y - spacing.offset.y)
    if (distance + recovery <= budget + 0.00001 && clears(point)) {
      spacing.offset = recovered
      return advance(Math.hypot(recovered.x, recovered.y) > 0.1 ? 'passing' : 'clear')
    }
    if (clears(intended)) {
      return advance('passing')
    }
    const returnToPath = towards(from, base, budget)
    if (
      Math.hypot(returnToPath.x - from.x, returnToPath.y - from.y) > 0.00001 &&
      clears(returnToPath)
    ) {
      spacing.offset = { x: returnToPath.x - base.x, y: returnToPath.y - base.y }
      spacing.status = 'passing'
      return false
    }
    spacing.status = 'yielding'
    return false
  }

  const intrusion = circleClearance(from, radius, blocker) < 0
  const walking = walkingActor(blocker)
  const queue = blocker.kind === 'traveler' ? queuedAnimal(id, blocker, others) : null
  if (queue && !intrusion) {
    if (clears(intended)) {
      return advance('passing')
    }
    spacing.yieldingTo = queue
    spacing.yieldingThrough = blocker.id
    spacing.status = 'yielding'
    return false
  }
  if (yieldToTravelers && walking && blocker.kind === 'traveler' && id < blocker.id && !intrusion) {
    spacing.status = 'yielding'
    return clears(intended) && advance('yielding')
  }
  if (spacing.blocker !== blocker.id) {
    spacing.blocker = blocker.id
    const normal = { x: -heading.y, y: heading.x }
    const sign = normal.y < -0.01 ? -1 : normal.y > 0.01 ? 1 : id.length % 2 ? -1 : 1
    spacing.side = sign
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
      return advance('passing')
    }
    const normal = { x: -heading.y, y: heading.x }
    const sign = spacing.side
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
  const oppositeLane = {
    x: 2 * spacing.offset.x - spacing.lane.x,
    y: 2 * spacing.offset.y - spacing.lane.y,
  }
  const oppositeOffset = towards(spacing.offset, oppositeLane, budget)
  const oppositeStep = { x: base.x + oppositeOffset.x, y: base.y + oppositeOffset.y }
  if (clears(oppositeStep)) {
    spacing.side = spacing.side === 1 ? -1 : 1
    spacing.lane = oppositeLane
    spacing.offset = oppositeOffset
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
    return advance('passing')
  }
  const clearance = (point: Point) =>
    Math.min(...others.map(other => circleClearance(point, radius, other)))
  let escape: Point | null = null
  let space = clearance(from)
  for (let index = 0; index < 16; index++) {
    const angle = (index * Math.PI) / 8
    const candidate = { x: from.x + Math.cos(angle) * budget, y: from.y + Math.sin(angle) * budget }
    const gap = clearance(candidate)
    if (gap > space + 0.00001 && clears(candidate)) {
      escape = candidate
      space = gap
    }
  }
  if (escape) {
    spacing.offset = { x: escape.x - base.x, y: escape.y - base.y }
    spacing.status = 'escaping'
    return false
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
  motionTime?: number,
  radius = 12,
) {
  const x = Number(point.x.toFixed(2))
  const y = Number(point.y.toFixed(2))
  if (spacing.placement) {
    const scene = element.ownerSVGElement
    const circle = { x, y }
    const { bounds, grounded } = spacing.placement
    if (
      scene &&
      x >= bounds.left &&
      x <= bounds.right &&
      y >= bounds.top &&
      y <= bounds.bottom &&
      grounded(circle, radius * scale) &&
      [...readActors(scene, false, element), ...readTrunks(scene, element)]
        .filter(other => other.id !== id)
        .every(other => circleClearance(circle, radius * scale, other) >= 0)
    ) {
      spacing.placement = null
      element.dataset.placement = 'ready'
    }
  }
  const placed = !spacing.placement
  const transform = `translate(${x} ${y}) scale(${scale.toFixed(4)})`
  if (element.getAttribute('transform') !== transform) element.setAttribute('transform', transform)
  const visibility = placed ? '' : 'hidden'
  if (element.style.visibility !== visibility) element.style.visibility = visibility
  const metadata = {
    trailCollision: 'traveler',
    collisionId: id,
    collisionActive: String(active && placed),
    collisionRadius: (radius * scale).toFixed(4),
    groundX: String(x),
    groundY: String(y),
    scale: scale.toFixed(4),
    spacing: spacing.status,
    yieldingTo: spacing.yieldingTo ?? '',
    yieldingThrough: spacing.yieldingThrough ?? '',
    yieldEpisode: String(spacing.yieldEpisode),
    pathOffset: Math.hypot(spacing.offset.x, spacing.offset.y).toFixed(2),
    ...(motionTime === undefined ? {} : { motionTime: String(motionTime) }),
  }
  for (const [key, value] of Object.entries(metadata))
    if (element.dataset[key] !== value) element.dataset[key] = value
}
