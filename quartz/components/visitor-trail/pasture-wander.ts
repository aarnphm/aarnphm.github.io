import type { WanderingAnimalKind } from './animals'
import {
  canMove,
  circleClearance,
  readActors,
  readTrunks,
  type CollisionActor,
} from './actor-spacing'
import { pondDistance, trailGroundScale, type Landscape, type Point } from './landscape'
import { trailRandom } from './random'

type Phase = 'waiting' | 'wandering' | 'resting' | 'departing' | 'away' | 'returning'
type WanderPath = { points: Point[]; distances: number[]; length: number }
type GooseFlight = { from: Point; to: Point; elapsed: number; duration: number; delay: number }
type Wanderer = {
  id: string
  kind: WanderingAnimalKind
  element: SVGGElement
  facing: SVGGElement
  home: Point
  position: Point
  path: WanderPath | null
  progress: number
  travelled: number
  direction: number
  phase: Phase
  remaining: number
  speed: number
  radius: number
  collisionRadius: number
  blockedFor: number
  yielding: boolean
  clearing: boolean
  flight: GooseFlight | null
  random: () => number
}

const habits = {
  cow: {
    speed: 2.4,
    variation: 1.2,
    radius: 42,
    rest: 5000,
    pause: 5000,
    clearance: 16,
    stride: 6,
    footprint: 9,
  },
  dog: {
    speed: 5,
    variation: 2,
    radius: 68,
    rest: 2000,
    pause: 3000,
    clearance: 12,
    stride: 5,
    footprint: 8,
  },
  cat: {
    speed: 3,
    variation: 1.8,
    radius: 44,
    rest: 3000,
    pause: 5000,
    clearance: 10,
    stride: 4,
    footprint: 5,
  },
  goose: {
    speed: 2.1,
    variation: 1.1,
    radius: 46,
    rest: 1500,
    pause: 2500,
    clearance: 0,
    stride: 6,
    footprint: 7,
  },
  bear: {
    speed: 3,
    variation: 1,
    radius: 58,
    rest: 4000,
    pause: 4000,
    clearance: 14,
    stride: 5,
    footprint: 9,
  },
  turkey: {
    speed: 6.5,
    variation: 2.5,
    radius: 86,
    rest: 1300,
    pause: 2200,
    clearance: 15,
    stride: 5,
    footprint: 10,
  },
  rabbit: {
    speed: 8,
    variation: 3,
    radius: 72,
    rest: 1600,
    pause: 2400,
    clearance: 12,
    stride: 5,
    footprint: 8,
  },
}

function animalKind(value: string | undefined): WanderingAnimalKind | null {
  return value === 'cow' ||
    value === 'dog' ||
    value === 'cat' ||
    value === 'goose' ||
    value === 'bear' ||
    value === 'turkey' ||
    value === 'rabbit'
    ? value
    : null
}

function pathFrom(points: Point[]): WanderPath {
  let length = 0
  const distances = points.map((point, index) => {
    if (index) length += Math.hypot(point.x - points[index - 1].x, point.y - points[index - 1].y)
    return length
  })
  return { points, distances, length }
}

function positionAlong(path: WanderPath, distance: number): Point {
  const index = path.distances.findIndex(value => value >= distance)
  if (index <= 0) return path.points[index < 0 ? path.points.length - 1 : 0]
  const from = path.points[index - 1]
  const to = path.points[index]
  const fraction =
    (distance - path.distances[index - 1]) /
    (path.distances[index] - path.distances[index - 1] || 1)
  return { x: from.x + (to.x - from.x) * fraction, y: from.y + (to.y - from.y) * fraction }
}

export function mountPastureWanderers(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) throw new Error('Animal wandering needs the trail document')
  const reducedMotion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let landscape: Landscape | undefined
  let seed: number | undefined
  let roots: Point[] = []
  let trunks: CollisionActor[] = []
  let wanderers: Wanderer[] = []
  let frame: number | null = null
  let timer: number | null = null
  let previousTime: number | null = null

  const flying = (animal: Wanderer) => animal.phase === 'departing' || animal.phase === 'returning'
  const migrate = (animal: Wanderer, outbound: boolean, index: number) => {
    if (!landscape) return
    animal.path = null
    animal.yielding = false
    animal.clearing = false
    animal.element.dataset.collisionActive = 'false'
    if (reducedMotion.matches) {
      animal.phase = outbound ? 'away' : 'resting'
      animal.position = animal.home
      animal.flight = null
      animal.remaining = habits.goose.pause
      return
    }
    const edge = { x: landscape.width + 96 + index * 18, y: landscape.alpine.top - 36 - index * 12 }
    animal.flight = {
      from: outbound || animal.phase !== 'away' ? animal.position : edge,
      to: outbound ? edge : animal.home,
      elapsed: 0,
      duration: 18_000 + index * 900,
      delay: index * 650,
    }
    animal.phase = outbound ? 'departing' : 'returning'
    animal.direction = outbound ? 1 : -1
  }
  const updateClimate = () => {
    const frozen = scene.dataset.waterFrozen === 'true'
    let index = 0
    for (const animal of wanderers) {
      if (animal.kind !== 'goose') continue
      if (frozen && animal.phase !== 'away' && animal.phase !== 'departing')
        migrate(animal, true, index)
      else if (!frozen && (animal.phase === 'away' || animal.phase === 'departing'))
        migrate(animal, false, index)
      index++
    }
  }

  const inHabitat = (point: Point, kind: WanderingAnimalKind) => {
    if (!landscape) return false
    if (kind === 'goose') return pondDistance(point, landscape.pond) < 0.86
    return (
      point.x >= 22 &&
      point.x <= landscape.width - 26 &&
      point.y >= (kind === 'bear' ? landscape.alpine.base - 26 : landscape.meadowTop + 64) &&
      point.y <= landscape.height - 28 &&
      pondDistance(point, landscape.pond) > 1.6 &&
      Math.hypot(point.x - landscape.camp.x, point.y - landscape.camp.y) > 90
    )
  }
  const safe = (point: Point, kind: WanderingAnimalKind) =>
    inHabitat(point, kind) &&
    (kind === 'goose' ||
      roots.every(root => Math.hypot(point.x - root.x, point.y - root.y) > habits[kind].clearance))
  const safeSite = (
    point: Point,
    kind: WanderingAnimalKind,
    available: (candidate: Point) => boolean = () => true,
  ) => {
    const fits = (candidate: Point) => {
      const drawn = { x: Math.round(candidate.x), y: Math.round(candidate.y) }
      return safe(drawn, kind) && available(drawn)
    }
    if (fits(point)) return point
    for (let index = 0; index < 96; index++) {
      const angle = index * 2.399963
      const radius = 5 + Math.floor(index / 12) * 7
      const candidate = {
        x: point.x + Math.cos(angle) * radius,
        y: point.y + Math.sin(angle) * radius,
      }
      if (fits(candidate)) return candidate
    }
    if (landscape) {
      if (kind === 'goose') {
        for (let index = 0; index < 128; index++) {
          const candidate = {
            x: landscape.pond.x + (((index * 0.618034) % 1) - 0.5) * landscape.pond.rx * 1.7,
            y: landscape.pond.y + (((index * 0.414214) % 1) - 0.5) * landscape.pond.ry * 1.7,
          }
          if (fits(candidate)) return candidate
        }
        return null
      }
      const height = Math.max(1, landscape.height - landscape.meadowTop - 92)
      for (let index = 0; index < 128; index++) {
        const candidate = {
          x: 22 + ((index * 0.618034) % 1) * Math.max(1, landscape.width - 48),
          y: landscape.meadowTop + 64 + ((index * 0.414214) % 1) * height,
        }
        if (fits(candidate)) return candidate
      }
    }
    return null
  }
  const readRoots = () => {
    trunks = readTrunks(scene)
    roots = trunks
  }
  const choosePath = (
    animal: Wanderer,
    obstacles: readonly CollisionActor[],
  ): WanderPath | null => {
    const waiting = obstacles.filter(
      other => other.element?.dataset.yieldingTo === `animal-${animal.id}`,
    )
    const away = waiting.length
      ? Math.atan2(
          animal.position.y - waiting.reduce((sum, other) => sum + other.y, 0) / waiting.length,
          animal.position.x - waiting.reduce((sum, other) => sum + other.x, 0) / waiting.length,
        )
      : 0
    let best: WanderPath | null = null
    let clearance = -Infinity
    for (let attempt = 0; attempt < 28; attempt++) {
      const angle = waiting.length && !attempt ? away : animal.random() * Math.PI * 2
      const reach = waiting.length ? 1.75 + Math.min(2, waiting.length - 1) * 0.5 : 1
      const radius = animal.radius * reach * (0.45 + animal.random() * 0.55)
      const target = {
        x: animal.home.x + Math.cos(angle) * radius,
        y: animal.home.y + Math.sin(angle) * radius * 0.6,
      }
      const { position: from } = animal
      const dx = target.x - from.x
      const dy = target.y - from.y
      const length = Math.hypot(dx, dy)
      if (
        length < 12 ||
        !safe(target, animal.kind) ||
        waiting.some(other => circleClearance(target, animal.collisionRadius, other) <= 8)
      )
        continue
      const bend = (animal.random() - 0.5) * Math.min(24, length * 0.65)
      const controlA = {
        x: from.x + dx / 3 - (dy / length) * bend,
        y: from.y + dy / 3 + (dx / length) * bend,
      }
      const controlB = {
        x: from.x + (dx * 2) / 3 - (dy / length) * bend,
        y: from.y + (dy * 2) / 3 + (dx / length) * bend,
      }
      const points = Array.from({ length: 33 }, (_, index) => {
        const t = index / 32
        const u = 1 - t
        return {
          x:
            u ** 3 * from.x +
            3 * u ** 2 * t * controlA.x +
            3 * u * t ** 2 * controlB.x +
            t ** 3 * target.x,
          y:
            u ** 3 * from.y +
            3 * u ** 2 * t * controlA.y +
            3 * u * t ** 2 * controlB.y +
            t ** 3 * target.y,
        }
      })
      if (
        points.every(
          (point, index) =>
            inHabitat(point, animal.kind) &&
            (!index || canMove(points[index - 1], point, animal.collisionRadius, obstacles)),
        )
      ) {
        const path = pathFrom(points)
        if (!waiting.length) return path
        const gap = Math.min(
          ...waiting.map(other => circleClearance(target, animal.collisionRadius, other)),
        )
        if (gap > clearance) {
          best = path
          clearance = gap
        }
      }
    }
    return best
  }
  const render = () => {
    for (const animal of wanderers) {
      const airborne = flying(animal)
      if (animal.kind === 'goose') {
        const visibility = animal.phase === 'away' ? 'hidden' : ''
        if (animal.element.style.visibility !== visibility)
          animal.element.style.visibility = visibility
        const active = String(!airborne && animal.phase !== 'away')
        if (animal.element.dataset.collisionActive !== active)
          animal.element.dataset.collisionActive = active
      }
      const x = String(Math.round(animal.position.x))
      const y = String(Math.round(animal.position.y))
      const transform = `translate(${x} ${y})`
      if (animal.element.getAttribute('transform') !== transform)
        animal.element.setAttribute('transform', transform)
      if (animal.element.dataset.groundX !== x) animal.element.dataset.groundX = x
      if (animal.element.dataset.groundY !== y) animal.element.dataset.groundY = y
      const phase =
        animal.phase === 'away'
          ? 'away'
          : reducedMotion.matches || animal.yielding
            ? 'resting'
            : animal.phase
      const motion = reducedMotion.matches
        ? 'still'
        : host.hidden || document.hidden || signal.aborted
          ? 'paused'
          : 'active'
      const stride = airborne
        ? String(Math.floor((animal.flight?.elapsed ?? 0) / 220) % 2)
        : phase === 'wandering'
          ? String(Math.floor(animal.travelled / habits[animal.kind].stride) % 2)
          : '0'
      if (animal.element.dataset.phase !== phase) animal.element.dataset.phase = phase
      if (animal.element.dataset.motion !== motion) animal.element.dataset.motion = motion
      if (animal.element.dataset.stride !== stride) animal.element.dataset.stride = stride
      const yielding = String(animal.yielding)
      if (animal.element.dataset.yielding !== yielding) animal.element.dataset.yielding = yielding
      const scale =
        animal.kind === 'bear' && landscape
          ? trailGroundScale(animal.position.y, landscape)
          : airborne && animal.flight
            ? 1 -
              0.45 *
                (animal.phase === 'departing'
                  ? Math.min(1, animal.flight.elapsed / animal.flight.duration)
                  : 1 - Math.min(1, animal.flight.elapsed / animal.flight.duration))
            : 1
      animal.element.dataset.scale = scale.toFixed(4)
      if (animal.kind === 'bear') {
        animal.collisionRadius =
          habits.bear.footprint * Number(animal.element.dataset.animalSize) * scale
        animal.element.dataset.collisionRadius = animal.collisionRadius.toFixed(4)
      }
      const facing = `scale(${animal.direction * scale} ${scale})`
      if (animal.facing.getAttribute('transform') !== facing)
        animal.facing.setAttribute('transform', facing)
    }
  }
  const pause = () => {
    if (frame !== null) view.cancelAnimationFrame(frame)
    if (timer !== null) view.clearTimeout(timer)
    frame = timer = previousTime = null
    for (const animal of wanderers)
      animal.element.dataset.motion = reducedMotion.matches ? 'still' : 'paused'
  }
  const step = (animal: Wanderer, elapsed: number, obstacles: readonly CollisionActor[]) => {
    if (animal.phase === 'away') return
    if (animal.flight) {
      const flight = animal.flight
      flight.elapsed += elapsed
      const t = Math.max(0, Math.min(1, (flight.elapsed - flight.delay) / flight.duration))
      animal.position = {
        x: flight.from.x + (flight.to.x - flight.from.x) * t,
        y: flight.from.y + (flight.to.y - flight.from.y) * t - Math.sin(t * Math.PI) * 40,
      }
      if (t === 1) {
        const returning = animal.phase === 'returning'
        animal.phase = returning ? 'resting' : 'away'
        animal.position = returning ? animal.home : flight.to
        animal.flight = null
        animal.remaining = habits.goose.pause
      }
      return
    }
    const clearing = obstacles.some(
      other => other.element?.dataset.yieldingTo === `animal-${animal.id}`,
    )
    if (clearing && !animal.clearing) {
      const path = choosePath(animal, obstacles)
      if (path) {
        animal.path = path
        animal.progress = 0
        animal.phase = 'wandering'
      }
    }
    animal.clearing = clearing
    if (animal.phase !== 'wandering') {
      animal.yielding = false
      if (
        animal.phase === 'resting' &&
        obstacles.some(other => {
          const gap = circleClearance(animal.position, animal.collisionRadius, other)
          return (
            gap < 0 ||
            ((other.kind === 'traveler' || other.kind === 'pet') &&
              (gap < 8 || other.element?.dataset.yieldingTo === `animal-${animal.id}`))
          )
        })
      )
        animal.remaining = Math.min(animal.remaining, 650)
      animal.remaining -= elapsed
      if (animal.remaining > 0) return
      animal.path = choosePath(animal, obstacles)
      animal.progress = 0
      animal.blockedFor = 0
      animal.phase = animal.path ? 'wandering' : 'resting'
      animal.remaining = 1000 + animal.random() * 1500
      return
    }
    if (!animal.path) return
    const distance = Math.min((animal.speed * elapsed) / 1000, animal.path.length - animal.progress)
    const position = positionAlong(animal.path, animal.progress + distance)
    const radius =
      animal.kind === 'bear' && landscape
        ? Math.max(
            animal.collisionRadius,
            habits.bear.footprint *
              Number(animal.element.dataset.animalSize) *
              trailGroundScale(position.y, landscape),
          )
        : animal.collisionRadius
    const drawnFrom = { x: Math.round(animal.position.x), y: Math.round(animal.position.y) }
    const drawnNext = { x: Math.round(position.x), y: Math.round(position.y) }
    if (
      !canMove(animal.position, position, radius, obstacles) ||
      !canMove(drawnFrom, drawnNext, radius, obstacles)
    ) {
      animal.yielding = true
      animal.blockedFor += elapsed
      if (animal.blockedFor >= 550) {
        animal.path = choosePath(animal, obstacles)
        animal.progress = 0
        animal.blockedFor = 0
        if (!animal.path) {
          animal.phase = 'resting'
          animal.remaining = 400 + animal.random() * 400
        }
      }
      return
    }
    animal.yielding = false
    animal.blockedFor = 0
    animal.progress += distance
    animal.travelled += distance
    if (Math.abs(position.x - animal.position.x) > 0.03)
      animal.direction = position.x < animal.position.x ? -1 : 1
    animal.position = position
    if (animal.progress >= animal.path.length) {
      animal.path = null
      animal.phase = 'resting'
      animal.remaining = habits[animal.kind].rest + animal.random() * habits[animal.kind].pause
    }
  }
  const schedule = () => {
    if (
      signal.aborted ||
      host.hidden ||
      document.hidden ||
      reducedMotion.matches ||
      !wanderers.length ||
      wanderers.every(animal => animal.phase === 'away') ||
      frame !== null ||
      timer !== null
    )
      return
    previousTime ??= view.performance.now()
    const moving = wanderers.some(animal => animal.phase === 'wandering' || flying(animal))
    const delay = moving
      ? 1000 / 15
      : Math.max(1, Math.min(500, ...wanderers.map(animal => animal.remaining)))
    timer = view.setTimeout(() => {
      timer = null
      frame = view.requestAnimationFrame(tick)
    }, delay)
  }
  const tick = (time: number) => {
    frame = null
    if (signal.aborted || host.hidden || document.hidden || reducedMotion.matches) return pause()
    const elapsed = previousTime === null ? 0 : Math.max(0, time - previousTime)
    let actors = [...readActors(scene), ...trunks]
    for (const animal of wanderers) {
      if (animal.phase === 'away') continue
      step(
        animal,
        animal.phase === 'wandering' || flying(animal) ? Math.min(200, elapsed) : elapsed,
        actors
          .filter(other => other.element !== animal.element)
          .map(other => {
            if (other.kind !== 'trunk') return other
            const radius = Math.max(
              other.radius,
              habits[animal.kind].clearance - animal.collisionRadius - 2,
            )
            return radius === other.radius ? other : { ...other, radius }
          }),
      )
      actors = actors.map(other =>
        other.element === animal.element
          ? { ...other, x: Math.round(animal.position.x), y: Math.round(animal.position.y) }
          : other,
      )
    }
    previousTime = time
    render()
    schedule()
  }
  const refreshForest = () => {
    readRoots()
    for (const animal of wanderers) {
      if (!animal.element.isConnected) continue
      if (animal.kind === 'goose' && (animal.phase === 'away' || flying(animal))) continue
      const clearTrunks = (point: Point) =>
        trunks.every(other => circleClearance(point, animal.collisionRadius, other) >= 0)
      animal.home = safeSite(animal.home, animal.kind, clearTrunks) ?? animal.home
      if (
        !safe(animal.position, animal.kind) ||
        animal.path?.points.some(point => !safe(point, animal.kind))
      ) {
        animal.path = null
        animal.phase = 'resting'
        animal.remaining = 500 + animal.random() * 1000
      }
    }
    render()
  }
  document.addEventListener(
    'visibilitychange',
    () => {
      if (document.hidden) pause()
      else {
        render()
        schedule()
      }
    },
    { signal },
  )
  reducedMotion.addEventListener(
    'change',
    () => {
      pause()
      if (reducedMotion.matches)
        for (const animal of wanderers)
          if (animal.kind === 'goose' && flying(animal))
            migrate(animal, scene.dataset.waterFrozen === 'true', 0)
      updateClimate()
      render()
      schedule()
    },
    { signal },
  )
  signal.addEventListener('abort', pause, { once: true })
  host.addEventListener(
    'trail:climate',
    () => {
      updateClimate()
      render()
      schedule()
    },
    { signal },
  )

  return {
    pause,
    refreshForest,
    update(nextLandscape: Landscape, nextSeed: number) {
      if (signal.aborted) return
      pause()
      const previous = new Map(wanderers.map(animal => [animal.id, animal]))
      const ratio = landscape
        ? Math.max(0.5, Math.min(1.5, nextLandscape.width / landscape.width))
        : 1
      const sameSeed = seed === nextSeed
      const sameLandscape = landscape === nextLandscape
      landscape = nextLandscape
      seed = nextSeed
      readRoots()
      const placed = [...readActors(scene).filter(other => other.kind === 'traveler'), ...trunks]
      const holdForSpace = (element: SVGGElement) => {
        element.style.visibility = 'hidden'
        element.dataset.collisionActive = 'false'
        element.dataset.motion = 'paused'
      }
      const elements = [...scene.querySelectorAll<SVGGElement>('[data-trail-wander]')]
      if (sameSeed && sameLandscape)
        elements.sort(
          (left, right) =>
            Number(previous.has(right.dataset.trailWander ?? '')) -
            Number(previous.has(left.dataset.trailWander ?? '')),
        )
      wanderers = elements.flatMap<Wanderer>((element, index) => {
        const id = element.dataset.trailWander
        const kind = animalKind(element.dataset.trailAnimal)
        const facing = element.querySelector<SVGGElement>('.trail-animal-facing')
        const anchor = {
          x: Number(element.dataset.animalHomeX),
          y: Number(element.dataset.animalHomeY),
        }
        if (!id || !kind || !facing || !Number.isFinite(anchor.x) || !Number.isFinite(anchor.y))
          return []
        const size = Number(element.dataset.animalSize)
        const radiusAt = (point: Point) =>
          habits[kind].footprint *
          (Number.isFinite(size) && size > 0 ? size : 2) *
          (kind === 'bear' ? trailGroundScale(point.y, nextLandscape) : 1)
        let collisionRadius = radiusAt(anchor)
        element.dataset.trailCollision = 'animal'
        element.dataset.collisionId = `animal-${id}`
        element.dataset.collisionActive = 'true'
        element.dataset.collisionRadius = String(collisionRadius)
        const old = sameSeed ? previous.get(id) : undefined
        const available = (candidate: Point) =>
          placed.every(other => circleClearance(candidate, radiusAt(candidate), other) >= 0)
        const home = old && sameLandscape ? old.home : safeSite(anchor, kind, available)
        if (!home) {
          holdForSpace(element)
          return []
        }
        collisionRadius = radiusAt(home)
        element.style.visibility = ''
        const remap = (point: Point) =>
          old
            ? {
                x: home.x + (point.x - old.home.x) * ratio,
                y: home.y + (point.y - old.home.y) * ratio,
              }
            : home
        if (old) {
          if (kind === 'goose' && (old.phase === 'away' || flying(old))) {
            element.dataset.collisionActive = 'false'
            placed.push({
              id: `animal-${id}`,
              kind: 'animal',
              element,
              x: Math.round(home.x),
              y: Math.round(home.y),
              radius: collisionRadius,
            })
            return [
              {
                ...old,
                element,
                facing,
                home,
                radius: habits[kind].radius * (nextLandscape.width < 700 ? 0.65 : 1),
                collisionRadius,
                position: remap(old.position),
                flight: old.flight
                  ? {
                      ...old.flight,
                      from: remap(old.flight.from),
                      to: old.phase === 'returning' ? home : remap(old.flight.to),
                    }
                  : null,
              },
            ]
          }
          const path = old.path ? pathFrom(old.path.points.map(remap)) : null
          const mappedPosition = remap(old.position)
          const position = safeSite(mappedPosition, kind, available)
          if (!position) {
            holdForSpace(element)
            return []
          }
          const relocated = position.x !== mappedPosition.x || position.y !== mappedPosition.y
          const allowed = !relocated && path?.points.every(point => safe(point, kind))
          const restarting = relocated || (old.phase === 'wandering' && !allowed)
          collisionRadius = radiusAt(position)
          placed.push({
            id: `animal-${id}`,
            kind: 'animal',
            element,
            x: Math.round(position.x),
            y: Math.round(position.y),
            radius: collisionRadius,
          })
          return [
            {
              ...old,
              element,
              facing,
              home,
              position,
              path: allowed ? path : null,
              progress:
                allowed && path && old.path ? (old.progress / old.path.length) * path.length : 0,
              phase: restarting ? 'resting' : old.phase,
              remaining: restarting ? 500 : old.remaining,
              blockedFor: restarting ? 0 : old.blockedFor,
              yielding: restarting ? false : old.yielding,
              radius: habits[kind].radius * (nextLandscape.width < 700 ? 0.65 : 1),
              collisionRadius,
            },
          ]
        }
        let hash = 0
        for (const character of id) hash = Math.imul(hash, 31) + character.charCodeAt(0)
        const random = trailRandom(nextSeed ^ hash ^ 0x58dfe793)
        const direction = random() < 0.5 ? -1 : 1
        const speed = habits[kind].speed + random() * habits[kind].variation
        placed.push({
          id: `animal-${id}`,
          kind: 'animal',
          element,
          x: Math.round(home.x),
          y: Math.round(home.y),
          radius: collisionRadius,
        })
        return [
          {
            id,
            kind,
            element,
            facing,
            home,
            position: home,
            path: null,
            progress: 0,
            travelled: 0,
            direction,
            phase: 'waiting',
            remaining: 1400 + index * 35 + random() * 500,
            speed,
            radius: habits[kind].radius * (nextLandscape.width < 700 ? 0.65 : 1),
            collisionRadius,
            blockedFor: 0,
            yielding: false,
            clearing: false,
            flight: null,
            random,
          } satisfies Wanderer,
        ]
      })
      wanderers.sort((a, b) => a.id.localeCompare(b.id))
      updateClimate()
      render()
      schedule()
    },
  }
}
