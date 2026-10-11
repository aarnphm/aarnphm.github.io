import type { TrailAnimalKind } from './animals'
import { pondDistance, type Landscape, type Point } from './landscape'
import { trailRandom } from './random'

type Phase = 'waiting' | 'wandering' | 'resting'
type WanderPath = { points: Point[]; distances: number[]; length: number }
type Wanderer = {
  id: string
  kind: TrailAnimalKind
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
  },
  dog: { speed: 5, variation: 2, radius: 68, rest: 2000, pause: 3000, clearance: 12, stride: 5 },
  cat: { speed: 3, variation: 1.8, radius: 44, rest: 3000, pause: 5000, clearance: 10, stride: 4 },
  goose: {
    speed: 2.1,
    variation: 1.1,
    radius: 46,
    rest: 1500,
    pause: 2500,
    clearance: 0,
    stride: 6,
  },
}

function animalKind(value: string | undefined): TrailAnimalKind | null {
  return value === 'cow' || value === 'dog' || value === 'cat' || value === 'goose' ? value : null
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
  let wanderers: Wanderer[] = []
  let frame: number | null = null
  let timer: number | null = null
  let previousTime: number | null = null

  const safe = (point: Point, kind: TrailAnimalKind) => {
    if (!landscape) return false
    if (kind === 'goose') return pondDistance(point, landscape.pond) < 0.86
    return (
      point.x >= 22 &&
      point.x <= landscape.width - 26 &&
      point.y >= landscape.meadowTop + 64 &&
      point.y <= landscape.height - 28 &&
      pondDistance(point, landscape.pond) > 1.6 &&
      Math.hypot(point.x - landscape.camp.x, point.y - landscape.camp.y) > 90 &&
      roots.every(root => Math.hypot(point.x - root.x, point.y - root.y) > habits[kind].clearance)
    )
  }
  const safeSite = (point: Point, kind: TrailAnimalKind) => {
    if (safe(point, kind)) return point
    for (let index = 0; index < 96; index++) {
      const angle = index * 2.399963
      const radius = 5 + Math.floor(index / 12) * 7
      const candidate = {
        x: point.x + Math.cos(angle) * radius,
        y: point.y + Math.sin(angle) * radius,
      }
      if (safe(candidate, kind)) return candidate
    }
    if (landscape) {
      if (kind === 'goose') return { x: landscape.pond.x, y: landscape.pond.y }
      const height = Math.max(1, landscape.height - landscape.meadowTop - 92)
      for (let index = 0; index < 128; index++) {
        const candidate = {
          x: 22 + ((index * 0.618034) % 1) * Math.max(1, landscape.width - 48),
          y: landscape.meadowTop + 64 + ((index * 0.414214) % 1) * height,
        }
        if (safe(candidate, kind)) return candidate
      }
    }
    return point
  }
  const readRoots = () => {
    roots = [...scene.querySelectorAll<SVGGElement>('.trail-tree')].flatMap(tree => {
      const transforms = tree.transform.baseVal
      if (!transforms.numberOfItems) return []
      const matrix = transforms.getItem(0).matrix
      return [{ x: matrix.e, y: matrix.f }]
    })
  }
  const choosePath = (animal: Wanderer): WanderPath | null => {
    for (let attempt = 0; attempt < 28; attempt++) {
      const angle = animal.random() * Math.PI * 2
      const radius = animal.radius * (0.45 + animal.random() * 0.55)
      const target = {
        x: animal.home.x + Math.cos(angle) * radius,
        y: animal.home.y + Math.sin(angle) * radius * 0.6,
      }
      const { position: from } = animal
      const dx = target.x - from.x
      const dy = target.y - from.y
      const length = Math.hypot(dx, dy)
      if (length < 12 || !safe(target, animal.kind)) continue
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
      if (points.every(point => safe(point, animal.kind))) return pathFrom(points)
    }
    return null
  }
  const render = () => {
    for (const animal of wanderers) {
      const transform = `translate(${Math.round(animal.position.x)} ${Math.round(animal.position.y)})`
      if (animal.element.getAttribute('transform') !== transform)
        animal.element.setAttribute('transform', transform)
      const phase = reducedMotion.matches ? 'resting' : animal.phase
      const motion = reducedMotion.matches
        ? 'still'
        : host.hidden || document.hidden || signal.aborted
          ? 'paused'
          : 'active'
      const stride =
        phase === 'wandering'
          ? String(Math.floor(animal.travelled / habits[animal.kind].stride) % 2)
          : '0'
      if (animal.element.dataset.phase !== phase) animal.element.dataset.phase = phase
      if (animal.element.dataset.motion !== motion) animal.element.dataset.motion = motion
      if (animal.element.dataset.stride !== stride) animal.element.dataset.stride = stride
      const facing = `scale(${animal.direction} 1)`
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
  const step = (animal: Wanderer, elapsed: number) => {
    if (animal.phase !== 'wandering') {
      animal.remaining -= elapsed
      if (animal.remaining > 0) return
      animal.path = choosePath(animal)
      animal.progress = 0
      animal.phase = animal.path ? 'wandering' : 'resting'
      animal.remaining = 1000 + animal.random() * 1500
      return
    }
    if (!animal.path) return
    const distance = Math.min((animal.speed * elapsed) / 1000, animal.path.length - animal.progress)
    animal.progress += distance
    animal.travelled += distance
    const position = positionAlong(animal.path, animal.progress)
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
      frame !== null ||
      timer !== null
    )
      return
    previousTime ??= view.performance.now()
    const moving = wanderers.some(animal => animal.phase === 'wandering')
    const delay = moving
      ? 1000 / 15
      : Math.max(1, Math.min(...wanderers.map(animal => animal.remaining)))
    timer = view.setTimeout(() => {
      timer = null
      frame = view.requestAnimationFrame(tick)
    }, delay)
  }
  const tick = (time: number) => {
    frame = null
    if (signal.aborted || host.hidden || document.hidden || reducedMotion.matches) return pause()
    const elapsed = previousTime === null ? 0 : Math.max(0, time - previousTime)
    for (const animal of wanderers)
      step(animal, animal.phase === 'wandering' ? Math.min(200, elapsed) : elapsed)
    previousTime = time
    render()
    schedule()
  }
  const refreshForest = () => {
    readRoots()
    for (const animal of wanderers) {
      if (!animal.element.isConnected) continue
      animal.home = safeSite(animal.home, animal.kind)
      animal.position = safeSite(animal.position, animal.kind)
      if (animal.path?.points.some(point => !safe(point, animal.kind))) {
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
      render()
      schedule()
    },
    { signal },
  )
  signal.addEventListener('abort', pause, { once: true })

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
      landscape = nextLandscape
      seed = nextSeed
      readRoots()
      wanderers = [...scene.querySelectorAll<SVGGElement>('[data-trail-wander]')].flatMap<Wanderer>(
        (element, index) => {
          const id = element.dataset.trailWander
          const kind = animalKind(element.dataset.trailAnimal)
          const facing = element.querySelector<SVGGElement>('.trail-animal-facing')
          const anchor = {
            x: Number(element.dataset.animalHomeX),
            y: Number(element.dataset.animalHomeY),
          }
          if (!id || !kind || !facing || !Number.isFinite(anchor.x) || !Number.isFinite(anchor.y))
            return []
          const home = safeSite(anchor, kind)
          const old = sameSeed ? previous.get(id) : undefined
          const remap = (point: Point) =>
            old
              ? {
                  x: home.x + (point.x - old.home.x) * ratio,
                  y: home.y + (point.y - old.home.y) * ratio,
                }
              : home
          if (old) {
            const path = old.path ? pathFrom(old.path.points.map(remap)) : null
            const allowed = path?.points.every(point => safe(point, kind))
            return [
              {
                ...old,
                element,
                facing,
                home,
                position: safeSite(remap(old.position), kind),
                path: allowed ? path : null,
                progress:
                  allowed && path && old.path ? (old.progress / old.path.length) * path.length : 0,
                phase: old.phase === 'wandering' && !allowed ? 'resting' : old.phase,
                remaining: old.phase === 'wandering' && !allowed ? 500 : old.remaining,
                radius: habits[kind].radius * (nextLandscape.width < 700 ? 0.65 : 1),
              },
            ]
          }
          let hash = 0
          for (const character of id) hash = Math.imul(hash, 31) + character.charCodeAt(0)
          const random = trailRandom(nextSeed ^ hash ^ 0x58dfe793)
          const direction = random() < 0.5 ? -1 : 1
          const speed = habits[kind].speed + random() * habits[kind].variation
          return [
            {
              id,
              kind,
              element,
              facing,
              home,
              position: safeSite(home, kind),
              path: null,
              progress: 0,
              travelled: 0,
              direction,
              phase: 'waiting',
              remaining: 1400 + index * 35 + random() * 500,
              speed,
              radius: habits[kind].radius * (nextLandscape.width < 700 ? 0.65 : 1),
              random,
            } satisfies Wanderer,
          ]
        },
      )
      render()
      schedule()
    },
  }
}
