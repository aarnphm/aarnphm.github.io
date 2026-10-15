import type { Alpine } from './landscape'
import type { MountainRange } from './mountain-profile'
import { climberSprites } from './climber-art'
import { buildClimbingRoutes, climbingPoint, type ClimbingRoute } from './climbing-route'
import { pixels } from './pixels'
import { trailRandom } from './random'
import { alpineRouteCapacity } from './scene-settings'

type Phase = 'ascending' | 'resting' | 'descending'
type Climber = {
  element: SVGGElement
  figure: SVGGElement
  holds: SVGPathElement
  route: ClimbingRoute
  progress: number
  phase: Phase
  next: 'ascending' | 'descending'
  restAt: number
  remaining: number
  duration: number
  random: () => number
  painted: string
}

const tones = {
  h: 'trail-camper-hair',
  s: 'trail-camper-skin',
  c: 'trail-climber-coat',
  w: 'trail-climber-chalk',
  d: 'trail-camper-trousers',
  b: 'trail-climber-shoe',
}
const cell = 1.5
const origin = { x: -10.5, y: -30 }
// The resting pose dips a hand in the chalk bag; the puff rises off it.
const artwork =
  Object.entries(climberSprites)
    .map(
      ([pose, rows]) =>
        `<g data-climbing-pose="${pose}" transform="translate(${origin.x} ${origin.y})">${pixels(rows, tones, cell)}</g>`,
    )
    .join('') +
  '<path class="trail-climber-chalk-puff" d="M-2.5-17h1.5v1.5h-1.5ZM0-19.5h1.5v1.5H0Z"/>'
// Cells of the planted hand in plantLeft and plantRight. The route chalks a hold under each, so
// every reach lands on a mark that earlier ascents left on the rock.
const plantedHands = [
  { x: 1, y: 0, at: 0 },
  { x: 12, y: 0, at: 0.5 },
]

const climbingCycle = [
  'liftLeft',
  'plantLeft',
  'stepLeft',
  'pullLeft',
  'liftRight',
  'plantRight',
  'stepRight',
  'pullRight',
]

function routePitches(route: ClimbingRoute) {
  const rise = Math.abs(route.points[0].y - route.points[route.points.length - 1].y)
  // A pull raises the torso one cell while the planted hand drops one cell in the sprite.
  return Math.max(1, Math.round(rise / (3 * route.scale)))
}

function chalkedHolds(route: ClimbingRoute) {
  const pitches = routePitches(route)
  const size = Math.max(0.6, cell * route.scale)
  let d = ''
  for (let pitch = 0; pitch < pitches; pitch++) {
    for (const hand of plantedHands) {
      const point = climbingPoint(route, (pitch + hand.at) / pitches)
      const x = point.x + (origin.x + hand.x * cell) * route.scale
      const y = point.y + (origin.y + hand.y * cell) * route.scale
      d += `M${x.toFixed(2)} ${y.toFixed(2)}h${size.toFixed(2)}v${size.toFixed(2)}h${(-size).toFixed(2)}Z`
    }
  }
  return d
}

function plantedProgress(route: ClimbingRoute, progress: number) {
  const pitches = routePitches(route)
  const stride = progress * pitches
  const fraction = stride % 1
  const pull = (value: number) => value * value * (3 - 2 * value)
  const advance =
    fraction < 0.375
      ? 0
      : fraction < 0.5
        ? pull((fraction - 0.375) / 0.125) / 2
        : fraction < 0.875
          ? 0.5
          : 0.5 + pull((fraction - 0.875) / 0.125) / 2
  return {
    position: Math.min(1, (Math.floor(stride) + advance) / pitches),
    pose: climbingCycle[Math.floor(fraction * climbingCycle.length)],
  }
}

// The alpine journey owns the animation clock and pauses; face climbers share its ticks.
export function createAlpineClimbers(scene: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  let seed: number | undefined
  let climbers: Climber[] = []
  let geometry = ''
  let availableRoutes: ClimbingRoute[] = []
  const clear = () => {
    for (const climber of climbers) climber.element.remove()
    climbers = []
  }
  signal.addEventListener('abort', clear, { once: true })
  const moving = () => climbers.some(climber => climber.phase !== 'resting')
  return {
    moving,
    present: () => climbers.length > 0,
    wait: () =>
      Math.min(
        ...climbers.map(climber =>
          climber.phase === 'resting'
            ? climber.remaining
            : Math.max(
                250,
                Math.min(
                  1000,
                  (0.2 * climber.duration) /
                    Math.abs(climber.route.points[0].y - climber.route.points[3].y),
                ),
              ),
        ),
      ),
    pause(still: boolean) {
      for (const climber of climbers) {
        climber.element.dataset.motion = still ? 'still' : 'paused'
        if (still) climber.element.dataset.pose = 'resting'
        climber.painted = ''
      }
    },
    update(alpine: Alpine, nextSeed: number, count: number, ranges: readonly MountainRange[]) {
      const layer = scene.querySelector<SVGGElement>('.trail-alpine-climbers')
      if (!layer || signal.aborted) return
      if (seed !== nextSeed) {
        clear()
        seed = nextSeed
      }
      const signature = `${nextSeed}|${alpine.left}|${alpine.top}|${alpine.width}|${alpine.height}|${alpine.base}`
      if (signature !== geometry) {
        availableRoutes = buildClimbingRoutes(
          alpine,
          nextSeed,
          Math.max(alpineRouteCapacity, count),
          ranges,
        )
        geometry = signature
      }
      const routes = availableRoutes.slice(0, count)
      for (const climber of climbers.splice(routes.length)) climber.element.remove()
      for (const [id, route] of routes.entries()) {
        const existing = climbers[id]
        if (existing) {
          existing.route = route
          continue
        }
        const random = trailRandom(nextSeed ^ (0x5731a9e5 + id * 0x27d4eb2d))
        const element = document.createElementNS('http://www.w3.org/2000/svg', 'g')
        element.dataset.alpineClimber = String(id)
        element.classList.add('trail-alpine-climber')
        const holds = document.createElementNS('http://www.w3.org/2000/svg', 'path')
        holds.classList.add('trail-climber-holds')
        const figure = document.createElementNS('http://www.w3.org/2000/svg', 'g')
        figure.classList.add('trail-climber-figure')
        figure.innerHTML = artwork
        element.append(holds, figure)
        const progress = 0.16 + random() * 0.42
        climbers.push({
          element,
          figure,
          holds,
          route,
          progress,
          phase: 'ascending',
          next: 'ascending',
          restAt: Math.min(1, progress + 0.16 + random() * 0.12),
          remaining: 0,
          duration: 240000 + random() * 180000,
          random,
          painted: '',
        })
      }
      for (const climber of climbers) {
        const { route } = climber
        climber.painted = ''
        climber.holds.setAttribute('d', chalkedHolds(route))
        climber.element.dataset.scale = String(route.scale)
        climber.element.dataset.mountainRange = String(route.rangeIndex)
        climber.element.dataset.climbingRoute = JSON.stringify(route)
      }
      layer.replaceChildren(...climbers.map(climber => climber.element))
    },
    step(elapsed: number) {
      for (const climber of climbers) {
        if (climber.phase === 'resting') {
          climber.remaining -= elapsed
          if (climber.remaining <= 0) climber.phase = climber.next
          continue
        }
        const direction = climber.phase === 'ascending' ? 1 : -1
        climber.progress = Math.max(
          0,
          Math.min(1, climber.progress + (direction * Math.min(elapsed, 1000)) / climber.duration),
        )
        const reached = direction > 0 ? climber.progress >= climber.restAt : climber.progress <= 0
        if (!reached) continue
        climber.progress = direction > 0 ? climber.restAt : 0
        const summit = climber.progress === 1
        climber.next = summit ? 'descending' : 'ascending'
        climber.phase = 'resting'
        climber.remaining = summit
          ? 45000 + climber.random() * 30000
          : 8000 + climber.random() * 12000
        climber.restAt = Math.min(1, climber.progress + 0.16 + climber.random() * 0.12)
      }
    },
    render(motion: 'active' | 'paused' | 'still') {
      for (const climber of climbers) {
        const { route, progress } = climber
        const planted = plantedProgress(route, progress)
        const exact = climbingPoint(route, planted.position)
        const point = { x: Math.round(exact.x * 5) / 5, y: Math.round(exact.y * 5) / 5 }
        const pose = motion === 'still' || climber.phase === 'resting' ? 'resting' : planted.pose
        const signature = `${point.x}|${point.y}|${pose}|${climber.phase}|${motion}`
        if (signature === climber.painted) continue
        climber.painted = signature
        climber.figure.setAttribute(
          'transform',
          `translate(${point.x.toFixed(3)} ${point.y.toFixed(3)}) scale(${route.scale})`,
        )
        const data = climber.element.dataset
        if (data.phase !== climber.phase) data.phase = climber.phase
        if (data.motion !== motion) data.motion = motion
        if (data.pose !== pose) data.pose = pose
        data.progress = planted.position.toFixed(6)
        data.faceX = point.x.toFixed(3)
        data.faceY = point.y.toFixed(3)
      }
    },
  }
}
