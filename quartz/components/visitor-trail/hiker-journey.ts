import type { TrailId } from './layout'
import {
  actorSpacing,
  placeOnReflow,
  publishActor,
  readActors,
  readStationaryActors,
  readTrunks,
  spacedPosition,
  spacedStep,
} from './actor-spacing'
import { bootImpression } from './footprints'
import { trailGroundScale, type Landscape, type Point } from './landscape'
import { detourTrail, replanOccupiedWaypoints } from './obstacle-route'
import { trailRandom } from './random'

export type JourneyRoute = {
  id: TrailId
  length: number
  readyAfter: number
  point: (distance: number) => Point
}
type Phase = 'waiting' | 'arriving' | 'unpacking' | 'resting' | 'packing' | 'departing' | 'gone'
type WalkingPhase = 'arriving' | 'departing'
const phases: Phase[] = [
  'waiting',
  'arriving',
  'unpacking',
  'resting',
  'packing',
  'departing',
  'gone',
]

export function mountHikerJourney(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const hiker = scene.querySelector<SVGGElement>('.trail-backpacker')
  const facing = scene.querySelector<SVGGElement>('.trail-hiker-facing')
  const pack = scene.querySelector<SVGGElement>('.trail-hiker-pack')
  const pole = scene.querySelector<SVGGElement>('.trail-hiker-held-pole')
  const tracks = scene.querySelector<SVGGElement>('.trail-hiker-tracks')
  if (!view || !hiker || !facing || !pack || !pole || !tracks)
    throw new Error('The backpacker needs the trail scene')
  const reducedMotion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let incoming: JourneyRoute | undefined
  let outgoing: JourneyRoute | undefined
  let landscape: Landscape | undefined
  let seed: number | undefined
  let phase: Phase = 'waiting'
  let elapsed = 0
  let restingUntil = 0
  let previousTime: number | null = null
  let renderedAt = -Infinity
  let frame: number | null = null
  let timer: number | null = null
  let spacing = actorSpacing()
  let position: Point = { x: 0, y: 0 }
  let waypoints: Point[] = []
  let roadLength = 0
  let roadStart = 0
  let progress = 0
  let roadReady = false
  let retryRoadAt = 0
  let stride = 0
  let stepDistance = 0
  let enabled = true

  const moving = () => phase === 'arriving' || phase === 'departing'
  const legLength = (leg: WalkingPhase) =>
    leg === 'arriving' ? (incoming?.length ?? 0) : (outgoing?.length ?? 0)
  const duration = () => {
    if (phase === 'waiting') return (incoming?.readyAfter ?? 0) + 250
    if (phase === 'arriving') return Math.max(20000, Math.min(50000, legLength(phase) * 28))
    if (phase === 'departing') return Math.max(22000, Math.min(52000, legLength(phase) * 28))
    if (phase === 'resting') return 45 * 60 * 1000
    return 1400
  }
  const onLeg = (leg: WalkingPhase, fraction: number): Point => {
    const route = leg === 'arriving' ? incoming : outgoing
    if (!route) return position
    const distance = Math.max(0, Math.min(1, fraction)) * route.length
    return route.point(leg === 'arriving' ? distance : route.length - distance)
  }
  const bounds = () => ({
    left: 12,
    right: (landscape?.width ?? 24) - 12,
    top: landscape ? Math.max(landscape.treeline, landscape.alpine.base - 18) : 0,
    bottom: (landscape?.height ?? 24) - 12,
  })
  const remainingRoad = () => {
    let remaining = 0
    let from = position
    for (const point of waypoints) {
      remaining += Math.hypot(point.x - from.x, point.y - from.y)
      from = point
    }
    return remaining
  }
  const setRoad = (points: Point[]) => {
    const first = points[0]
    waypoints =
      first && Math.hypot(first.x - position.x, first.y - position.y) < 0.1
        ? points.slice(1)
        : points
    roadStart = progress
    roadLength = remainingRoad()
    roadReady = true
  }
  const planLeg = () => {
    if (!moving() || !landscape) return
    const leg: WalkingPhase = phase === 'arriving' ? 'arriving' : 'departing'
    const steps = Math.max(1, Math.ceil(((1 - progress) * legLength(leg)) / 16))
    const road = [
      position,
      ...Array.from({ length: steps + 1 }, (_, index) =>
        onLeg(leg, progress + ((1 - progress) * index) / steps),
      ),
    ]
    const obstacles = [
      ...readTrunks(scene),
      ...readStationaryActors(scene, 'hiker-camp'),
      ...readActors(scene).filter(actor => actor.kind === 'pet'),
    ]
    const planned = detourTrail(road, 12, obstacles, bounds())
    if (planned) setRoad(planned.points)
    else {
      roadReady = false
      retryRoadAt = view.performance.now() + 500
    }
  }
  const placeHiker = () => {
    if (!landscape || !incoming) return
    const terrain = landscape
    const base = onLeg(
      phase === 'departing' ? 'departing' : 'arriving',
      reducedMotion.matches ? 1 : moving() ? progress : phase === 'waiting' ? 0 : 1,
    )
    placeOnReflow(
      hiker,
      'hiker-camp',
      base,
      point => 12 * trailGroundScale(point.y, terrain),
      spacing,
      [...readActors(scene, true), ...readTrunks(scene)],
      bounds(),
    )
    position = spacedPosition(base, spacing)
    spacing.lane.x -= spacing.offset.x
    spacing.lane.y -= spacing.offset.y
    spacing.offset = { x: 0, y: 0 }
  }
  const leaveTracks = (before: Point) => {
    const dx = position.x - before.x
    const dy = position.y - before.y
    stepDistance += Math.hypot(dx, dy)
    if (stepDistance < 10) return
    stepDistance %= 10
    stride++
    const side = stride % 2 ? -1 : 1
    const scale = landscape ? trailGroundScale(position.y, landscape) : 1
    const angle = (Math.atan2(dy, dx) * 180) / Math.PI
    const mark = bootImpression(document)
    mark.setAttribute(
      'transform',
      `translate(${position.x.toFixed(2)} ${position.y.toFixed(2)}) rotate(${angle.toFixed(2)}) scale(${scale.toFixed(4)}) translate(0 ${side * 2}) scale(1 ${side})`,
    )
    tracks.append(mark)
    if (tracks.childElementCount > 128) tracks.firstElementChild?.remove()
  }
  const render = () => {
    if (!incoming) return
    hiker.dataset.enabled = String(enabled)
    if (!enabled) {
      hiker.style.visibility = 'hidden'
      hiker.dataset.collisionActive = 'false'
      return
    }
    hiker.dataset.phase = phase
    hiker.dataset.motion = reducedMotion.matches
      ? 'still'
      : host.hidden || document.hidden
        ? 'paused'
        : spacing.yieldingTo
          ? 'yielding'
          : 'walking'
    hiker.dataset.progress = (moving() ? progress : elapsed / duration()).toFixed(6)
    hiker.dataset.walkingDuration = moving() ? String(duration()) : ''
    hiker.dataset.restUntil = phase === 'resting' ? String(restingUntil) : ''
    hiker.dataset.waypointsRemaining = String(waypoints.length)
    const target = waypoints[0] ?? position
    hiker.dataset.targetX = target.x.toFixed(2)
    hiker.dataset.targetY = target.y.toFixed(2)
    const scale = landscape ? trailGroundScale(position.y, landscape) : 1
    const leg: WalkingPhase | null = moving()
      ? phase === 'arriving'
        ? 'arriving'
        : 'departing'
      : null
    hiker.dataset.moveSpeed = leg ? ((legLength(leg) * 1000 * scale) / duration()).toFixed(4) : '0'
    publishActor(
      hiker,
      'hiker-camp',
      position,
      scale,
      spacing,
      phase !== 'waiting' && phase !== 'gone',
      previousTime ?? view.performance.now(),
    )
    if (leg) {
      if (Math.abs(target.x - position.x) > 0.1)
        facing.setAttribute('transform', `scale(${target.x < position.x ? -1 : 1} 1)`)
      hiker.dataset.route = leg === 'arriving' ? incoming.id : outgoing?.id
      hiker.dataset.stride = String(stride % 2)
      pole.setAttribute('transform', `rotate(${stride % 2 ? -12 : 8} 10 -12)`)
    } else facing.setAttribute('transform', 'scale(1 1)')
    const fraction = Math.max(0, Math.min(1, elapsed / duration()))
    const grounded = phase === 'resting' || phase === 'unpacking' || phase === 'packing'
    const lift =
      phase === 'unpacking' ? 1 - fraction : phase === 'packing' ? fraction : grounded ? 0 : 1
    pack.setAttribute(
      'transform',
      `translate(${Math.round(-18 + lift * 6)} ${Math.round(-14 - lift * 8)})`,
    )
    hiker.dataset.pack = grounded ? (phase === 'resting' ? 'ground' : 'lifting') : 'shoulder'
  }
  const pause = () => {
    if (frame !== null) view.cancelAnimationFrame(frame)
    if (timer !== null) view.clearTimeout(timer)
    frame = timer = previousTime = null
    hiker.dataset.motion = reducedMotion.matches ? 'still' : 'paused'
  }
  const schedule = () => {
    if (
      signal.aborted ||
      host.hidden ||
      document.hidden ||
      reducedMotion.matches ||
      !enabled ||
      !incoming ||
      phase === 'gone'
    )
      return
    if (frame !== null || timer !== null) return
    previousTime ??= view.performance.now()
    if (moving() || phase === 'unpacking' || phase === 'packing')
      frame = view.requestAnimationFrame(tick)
    else
      timer = view.setTimeout(
        () => tick(view.performance.now()),
        Math.max(1, phase === 'resting' ? restingUntil - Date.now() : duration() - elapsed),
      )
  }
  const advancePhase = () => {
    elapsed = 0
    progress = 0
    waypoints = []
    phase = phases[Math.min(phases.length - 1, phases.indexOf(phase) + 1)]
    if (phase === 'resting') restingUntil = Date.now() + duration()
    if (moving()) {
      if (phase === 'arriving') placeHiker()
      stride = 0
      stepDistance = 0
      planLeg()
    }
    renderedAt = -Infinity
  }
  const walk = (delta: number, time: number) => {
    if (!landscape) return
    if (!roadReady) {
      if (time >= retryRoadAt) planLeg()
      return
    }
    const scale = trailGroundScale(position.y, landscape)
    const radius = 12 * scale
    const obstacles = [...readActors(scene), ...readTrunks(scene)]
    const replacement = replanOccupiedWaypoints(
      'hiker-camp',
      position,
      waypoints,
      radius,
      obstacles,
      bounds(),
      spacing.yieldingTo,
    )
    if (replacement?.length) setRoad(replacement)
    const target = waypoints[0]
    if (!target) return advancePhase()
    const before = position
    const distance = Math.hypot(target.x - position.x, target.y - position.y)
    const leg: WalkingPhase = phase === 'arriving' ? 'arriving' : 'departing'
    const budget = (legLength(leg) * Math.min(200, delta) * scale) / duration()
    const fraction = Math.min(1, budget / Math.max(1, distance))
    const next = {
      x: position.x + (target.x - position.x) * fraction,
      y: position.y + (target.y - position.y) * fraction,
    }
    if (
      spacedStep(
        'hiker-camp',
        position,
        next,
        radius,
        budget,
        spacing,
        obstacles,
        bounds(),
        true,
        target,
        true,
      )
    )
      position = next
    position = spacedPosition(position, spacing)
    spacing.lane.x -= spacing.offset.x
    spacing.lane.y -= spacing.offset.y
    spacing.offset = { x: 0, y: 0 }
    leaveTracks(before)
    if (Math.hypot(target.x - position.x, target.y - position.y) < 2) waypoints.shift()
    progress =
      roadStart +
      (1 - roadStart) * Math.max(0, Math.min(1, 1 - remainingRoad() / (roadLength || 1)))
    if (!waypoints.length) advancePhase()
  }
  const tick = (time: number) => {
    frame = timer = null
    if (host.hidden || document.hidden || signal.aborted) return pause()
    const delta = previousTime === null ? 0 : Math.max(0, time - previousTime)
    if (moving() && time - renderedAt < 1000 / 30) return schedule()
    if (moving()) walk(delta, time)
    else if (phase === 'resting') elapsed = duration() - Math.max(0, restingUntil - Date.now())
    else elapsed += delta
    previousTime = time
    if (!moving() && elapsed >= duration()) advancePhase()
    if (time - renderedAt >= 1000 / 30 || !moving()) {
      render()
      renderedAt = time
    }
    schedule()
  }
  const still = () => {
    pause()
    phase = 'resting'
    elapsed = 0
    restingUntil = Date.now() + duration()
    waypoints = []
    tracks.replaceChildren()
    placeHiker()
    render()
  }
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : schedule()), {
    signal,
  })
  reducedMotion.addEventListener(
    'change',
    () => {
      if (reducedMotion.matches) still()
      else {
        phase = 'waiting'
        elapsed = progress = 0
        placeHiker()
        render()
        schedule()
      }
    },
    { signal },
  )
  signal.addEventListener('abort', pause, { once: true })

  return {
    pause,
    update(
      routes: JourneyRoute[],
      nextLandscape: Landscape,
      nextSeed: number,
      nextEnabled: boolean,
    ) {
      pause()
      const resumed = !enabled && nextEnabled
      enabled = nextEnabled
      landscape = nextLandscape
      incoming = routes.find(route => route.id === 'main')
      const departures = routes.filter(
        route => route.id === 'pines' || route.id === 'ridge' || route.id === 'meadow',
      )
      outgoing = departures[Math.floor(trailRandom(nextSeed ^ 0x78ae12d3)() * departures.length)]
      if (!incoming || !outgoing) return pause()
      if (nextSeed !== seed) {
        seed = nextSeed
        phase = 'waiting'
        elapsed = progress = stride = stepDistance = 0
        renderedAt = -Infinity
        spacing = actorSpacing()
        tracks.replaceChildren()
      } else if (resumed || !enabled) tracks.replaceChildren()
      placeHiker()
      if (moving()) planLeg()
      if (reducedMotion.matches) still()
      else {
        render()
        schedule()
      }
    },
  }
}
