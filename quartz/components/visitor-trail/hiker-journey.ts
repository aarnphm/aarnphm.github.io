import type { TrailId } from './layout'
import {
  actorSpacing,
  publishActor,
  readActors,
  readTrunks,
  spacedPosition,
  spacedStep,
} from './actor-spacing'
import { bootImpression } from './footprints'
import { trailGroundScale, type Landscape } from './landscape'
import { trailRandom } from './random'

type Point = { x: number; y: number }
export type JourneyRoute = {
  id: TrailId
  length: number
  readyAfter: number
  point: (distance: number) => Point
}
type Phase = 'waiting' | 'arriving' | 'unpacking' | 'resting' | 'packing' | 'departing' | 'gone'
type WalkingPhase = 'arriving' | 'departing'
type Impression = {
  phase: WalkingPhase
  progress: number
  side: number
  offset: Point
  element: SVGPathElement
}
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
  let previousTime: number | null = null
  let renderedAt = -Infinity
  let frame: number | null = null
  let timer: number | null = null
  let nextStep = 10
  const impressions: Impression[] = []
  let spacing = actorSpacing()

  const duration = () => {
    if (phase === 'waiting') return (incoming?.readyAfter ?? 0) + 250
    if (phase === 'arriving') return Math.max(20000, Math.min(50000, (incoming?.length ?? 0) * 28))
    if (phase === 'departing') return Math.max(22000, Math.min(52000, (outgoing?.length ?? 0) * 28))
    if (phase === 'resting') return 9000
    return 1400
  }
  const moving = () => phase === 'arriving' || phase === 'departing'
  const legLength = (leg: WalkingPhase) =>
    leg === 'arriving' ? (incoming?.length ?? 0) : (outgoing?.length ?? 0)
  const onLeg = (leg: WalkingPhase, progress: number): Point => {
    const route = leg === 'arriving' ? incoming : outgoing
    if (!route) return { x: 0, y: 0 }
    const distance = Math.max(0, Math.min(1, progress)) * legLength(leg)
    const remaining = leg === 'arriving' ? route.length - distance : distance
    const point = route.point(leg === 'arriving' ? distance : route.length - distance)
    const approach = Math.max(0, 1 - remaining / 110)
    const blend = approach * approach * (3 - 2 * approach)
    return { x: point.x - 48 * blend, y: point.y + 30 * blend }
  }
  const impressionTransform = (stamp: Impression) => {
    const base = onLeg(stamp.phase, stamp.progress)
    const point = { x: base.x + stamp.offset.x, y: base.y + stamp.offset.y }
    const ahead = onLeg(stamp.phase, Math.min(1, stamp.progress + 1 / legLength(stamp.phase)))
    const scale = landscape ? trailGroundScale(point.y, landscape) : 1
    const angle = (Math.atan2(ahead.y - base.y, ahead.x - base.x) * 180) / Math.PI
    return `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)}) rotate(${angle.toFixed(2)}) scale(${scale.toFixed(4)}) translate(0 ${stamp.side * 2}) scale(1 ${stamp.side})`
  }
  const render = () => {
    if (!incoming) return
    hiker.dataset.phase = phase
    hiker.dataset.motion = reducedMotion.matches
      ? 'still'
      : host.hidden || document.hidden
        ? 'paused'
        : 'walking'
    const progress = Math.min(1, elapsed / duration())
    hiker.dataset.progress = progress.toFixed(3)
    hiker.dataset.walkingDuration = moving() ? String(duration()) : ''
    const leg: WalkingPhase | null = phase === 'arriving' || phase === 'departing' ? phase : null
    const base = leg ? onLeg(leg, progress) : onLeg('arriving', phase === 'waiting' ? 0 : 1)
    const position = spacedPosition(base, spacing)
    const scale = landscape ? trailGroundScale(position.y, landscape) : 1
    hiker.dataset.moveSpeed = leg ? ((legLength(leg) * 1000 * scale) / duration()).toFixed(4) : '0'
    publishActor(
      hiker,
      'hiker-camp',
      position,
      scale,
      spacing,
      phase !== 'waiting' && phase !== 'gone',
    )
    if (leg) {
      const ahead = onLeg(leg, Math.min(1, progress + 2 / legLength(leg)))
      if (Math.abs(ahead.x - position.x) > 0.1)
        facing.setAttribute('transform', `scale(${ahead.x < position.x ? -1 : 1} 1)`)
      hiker.dataset.route = leg === 'arriving' ? incoming.id : outgoing?.id
      const distance = progress * legLength(leg)
      const stride = Math.floor(distance / 10) % 2
      hiker.dataset.stride = String(stride)
      pole.setAttribute('transform', `rotate(${stride ? -12 : 8} 10 -12)`)
      while (nextStep < distance - 6) {
        const side = Math.floor(nextStep / 10) % 2 ? -1 : 1
        const mark = bootImpression(document)
        const stamp = {
          phase: leg,
          progress: nextStep / legLength(leg),
          side,
          offset: { ...spacing.offset },
          element: mark,
        }
        mark.setAttribute('transform', impressionTransform(stamp))
        tracks.append(mark)
        impressions.push(stamp)
        if (impressions.length > 128) impressions.shift()?.element.remove()
        nextStep += 10
      }
    } else facing.setAttribute('transform', 'scale(1 1)')
    const grounded = phase === 'resting' || phase === 'unpacking' || phase === 'packing'
    const lift =
      phase === 'unpacking' ? 1 - progress : phase === 'packing' ? progress : grounded ? 0 : 1
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
      !incoming ||
      phase === 'gone'
    )
      return
    if (frame !== null || timer !== null) return
    previousTime ??= view.performance.now()
    if (moving() || phase === 'unpacking' || phase === 'packing')
      frame = view.requestAnimationFrame(tick)
    else
      timer = view.setTimeout(() => tick(view.performance.now()), Math.max(1, duration() - elapsed))
  }
  const tick = (time: number) => {
    frame = timer = null
    if (host.hidden || document.hidden || signal.aborted) return pause()
    const delta = previousTime === null ? 0 : Math.max(0, time - previousTime)
    if (moving() && time - renderedAt < 1000 / 30) return schedule()
    if (moving() && landscape) {
      const leg: WalkingPhase = phase === 'arriving' ? 'arriving' : 'departing'
      const progress = Math.min(1, elapsed / duration())
      const base = onLeg(leg, progress)
      const scale = trailGroundScale(spacedPosition(base, spacing).y, landscape)
      const interval = Math.min(200, delta) * scale
      const budget = (legLength(leg) * interval) / duration()
      const factor = Math.hypot(spacing.offset.x, spacing.offset.y) > 0.1 ? 0.7 : 1
      const advance = budget * factor
      let increment = interval * factor
      let next = onLeg(leg, Math.min(1, (elapsed + increment) / duration()))
      for (let iteration = 0; iteration < 2; iteration++) {
        const distance = Math.hypot(next.x - base.x, next.y - base.y)
        if (distance <= advance || !distance) break
        increment *= (advance / distance) * 0.999
        next = onLeg(leg, Math.min(1, (elapsed + increment) / duration()))
      }
      const radius =
        12 *
        Math.max(
          scale,
          trailGroundScale(Math.max(base.y, next.y) + spacing.offset.y + budget, landscape),
        )
      if (
        spacedStep(
          'hiker-camp',
          base,
          next,
          radius,
          budget,
          spacing,
          [...readActors(scene), ...readTrunks(scene)],
          {
            left: radius,
            right: landscape.width - radius,
            top: Math.max(landscape.treeline, landscape.alpine.base - 18),
            bottom: landscape.height - radius,
          },
        )
      )
        elapsed += increment
    } else elapsed += delta
    previousTime = time
    if (elapsed >= duration()) {
      elapsed = 0
      phase = phases[Math.min(phases.length - 1, phases.indexOf(phase) + 1)]
      nextStep = 10
      renderedAt = -Infinity
    }
    // Thirty draws per second suit the pixel stride. Timed pauses need no frame loop.
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
    tracks.replaceChildren()
    impressions.length = 0
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
        elapsed = 0
        nextStep = 10
        render()
        schedule()
      }
    },
    { signal },
  )
  signal.addEventListener('abort', pause, { once: true })

  return {
    pause,
    update(routes: JourneyRoute[], nextLandscape: Landscape, nextSeed: number) {
      const fraction = elapsed / duration()
      landscape = nextLandscape
      incoming = routes.find(route => route.id === 'main')
      const departures = routes.filter(
        route => route.id === 'pines' || route.id === 'ridge' || route.id === 'meadow',
      )
      outgoing = departures[Math.floor(trailRandom(nextSeed ^ 0x78ae12d3)() * departures.length)]
      if (!incoming || !outgoing) return pause()
      if (nextSeed !== seed) {
        pause()
        seed = nextSeed
        phase = 'waiting'
        elapsed = 0
        renderedAt = -Infinity
        nextStep = 10
        tracks.replaceChildren()
        impressions.length = 0
        spacing = actorSpacing()
      } else {
        elapsed = fraction * duration()
        if (phase === 'arriving' || phase === 'departing')
          nextStep = (Math.floor((fraction * legLength(phase)) / 10) + 1) * 10
        for (const stamp of impressions)
          stamp.element.setAttribute('transform', impressionTransform(stamp))
      }
      if (reducedMotion.matches) still()
      else {
        render()
        schedule()
      }
    },
  }
}
