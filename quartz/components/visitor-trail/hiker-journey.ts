import type { TrailId } from './layout'
import { bootImpression, footprintTransform } from './footprints'
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
type Impression = { phase: WalkingPhase; progress: number; side: number; element: SVGPathElement }
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
  let exit: Point = { x: 0, y: 0 }
  let exitLength = 0
  let seed: number | undefined
  let phase: Phase = 'waiting'
  let elapsed = 0
  let previousTime: number | null = null
  let renderedAt = -Infinity
  let frame: number | null = null
  let timer: number | null = null
  let nextStep = 10
  const impressions: Impression[] = []

  const duration = () => {
    if (phase === 'waiting') return (incoming?.readyAfter ?? 0) + 250
    if (phase === 'arriving') return Math.max(20000, Math.min(50000, (incoming?.length ?? 0) * 28))
    if (phase === 'departing')
      return Math.max(22000, Math.min(52000, ((outgoing?.length ?? 0) + exitLength) * 28))
    if (phase === 'resting') return 9000
    return 1400
  }
  const moving = () => phase === 'arriving' || phase === 'departing'
  const legLength = (leg: WalkingPhase) =>
    leg === 'arriving' ? (incoming?.length ?? 0) : (outgoing?.length ?? 0) + exitLength
  const onLeg = (leg: WalkingPhase, progress: number): Point => {
    const route = leg === 'arriving' ? incoming : outgoing
    if (!route) return { x: 0, y: 0 }
    const distance = Math.max(0, Math.min(1, progress)) * legLength(leg)
    if (leg === 'arriving') return route.point(distance)
    if (distance <= route.length) return route.point(route.length - distance)
    const start = route.point(0)
    const fraction = exitLength ? (distance - route.length) / exitLength : 1
    return {
      x: start.x + (exit.x - start.x) * fraction,
      y: start.y + (exit.y - start.y) * fraction,
    }
  }
  const impressionTransform = (leg: WalkingPhase, progress: number, side: number) => {
    const point = onLeg(leg, progress)
    const ahead = onLeg(leg, Math.min(1, progress + 1 / legLength(leg)))
    return footprintTransform(
      point.x,
      point.y,
      Math.atan2(ahead.y - point.y, ahead.x - point.x),
      side,
    )
  }
  const render = () => {
    if (!incoming) return
    hiker.dataset.phase = phase
    hiker.dataset.motion = reducedMotion.matches ? 'still' : 'walking'
    const progress = Math.min(1, elapsed / duration())
    hiker.dataset.progress = progress.toFixed(3)
    hiker.dataset.walkingDuration = moving() ? String(duration()) : ''
    const leg: WalkingPhase | null = phase === 'arriving' || phase === 'departing' ? phase : null
    const position = leg ? onLeg(leg, progress) : incoming.point(incoming.length)
    hiker.setAttribute(
      'transform',
      `translate(${Math.round(position.x)} ${Math.round(position.y)})`,
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
        const stamp = { phase: leg, progress: nextStep / legLength(leg), side, element: mark }
        mark.setAttribute('transform', impressionTransform(stamp.phase, stamp.progress, side))
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
    elapsed += moving() ? Math.min(200, delta) : delta
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
    update(routes: JourneyRoute[], width: number, nextSeed: number) {
      const fraction = elapsed / duration()
      incoming = routes.find(route => route.id === 'main')
      const departures = routes.filter(
        route => route.id === 'pines' || route.id === 'ridge' || route.id === 'meadow',
      )
      outgoing = departures[Math.floor(trailRandom(nextSeed ^ 0x78ae12d3)() * departures.length)]
      if (!incoming || !outgoing) return pause()
      const start = outgoing.point(0)
      exit =
        outgoing.id === 'pines'
          ? { x: start.x + 16, y: -30 }
          : outgoing.id === 'ridge'
            ? { x: width + 30, y: start.y - 16 }
            : { x: -30, y: start.y - 10 }
      exitLength = Math.hypot(exit.x - start.x, exit.y - start.y)
      if (nextSeed !== seed) {
        pause()
        seed = nextSeed
        phase = 'waiting'
        elapsed = 0
        renderedAt = -Infinity
        nextStep = 10
        tracks.replaceChildren()
        impressions.length = 0
      } else {
        elapsed = fraction * duration()
        if (phase === 'arriving' || phase === 'departing')
          nextStep = (Math.floor((fraction * legLength(phase)) / 10) + 1) * 10
        for (const stamp of impressions)
          stamp.element.setAttribute(
            'transform',
            impressionTransform(stamp.phase, stamp.progress, stamp.side),
          )
      }
      if (reducedMotion.matches) still()
      else {
        render()
        schedule()
      }
    },
  }
}
