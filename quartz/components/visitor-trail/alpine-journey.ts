import type { Landscape, Point } from './landscape'
import { bootImpression } from './footprints'
import { pixels } from './pixels'
import { trailRandom } from './random'

type MovingPhase = 'climbing' | 'descending'
type Phase = MovingPhase | 'reading' | 'resting'
type Impression = { progress: number; side: number; direction: number; element: SVGPathElement }
type Route = { length: number; point: (progress: number) => Point }
type Traveler = {
  id: number
  element: SVGGElement
  facing: SVGGElement
  pole: SVGGElement
  tracks: SVGGElement
  route: Route
  phase: Phase
  next: MovingPhase
  progress: number
  stamped: number
  step: number
  checkAt: number | null
  remaining: number
  duration: number
  random: () => number
  impressions: Impression[]
}

const perspective = (progress: number) => 0.65 - 0.47 * progress

function alpineBackpacker() {
  const tones = {
    H: 'trail-hiker-hat',
    S: 'trail-camper-skin',
    C: 'trail-hiker-coat',
    D: 'trail-camper-trousers',
    B: 'trail-hiker-boot',
  }
  const strides = [
    ['.D.D...', '.D.D...', 'D...D..', 'BB..BB.'],
    ['.D.D...', '..DD...', '..DD...', '..BBB..'],
  ]
  return `<path class="trail-hiker-shadow" d="M-8 0h16v2H-8Z"/>
    <g class="trail-alpine-facing">
      <g transform="translate(-12 -22)">${pixels(['RRRRR', '.PPPP', '.PLPP', '.PPPP', '.PPPP', '.PLPP', '.PPPP'], { P: 'trail-hiker-pack-cloth', L: 'trail-hiker-pack-strap', R: 'trail-hiker-bedroll' }, 2)}</g>
      <g data-alpine-pose="walk">
        <g transform="translate(-4 -24)">${pixels(['.HHH...', '.HHHH..', '.HSS...', '.HSSS..', '.CCCC..', '.CCCCS.', '.CCC.S.', '.CCC...'], tones, 2)}</g>
        ${strides.map((rows, index) => `<g class="trail-alpine-stride" data-stride="${index}" transform="translate(-4 -8)">${pixels(rows, tones, 2)}</g>`).join('')}
        <g class="trail-alpine-pole">
          <path class="trail-hiker-pole" d="M10-18h1V1h-1Z"/>
          <path class="trail-hiker-grip" d="M9-16h3v3H9Z"/>
          <path class="trail-camper-skin" d="M6-12h4v2H6Z"/>
        </g>
      </g>
      <g data-alpine-pose="read">
        <g transform="translate(-4 -24)">${pixels(['.HHH...', '.HHHH..', '.HSS...', '..SSS..', '.CCCC..', '.CCCCS.', '.CCCSS.', '.CCC...'], tones, 2)}</g>
        <g transform="translate(-4 -8)">${pixels(strides[0], tones, 2)}</g>
        <path class="trail-hiker-pole" d="M-14-23h1V1h-1Z"/>
        <path class="trail-map-edge" d="M4-17h5v1h5v-1h4v11h-4v1H9v-1H4Z"/>
        <path class="trail-map-paper" d="M5-16h3v1h7v-1h2v9h-3v1H9v-1H5Z"/>
        <path class="trail-map-fold" d="M9-15h1v8H9ZM13-15h1v8h-1Z"/>
        <path class="trail-map-route" d="M6-10h4v-2h4v-2h2v1h-1v2h-4v2H6Z"/>
      </g>
    </g>`
}

export function mountAlpineJourneys(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) throw new Error('Alpine travelers need a visible document')
  const reducedMotion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let landscape: Landscape | undefined
  let travelerLayer: SVGGElement | null = null
  let trackLayer: SVGGElement | null = null
  let travelers: Traveler[] = []
  let seed: number | undefined
  let frame: number | null = null
  let timer: number | null = null
  let previousTime: number | null = null
  let renderedAt = -Infinity

  const layerMotion = (motion: string) => {
    for (const layer of [travelerLayer, trackLayer])
      if (layer && layer.dataset.motion !== motion) layer.dataset.motion = motion
  }

  const ascent = (id: number, nextSeed: number): Route => {
    if (!landscape) throw new Error('Alpine routes need their terrain')
    const random = trailRandom(nextSeed ^ (0x7431c59f + id * 0x27d4eb2d))
    const { alpine, meadowTop } = landscape
    const peaks = [
      { x: 0.24, y: 0.45 },
      { x: 0.46, y: 0.5 },
      { x: 0.68, y: 0.55 },
    ]
    const peak = peaks[id % peaks.length]
    const target = {
      x: alpine.left + alpine.width * peak.x,
      y: alpine.top + alpine.height * peak.y,
    }
    const desired = { x: target.x + (random() - 0.5) * alpine.width * 0.08, y: meadowTop + 80 }
    let start = desired
    let score = Infinity
    for (const path of scene.querySelectorAll<SVGPathElement>('[data-trail-route]')) {
      const length = path.getTotalLength()
      for (let distance = 0; distance <= length; distance += 16) {
        const point = path.getPointAtLength(distance)
        if (point.y < meadowTop + 32 || point.y > meadowTop + 140) continue
        const candidate = Math.abs(point.y - desired.y) * 2 + Math.abs(point.x - desired.x)
        if (candidate < score) {
          score = candidate
          start = { x: point.x, y: point.y }
        }
      }
    }
    const phase = random() * Math.PI * 2
    const amplitude = alpine.width * (0.045 + random() * 0.018)
    const point = (progress: number) => {
      const fraction = Math.max(0, Math.min(1, progress))
      const blend = fraction * fraction * (3 - 2 * fraction)
      const switchback =
        Math.sin(phase + fraction * Math.PI * 6) *
        Math.sin(fraction * Math.PI) *
        Math.pow(1 - fraction, 1.2)
      return {
        x: start.x + (target.x - start.x) * blend + amplitude * switchback,
        y: start.y + (target.y - start.y) * fraction,
      }
    }
    let length = 0
    let before = point(0)
    for (let step = 1; step <= 128; step++) {
      const next = point(step / 128)
      length += Math.hypot(next.x - before.x, next.y - before.y)
      before = next
    }
    return { length, point }
  }
  const placeImpression = (traveler: Traveler, impression: Impression) => {
    const point = traveler.route.point(impression.progress)
    const ahead = traveler.route.point(
      Math.max(0, Math.min(1, impression.progress + impression.direction * 0.001)),
    )
    const scale = perspective(impression.progress)
    const angle = (Math.atan2(ahead.y - point.y, ahead.x - point.x) * 180) / Math.PI
    impression.element.setAttribute(
      'transform',
      `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)}) rotate(${angle.toFixed(2)}) scale(${scale.toFixed(4)}) translate(0 ${impression.side * 2}) scale(1 ${impression.side})`,
    )
    impression.element.dataset.progress = impression.progress.toFixed(6)
    impression.element.dataset.scale = scale.toFixed(4)
    impression.element.dataset.direction = String(impression.direction)
    impression.element.dataset.groundX = point.x.toFixed(2)
    impression.element.dataset.groundY = point.y.toFixed(2)
  }
  const leaveTracks = (traveler: Traveler) => {
    if (
      signal.aborted ||
      host.hidden ||
      document.hidden ||
      reducedMotion.matches ||
      (traveler.phase !== 'climbing' && traveler.phase !== 'descending')
    )
      return
    const direction = traveler.phase === 'climbing' ? 1 : -1
    let interval = (8 * perspective(traveler.stamped)) / traveler.route.length
    while (direction * (traveler.progress - traveler.stamped) >= interval) {
      traveler.stamped += direction * interval
      const impression = {
        progress: traveler.stamped,
        side: traveler.step++ % 2 ? -1 : 1,
        direction,
        element: bootImpression(document),
      }
      placeImpression(traveler, impression)
      traveler.tracks.append(impression.element)
      traveler.impressions.push(impression)
      if (traveler.impressions.length > 64) traveler.impressions.shift()?.element.remove()
      interval = (8 * perspective(traveler.stamped)) / traveler.route.length
    }
  }
  const render = () => {
    if (!landscape) return
    layerMotion(
      reducedMotion.matches ? 'still' : host.hidden || document.hidden ? 'paused' : 'active',
    )
    for (const traveler of travelers) {
      const from = traveler.route.point(0)
      const to = traveler.route.point(1)
      const stillProgress =
        (from.y - (landscape.alpine.base - landscape.alpine.height * 0.1)) / (from.y - to.y)
      const progress = reducedMotion.matches
        ? Math.max(0, Math.min(1, stillProgress))
        : traveler.progress
      const point = traveler.route.point(progress)
      const scale = perspective(progress)
      const direction = traveler.phase === 'descending' || traveler.next === 'descending' ? -1 : 1
      const ahead = traveler.route.point(Math.max(0, Math.min(1, progress + direction * 0.003)))
      traveler.element.setAttribute(
        'transform',
        `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)}) scale(${scale.toFixed(4)})`,
      )
      traveler.element.dataset.phase = reducedMotion.matches ? 'reading' : traveler.phase
      traveler.element.dataset.progress = progress.toFixed(6)
      traveler.element.dataset.scale = scale.toFixed(4)
      traveler.element.dataset.groundX = point.x.toFixed(2)
      traveler.element.dataset.groundY = point.y.toFixed(2)
      traveler.element.dataset.motion = reducedMotion.matches
        ? 'still'
        : host.hidden || document.hidden
          ? 'paused'
          : 'active'
      if (Math.abs(ahead.x - point.x) > 0.01)
        traveler.facing.setAttribute('transform', `scale(${ahead.x < point.x ? -1 : 1} 1)`)
      const stride = Math.floor((progress * traveler.route.length) / (8 * scale)) % 2
      traveler.element.dataset.stride = String(stride)
      traveler.pole.setAttribute('transform', `rotate(${stride ? -12 : 8} 10 -12)`)
      leaveTracks(traveler)
    }
  }
  const pause = () => {
    if (frame !== null) view.cancelAnimationFrame(frame)
    if (timer !== null) view.clearTimeout(timer)
    frame = timer = previousTime = null
    layerMotion(reducedMotion.matches ? 'still' : 'paused')
    for (const traveler of travelers)
      traveler.element.dataset.motion = reducedMotion.matches ? 'still' : 'paused'
  }
  const moving = (traveler: Traveler) =>
    traveler.phase === 'climbing' || traveler.phase === 'descending'
  const step = (traveler: Traveler, elapsed: number) => {
    if (!moving(traveler)) {
      traveler.remaining -= elapsed
      if (traveler.remaining <= 0) {
        traveler.phase = traveler.next
        traveler.stamped = traveler.progress
        if (traveler.progress === 0 || traveler.progress === 1) {
          traveler.checkAt =
            traveler.phase === 'climbing'
              ? 0.45 + traveler.random() * 0.14
              : 0.35 + traveler.random() * 0.15
          traveler.tracks.replaceChildren()
          traveler.impressions.length = 0
        }
      }
      return
    }
    const direction = traveler.phase === 'climbing' ? 1 : -1
    const target = traveler.checkAt ?? (direction > 0 ? 1 : 0)
    const next = traveler.progress + (direction * elapsed) / traveler.duration
    const reached = direction > 0 ? next >= target : next <= target
    traveler.progress = reached ? target : next
    if (!reached) return
    traveler.next = traveler.phase === 'climbing' ? 'climbing' : 'descending'
    if (traveler.checkAt !== null) {
      traveler.phase = 'reading'
      traveler.checkAt = null
      traveler.remaining = 2500 + traveler.random() * 2500
    } else {
      traveler.phase = 'resting'
      traveler.next = direction > 0 ? 'descending' : 'climbing'
      traveler.remaining = 5500 + traveler.random() * 4500
    }
  }
  const schedule = () => {
    if (
      signal.aborted ||
      host.hidden ||
      document.hidden ||
      reducedMotion.matches ||
      !travelers.length
    )
      return
    if (frame !== null || timer !== null) return
    previousTime ??= view.performance.now()
    if (travelers.some(moving)) frame = view.requestAnimationFrame(tick)
    else
      timer = view.setTimeout(
        () => tick(view.performance.now()),
        Math.max(1, Math.min(...travelers.map(traveler => traveler.remaining))),
      )
  }
  const tick = (time: number) => {
    frame = timer = null
    if (signal.aborted || host.hidden || document.hidden || reducedMotion.matches) return pause()
    const elapsed = previousTime === null ? 0 : Math.max(0, time - previousTime)
    for (const traveler of travelers)
      step(traveler, moving(traveler) ? Math.min(200, elapsed) : elapsed)
    previousTime = time
    if (time - renderedAt >= 1000 / 24 || !travelers.some(moving)) {
      render()
      renderedAt = time
    }
    schedule()
  }
  const resume = () => {
    render()
    schedule()
  }
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : resume()), {
    signal,
  })
  reducedMotion.addEventListener(
    'change',
    () => {
      pause()
      if (reducedMotion.matches)
        for (const traveler of travelers) {
          traveler.tracks.replaceChildren()
          traveler.impressions.length = 0
          traveler.stamped = traveler.progress
        }
      resume()
    },
    { signal },
  )
  signal.addEventListener('abort', pause, { once: true })

  return {
    pause,
    resume,
    update(nextLandscape: Landscape, nextSeed: number) {
      pause()
      landscape = nextLandscape
      const group = scene.querySelector<SVGGElement>('.trail-alpine-travelers')
      const tracks = scene.querySelector<SVGGElement>('.trail-alpine-tracks')
      if (!group || !tracks || signal.aborted) return
      travelerLayer = group
      trackLayer = tracks
      if (seed !== nextSeed) {
        seed = nextSeed
        travelers = []
        renderedAt = -Infinity
      }
      const count = landscape.width < 700 ? 2 : 3
      if (travelers.length > count)
        for (const traveler of travelers.splice(count)) {
          traveler.element.remove()
          traveler.tracks.remove()
        }
      while (travelers.length < count) {
        const id = travelers.length
        const random = trailRandom(nextSeed ^ (0x19a249e3 + id * 0x27d4eb2d))
        const element = document.createElementNS('http://www.w3.org/2000/svg', 'g')
        element.classList.add('trail-alpine-traveler')
        element.dataset.alpineTraveler = String(id)
        element.innerHTML = alpineBackpacker()
        const facing = element.querySelector<SVGGElement>('.trail-alpine-facing')
        const pole = element.querySelector<SVGGElement>('.trail-alpine-pole')
        if (!facing || !pole) throw new Error('An Alpine backpacker needs a pole and pose')
        const marks = document.createElementNS('http://www.w3.org/2000/svg', 'g')
        marks.dataset.alpineTrack = String(id)
        const progress = 0.06 + id * 0.14 + random() * 0.06
        travelers.push({
          id,
          element,
          facing,
          pole,
          tracks: marks,
          route: ascent(id, nextSeed),
          phase: 'climbing',
          next: 'climbing',
          progress,
          stamped: progress,
          step: 0,
          checkAt: 0.45 + random() * 0.14,
          remaining: 0,
          duration: 55000 + random() * 35000,
          random,
          impressions: [],
        })
      }
      group.replaceChildren(...travelers.map(traveler => traveler.element))
      tracks.replaceChildren(...travelers.map(traveler => traveler.tracks))
      for (const traveler of travelers) {
        traveler.route = ascent(traveler.id, nextSeed)
        for (const impression of traveler.impressions) placeImpression(traveler, impression)
      }
      resume()
    },
  }
}
