import type { JourneyRoute } from './hiker-journey'
import type { TrailId } from './layout'
import { bootImpression, footprintTransform } from './footprints'
import { onDryLand, type Landscape } from './landscape'
import { trailRandom } from './random'
import { trailMapReader } from './wayfarers'

type Phase = 'waiting' | 'walking' | 'reading'
type Stop = 'middle' | 'camp' | 'turn'
type Impression = {
  route: TrailId
  progress: number
  side: number
  direction: number
  element: SVGPathElement
}
type Reader = {
  element: SVGGElement
  facing: SVGGElement
  pole: SVGGElement
  tracks: SVGGElement
  impressions: Impression[]
  stampedProgress: number
  step: number
  route: TrailId
  phase: Phase
  stop: Stop
  progress: number
  target: number
  checkAt: number | null
  direction: number
  remaining: number
  speed: number
  lap: number
  random: () => number
}

export function mountMapReaderJourneys(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const group = scene.querySelector<SVGGElement>('.trail-wayfarers')
  const tracks = scene.querySelector<SVGGElement>('.trail-reader-tracks')
  if (!view || !group || !tracks) throw new Error('Map readers need the trail scene')
  const reducedMotion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let readers: Reader[] = []
  let routes: JourneyRoute[] = []
  let landscape: Landscape | undefined
  let seed: number | undefined
  let frame: number | null = null
  let timer: number | null = null
  let previousTime: number | null = null
  let renderedAt = -Infinity

  const routeFor = (reader: Reader) => routes.find(route => route.id === reader.route)
  const placeImpression = (stamp: Impression) => {
    const route = routes.find(route => route.id === stamp.route)
    if (!route) return
    const distance = stamp.progress * route.length
    const point = route.point(distance)
    const ahead = route.point(Math.max(0, Math.min(route.length, distance + stamp.direction)))
    stamp.element.setAttribute(
      'transform',
      footprintTransform(
        point.x,
        point.y,
        Math.atan2(ahead.y - point.y, ahead.x - point.x),
        stamp.side,
      ),
    )
  }
  const leaveTracks = (reader: Reader, route: JourneyRoute) => {
    if (reducedMotion.matches || reader.phase !== 'walking' || host.hidden || document.hidden)
      return
    while (Math.abs(reader.progress - reader.stampedProgress) * route.length >= 12) {
      reader.stampedProgress += (reader.direction * 12) / route.length
      const element = bootImpression(document)
      element.dataset.route = route.id
      element.dataset.progress = reader.stampedProgress.toFixed(6)
      element.dataset.direction = String(reader.direction)
      element.dataset.lap = String(reader.lap)
      const stamp = {
        route: route.id,
        progress: reader.stampedProgress,
        side: reader.step++ % 2 ? -1 : 1,
        direction: reader.direction,
        element,
      }
      placeImpression(stamp)
      reader.tracks.append(element)
      reader.impressions.push(stamp)
      if (reader.impressions.length > 48) reader.impressions.shift()?.element.remove()
    }
  }
  const clearingStart = (route: JourneyRoute) => {
    if (!landscape) return 0.5
    // The final continuous clearing section keeps walkers below the reading text.
    for (let index = 63; index >= 0; index--) {
      if (route.point((route.length * index) / 64).y < landscape.meadowTop + 64)
        return Math.min(0.92, (index + 1) / 64)
    }
    return 0
  }
  const turningPoint = (route: JourneyRoute, reader: Reader) => {
    const start = clearingStart(route)
    return start + (1 - start) * (0.08 + reader.random() * 0.18)
  }
  const render = () => {
    if (!landscape) return
    for (const reader of readers) {
      const route = routeFor(reader)
      if (!route) continue
      const position = route.point(reader.progress * route.length)
      const location = reducedMotion.matches
        ? onDryLand({ x: position.x + 12, y: position.y }, landscape, 1.8)
        : position
      reader.element.setAttribute(
        'transform',
        `translate(${Math.round(location.x)} ${Math.round(location.y)})`,
      )
      reader.element.dataset.phase = reducedMotion.matches ? 'reading' : reader.phase
      reader.element.dataset.motion = reducedMotion.matches
        ? 'still'
        : host.hidden || document.hidden
          ? 'paused'
          : 'active'
      reader.element.dataset.route = route.id
      reader.element.dataset.progress = reader.progress.toFixed(6)
      reader.element.dataset.direction = String(reader.direction)
      reader.element.dataset.lap = String(reader.lap)
      const ahead = route.point(
        Math.max(0, Math.min(route.length, reader.progress * route.length + reader.direction * 3)),
      )
      if (Math.abs(ahead.x - position.x) > 0.1)
        reader.facing.setAttribute('transform', `scale(${ahead.x < position.x ? -1 : 1} 1)`)
      const stride = Math.floor((reader.progress * route.length) / 10) % 2
      reader.element.dataset.stride = String(stride)
      reader.pole.setAttribute('transform', `rotate(${stride ? -10 : 8} 10 -13)`)
      leaveTracks(reader, route)
    }
  }
  const pause = () => {
    if (frame !== null) view.cancelAnimationFrame(frame)
    if (timer !== null) view.clearTimeout(timer)
    frame = timer = previousTime = null
    for (const reader of readers)
      reader.element.dataset.motion = reducedMotion.matches ? 'still' : 'paused'
  }
  const checkMap = (reader: Reader, stop: Stop) => {
    reader.phase = 'reading'
    reader.stop = stop
    reader.remaining = (stop === 'camp' ? 2200 : 1800) + reader.random() * 1800
  }
  const resume = (reader: Reader) => {
    reader.phase = 'walking'
    if (reader.stop === 'middle') return
    const current = routeFor(reader)
    if (!current) return
    if (reader.stop === 'camp') {
      const choices = routes.filter(route => route.id !== reader.route)
      const next = choices[Math.floor(reader.random() * choices.length)]
      if (!next) return
      // Every route shares this endpoint, so changing paths keeps position continuous.
      reader.route = next.id
      reader.progress = 1
      reader.stampedProgress = 1
      reader.direction = -1
      reader.target = turningPoint(next, reader)
      reader.checkAt = 1 - (1 - reader.target) * (0.35 + reader.random() * 0.25)
      reader.lap++
    } else {
      reader.direction = 1
      reader.stampedProgress = reader.progress
      reader.target = 1
      reader.checkAt = null
    }
  }
  const step = (reader: Reader, elapsed: number) => {
    if (reader.phase !== 'walking') {
      reader.remaining -= elapsed
      if (reader.remaining <= 0) {
        if (reader.phase === 'waiting') reader.phase = 'walking'
        else resume(reader)
      }
      return
    }
    const route = routeFor(reader)
    if (!route?.length) return
    const next =
      reader.progress + (reader.direction * reader.speed * elapsed) / (1000 * route.length)
    const stop = reader.checkAt ?? reader.target
    const reached = reader.direction > 0 ? next >= stop : next <= stop
    reader.progress = reached ? stop : Math.max(0, Math.min(1, next))
    if (reached) {
      const middle = reader.checkAt !== null
      reader.checkAt = null
      checkMap(reader, middle ? 'middle' : reader.direction > 0 ? 'camp' : 'turn')
    }
  }
  const schedule = () => {
    if (
      signal.aborted ||
      host.hidden ||
      document.hidden ||
      reducedMotion.matches ||
      !readers.length
    )
      return
    if (frame !== null || timer !== null) return
    previousTime ??= view.performance.now()
    if (readers.some(reader => reader.phase === 'walking')) frame = view.requestAnimationFrame(tick)
    else
      timer = view.setTimeout(
        () => tick(view.performance.now()),
        Math.max(1, Math.min(...readers.map(reader => reader.remaining))),
      )
  }
  const tick = (time: number) => {
    frame = timer = null
    if (host.hidden || document.hidden || signal.aborted || reducedMotion.matches) return pause()
    const delta = previousTime === null ? 0 : Math.max(0, time - previousTime)
    for (const reader of readers)
      step(reader, reader.phase === 'walking' ? Math.min(200, delta) : delta)
    const walking = readers.some(reader => reader.phase === 'walking')
    previousTime = time
    if (time - renderedAt >= 1000 / 24 || !walking) {
      render()
      renderedAt = time
    }
    schedule()
  }
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : schedule()), {
    signal,
  })
  reducedMotion.addEventListener(
    'change',
    () => {
      pause()
      if (reducedMotion.matches)
        for (const reader of readers) {
          reader.tracks.replaceChildren()
          reader.impressions.length = 0
          reader.stampedProgress = reader.progress
        }
      render()
      schedule()
    },
    { signal },
  )
  signal.addEventListener('abort', pause, { once: true })

  return {
    pause,
    update(nextRoutes: JourneyRoute[], nextLandscape: Landscape, nextSeed: number) {
      routes = nextRoutes
      landscape = nextLandscape
      const count = landscape.width < 700 ? 2 : 3
      if (seed !== nextSeed) {
        pause()
        seed = nextSeed
        readers = []
        group.replaceChildren()
        tracks.replaceChildren()
        renderedAt = -Infinity
      }
      if (readers.length > count) {
        for (const reader of readers.splice(count)) {
          reader.element.remove()
          reader.tracks.remove()
        }
      }
      while (readers.length < count) {
        const index = readers.length
        const random = trailRandom(nextSeed ^ (0x381c72ad + index * 0x27d4eb2d))
        const route =
          routes.find(route => route.id === ['grove', 'ridge', 'meadow'][index]) ?? routes[index]
        if (!route) break
        group.insertAdjacentHTML('beforeend', trailMapReader(index))
        const element = group.querySelector<SVGGElement>(`[data-reader="${index}"]`)
        const facing = element?.querySelector<SVGGElement>('.trail-reader-facing')
        const pole = element?.querySelector<SVGGElement>('.trail-reader-pole')
        if (!element || !facing || !pole) throw new Error('A map reader needs its walking pose')
        const start = clearingStart(route)
        const progress = Math.max(start, 1 - (110 + random() * 130) / route.length)
        const readerTracks = document.createElementNS('http://www.w3.org/2000/svg', 'g')
        readerTracks.dataset.readerTrack = String(index)
        tracks.append(readerTracks)
        readers.push({
          element,
          facing,
          pole,
          tracks: readerTracks,
          impressions: [],
          stampedProgress: progress,
          step: 0,
          route: route.id,
          phase: 'waiting',
          stop: 'middle',
          progress,
          target: 1,
          checkAt: progress + (1 - progress) * (0.3 + random() * 0.25),
          direction: 1,
          remaining: route.readyAfter + 250 + index * 450,
          speed: 22 + random() * 10,
          lap: 0,
          random,
        })
      }
      for (const reader of readers) for (const stamp of reader.impressions) placeImpression(stamp)
      render()
      schedule()
    },
  }
}
