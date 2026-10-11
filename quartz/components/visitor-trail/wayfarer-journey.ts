import type { JourneyRoute } from './hiker-journey'
import type { TrailId } from './layout'
import {
  actorSpacing,
  publishActor,
  readActors,
  readTrunks,
  spacedPosition,
  spacedStep,
  type ActorSpacing,
  type CollisionActor,
} from './actor-spacing'
import { bootImpression } from './footprints'
import { trailGroundScale, type Landscape, type Point } from './landscape'
import { trailRandom } from './random'
import { trailMapReader } from './wayfarers'

type Phase = 'waiting' | 'walking' | 'reading'
type Stop = 'middle' | 'camp' | 'turn'
type Impression = {
  route: TrailId
  progress: number
  side: number
  direction: number
  offset: Point
  element: SVGPathElement
}
type Reader = {
  id: string
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
  spacing: ActorSpacing
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
  const positionAt = (reader: Reader, route: JourneyRoute, progress: number) => {
    const point = route.point(progress * route.length)
    const approach = Math.max(0, 1 - ((1 - progress) * route.length) / 110)
    const blend = approach * approach * (3 - 2 * approach)
    const index = Number(reader.element.dataset.reader)
    return { x: point.x + (-14 + index * 34) * blend, y: point.y + (38 + (index % 2) * 6) * blend }
  }
  const placeImpression = (reader: Reader, stamp: Impression) => {
    const route = routes.find(route => route.id === stamp.route)
    if (!route || !landscape) return
    const base = positionAt(reader, route, stamp.progress)
    const point = { x: base.x + stamp.offset.x, y: base.y + stamp.offset.y }
    const ahead = positionAt(
      reader,
      route,
      Math.max(0, Math.min(1, stamp.progress + stamp.direction / route.length)),
    )
    const scale = trailGroundScale(point.y, landscape)
    const angle = (Math.atan2(ahead.y - base.y, ahead.x - base.x) * 180) / Math.PI
    stamp.element.setAttribute(
      'transform',
      `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)}) rotate(${angle.toFixed(2)}) scale(${scale.toFixed(4)}) translate(0 ${stamp.side * 2}) scale(1 ${stamp.side})`,
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
        offset: { ...reader.spacing.offset },
        element,
      }
      reader.tracks.append(element)
      reader.impressions.push(stamp)
      placeImpression(reader, stamp)
      if (reader.impressions.length > 48) reader.impressions.shift()?.element.remove()
    }
  }
  const clearingStart = (route: JourneyRoute) => {
    if (!landscape) return 0.5
    // Far roots stay on the rock foot; older routes must also clear all homepage text.
    for (let index = 63; index >= 0; index--) {
      if (route.point((route.length * index) / 64).y < landscape.alpine.base - 18)
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
      const position = positionAt(reader, route, reader.progress)
      const location = spacedPosition(position, reader.spacing)
      const scale = trailGroundScale(location.y, landscape)
      reader.element.dataset.moveSpeed = (reader.speed * scale).toFixed(4)
      publishActor(
        reader.element,
        reader.id,
        location,
        scale,
        reader.spacing,
        reducedMotion.matches || reader.phase !== 'waiting',
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
      const ahead = positionAt(
        reader,
        route,
        Math.max(0, Math.min(1, reader.progress + (reader.direction * 3) / route.length)),
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
  const step = (reader: Reader, elapsed: number, obstacles: CollisionActor[]) => {
    if (reader.phase !== 'walking') {
      reader.remaining -= elapsed
      if (reader.remaining <= 0) {
        if (reader.phase === 'waiting') reader.phase = 'walking'
        else resume(reader)
      }
      return
    }
    const route = routeFor(reader)
    if (!route?.length || !landscape) return
    const base = positionAt(reader, route, reader.progress)
    const scale = trailGroundScale(spacedPosition(base, reader.spacing).y, landscape)
    const budget = (reader.speed * elapsed * scale) / 1000
    const advance = budget * (Math.hypot(reader.spacing.offset.x, reader.spacing.offset.y) > 0.1 ? 0.7 : 1)
    const next = reader.progress + (reader.direction * advance) / route.length
    const stop = reader.checkAt ?? reader.target
    const reached = reader.direction > 0 ? next >= stop : next <= stop
    let progress = reached ? stop : Math.max(0, Math.min(1, next))
    let destination = positionAt(reader, route, progress)
    for (let iteration = 0; iteration < 2; iteration++) {
      const distance = Math.hypot(destination.x - base.x, destination.y - base.y)
      if (distance <= advance || !distance) break
      progress = reader.progress + (progress - reader.progress) * advance / distance * 0.999
      destination = positionAt(reader, route, progress)
    }
    const radius = 12 * Math.max(scale, trailGroundScale(Math.max(base.y, destination.y) + reader.spacing.offset.y + budget, landscape))
    const advances = spacedStep(
      reader.id,
      base,
      destination,
      radius,
      budget,
      reader.spacing,
      obstacles,
      { left: radius, right: landscape.width - radius, top: landscape.alpine.base - 18, bottom: landscape.height - radius },
    )
    if (advances) reader.progress = progress
    const point = spacedPosition(positionAt(reader, route, reader.progress), reader.spacing)
    const circle = obstacles.find(other => other.id === reader.id)
    if (circle) Object.assign(circle, point, { radius: 12 * trailGroundScale(point.y, landscape) })
    else obstacles.push({ ...point, id: reader.id, kind: 'traveler', radius: 12 * scale })
    if (progress === stop && advances) {
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
    if (time - renderedAt < 1000 / 24 && readers.some(reader => reader.phase === 'walking')) {
      schedule()
      return
    }
    const obstacles = [...readActors(scene), ...readTrunks(scene)]
    for (const reader of readers)
      step(reader, reader.phase === 'walking' ? Math.min(200, delta) : delta, obstacles)
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
        const progress = start + (1 - start) * (0.015 + index * 0.035 + random() * 0.03)
        const readerTracks = document.createElementNS('http://www.w3.org/2000/svg', 'g')
        readerTracks.dataset.readerTrack = String(index)
        tracks.append(readerTracks)
        readers.push({
          id: `reader-${index}`,
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
          checkAt: progress + (1 - progress) * (0.08 + random() * 0.07),
          direction: 1,
          remaining: route.readyAfter + 250 + index * 450,
          speed: 22 + random() * 10,
          lap: 0,
          random,
          spacing: actorSpacing(),
        })
      }
      for (const reader of readers)
        for (const stamp of reader.impressions) placeImpression(reader, stamp)
      render()
      schedule()
    },
  }
}
