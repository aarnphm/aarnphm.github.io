import type { JourneyRoute } from './hiker-journey'
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
  type ActorSpacing,
  type CollisionActor,
} from './actor-spacing'
import { bootImpression } from './footprints'
import { trailGroundScale, type Landscape, type Point } from './landscape'
import { detourTrail, navigableTrail, replanOccupiedWaypoints } from './obstacle-route'
import { trailRandom } from './random'
import { trailMapReader } from './wayfarers'

type Phase = 'waiting' | 'walking' | 'reading'
type Stop = 'middle' | 'camp' | 'turn'
type Impression = {
  route: TrailId
  progress: number
  side: number
  offset: Point
  angle: number
  element: SVGPathElement
}
type Reader = {
  id: string
  element: SVGGElement
  facing: SVGGElement
  pole: SVGGElement
  tracks: SVGGElement
  impressions: Impression[]
  lastStep: Point
  step: number
  route: TrailId
  routes: JourneyRoute[]
  phase: Phase
  stop: Stop
  position: Point
  waypoints: Point[]
  stopPoint: Point
  roadStart: number
  roadEnd: number
  roadLength: number
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

function roadLength(from: Point, points: readonly Point[]) {
  let length = 0
  for (const point of points) {
    length += Math.hypot(point.x - from.x, point.y - from.y)
    from = point
  }
  return length
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
  let solids: readonly CollisionActor[] = []
  let landscape: Landscape | undefined
  let seed: number | undefined
  let frame: number | null = null
  let timer: number | null = null
  let previousTime: number | null = null
  let renderedAt = -Infinity

  const routeFor = (reader: Reader) => reader.routes.find(route => route.id === reader.route)
  const positionAt = (route: JourneyRoute, progress: number) => route.point(progress * route.length)
  const bounds = () => ({
    left: 12,
    right: (landscape?.width ?? 24) - 12,
    top: Math.max(landscape?.treeline ?? 0, (landscape?.alpine.base ?? 18) - 18),
    bottom: (landscape?.height ?? 24) - 12,
  })
  const planningBodies = (id: string) => [
    ...solids,
    ...readStationaryActors(scene, id),
    ...readActors(scene).filter(actor => actor.kind === 'pet'),
  ]
  const pathsFor = (index: number, id: string) => {
    if (!landscape) return []
    const terrain = landscape
    const bodies = planningBodies(id)
    return routes.flatMap(route => {
      const planned = navigableTrail(
        {
          ...route,
          point: distance => {
            const point = route.point(distance)
            const approach = Math.max(0, 1 - (route.length - distance) / 110)
            const blend = approach * approach * (3 - 2 * approach)
            return {
              x: point.x + (-14 + index * 34) * blend,
              y: point.y + (38 + (index % 2) * 6) * blend,
            }
          },
        },
        terrain,
        bodies,
      )
      return planned ? [planned] : []
    })
  }
  const setRoad = (reader: Reader, points: Point[]) => {
    reader.waypoints = points
    reader.roadStart = reader.progress
    reader.roadEnd = reader.checkAt ?? reader.target
    reader.roadLength = roadLength(reader.position, points)
    reader.stopPoint = points.at(-1) ?? reader.position
  }
  const planWalk = (reader: Reader) => {
    const route = routeFor(reader)
    if (!route || !landscape) return
    const stop = reader.checkAt ?? reader.target
    const distance = Math.abs(stop - reader.progress) * route.length
    const steps = Math.max(1, Math.ceil(distance / 16))
    const points = Array.from({ length: steps }, (_, index) =>
      positionAt(route, reader.progress + ((stop - reader.progress) * (index + 1)) / steps),
    )
    const planned = detourTrail(
      [reader.position, ...points],
      12,
      planningBodies(reader.id),
      bounds(),
    )
    setRoad(reader, planned?.points.slice(1) ?? points)
  }
  const placeReader = (reader: Reader, obstacles: CollisionActor[]) => {
    const route = routeFor(reader)
    if (!route || !landscape) return
    const terrain = landscape
    placeOnReflow(
      reader.element,
      reader.id,
      positionAt(route, reader.progress),
      point => 12 * trailGroundScale(point.y, terrain),
      reader.spacing,
      obstacles,
      bounds(),
    )
    reader.position = spacedPosition(positionAt(route, reader.progress), reader.spacing)
    reader.spacing.offset = { x: 0, y: 0 }
    reader.spacing.lane = { x: 0, y: 0 }
    reader.lastStep = { ...reader.position }
  }
  const placeImpression = (reader: Reader, stamp: Impression) => {
    const route = reader.routes.find(route => route.id === stamp.route)
    if (!route || !landscape) return
    const base = positionAt(route, stamp.progress)
    const point = { x: base.x + stamp.offset.x, y: base.y + stamp.offset.y }
    const scale = trailGroundScale(point.y, landscape)
    stamp.element.setAttribute(
      'transform',
      `translate(${point.x.toFixed(2)} ${point.y.toFixed(2)}) rotate(${stamp.angle.toFixed(2)}) scale(${scale.toFixed(4)}) translate(0 ${stamp.side * 2}) scale(1 ${stamp.side})`,
    )
  }
  const leaveTracks = (reader: Reader, route: JourneyRoute) => {
    if (reducedMotion.matches || reader.phase !== 'walking' || host.hidden || document.hidden)
      return
    const point = reader.position
    if (Math.hypot(point.x - reader.lastStep.x, point.y - reader.lastStep.y) < 12) return
    const base = positionAt(route, reader.progress)
    const element = bootImpression(document)
    element.dataset.route = route.id
    element.dataset.progress = reader.progress.toFixed(6)
    element.dataset.direction = String(reader.direction)
    element.dataset.lap = String(reader.lap)
    const stamp: Impression = {
      route: route.id,
      progress: reader.progress,
      side: reader.step++ % 2 ? -1 : 1,
      offset: { x: point.x - base.x, y: point.y - base.y },
      angle: (Math.atan2(point.y - reader.lastStep.y, point.x - reader.lastStep.x) * 180) / Math.PI,
      element,
    }
    reader.lastStep = { ...point }
    reader.tracks.append(element)
    reader.impressions.push(stamp)
    placeImpression(reader, stamp)
    if (reader.impressions.length > 48) reader.impressions.shift()?.element.remove()
  }
  const clearingStart = (route: JourneyRoute) => {
    if (!landscape) return 0.5
    for (let index = 63; index >= 0; index--)
      if (route.point((route.length * index) / 64).y < landscape.alpine.base - 18)
        return Math.min(0.92, (index + 1) / 64)
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
      const scale = trailGroundScale(reader.position.y, landscape)
      reader.element.dataset.moveSpeed = (reader.speed * scale).toFixed(4)
      publishActor(
        reader.element,
        reader.id,
        reader.position,
        scale,
        reader.spacing,
        reducedMotion.matches || reader.phase !== 'waiting',
        previousTime ?? view.performance.now(),
      )
      reader.element.dataset.phase = reducedMotion.matches ? 'reading' : reader.phase
      reader.element.dataset.motion = reducedMotion.matches
        ? 'still'
        : host.hidden || document.hidden
          ? 'paused'
          : reader.spacing.yieldingTo
            ? 'yielding'
            : 'active'
      reader.element.dataset.route = route.id
      reader.element.dataset.progress = reader.progress.toFixed(6)
      reader.element.dataset.direction = String(reader.direction)
      reader.element.dataset.lap = String(reader.lap)
      reader.element.dataset.waypointsRemaining = String(reader.waypoints.length)
      const target = reducedMotion.matches
        ? reader.position
        : (reader.waypoints[0] ?? reader.stopPoint)
      reader.element.dataset.targetX = target.x.toFixed(2)
      reader.element.dataset.targetY = target.y.toFixed(2)
      if (Math.abs(target.x - reader.position.x) > 0.1)
        reader.facing.setAttribute('transform', `scale(${target.x < reader.position.x ? -1 : 1} 1)`)
      const stride = reader.step % 2
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
    const current = routeFor(reader)
    if (!current) return
    if (reader.stop === 'camp') {
      const choices = reader.routes.filter(route => route.id !== reader.route)
      const next = choices[Math.floor(reader.random() * choices.length)]
      if (!next) return
      reader.route = next.id
      reader.progress = 1
      reader.direction = -1
      reader.target = turningPoint(next, reader)
      reader.checkAt = 1 - (1 - reader.target) * (0.35 + reader.random() * 0.25)
      reader.lap++
    } else if (reader.stop === 'turn') {
      reader.direction = 1
      reader.target = 1
      reader.checkAt = null
    }
    planWalk(reader)
  }
  const step = (reader: Reader, elapsed: number, obstacles: CollisionActor[]) => {
    if (reader.phase !== 'walking') {
      reader.remaining -= elapsed
      if (reader.remaining <= 0) {
        if (reader.phase === 'waiting') {
          placeReader(reader, obstacles)
          reader.phase = 'walking'
          planWalk(reader)
        } else resume(reader)
      }
      return
    }
    if (!landscape) return
    const scale = trailGroundScale(reader.position.y, landscape)
    const radius = 12 * scale
    const replacement = replanOccupiedWaypoints(
      reader.id,
      reader.position,
      reader.waypoints,
      radius,
      obstacles,
      bounds(),
      reader.spacing.yieldingTo,
    )
    if (replacement?.length) setRoad(reader, replacement)
    const target = reader.waypoints[0] ?? reader.stopPoint
    const distance = Math.hypot(target.x - reader.position.x, target.y - reader.position.y)
    const budget = (reader.speed * elapsed * scale) / 1000
    const fraction = Math.min(1, budget / Math.max(1, distance))
    const next = {
      x: reader.position.x + (target.x - reader.position.x) * fraction,
      y: reader.position.y + (target.y - reader.position.y) * fraction,
    }
    if (
      spacedStep(
        reader.id,
        reader.position,
        next,
        radius,
        budget,
        reader.spacing,
        obstacles,
        bounds(),
        true,
        target,
        true,
      )
    )
      reader.position = next
    reader.position = spacedPosition(reader.position, reader.spacing)
    reader.spacing.lane.x -= reader.spacing.offset.x
    reader.spacing.lane.y -= reader.spacing.offset.y
    reader.spacing.offset = { x: 0, y: 0 }
    if (Math.hypot(target.x - reader.position.x, target.y - reader.position.y) < 2)
      reader.waypoints.shift()
    const remaining = roadLength(reader.position, reader.waypoints)
    const fractionDone = Math.max(0, Math.min(1, 1 - remaining / (reader.roadLength || 1)))
    reader.progress = reader.roadStart + (reader.roadEnd - reader.roadStart) * fractionDone
    const body = obstacles.find(other => other.id === reader.id)
    if (body)
      Object.assign(body, reader.position, {
        radius: 12 * trailGroundScale(reader.position.y, landscape),
      })
    else obstacles.push({ ...reader.position, id: reader.id, kind: 'traveler', radius })
    if (!reader.waypoints.length) {
      reader.progress = reader.roadEnd
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
          reader.lastStep = { ...reader.position }
        }
      render()
      schedule()
    },
    { signal },
  )
  signal.addEventListener('abort', pause, { once: true })

  return {
    pause,
    update(
      nextRoutes: JourneyRoute[],
      nextLandscape: Landscape,
      nextSeed: number,
      count: number,
      nextSolids: readonly CollisionActor[],
    ) {
      pause()
      const reflow =
        landscape?.width !== nextLandscape.width ||
        landscape?.height !== nextLandscape.height ||
        landscape?.camp.x !== nextLandscape.camp.x ||
        landscape?.camp.y !== nextLandscape.camp.y ||
        seed !== nextSeed
      const existing = new Set(readers)
      routes = nextRoutes
      landscape = nextLandscape
      solids = nextSolids
      if (seed !== nextSeed) {
        seed = nextSeed
        readers = []
        group.replaceChildren()
        tracks.replaceChildren()
        renderedAt = -Infinity
      }
      if (readers.length > count)
        for (const reader of readers.splice(count)) {
          reader.element.remove()
          reader.tracks.remove()
        }
      const plannedRoutes = Array.from({ length: count }, (_, index) =>
        pathsFor(index, `reader-${index}`),
      )
      while (readers.length < count) {
        const index = readers.length
        const random = trailRandom(nextSeed ^ (0x381c72ad + index * 0x27d4eb2d))
        const readerRoutes = plannedRoutes[index]
        const route =
          readerRoutes.find(route => route.id === ['grove', 'ridge', 'meadow'][index]) ??
          readerRoutes[index]
        if (!route) break
        group.insertAdjacentHTML('beforeend', trailMapReader(index))
        const element = group.querySelector<SVGGElement>(`[data-reader="${index}"]`)
        const facing = element?.querySelector<SVGGElement>('.trail-reader-facing')
        const pole = element?.querySelector<SVGGElement>('.trail-reader-pole')
        if (!element || !facing || !pole) throw new Error('A map reader needs its walking pose')
        const start = clearingStart(route)
        const progress = start + (1 - start) * (0.015 + index * 0.035 + random() * 0.03)
        const position = positionAt(route, progress)
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
          lastStep: position,
          step: 0,
          route: route.id,
          routes: readerRoutes,
          phase: 'waiting',
          stop: 'middle',
          position,
          waypoints: [],
          stopPoint: position,
          roadStart: progress,
          roadEnd: progress,
          roadLength: 0,
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
      const obstacles = [...readActors(scene, true), ...readTrunks(scene)]
      for (const [index, reader] of readers.entries()) {
        reader.routes = plannedRoutes[index]
        if (reflow || !existing.has(reader)) placeReader(reader, obstacles)
        if (reader.phase !== 'reading') planWalk(reader)
        else reader.stopPoint = { ...reader.position }
        for (const stamp of reader.impressions) placeImpression(reader, stamp)
      }
      render()
      schedule()
    },
  }
}
