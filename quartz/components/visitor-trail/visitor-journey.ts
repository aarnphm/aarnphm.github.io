import type { JourneyRoute } from './hiker-journey'
import type { CampSettings } from './scene-settings'
import { svgPathGeometry } from '../../util/svg-geometry'
import {
  actorSpacing,
  circleClearance,
  placeOnReflow,
  publishActor,
  readActors,
  readStationaryActors,
  readTrunks,
  spacedPosition,
  spacedStep,
  type ActorSpacing,
} from './actor-spacing'
import { campSites, type CampSite } from './camp-layout'
import { campDeparture, campPatrolRoute } from './camp-routine'
import { campFigure } from './campsite'
import { bootImpression, footprintTransform } from './footprints'
import { onDryLand, trailGroundScale, type Landscape, type Point } from './landscape'
import { detourTrail, replanOccupiedWaypoints } from './obstacle-route'
import { anonymousVisitor, visitorHash } from './visitor-name'

export type TrailVisitor = { id: string; camp: number; own: boolean }
type Phase = 'waiting' | 'arriving' | 'attending' | 'departing'
type Visitor = TrailVisitor & {
  actorId: string
  element: SVGGElement
  art: SVGGElement
  tracks: SVGGElement
  button: HTMLButtonElement
  popover: HTMLDivElement
  present: boolean
  site: number
  phase: Phase
  position: Point
  goal: Point
  road: Point[]
  spacing: ActorSpacing
  lastStep: Point
  step: number
  route: JourneyRoute
}

export function mountVisitorJourneys(
  scene: HTMLElement,
  host: HTMLElement,
  signal: AbortSignal,
  onStructure: () => void,
) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const canvas = scene.querySelector<SVGSVGElement>('.trail-canvas')
  const layer = scene.querySelector<SVGGElement>('.trail-visitors')
  const tracks = scene.querySelector<SVGGElement>('.trail-visitor-tracks')
  const labels = scene.querySelector<HTMLElement>('.trail-visitor-labels')
  if (!view || !canvas || !layer || !tracks || !labels)
    throw new Error('Visitors need their trail layers')
  const reduced = view.matchMedia('(prefers-reduced-motion: reduce)')
  const visitors = new Map<string, Visitor>()
  let roster: TrailVisitor[] = []
  let routes: JourneyRoute[] = []
  let sites: CampSite[] = []
  let landscape: Landscape | undefined
  let seed: number | undefined
  let frame = 0
  let timer = 0
  let previous = 0
  let painted = -Infinity
  let paused = true

  const bounds = () => ({
    left: 18,
    right: (landscape?.width ?? 36) - 18,
    top: (landscape?.alpine.top ?? 0) + 4,
    bottom: (landscape?.height ?? 36) - 18,
  })
  const scaleAt = (point: Point) => {
    if (!landscape) return 1
    if (point.y >= landscape.treeline) return trailGroundScale(point.y, landscape)
    const depth = (point.y - landscape.alpine.top) / (landscape.treeline - landscape.alpine.top)
    return 0.18 + 0.17 * Math.max(0, Math.min(1, depth))
  }
  const available = (site: number) =>
    ['staying', 'sleeping'].includes(
      scene.querySelector<SVGGElement>(`.trail-camp[data-camp-site="${site}"]`)?.dataset
        .campPhase ?? '',
    )
  const bodies = (visitor: Visitor) => [
    ...readTrunks(scene),
    ...readStationaryActors(scene, visitor.actorId),
    ...readActors(scene).filter(body => body.kind === 'pet'),
  ]
  const mountainRoad = (from: Point) => {
    let path: SVGPathElement | undefined
    let gap = Infinity
    for (const candidate of scene.querySelectorAll<SVGPathElement>('[data-mountain-track]')) {
      const point = svgPathGeometry(candidate).getPointAtLength(0)
      const distance = Math.hypot(point.x - from.x, point.y - from.y)
      if (distance < gap) {
        gap = distance
        path = candidate
      }
    }
    if (!path) return [from]
    const geometry = svgPathGeometry(path)
    const length = geometry.getTotalLength() * 0.6
    const steps = Math.max(1, Math.ceil(length / 12))
    return Array.from({ length: steps + 1 }, (_, index) => {
      const point = geometry.getPointAtLength((length * index) / steps)
      return { x: point.x, y: point.y }
    })
  }
  const chooseSeat = (visitor: Visitor) => {
    if (!landscape) return false
    const requested = visitor.camp % sites.length
    const site =
      sites.find(site => site.id === requested && available(site.id)) ??
      sites.find(site => available(site.id))
    if (!site) return false
    const occupied = [...readActors(scene), ...readTrunks(scene)].filter(
      body => body.id !== visitor.actorId,
    )
    for (const other of visitors.values())
      if (other !== visitor && other.present && other.phase !== 'waiting')
        occupied.push({
          ...other.goal,
          radius: 12 * scaleAt(other.goal),
          id: other.actorId,
          kind: 'traveler',
        })
    const offset = visitorHash(visitor.id) % 12
    for (let ring = 0; ring < 3; ring++)
      for (let slot = 0; slot < 12; slot++) {
        const angle = ((slot + offset) * Math.PI * 2) / 12
        const point = onDryLand(
          {
            x: site.x + Math.cos(angle) * (72 + ring * 24),
            y: site.y + 22 + Math.sin(angle) * (32 + ring * 16),
          },
          landscape,
          1.8,
        )
        const radius = 12 * scaleAt(point)
        if (
          point.x >= bounds().left + radius &&
          point.x <= bounds().right - radius &&
          point.y >= landscape.meadowTop + 20 &&
          point.y <= bounds().bottom - radius &&
          occupied.every(body => circleClearance(point, radius, body) >= 3)
        ) {
          visitor.site = site.id
          visitor.goal = point
          return true
        }
      }
    return false
  }
  const plan = (visitor: Visitor, points: Point[]) => {
    const planned = detourTrail([visitor.position, ...points], 12, bodies(visitor), bounds())
    visitor.road = planned?.points.slice(1) ?? []
    return visitor.road.length > 0
  }
  const arrive = (visitor: Visitor) => {
    if (!landscape || !chooseSeat(visitor)) {
      visitor.phase = 'waiting'
      visitor.road = []
      return
    }
    const mountain = visitor.position.y < landscape.treeline ? mountainRoad(visitor.position) : []
    let nearest = 0
    let distance = Infinity
    for (const [index, point] of mountain.entries()) {
      const current = Math.hypot(point.x - visitor.position.x, point.y - visitor.position.y)
      if (current < distance) {
        distance = current
        nearest = index
      }
    }
    const descending = mountain.slice(0, nearest + 1).reverse()
    const start = descending.at(-1) ?? visitor.position
    const road = campPatrolRoute(start, visitor.goal, routes, landscape, bodies(visitor))
    visitor.phase = plan(visitor, [...descending, ...road, visitor.goal]) ? 'arriving' : 'waiting'
  }
  const leave = (visitor: Visitor) => {
    if (!landscape) return
    if (reduced.matches) return remove(visitor)
    const road = campDeparture(visitor.position, routes, landscape, bodies(visitor))
    const mountain = mountainRoad(road.at(-1) ?? visitor.position)
    visitor.goal = mountain.at(-1) ?? visitor.position
    visitor.phase = 'departing'
    plan(visitor, [...road, ...mountain])
    visitor.element.dataset.leftAt = String(Date.now())
  }
  const positionPopover = (visitor: Visitor) => {
    if (!visitor.popover.matches(':popover-open')) return
    const rect = visitor.button.getBoundingClientRect()
    const popup = visitor.popover.getBoundingClientRect()
    visitor.popover.style.left = `${Math.max(8, Math.min(view.innerWidth - popup.width - 8, rect.left + rect.width / 2 - popup.width / 2))}px`
    visitor.popover.style.top = `${Math.max(8, rect.top - popup.height - 8)}px`
  }
  const render = (visitor: Visitor, time: number) => {
    const moving = visitor.phase === 'arriving' || visitor.phase === 'departing'
    const scale = scaleAt(visitor.position)
    visitor.element.dataset.phase = visitor.phase
    visitor.element.dataset.present = String(visitor.present)
    visitor.element.dataset.campSite = String(visitor.site)
    visitor.element.dataset.region =
      visitor.position.y < (landscape?.treeline ?? 0) ? 'mountain' : 'valley'
    visitor.element.dataset.motion = reduced.matches
      ? 'still'
      : paused
        ? 'paused'
        : visitor.spacing.yieldingTo
          ? 'yielding'
          : moving
            ? 'active'
            : 'still'
    visitor.element.dataset.moveSpeed = (28 * scale).toFixed(4)
    visitor.element.dataset.waypointsRemaining = String(visitor.road.length)
    const target = visitor.road[0] ?? visitor.position
    visitor.element.dataset.targetX = target.x.toFixed(2)
    visitor.element.dataset.targetY = target.y.toFixed(2)
    publishActor(
      visitor.element,
      visitor.actorId,
      visitor.position,
      scale,
      visitor.spacing,
      true,
      time,
    )
    const pose = visitor.phase === 'attending' ? 'sit' : 'walk'
    const facing =
      pose === 'sit'
        ? (sites[visitor.site]?.x ?? target.x) < visitor.position.x
        : target.x < visitor.position.x
    if (visitor.element.dataset.pose !== pose) {
      visitor.element.dataset.pose = pose
      visitor.art.innerHTML = campFigure(pose, pose === 'walk')
    }
    visitor.art.setAttribute(
      'transform',
      `translate(${facing ? 4 : -4} -12) scale(${facing ? -1 : 1} 1)`,
    )
    visitor.button.hidden = visitor.element.dataset.collisionActive !== 'true'
    visitor.button.style.left = `${visitor.position.x}px`
    visitor.button.style.top = `${visitor.position.y - 12 * scale}px`
    visitor.button.dataset.phase = visitor.phase
    visitor.button.setAttribute(
      'aria-label',
      `${anonymousVisitor(visitor.id)}${visitor.own ? ' (you)' : ''}, ${pose === 'sit' ? 'at' : visitor.phase === 'departing' ? 'leaving' : 'walking to'} camp ${visitor.site + 1}`,
    )
    positionPopover(visitor)
  }
  const remove = (visitor: Visitor) => {
    visitor.popover.hidePopover()
    visitor.popover.remove()
    visitor.button.remove()
    visitor.element.remove()
    visitor.tracks.remove()
    visitors.delete(visitor.id)
    onStructure()
  }
  const add = (peer: TrailVisitor) => {
    if (!landscape || !routes.length) return
    const route = routes[visitorHash(peer.id) % routes.length]
    const mountain = mountainRoad(route.point(0))
    const element = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    element.classList.add('trail-visitor')
    element.dataset.trailVisitor = peer.id
    element.dataset.own = String(peer.own)
    element.dataset.route = route.id
    const art = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    art.classList.add('trail-visitor-art')
    element.append(art)
    layer.append(element)
    const ownTracks = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    ownTracks.dataset.visitorTracks = peer.id
    tracks.append(ownTracks)
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'trail-visitor-label'
    button.dataset.visitorLabel = peer.id
    const popover = document.createElement('div')
    popover.className = 'trail-visitor-popover'
    popover.popover = 'auto'
    popover.id = `trail-visitor-${crypto.randomUUID()}`
    const name = document.createElement('span')
    name.textContent = anonymousVisitor(peer.id)
    popover.append(name)
    popover.setAttribute('role', 'tooltip')
    button.setAttribute('aria-describedby', popover.id)
    button.setAttribute('aria-expanded', 'false')
    labels.append(button, popover)
    const position = mountain.at(-1) ?? route.point(0)
    const visitor: Visitor = {
      ...peer,
      actorId: `visitor-${peer.id}`,
      element,
      art,
      tracks: ownTracks,
      button,
      popover,
      present: true,
      site: peer.camp % sites.length,
      phase: 'waiting',
      position,
      goal: position,
      road: [],
      spacing: actorSpacing(),
      lastStep: position,
      step: 0,
      route,
    }
    const reveal = () => {
      if (host.hidden || visitor.button.hidden) return
      popover.showPopover()
      button.setAttribute('aria-expanded', 'true')
      positionPopover(visitor)
    }
    const conceal = () => {
      if (button.matches(':focus-visible')) return
      popover.hidePopover()
      button.setAttribute('aria-expanded', 'false')
    }
    button.addEventListener('pointerenter', reveal, { signal })
    button.addEventListener(
      'pointerleave',
      event => {
        if (event.pointerType !== 'touch') conceal()
      },
      { signal },
    )
    button.addEventListener('focus', reveal, { signal })
    button.addEventListener('blur', conceal, { signal })
    button.addEventListener('click', reveal, { signal })
    button.addEventListener(
      'keydown',
      event => {
        if (event.key !== 'Escape' || !popover.matches(':popover-open')) return
        event.preventDefault()
        event.stopPropagation()
        popover.hidePopover()
        button.setAttribute('aria-expanded', 'false')
      },
      { signal },
    )
    popover.addEventListener(
      'toggle',
      () => button.setAttribute('aria-expanded', String(popover.matches(':popover-open'))),
      { signal },
    )
    visitors.set(peer.id, visitor)
    placeOnReflow(
      element,
      visitor.actorId,
      position,
      point => 12 * scaleAt(point),
      visitor.spacing,
      [...readActors(scene), ...readTrunks(scene)],
      bounds(),
    )
    visitor.position = spacedPosition(position, visitor.spacing)
    visitor.spacing.offset = { x: 0, y: 0 }
    visitor.spacing.lane = { x: 0, y: 0 }
    visitor.lastStep = { ...visitor.position }
    arrive(visitor)
    render(visitor, view.performance.now())
    onStructure()
  }
  const sync = () => {
    if (signal.aborted || !landscape || !routes.length || !sites.length) return
    const peers = new Map(roster.map(peer => [peer.id, peer]))
    for (const visitor of visitors.values()) {
      const peer = peers.get(visitor.id)
      if (!peer) {
        if (visitor.present) {
          visitor.present = false
          leave(visitor)
          if (visitors.has(visitor.id)) render(visitor, view.performance.now())
        }
      } else {
        const changed = peer.camp !== visitor.camp || !visitor.present
        visitor.present = true
        visitor.camp = peer.camp
        if (changed) arrive(visitor)
      }
    }
    for (const peer of roster) if (!visitors.has(peer.id)) add(peer)
    schedule()
  }
  const pause = () => {
    paused = true
    view.cancelAnimationFrame(frame)
    view.clearTimeout(timer)
    frame = timer = previous = 0
    for (const visitor of visitors.values()) {
      visitor.element.dataset.motion = reduced.matches ? 'still' : 'paused'
      visitor.popover.hidePopover()
    }
    scene.dataset.visitorMotion = 'paused'
  }
  const step = (visitor: Visitor, delta: number) => {
    if (!landscape || !['arriving', 'departing'].includes(visitor.phase)) return
    const obstacles = [...readActors(scene), ...readTrunks(scene)]
    const radius = 12 * scaleAt(visitor.position)
    const replacement = replanOccupiedWaypoints(
      visitor.actorId,
      visitor.position,
      visitor.road,
      radius,
      obstacles,
      bounds(),
      visitor.spacing.yieldingTo,
    )
    if (replacement?.length) {
      visitor.road = replacement
      visitor.goal = replacement.at(-1) ?? visitor.goal
    }
    if (!visitor.road.length) {
      if (visitor.present) arrive(visitor)
      else leave(visitor)
      return
    }
    const target = visitor.road[0]
    const distance = Math.hypot(target.x - visitor.position.x, target.y - visitor.position.y)
    const budget = (28 * scaleAt(visitor.position) * delta) / 1000
    const fraction = Math.min(1, budget / Math.max(1, distance))
    const next = {
      x: visitor.position.x + (target.x - visitor.position.x) * fraction,
      y: visitor.position.y + (target.y - visitor.position.y) * fraction,
    }
    if (
      spacedStep(
        visitor.actorId,
        visitor.position,
        next,
        radius,
        budget,
        visitor.spacing,
        obstacles,
        bounds(),
        true,
        target,
        true,
      )
    )
      visitor.position = next
    visitor.position = spacedPosition(visitor.position, visitor.spacing)
    visitor.spacing.lane.x -= visitor.spacing.offset.x
    visitor.spacing.lane.y -= visitor.spacing.offset.y
    visitor.spacing.offset = { x: 0, y: 0 }
    if (
      Math.hypot(
        visitor.position.x - visitor.lastStep.x,
        visitor.position.y - visitor.lastStep.y,
      ) >=
      8 * scaleAt(visitor.position)
    ) {
      const mark = bootImpression(document)
      mark.dataset.footprintOwner = visitor.actorId
      mark.setAttribute(
        'transform',
        footprintTransform(
          visitor.position.x,
          visitor.position.y,
          Math.atan2(
            visitor.position.y - visitor.lastStep.y,
            visitor.position.x - visitor.lastStep.x,
          ),
          visitor.step++ % 2 ? -1 : 1,
        ) + ` scale(${scaleAt(visitor.position).toFixed(4)})`,
      )
      visitor.tracks.append(mark)
      while (visitor.tracks.childElementCount > 64) visitor.tracks.firstElementChild?.remove()
      visitor.lastStep = { ...visitor.position }
    }
    if (Math.hypot(target.x - visitor.position.x, target.y - visitor.position.y) < 2) {
      visitor.road.shift()
      if (!visitor.road.length) {
        if (!visitor.present) return remove(visitor)
        visitor.phase = 'attending'
        visitor.goal = { ...visitor.position }
        visitor.element.dataset.arrivedAt = String(Date.now())
      }
    }
  }
  const schedule = () => {
    if (
      paused ||
      signal.aborted ||
      host.hidden ||
      document.hidden ||
      reduced.matches ||
      frame ||
      timer ||
      !visitors.size
    )
      return
    if (
      [...visitors.values()].some(
        visitor =>
          visitor.road.length > 0 &&
          (visitor.phase === 'arriving' || visitor.phase === 'departing'),
      )
    )
      frame = view.requestAnimationFrame(tick)
    else timer = view.setTimeout(() => tick(view.performance.now()), 2000)
  }
  const tick = (time: number) => {
    frame = timer = 0
    if (paused || signal.aborted || host.hidden || document.hidden || reduced.matches)
      return pause()
    if (time - painted < 1000 / 24) return schedule()
    const delta = previous ? Math.min(160, time - previous) : 0
    previous = painted = time
    for (const visitor of visitors.values()) {
      if (
        visitor.present &&
        (visitor.phase === 'waiting' || (visitor.phase === 'attending' && !available(visitor.site)))
      )
        arrive(visitor)
      step(visitor, delta)
      if (visitors.has(visitor.id)) render(visitor, time)
    }
    scene.dataset.visitorMotion = [...visitors.values()].some(visitor =>
      ['arriving', 'departing'].includes(visitor.phase),
    )
      ? 'walking'
      : 'idle'
    schedule()
  }
  const resume = () => {
    paused = false
    previous = 0
    if (reduced.matches) {
      for (const visitor of visitors.values()) {
        if (!visitor.present) {
          remove(visitor)
          continue
        }
        if (!chooseSeat(visitor)) continue
        visitor.position = { ...visitor.goal }
        visitor.spacing = actorSpacing()
        visitor.phase = 'attending'
        visitor.road = []
        render(visitor, view.performance.now())
      }
      scene.dataset.visitorMotion = 'reduced'
      return
    }
    schedule()
  }
  host.addEventListener('trail:hide', pause, { signal })
  host.addEventListener('trail:show', resume, { signal })
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : resume()), {
    signal,
  })
  reduced.addEventListener(
    'change',
    () => {
      pause()
      resume()
    },
    { signal },
  )
  view.addEventListener(
    'scroll',
    () => {
      for (const visitor of visitors.values()) positionPopover(visitor)
    },
    { signal, passive: true },
  )
  signal.addEventListener(
    'abort',
    () => {
      pause()
      for (const visitor of visitors.values()) remove(visitor)
    },
    { once: true },
  )
  return {
    pause,
    update(
      nextRoutes: JourneyRoute[],
      nextLandscape: Landscape,
      nextSeed: number,
      settings: CampSettings,
    ) {
      const reflow =
        landscape?.width !== nextLandscape.width ||
        landscape?.height !== nextLandscape.height ||
        seed !== nextSeed
      pause()
      routes = nextRoutes
      landscape = nextLandscape
      seed = nextSeed
      sites = campSites(landscape, seed, settings)
      if (reflow) for (const visitor of visitors.values()) remove(visitor)
      else
        for (const visitor of visitors.values()) {
          if (visitor.phase === 'arriving') arrive(visitor)
          else if (visitor.phase === 'departing') leave(visitor)
        }
      sync()
      resume()
    },
    setVisitors(peers: TrailVisitor[]) {
      roster = peers
      sync()
      if (reduced.matches) resume()
    },
  }
}
