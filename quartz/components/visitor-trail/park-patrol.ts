import type { JourneyRoute } from './hiker-journey'
import {
  actorSpacing,
  circleClearance,
  placeOnReflow,
  publishActor,
  readActors,
  readTrunks,
  spacedPosition,
  spacedStep,
  type ActorSpacing,
} from './actor-spacing'
import { campPatrolRoute } from './camp-routine'
import { footprint, footprintTransform } from './footprints'
import { pondDistance, trailGroundScale, type Landscape, type Point } from './landscape'
import { pixels } from './pixels'

type Ranger = {
  id: string
  element: SVGGElement
  facing: SVGGElement
  position: Point
  spacing: ActorSpacing
  route: Point[]
  goal: Stop
  site: number
  phase: 'waiting' | 'patrolling' | 'checking'
  remaining: number
  lastStep: Point
  stride: number
}
type Stop = Point & { id: string }

function rider() {
  return `<g class="trail-ranger-facing"><g transform="translate(-18 -20)">
    ${pixels(['................MM......', '...............MBMM.....', 'TT..BBBBBBBBB..BBBE.....', 'T..BBBBBBBBBBBBBBBNN....', '...BBBBSSSBBBBBBBB......', '...BBBSSSSSBBBBB........', '....BBBBBBBBBBB.........', '....BBB.....BBB.........'], { B: 'trail-park-horse', M: 'trail-park-mane', S: 'trail-park-saddle', E: 'trail-animal-eye', N: 'trail-park-muzzle', T: 'trail-park-mane' }, 1.5)}
    <g transform="translate(0 12)"><g class="trail-park-front-leg">${pixels(['..............B.B.......', '..............B.B.......', '..............M.M.......'], { B: 'trail-park-horse', M: 'trail-park-mane' }, 1.5)}</g><g class="trail-park-back-leg">${pixels(['....B.B.................', '....B.B.................', '....M.M.................'], { B: 'trail-park-horse', M: 'trail-park-mane' }, 1.5)}</g></g>
    <g transform="translate(11 -13)">${pixels(['..HH...', '.HHHHH.', '..SSS..', '.CCCC..', '.CCCSS.', '..DD...', '..D.D..'], { H: 'trail-park-hat', S: 'trail-camper-skin', C: 'trail-park-uniform', D: 'trail-camper-trousers' }, 2)}</g><path class="trail-park-rein" d="M22 0 30 6 33 7"/>
    </g></g><text class="trail-park-check" y="-39" text-anchor="middle" aria-hidden="true">✓</text>`
}

export function mountParkPatrol(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const layer = scene.querySelector<SVGGElement>('.trail-park-patrol')
  const tracks = scene.querySelector<SVGGElement>('.trail-ranger-tracks')
  if (!view || !layer || !tracks) throw new Error('Park patrol needs its trail layers')
  const reduced = view.matchMedia('(prefers-reduced-motion: reduce)')
  let landscape: Landscape | null = null
  let routes: JourneyRoute[] = []
  let rangers: Ranger[] = []
  let stops: Stop[] = []
  let signature = ''
  let frame = 0
  let last = 0
  let elapsed = 0
  const bounds = () => ({
    left: 18,
    right: (landscape?.width ?? 36) - 18,
    top: (landscape?.treeline ?? 0) + 8,
    bottom: (landscape?.height ?? 36) - 18,
  })
  const inspectionStop = (site: number, index: number, id: string, position: Point): Stop => {
    const camp = stops[site]
    if (!camp || !landscape) return { ...position, id: camp?.id ?? '' }
    const obstacles = [...readActors(scene, true), ...readTrunks(scene)].filter(
      body => body.id !== id,
    )
    const limits = bounds()
    const reach = 118 + Math.floor(index / 2) * 42
    const preferred = index % 2 ? 0 : Math.PI
    let fallback: Stop | null = null
    for (const angle of [preferred, preferred + Math.PI, Math.PI / 2, -Math.PI / 2]) {
      const point = {
        x: camp.x + Math.cos(angle) * reach,
        y: camp.y + 32 + Math.sin(angle) * reach * 0.45,
      }
      if (
        point.x < limits.left ||
        point.x > limits.right ||
        point.y < limits.top ||
        point.y > limits.bottom ||
        pondDistance(point, landscape.pond) < 1.7
      )
        continue
      const radius = 18 * trailGroundScale(point.y, landscape)
      if (obstacles.some(body => body.kind === 'trunk' && circleClearance(point, radius, body) < 4))
        continue
      fallback ??= { ...point, id: camp.id }
      if (obstacles.every(body => circleClearance(point, radius, body) >= 4))
        return { ...point, id: camp.id }
    }
    return fallback ?? { ...position, id: camp.id }
  }
  const render = (ranger: Ranger) => {
    if (!landscape) return
    const point = spacedPosition(ranger.position, ranger.spacing)
    const scale = trailGroundScale(point.y, landscape)
    publishActor(
      ranger.element,
      ranger.id,
      point,
      scale,
      ranger.spacing,
      ranger.phase !== 'waiting',
      elapsed,
      18,
    )
    ranger.element.dataset.phase = ranger.phase
    ranger.element.dataset.motion = reduced.matches
      ? 'still'
      : ranger.spacing.yieldingTo
        ? 'yielding'
        : ranger.phase
    ranger.element.dataset.moveSpeed = '36'
    ranger.element.dataset.stride = String(ranger.stride % 2)
    ranger.element.dataset.campSite = stops[ranger.site]?.id ?? ''
    const next = ranger.route[0] ?? ranger.goal
    ranger.element.dataset.targetX = next?.x.toFixed(2) ?? ''
    ranger.element.dataset.targetY = next?.y.toFixed(2) ?? ''
    ranger.element.dataset.waypointsRemaining = String(ranger.route.length)
    ranger.facing.setAttribute('transform', `scale(${next && next.x < point.x ? -1 : 1} 1)`)
    if (ranger.phase === 'waiting') ranger.element.style.visibility = 'hidden'
  }
  const visit = (ranger: Ranger) => {
    if (!landscape || !stops.length) return
    ranger.phase = 'patrolling'
    ranger.goal = inspectionStop(
      ranger.site,
      Number(ranger.element.dataset.parkRanger),
      ranger.id,
      ranger.position,
    )
    ranger.route = campPatrolRoute(
      ranger.position,
      ranger.goal,
      routes,
      landscape,
      readTrunks(scene),
    )
  }
  const tick = (time: number) => {
    frame = 0
    if (host.hidden || document.hidden || signal.aborted || reduced.matches || !landscape) return
    if (!last) last = time
    if (time - last >= 80) {
      const delta = Math.min(160, time - last)
      last = time
      elapsed += delta
      const obstacles = [...readActors(scene), ...readTrunks(scene)]
      for (const ranger of rangers) {
        if (ranger.phase === 'waiting' || ranger.phase === 'checking') {
          ranger.remaining -= delta
          if (ranger.remaining <= 0) {
            if (ranger.phase === 'checking') ranger.site = (ranger.site + 1) % stops.length
            visit(ranger)
          }
        }
        if (ranger.phase === 'patrolling') {
          const target = ranger.route[0] ?? ranger.goal
          const distance = Math.hypot(target.x - ranger.position.x, target.y - ranger.position.y)
          const budget = delta * 0.024
          const ratio = Math.min(1, budget / Math.max(1, distance))
          const next = {
            x: ranger.position.x + (target.x - ranger.position.x) * ratio,
            y: ranger.position.y + (target.y - ranger.position.y) * ratio,
          }
          const scale = trailGroundScale(ranger.position.y, landscape)
          if (
            spacedStep(
              ranger.id,
              ranger.position,
              next,
              18 * scale,
              budget * 1.5,
              ranger.spacing,
              obstacles,
              bounds(),
              true,
              target,
              true,
            )
          )
            ranger.position = next
          ranger.position = spacedPosition(ranger.position, ranger.spacing)
          ranger.spacing.lane.x -= ranger.spacing.offset.x
          ranger.spacing.lane.y -= ranger.spacing.offset.y
          ranger.spacing.offset = { x: 0, y: 0 }
          if (
            Math.hypot(
              ranger.position.x - ranger.lastStep.x,
              ranger.position.y - ranger.lastStep.y,
            ) >= 10
          ) {
            tracks.insertAdjacentHTML(
              'beforeend',
              `<g data-footprint-owner="${ranger.id}">${footprint('hoof', `${footprintTransform(ranger.position.x, ranger.position.y, Math.atan2(ranger.position.y - ranger.lastStep.y, ranger.position.x - ranger.lastStep.x), ranger.stride++ % 2 ? 1 : -1)} scale(${scale.toFixed(3)})`)}</g>`,
            )
            while (tracks.childElementCount > 100) tracks.firstElementChild?.remove()
            ranger.lastStep = { ...ranger.position }
          }
          if (Math.hypot(target.x - ranger.position.x, target.y - ranger.position.y) < 2) {
            if (ranger.route.length > 1) ranger.route.shift()
            else {
              ranger.phase = 'checking'
              ranger.remaining = 5200 + ranger.site * 700
              ranger.element.dataset.inspection =
                scene.querySelector<SVGGElement>(
                  `.trail-camp[data-camp-site="${stops[ranger.site].id}"]`,
                )?.dataset.fireState === 'out'
                  ? 'fire-out'
                  : 'camp-in-use'
              ranger.element.dataset.inspections = String(
                Number(ranger.element.dataset.inspections ?? 0) + 1,
              )
            }
          }
        }
        render(ranger)
        const body = obstacles.find(other => other.id === ranger.id)
        if (body) Object.assign(body, ranger.position)
      }
    }
    frame = view.requestAnimationFrame(tick)
  }
  const pause = () => {
    view.cancelAnimationFrame(frame)
    frame = 0
    last = 0
    scene.dataset.parkMotion = 'paused'
  }
  const resume = () => {
    pause()
    if (host.hidden || document.hidden || signal.aborted || !landscape) return
    scene.dataset.parkMotion = reduced.matches ? 'still' : 'running'
    if (!reduced.matches && rangers.length) frame = view.requestAnimationFrame(tick)
  }
  signal.addEventListener('abort', pause, { once: true })
  host.addEventListener('trail:hide', pause, { signal })
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : resume()), {
    signal,
  })
  reduced.addEventListener('change', resume, { signal })
  return {
    pause,
    update(current: Landscape, seed: number, paths: JourneyRoute[], count: number) {
      const camps = [...scene.querySelectorAll<SVGGElement>('.trail-camp[data-camp-site]')].sort(
        (a, b) => Number(a.dataset.campSite) - Number(b.dataset.campSite),
      )
      const next = `${seed}|${current.width}|${current.height}|${count}|${camps.map(camp => `${camp.dataset.groundX},${camp.dataset.groundY}`).join('|')}`
      const rerouted = routes !== paths
      landscape = current
      routes = paths
      if (next === signature && !rerouted) return resume()
      pause()
      signature = next
      for (const ranger of rangers) ranger.element.remove()
      tracks.replaceChildren()
      stops = camps.map(camp => ({
        id: camp.dataset.campSite ?? '0',
        x: Number(camp.dataset.groundX),
        y: Number(camp.dataset.groundY),
      }))
      rangers = []
      for (let index = 0; index < count && stops.length; index++) {
        const road = paths[index % paths.length]
        if (!road) continue
        const position = road.point(0)
        const element = document.createElementNS('http://www.w3.org/2000/svg', 'g')
        element.classList.add('trail-park-ranger')
        element.dataset.parkRanger = String(index)
        element.innerHTML = rider()
        layer.append(element)
        const facing = element.querySelector<SVGGElement>('.trail-ranger-facing')
        if (!facing) continue
        const spacing = actorSpacing()
        const id = `park-ranger-${index}`
        placeOnReflow(
          element,
          id,
          position,
          point => 18 * trailGroundScale(point.y, current),
          spacing,
          [...readActors(scene), ...readTrunks(scene)],
          bounds(),
          point => pondDistance(point, current.pond) > 1.7,
        )
        const placed = spacedPosition(position, spacing)
        spacing.offset = { x: 0, y: 0 }
        spacing.lane = { x: 0, y: 0 }
        const ranger: Ranger = {
          id,
          element,
          facing,
          position: placed,
          spacing,
          route: [],
          goal: inspectionStop(index % stops.length, index, id, position),
          site: index % stops.length,
          phase: reduced.matches ? 'checking' : 'waiting',
          remaining: reduced.matches ? Infinity : road.readyAfter + index * 1500,
          lastStep: placed,
          stride: 0,
        }
        if (reduced.matches) {
          ranger.position = ranger.goal
          placeOnReflow(
            element,
            id,
            ranger.position,
            point => 18 * trailGroundScale(point.y, current),
            spacing,
            [...readActors(scene), ...readTrunks(scene)],
            bounds(),
            point => pondDistance(point, current.pond) > 1.7,
          )
        }
        rangers.push(ranger)
        render(ranger)
      }
      resume()
    },
  }
}
