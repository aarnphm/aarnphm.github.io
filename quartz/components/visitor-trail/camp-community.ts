import type { JourneyRoute } from './hiker-journey'
import { subscribeVisitorTrailWeather } from '../../util/visitor-trail-weather'
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
} from './actor-spacing'
import { campSites, type CampSite } from './camp-layout'
import {
  campDeparture,
  campMayDepart,
  campMaySleep,
  campRoutine,
  type CampRoutine,
} from './camp-routine'
import { defaultCampSettings, type CampSettings } from './camp-settings'
import { campFigure, trailCampsite } from './campsite'
import { bootImpression, footprintTransform } from './footprints'
import {
  followTerrain,
  onDryLand,
  pondDistance,
  trailGroundScale,
  type Landscape,
  type Point,
} from './landscape'
import { detourTrail, replanOccupiedWaypoints } from './obstacle-route'
import { pixels } from './pixels'

type Person = {
  id: string
  site: number
  slot: number
  role: 'camper' | 'builder'
  element: SVGGElement
  position: Point
  spacing: ActorSpacing
  target: Point
  tent: number
  exitTent: number
  buildTime: number
  phase:
    | 'walking'
    | 'building'
    | 'attending'
    | 'sleeping'
    | 'packing'
    | 'dousing'
    | 'departing'
    | 'gone'
  carrying: boolean
  wakeAt: number
  departAt: number
  walked: number
  departureLength: number
  packingFrom: number
  lastStep: Point
  stride: number
  stalled: number
  remaining: number
  route: Point[]
  look: Point | null
  surpriseUntil: number
}
type Tent = { element: SVGGElement; position: Point; progress: number }

function tentArt() {
  return `<path class="trail-tent-ground" d="M-25 0h50v3h-50Z"/>
    <g class="trail-tent-canvas" transform="translate(-23.75 -22.5)">${pixels(
      [
        '.........R.........',
        '........RRR........',
        '.......RRRRR.......',
        '......RRRLLRR......',
        '.....RRRLLLDRR.....',
        '....RRRLLLLDDRR....',
        '...RRRLLLLLDDDRR...',
        '..RRRLLLLLLDDDDRR..',
        '.RRRLLLLLLLDDDDDRR.',
      ],
      { R: 'trail-tent-roof', L: 'trail-tent-light', D: 'trail-tent-door' },
      2.5,
    )}</g>
    <path class="trail-tent-poles" d="M-22 1 0-25 22 1M-28 3-21-2M21-2l7 5"/>
    <path class="trail-tent-sleeper" d="M-4-3h8v2h-8Z"/>
    <g class="trail-tent-lantern"><path class="trail-tent-lamp-pool" d="M12 1h16v2h4v2H8V3h4Z"/><path class="trail-tent-window" d="M2-13h3v3H2ZM5-10h3v3H5ZM8-7h3v3H8Z"/><g transform="translate(17 -12)">${pixels(['.HH.', 'H..H', 'HLLH', 'HLLH', '.HH.'], { H: 'trail-tent-lamp-frame', L: 'trail-tent-lamp' }, 1.5)}</g></g>`
}

export function mountCampCommunity(
  scene: HTMLElement,
  host: HTMLElement,
  signal: AbortSignal,
  onStructure: () => void,
) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const layer = scene.querySelector<SVGGElement>('.trail-community')
  const tracks = scene.querySelector<SVGGElement>('.trail-camp-tracks')
  const primary = scene.querySelector<SVGGElement>('.trail-campsite .trail-camp')
  if (!view || !layer || !tracks || !primary) throw new Error('Camps need their scene layers')
  const reduced = view.matchMedia('(prefers-reduced-motion: reduce)')
  let landscape: Landscape | null = null
  let settings = { ...defaultCampSettings }
  let seed = 0
  let sites: CampSite[] = []
  let fires: SVGGElement[] = []
  let people: Person[] = []
  let tents: Tent[][] = []
  let routines: CampRoutine[] = []
  let trailRoutes: JourneyRoute[] = []
  let forecastDay: boolean | null = null
  let frame = 0
  let last = 0
  let elapsed = 0
  let paused = true
  scene.addEventListener(
    'animationend',
    event => {
      if (
        event.animationName === 'trail-arrive' &&
        event.target instanceof view.SVGElement &&
        event.target.matches('.trail-extra-camp, .trail-tent')
      )
        event.target.dataset.arrived = 'true'
    },
    { signal },
  )
  const night = () =>
    settings.light === 'night' ||
    (settings.light === 'weather' &&
      (forecastDay === null
        ? new Date().getHours() < 6 || new Date().getHours() >= 20
        : !forecastDay))
  const bedtime = (site: number) => campMaySleep(routines[site], Date.now(), night())
  const bounds = (departing = false) => ({
    left: 18,
    right: (landscape?.width ?? 36) - 18,
    top: departing ? (landscape?.treeline ?? 0) + 8 : (landscape?.meadowTop ?? 0) + 20,
    bottom: (landscape?.height ?? 48) - 18,
  })
  const seat = (person: Person) => {
    const site = sites[person.site]
    const index = person.slot
    if (index < 2) return { x: site.x + (index ? 39 : -39), y: site.y + 19 }
    const ring = Math.floor((index - 2) / 4)
    const slot = (index - 2) % 4
    const point = {
      x: site.x + (slot % 2 ? -1 : 1) * (slot < 2 ? 76 + ring * 30 : 56 + ring * 30),
      y: site.y + (slot < 2 ? 18 : -10),
    }
    if (!landscape) return point
    const limits = bounds()
    return onDryLand(
      {
        x: Math.max(limits.left, Math.min(limits.right, point.x)),
        y: Math.max(limits.top, Math.min(limits.bottom, point.y)),
      },
      landscape,
      1.8,
    )
  }
  const renderPerson = (person: Person, active = true) => {
    if (!landscape) return
    const target = person.route[0] ?? person.target
    person.element.dataset.targetX = target.x.toFixed(2)
    person.element.dataset.targetY = target.y.toFixed(2)
    person.element.dataset.waypointsRemaining = String(person.route.length)
    const point = spacedPosition(person.position, person.spacing)
    const scale = trailGroundScale(point.y, landscape)
    if (
      person.element.dataset.phase === person.phase &&
      person.element.dataset.groundX === String(Number(point.x.toFixed(2))) &&
      person.element.dataset.groundY === String(Number(point.y.toFixed(2))) &&
      person.element.dataset.scale === scale.toFixed(4) &&
      person.element.dataset.collisionActive === String(active) &&
      person.element.dataset.spacing === person.spacing.status &&
      person.element.dataset.yieldingTo === (person.spacing.yieldingTo ?? '') &&
      person.element.dataset.yieldingThrough === (person.spacing.yieldingThrough ?? '')
    )
      return
    person.element.dataset.phase = person.phase
    person.element.dataset.campSite = String(person.site)
    person.element.dataset.stride = String(person.stride % 2)
    person.element.dataset.moveSpeed = '27'
    person.element.dataset.motion = person.spacing.status === 'yielding' ? 'yielding' : person.phase
    publishActor(person.element, person.id, point, scale, person.spacing, active, elapsed)
    person.element.style.visibility =
      person.phase === 'sleeping' || person.phase === 'gone' ? 'hidden' : ''
  }
  const lighting = () => {
    const dark = String(night())
    scene.dataset.campNight = dark
    for (const fire of fires) fire.dataset.night = dark
    for (let site = 0; site < tents.length; site++) {
      const resting = routines[site]?.phase === 'staying' || routines[site]?.phase === 'sleeping'
      for (const tent of tents[site])
        tent.element.dataset.lit = String(dark === 'true' && resting && tent.progress >= 0.95)
    }
  }
  const approachingPet = (): Point | null => {
    const pet = document.querySelector<HTMLElement>('.garden-pet:not([hidden])')
    const canvas = scene.querySelector<SVGSVGElement>('.trail-canvas')
    const matrix = canvas?.getScreenCTM()
    if (!pet || !matrix) return null
    const style = view.getComputedStyle(pet)
    if (style.visibility === 'hidden' || Number(style.opacity) === 0) return null
    const rect = pet.getBoundingClientRect()
    if (!rect.width || !rect.height) return null
    // Compare feet on the walking plane, rather than the top of the blob's hat.
    const point = new DOMPoint(
      rect.left + rect.width / 2,
      rect.top + rect.height * 0.9,
    ).matrixTransform(matrix.inverse())
    return { x: point.x, y: point.y }
  }
  const noticePet = (person: Person, pet: Point | null) => {
    const site = sites[person.site]
    const distance = pet ? Math.hypot(pet.x - site.x, pet.y - site.y) : Infinity
    const near = person.phase === 'attending' && distance < 140
    if (near && !person.look) person.surpriseUntil = elapsed + 1800
    person.look = near ? pet : null
    person.element.dataset.petDistance = Number.isFinite(distance) ? distance.toFixed(1) : ''
    person.element.dataset.petNotice = near
      ? !reduced.matches && elapsed < person.surpriseUntil
        ? 'surprised'
        : 'looking'
      : 'none'
  }
  const setPose = (person: Person) => {
    const pose =
      person.phase === 'building' || person.phase === 'packing' || person.phase === 'dousing'
        ? 'build'
        : person.phase === 'attending'
          ? 'sit'
          : 'walk'
    const target =
      person.look ?? (pose === 'sit' ? sites[person.site] : (person.route[0] ?? person.target))
    const facing = target.x < person.position.x ? 'left' : 'right'
    if (
      person.element.dataset.pose === pose &&
      person.element.dataset.facing === facing &&
      person.element.dataset.pack === String(person.carrying)
    )
      return
    person.element.dataset.pose = pose
    person.element.dataset.facing = facing
    person.element.dataset.pack = String(person.carrying)
    const art = person.element.querySelector('.trail-camp-person-art')
    if (art)
      art.innerHTML =
        facing === 'left'
          ? `<g transform="translate(8 0) scale(-1 1)">${campFigure(pose, person.carrying)}</g>`
          : campFigure(pose, person.carrying)
  }
  const travelTo = (person: Person, point: Point) => {
    person.target = point
    person.route = landscape ? followTerrain([person.position, point], landscape).slice(1) : [point]
    if (landscape) {
      const ownTent = tents[person.site][person.tent]?.element.dataset.campTent
      const exitTent = tents[person.site][person.exitTent]?.element.dataset.campTent
      const solids = [...readTrunks(scene), ...readStationaryActors(scene, person.id)].filter(
        solid => solid.id !== `tent-${ownTent}` && solid.id !== `tent-${exitTent}`,
      )
      const planned = detourTrail([person.position, ...person.route], 12, solids, bounds())
      if (planned) person.route = planned.points.slice(1)
    }
    person.stalled = 0
    person.remaining = Infinity
  }
  const placePerson = (person: Person, enterTent = false) => {
    if (!landscape) return
    const current = landscape
    const ownTent = enterTent ? tents[person.site][person.tent]?.element.dataset.campTent : null
    placeOnReflow(
      person.element,
      person.id,
      person.position,
      point => 12 * trailGroundScale(point.y, current),
      person.spacing,
      [...readActors(scene), ...readTrunks(scene)].filter(other => other.id !== `tent-${ownTent}`),
      bounds(),
      point => pondDistance(point, current.pond) > 1.7,
    )
    person.position = spacedPosition(person.position, person.spacing)
    person.spacing.offset = { x: 0, y: 0 }
    person.spacing.lane = { x: 0, y: 0 }
    person.lastStep = { ...person.position }
  }
  const nextTent = (person: Person) => {
    const candidates = tents[person.site]
    return candidates.findIndex(
      (tent, index) =>
        tent.progress < 1 &&
        !people.some(
          other =>
            other !== person &&
            other.site === person.site &&
            other.role === 'builder' &&
            other.tent === index &&
            other.phase !== 'attending',
        ),
    )
  }
  const sendToTent = (person: Person) => {
    const ready = tents[person.site].filter(tent => tent.progress >= 1)
    if (!ready.length) return false
    const index = person.slot
    person.tent = tents[person.site].indexOf(ready[index % ready.length])
    const tent = tents[person.site][person.tent]
    travelTo(person, { x: tent.position.x, y: tent.position.y + 9 })
    person.phase = 'walking'
    person.element.dataset.destination = 'sleep'
    return true
  }
  const phase = (site: number, next: CampRoutine['phase']) => {
    const routine = routines[site]
    if (routine.phase === next) return
    routine.phase = next
    routine.since = Date.now()
    fires[site].dataset.campPhase = next
    for (const tent of tents[site]) tent.element.dataset.campPhase = next
  }
  const nextPackingTent = (person: Person) =>
    tents[person.site].findIndex(
      (tent, index) =>
        tent.progress > 0 &&
        !people.some(
          other =>
            other.site === person.site &&
            other.tent === index &&
            (other.phase === 'sleeping' ||
              (other !== person &&
                (other.phase === 'packing' ||
                  (other.phase === 'walking' && other.element.dataset.destination === 'pack')))),
        ),
    )
  const assignPacking = (person: Person) => {
    person.tent = nextPackingTent(person)
    person.buildTime = 0
    person.phase = 'walking'
    const tent = tents[person.site][person.tent]
    person.packingFrom = tent?.progress ?? 0
    person.element.dataset.destination = tent ? 'pack' : 'fire'
    travelTo(person, tent ? { x: tent.position.x, y: tent.position.y + 17 } : seat(person))
  }
  const finishCamp = (site: number) => {
    for (const tent of tents[site]) {
      tent.progress = 0
      tent.element.dataset.progress = '0'
      tent.element.dataset.phase = 'packed'
      tent.element.dataset.occupied = 'false'
      tent.element.dataset.carried = 'true'
      tent.element.removeAttribute('data-camp-solid')
    }
    for (const person of people.filter(person => person.site === site)) {
      person.phase = 'gone'
      renderPerson(person, false)
    }
    fires[site].dataset.fireState = 'out'
    fires[site].style.setProperty('--trail-fire-strength', '0')
    phase(site, 'empty')
  }
  const packUp = (site: number) => {
    const routine = routines[site]
    if (!routine || !['staying', 'sleeping'].includes(routine.phase)) return
    phase(site, 'packing')
    for (const tent of tents[site]) {
      if (tent.progress === 0) {
        tent.element.removeAttribute('data-camp-solid')
        tent.element.dataset.carried = 'true'
      }
    }
    for (const person of people.filter(person => person.site === site)) {
      person.look = null
      person.element.dataset.petNotice = 'none'
      if (person.phase === 'sleeping') person.wakeAt = elapsed + person.slot * 900
      else assignPacking(person)
    }
    if (reduced.matches) {
      finishCamp(site)
    }
    lighting()
  }
  const extinguish = (site: number) => {
    phase(site, 'extinguishing')
    fires[site].dataset.fireState = 'dousing'
    for (const person of people.filter(person => person.site === site && person.phase !== 'gone')) {
      person.phase = 'walking'
      person.tent = -1
      person.carrying = true
      person.element.dataset.destination = person.slot === 0 ? 'douse' : 'fire'
      travelTo(person, seat(person))
    }
  }
  const depart = (site: number) => {
    if (!landscape) return
    phase(site, 'departing')
    fires[site].dataset.fireState = 'out'
    fires[site].style.setProperty('--trail-fire-strength', '0')
    for (const person of people.filter(person => person.site === site && person.phase !== 'gone')) {
      person.phase = 'departing'
      person.carrying = true
      person.departAt = elapsed + person.slot * 1500
      person.walked = 0
      person.route = campDeparture(person.position, trailRoutes, landscape, [
        ...readTrunks(scene),
        ...readStationaryActors(scene, person.id),
      ])
      person.target = person.route[person.route.length - 1] ?? person.position
      person.departureLength = person.route.reduce((length, point, index) => {
        const previous = index ? person.route[index - 1] : person.position
        return length + Math.hypot(point.x - previous.x, point.y - previous.y)
      }, 0)
      person.element.dataset.destination = 'hike'
      person.element.dataset.departureProgress = '0'
      person.remaining = Infinity
      person.stalled = 0
      setPose(person)
    }
  }
  const updateRoutines = () => {
    const now = Date.now()
    for (let site = 0; site < routines.length; site++) {
      const routine = routines[site]
      const group = people.filter(person => person.site === site)
      if (routine.phase === 'staying' || routine.phase === 'sleeping') {
        if (campMayDepart(routine, now, night()) && tents[site].every(tent => tent.progress === 1))
          packUp(site)
        else if (night() && group.length && group.every(person => person.phase === 'sleeping'))
          phase(site, 'sleeping')
        else if (!night() && routine.phase === 'sleeping') phase(site, 'staying')
      } else if (routine.phase === 'packing') {
        if (
          tents[site].every(tent => tent.progress === 0) &&
          group.every(person => person.phase !== 'sleeping')
        )
          extinguish(site)
      } else if (routine.phase === 'extinguishing') {
        const helper = group.find(person => person.phase === 'dousing')
        const strength = helper ? Math.max(0, 1 - helper.buildTime / 3200) : 1
        fires[site].style.setProperty('--trail-fire-strength', strength.toFixed(3))
        if (strength === 0) depart(site)
      } else if (routine.phase === 'departing') {
        if (group.every(person => person.phase === 'gone')) phase(site, 'empty')
      } else if (now - routine.since >= routine.absence) resetCamp(site)
    }
  }
  const updatePeople = (delta: number) => {
    if (!landscape) return
    const pet = approachingPet()
    const obstacles = [...readActors(scene), ...readTrunks(scene)]
    for (const person of people) {
      const routine = routines[person.site]
      if (person.phase === 'gone') continue
      if (person.phase === 'sleeping') {
        if (
          (!bedtime(person.site) && routine.phase !== 'packing') ||
          (routine.phase === 'packing' && elapsed >= person.wakeAt)
        ) {
          person.exitTent = person.tent
          person.phase = 'walking'
          placePerson(person, true)
          if (routine.phase === 'packing') assignPacking(person)
          else {
            travelTo(person, seat(person))
            person.element.dataset.destination = 'fire'
          }
        } else continue
      }
      if (
        person.phase === 'attending' &&
        routine.phase === 'packing' &&
        nextPackingTent(person) >= 0
      )
        assignPacking(person)
      if (
        person.phase === 'attending' &&
        bedtime(person.site) &&
        ['staying', 'sleeping'].includes(routine.phase)
      )
        sendToTent(person)
      if (person.phase === 'packing') {
        person.buildTime += delta
        const tent = tents[person.site][person.tent]
        if (tent) {
          tent.progress = Math.max(0, person.packingFrom * (1 - person.buildTime / 3000))
          tent.element.dataset.progress = tent.progress.toFixed(3)
          tent.element.dataset.phase = tent.progress ? 'packing' : 'packed'
          tent.element.style.setProperty('--tent-progress', String(tent.progress))
        }
        if (!tent || tent.progress === 0) {
          if (tent) {
            tent.element.removeAttribute('data-camp-solid')
            tent.element.dataset.carried = 'true'
          }
          person.carrying = true
          person.exitTent = person.tent
          assignPacking(person)
        }
      }
      if (person.phase === 'dousing') person.buildTime += delta
      if (person.phase === 'building') {
        person.buildTime += delta
        const tent = tents[person.site][person.tent]
        if (tent) {
          tent.progress = Math.min(1, person.buildTime / 4500)
          tent.element.dataset.progress = tent.progress.toFixed(3)
          tent.element.dataset.phase = tent.progress === 1 ? 'ready' : 'building'
          tent.element.style.setProperty('--tent-progress', String(Math.max(0.15, tent.progress)))
        }
        if (!tent || tent.progress >= 1) {
          person.exitTent = person.tent
          person.tent = nextTent(person)
          person.buildTime = 0
          person.phase = 'walking'
          if (person.tent >= 0) {
            const next = tents[person.site][person.tent]
            travelTo(person, { x: next.position.x, y: next.position.y + 17 })
            person.element.dataset.destination = 'build'
          } else {
            travelTo(person, seat(person))
            person.element.dataset.destination = 'fire'
          }
        }
      }
      if (
        person.phase === 'walking' ||
        (person.phase === 'departing' && elapsed >= person.departAt)
      ) {
        const before = { ...person.position }
        const ownTent = tents[person.site][person.tent]?.element.dataset.campTent
        const exit = tents[person.site][person.exitTent]
        if (
          exit &&
          Math.hypot(person.position.x - exit.position.x, person.position.y - exit.position.y) > 34
        )
          person.exitTent = -1
        const leavingTent = tents[person.site][person.exitTent]?.element.dataset.campTent
        const nearby = obstacles.filter(
          obstacle =>
            obstacle.id !== person.id &&
            obstacle.id !== `tent-${ownTent}` &&
            obstacle.id !== `tent-${leavingTent}`,
        )
        const scale = trailGroundScale(person.position.y, landscape)
        const replacement = replanOccupiedWaypoints(
          person.id,
          person.position,
          person.route,
          12 * scale,
          nearby,
          bounds(person.phase === 'departing'),
          person.spacing.yieldingTo,
        )
        if (replacement?.length) {
          person.route = replacement
          person.target = replacement[replacement.length - 1]
          person.remaining = Infinity
          person.stalled = 0
        }
        const destinationPoint = person.route[0] ?? person.target
        const distance = Math.hypot(
          destinationPoint.x - person.position.x,
          destinationPoint.y - person.position.y,
        )
        const budget = (delta / 1000) * 18
        person.stalled = distance < person.remaining - 0.15 ? 0 : person.stalled + delta
        person.remaining = distance
        const fraction = Math.min(1, budget / Math.max(1, distance))
        const intended = {
          x: person.position.x + (destinationPoint.x - person.position.x) * fraction,
          y: person.position.y + (destinationPoint.y - person.position.y) * fraction,
        }
        if (
          spacedStep(
            person.id,
            person.position,
            intended,
            12 * scale,
            budget * 1.5,
            person.spacing,
            nearby,
            bounds(person.phase === 'departing'),
            true,
            destinationPoint,
            true,
          )
        )
          person.position = intended
        const current = spacedPosition(person.position, person.spacing)
        // Campsite walkers steer in world space; they have no path offset to recover.
        person.position = current
        person.spacing.lane.x -= person.spacing.offset.x
        person.spacing.lane.y -= person.spacing.offset.y
        person.spacing.offset = { x: 0, y: 0 }
        if (Math.hypot(current.x - person.lastStep.x, current.y - person.lastStep.y) >= 8) {
          const mark = bootImpression(document)
          mark.dataset.footprintOwner = person.id
          mark.setAttribute(
            'transform',
            footprintTransform(
              current.x,
              current.y,
              Math.atan2(current.y - person.lastStep.y, current.x - person.lastStep.x),
              person.stride++ % 2 ? 1 : -1,
            ) + ` scale(${scale.toFixed(3)})`,
          )
          tracks.append(mark)
          while (tracks.childElementCount > 80) tracks.firstElementChild?.remove()
          person.lastStep = current
        }
        const arrived =
          Math.hypot(current.x - destinationPoint.x, current.y - destinationPoint.y) < 2
        const destination = person.element.dataset.destination
        const clear = nearby.every(
          other =>
            Math.hypot(current.x - other.x, current.y - other.y) >= 12 * scale + other.radius + 1.9,
        )
        const workTent = tents[person.site][person.tent]
        const canBuild =
          (destination === 'build' || destination === 'pack') &&
          workTent &&
          distance < 18 &&
          current.y >= workTent.position.y + 8 &&
          clear
        const settled = destination === 'fire' && person.stalled > 1500 && distance < 24 && clear
        if (arrived && person.route.length > 1) {
          person.route.shift()
          person.remaining = Infinity
        } else if (
          arrived ||
          (person.route.length <= 1 &&
            (canBuild ||
              settled ||
              (destination === 'sleep' && distance < 10 && person.spacing.status === 'clear')))
        ) {
          person.phase =
            destination === 'hike'
              ? 'gone'
              : destination === 'douse'
                ? 'dousing'
                : destination === 'pack'
                  ? 'packing'
                  : destination === 'build'
                    ? 'building'
                    : destination === 'sleep'
                      ? 'sleeping'
                      : 'attending'
          person.buildTime = 0
          if (person.phase === 'sleeping')
            tents[person.site][person.tent].element.dataset.occupied = 'true'
        }
        if (destination === 'hike') {
          person.walked += Math.hypot(current.x - before.x, current.y - before.y)
          person.element.dataset.departureProgress = Math.min(
            1,
            person.walked / Math.max(1, person.departureLength),
          ).toFixed(4)
        }
      }
      noticePet(person, pet)
      setPose(person)
      renderPerson(person, person.phase !== 'sleeping' && person.phase !== 'gone')
      const reservation = obstacles.find(obstacle => obstacle.id === person.id)
      if (reservation) Object.assign(reservation, spacedPosition(person.position, person.spacing))
    }
    for (let site = 0; site < tents.length; site++)
      for (let index = 0; index < tents[site].length; index++)
        tents[site][index].element.dataset.occupied = String(
          people.some(
            person => person.site === site && person.tent === index && person.phase === 'sleeping',
          ),
        )
  }
  const tick = (time: number) => {
    frame = 0
    if (paused || host.hidden || document.hidden || signal.aborted || reduced.matches) return
    if (!last) last = time
    if (time - last >= 80) {
      const delta = Math.min(160, time - last)
      last = time
      elapsed += delta
      scene.dataset.campElapsed = elapsed.toFixed(1)
      updatePeople(delta)
      updateRoutines()
      lighting()
    }
    frame = view.requestAnimationFrame(tick)
  }
  const pause = () => {
    paused = true
    if (frame) view.cancelAnimationFrame(frame)
    frame = 0
    last = 0
    scene.dataset.campMotion = 'paused'
  }
  const resume = () => {
    pause()
    paused = false
    updateRoutines()
    lighting()
    if (reduced.matches) {
      const pet = approachingPet()
      for (let site = 0; site < routines.length; site++) {
        if (['packing', 'extinguishing', 'departing'].includes(routines[site].phase))
          finishCamp(site)
      }
      for (const group of tents)
        for (const tent of group) {
          if (tent.element.dataset.campPhase === 'empty') continue
          tent.progress = 1
          tent.element.dataset.progress = '1'
          tent.element.dataset.phase = 'ready'
          tent.element.dataset.occupied = 'false'
          tent.element.style.setProperty('--tent-progress', '1')
        }
      for (const person of people) {
        if (routines[person.site]?.phase === 'empty') continue
        person.position = seat(person)
        person.spacing = actorSpacing()
        person.phase = bedtime(person.site) && tents[person.site].length ? 'sleeping' : 'attending'
        if (person.phase === 'sleeping') {
          person.tent = person.slot % tents[person.site].length
          person.target = { ...tents[person.site][person.tent].position }
          tents[person.site][person.tent].element.dataset.occupied = 'true'
        } else placePerson(person)
        noticePet(person, pet)
        setPose(person)
        renderPerson(person, person.phase !== 'sleeping')
      }
      scene.dataset.campMotion = 'reduced'
      lighting()
      return
    }
    if (host.hidden || document.hidden || !landscape) return
    scene.dataset.campMotion = 'running'
    frame = view.requestAnimationFrame(tick)
  }
  const addPerson = (id: string, site: number, role: Person['role']) => {
    if (!landscape) return
    const element = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    element.classList.add('trail-camp-person')
    element.dataset.campPerson = id
    element.dataset.destination = role === 'builder' && tents[site].length ? 'build' : 'fire'
    element.innerHTML = `<g class="trail-camp-person-art" transform="translate(-4 -12)"></g><text class="trail-camp-question" y="-25" text-anchor="middle" aria-hidden="true">?</text>`
    layer.append(element)
    const person: Person = {
      id,
      site,
      slot: 0,
      role,
      element,
      position: { x: sites[site].x - 68, y: sites[site].y - 58 },
      spacing: actorSpacing(),
      target: sites[site],
      tent: -1,
      exitTent: -1,
      buildTime: 0,
      phase: 'walking',
      carrying: false,
      wakeAt: Infinity,
      departAt: 0,
      walked: 0,
      departureLength: 0,
      packingFrom: 1,
      lastStep: sites[site],
      stride: 0,
      stalled: 0,
      remaining: Infinity,
      route: [],
      look: null,
      surpriseUntil: 0,
    }
    while (people.some(other => other.site === site && other.slot === person.slot)) person.slot++
    people.push(person)
    if (role === 'camper') {
      person.position = seat(person)
      person.phase = 'attending'
    } else {
      person.tent = nextTent(person)
      if (person.tent >= 0) {
        const tent = tents[site][person.tent]
        person.target = { x: tent.position.x, y: tent.position.y + 17 }
        person.position = { x: tent.position.x - 25, y: tent.position.y + 27 }
      } else {
        person.target = seat(person)
        element.dataset.destination = 'fire'
      }
    }
    person.position = onDryLand(person.position, landscape, 1.8)
    placePerson(person, person.role === 'builder')
    setPose(person)
    person.position = spacedPosition(person.position, person.spacing)
    person.spacing.offset = { x: 0, y: 0 }
    travelTo(person, person.target)
    renderPerson(person, person.phase !== 'sleeping')
  }
  const resetCamp = (site: number) => {
    for (const person of people.filter(person => person.site === site)) person.element.remove()
    people = people.filter(person => person.site !== site)
    routines[site] = campRoutine(seed, site, Date.now())
    fires[site].dataset.campStay = String(routines[site].stay)
    fires[site].dataset.campLeaveAt = String(routines[site].leaveAt)
    fires[site].dataset.campSleepAt = String(routines[site].sleepAt)
    fires[site].dataset.campPhase = 'staying'
    fires[site].dataset.fireState = 'burning'
    fires[site].style.setProperty('--trail-fire-strength', '1')
    for (const tent of tents[site]) {
      tent.progress = settings.builders ? 0 : 1
      tent.element.dataset.progress = String(tent.progress)
      tent.element.dataset.phase = tent.progress ? 'ready' : 'packed'
      tent.element.dataset.campPhase = 'staying'
      tent.element.dataset.occupied = 'false'
      tent.element.dataset.carried = 'false'
      tent.element.dataset.campSolid = `tent-${tent.element.dataset.campTent}`
      tent.element.style.setProperty('--tent-progress', String(Math.max(0.15, tent.progress)))
    }
    for (let index = 0; index < 2; index++) addPerson(`camper-${site}-${index}`, site, 'camper')
    for (let index = 0; index < settings.builders; index++)
      addPerson(`builder-${site}-${index}`, site, 'builder')
    lighting()
    onStructure()
  }
  const update = (
    current: Landscape,
    currentSeed: number,
    currentSettings: CampSettings,
    routes: JourneyRoute[],
  ) => {
    const rerouted = trailRoutes !== routes
    trailRoutes = routes
    const same =
      landscape?.width === current.width &&
      landscape.height === current.height &&
      landscape.camp.x === current.camp.x &&
      landscape.camp.y === current.camp.y &&
      seed === currentSeed &&
      settings.sites === currentSettings.sites &&
      settings.tents === currentSettings.tents &&
      settings.builders === currentSettings.builders
    if (same) {
      settings = currentSettings
      if (rerouted) {
        for (const person of people) {
          if (person.phase === 'walking') travelTo(person, person.target)
          else if (person.phase === 'departing') {
            person.route = campDeparture(person.position, routes, current, [
              ...readTrunks(scene),
              ...readStationaryActors(scene, person.id),
            ])
            person.target = person.route[person.route.length - 1] ?? person.position
            person.remaining = Infinity
          }
        }
      }
      lighting()
      resume()
      return
    }
    scene.dataset.campSetup = landscape && seed === currentSeed ? 'editing' : 'opening'
    pause()
    landscape = current
    seed = currentSeed
    settings = currentSettings
    sites = campSites(current, seed, settings)
    layer.replaceChildren()
    tracks.replaceChildren()
    people = []
    tents = []
    fires = []
    routines = sites.map(site => campRoutine(seed, site.id, Date.now()))
    for (const site of sites) {
      const camp =
        site.id === 0 ? primary : document.createElementNS('http://www.w3.org/2000/svg', 'g')
      camp.dataset.campSite = String(site.id)
      camp.dataset.campPhase = 'staying'
      camp.dataset.campStay = String(routines[site.id].stay)
      camp.dataset.campLeaveAt = String(routines[site.id].leaveAt)
      camp.dataset.campSleepAt = String(routines[site.id].sleepAt)
      camp.dataset.campVariation = routines[site.id].variation
      camp.dataset.fireState = 'burning'
      camp.style.setProperty('--trail-fire-strength', '1')
      camp.dataset.campSolid = `hearth-${site.id}`
      camp.dataset.groundX = String(site.x)
      camp.dataset.groundY = String(site.y + 16)
      camp.dataset.solidRadius = '16'
      fires[site.id] = camp
      if (site.id) {
        camp.classList.add('trail-camp', 'trail-extra-camp')
        camp.setAttribute('transform', `translate(${site.x} ${site.y + 16})`)
        camp.innerHTML = trailCampsite(false)
        layer.append(camp)
      }
      tents[site.id] = site.tents.map((point, index) => {
        const tent = document.createElementNS('http://www.w3.org/2000/svg', 'g')
        tent.classList.add('trail-tent')
        tent.dataset.campTent = `${site.id}-${index}`
        tent.dataset.campSolid = `tent-${site.id}-${index}`
        tent.dataset.groundX = String(point.x)
        tent.dataset.groundY = String(point.y)
        tent.dataset.solidRadius = '28'
        tent.dataset.progress = settings.builders ? '0' : '1'
        tent.dataset.phase = settings.builders ? 'packed' : 'ready'
        tent.dataset.campPhase = 'staying'
        tent.setAttribute('transform', `translate(${point.x} ${point.y})`)
        tent.style.setProperty('--tent-progress', settings.builders ? '0.15' : '1')
        tent.innerHTML = tentArt()
        layer.append(tent)
        return { element: tent, position: point, progress: settings.builders ? 0 : 1 }
      })
    }
    for (const site of sites) {
      for (let index = 0; index < 2; index++)
        addPerson(`camper-${site.id}-${index}`, site.id, 'camper')
      for (let index = 0; index < settings.builders; index++)
        addPerson(`builder-${site.id}-${index}`, site.id, 'builder')
    }
    lighting()
    if (!host.hidden) resume()
  }
  subscribeVisitorTrailWeather(
    document,
    weather => {
      forecastDay = weather?.day ?? null
      lighting()
    },
    signal,
  )
  host.addEventListener('trail:hide', pause, { signal })
  host.addEventListener(
    'trail:show',
    () => {
      primary.dataset.campPhase = 'staying'
      primary.dataset.fireState = 'burning'
      primary.style.setProperty('--trail-fire-strength', '1')
    },
    { signal },
  )
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : resume()), {
    signal,
  })
  reduced.addEventListener('change', resume, { signal })
  signal.addEventListener('abort', pause, { once: true })
  return { update, pause, resume, packUp }
}
