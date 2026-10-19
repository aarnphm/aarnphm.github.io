import type { JourneyRoute } from './hiker-journey'
import { readActors } from './actor-spacing'
import { footprint, footprintTransform } from './footprints'
import { trailGroundScale, type Landscape, type Point } from './landscape'
import { pixels } from './pixels'
import { trailRandom } from './random'
import { sceneMotionReady } from './scene-reveal'

// Now and then an ultrarunner and a mountain biker come down off the mountain on the trails. The
// runner runs one trail down to the camp, stops at the aid station, and runs back up another. The
// biker descends one trail at speed and climbs back up another in a low gear. Between outings both
// are away in the mountains. They give way to everyone: on a trail a runner or a rider swings out
// round hikers, horses and cows with room to spare, braking until clear and stopping behind a
// crowd, and nobody steps aside for them, so they publish no collision body. Runners leave shoe prints and the bike a tyre line, both on the
// track layer, so the moss takes the damage. Weather keeps the biker home when the trails are wet,
// snowed over or dark; only a storm keeps the runner in, who goes out at night with a headlamp.

type Kind = 'runner' | 'biker'
type Leg = { route: JourneyRoute; down: boolean; pace: number }
type Athlete = {
  kind: Kind
  element: SVGGElement
  facing: SVGGElement
  legs: Leg[]
  leg: number
  distance: number
  // Milliseconds left away, or at the aid station.
  rest: number
  phase: 'away' | 'out' | 'aid'
  lane: number
  lastMark: Point | null
  stride: number
  cadence: number
  // Milliseconds held up behind someone with no way round.
  held: number
}

const runnerTones = {
  H: 'trail-athlete-cap',
  S: 'trail-camper-skin',
  C: 'trail-athlete-shirt',
  V: 'trail-athlete-vest',
  W: 'trail-athlete-bib',
  D: 'trail-athlete-shorts',
  B: 'trail-athlete-shoe',
}
const runnerBody = [
  '..HHH..',
  '..HHHH.',
  '..SSS..',
  '..SSS..',
  '.CVVC..',
  'SCVWCS.',
  '.CVVC..',
  '.DDDD..',
]
// Push-off with the back foot kicked up, and the swing leg passing under the hips.
const runnerLegs = [
  ['.S.S...', 'S...S..', 'B...S..', '....BB.'],
  ['..SS...', '..SS...', '..S.B..', '..BB...'],
]
const bikerTones = {
  H: 'trail-athlete-helmet',
  V: 'trail-athlete-visor',
  S: 'trail-camper-skin',
  J: 'trail-athlete-jersey',
  D: 'trail-athlete-shorts',
  E: 'trail-athlete-saddle',
  F: 'trail-athlete-frame',
  K: 'trail-athlete-bar',
  O: 'trail-athlete-tyre',
  X: 'trail-athlete-hub',
  C: 'trail-athlete-hub',
  B: 'trail-athlete-shoe',
}
// A hardtail facing right, the rider low over the bars; the leg is drawn per crank position.
const bike = [
  '........HHH...',
  '.......HHHHV..',
  '........SSS...',
  '......JJJJ....',
  '.....JJJJ.S...',
  '....JJJ....S..',
  '....DDD....SK.',
  '....EE.....F..',
  '.....FFF.FFF..',
  '.OOO.F....OFO.',
  'O...OF.F.O.F.O',
  'O.XFFFC..O.X.O',
  'O...O....O...O',
  '.OOO......OOO.',
]
// Cranks level, the near foot forward, which is also how a rider stands to descend; and the near
// foot at the bottom of the stroke.
const bikerLegs = [
  ['.......D......', '........S.....', '........S.....', '........B.....'],
  [
    '......D.......',
    '......D.......',
    '......S.......',
    '......S.......',
    '.......S......',
    '.......B......',
  ],
]

export function mountTrailAthletes(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const layer = scene.querySelector<SVGGElement>('.trail-athletes')
  const tracks = scene.querySelector<SVGGElement>('.trail-athlete-tracks')
  if (!view || !layer || !tracks) throw new Error('Trail athletes need their trail layers')
  const reduced = view.matchMedia('(prefers-reduced-motion: reduce)')
  let landscape: Landscape | null = null
  let routes: JourneyRoute[] = []
  let athletes: Athlete[] = []
  let random = trailRandom(1)
  let signature = ''
  let frame = 0
  let last = 0

  const build = (kind: Kind, index: number): Athlete => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    element.setAttribute('class', 'trail-athlete')
    element.dataset.trailAthlete = kind
    element.dataset.phase = 'away'
    element.dataset.stride = '0'
    element.style.visibility = 'hidden'
    element.innerHTML =
      kind === 'runner'
        ? `<g class="trail-athlete-facing"><g transform="translate(-7 -24)">${pixels(runnerBody, runnerTones, 2)}<path class="trail-athlete-beam" d="M12 3 38 8V22Z"/><path class="trail-athlete-lamp" d="M10 2h2v2h-2Z"/></g>${runnerLegs.map((rows, stride) => `<g class="trail-athlete-stride" data-stride="${stride}" transform="translate(-7 -8)">${pixels(rows, runnerTones, 2)}</g>`).join('')}</g>`
        : `<g class="trail-athlete-facing"><g transform="translate(-14 -28)">${pixels(bike, bikerTones, 2)}${bikerLegs.map((rows, stride) => `<g class="trail-athlete-stride" data-stride="${stride}" transform="translate(0 14)">${pixels(rows, bikerTones, 2)}</g>`).join('')}</g></g>`
    layer.append(element)
    return {
      kind,
      element,
      facing: element.querySelector<SVGGElement>('.trail-athlete-facing')!,
      legs: [],
      leg: 0,
      distance: 0,
      // The first outing comes within a couple of minutes of the trail opening.
      rest: 20_000 + index * 35_000 + random() * 60_000,
      phase: 'away',
      lane: 0,
      lastMark: null,
      stride: 0,
      cadence: 0,
      held: 0,
    }
  }
  // Whether the weather lets this athlete out now.
  const fit = (kind: Kind) => {
    const { weatherStorm, night, weatherMode, snowCover } = scene.dataset
    if (weatherStorm === 'true') return false
    if (kind === 'runner') return true
    return night !== 'true' && weatherMode !== 'rain' && Number(snowCover ?? 0) < 0.5
  }
  // Down one trail and up another, so they come back into the mountains by a different way.
  const plan = (athlete: Athlete) => {
    const long = routes.filter(route => route.length > 240)
    if (long.length < 2) return false
    const down = long[Math.floor(random() * long.length)]
    const others = long.filter(route => route !== down)
    const up = others[Math.floor(random() * others.length)]
    // Scene units a second: a runner at an ultra pace, slower going back up; a rider descending
    // about twice as fast, and climbing at walking pace.
    athlete.legs =
      athlete.kind === 'runner'
        ? [
            { route: down, down: true, pace: 46 + random() * 8 },
            { route: up, down: false, pace: 30 + random() * 6 },
          ]
        : [
            { route: down, down: true, pace: 92 + random() * 18 },
            { route: up, down: false, pace: 24 + random() * 4 },
          ]
    athlete.leg = 0
    athlete.distance = 0
    athlete.lane = 0
    athlete.held = 0
    athlete.lastMark = null
    return true
  }
  const at = (leg: Leg, distance: number) =>
    leg.route.point(leg.down ? distance : leg.route.length - distance)
  const away = (athlete: Athlete) => {
    athlete.phase = 'away'
    athlete.rest = 90_000 + random() * 150_000
    athlete.element.dataset.phase = 'away'
    athlete.element.style.visibility = 'hidden'
    delete athlete.element.dataset.groundY
  }
  // Prints every few scene units along the line of travel: alternate shoes for the runner, a
  // short tyre line for the bike.
  const mark = (athlete: Athlete, point: Point, scale: number) => {
    const spacing = (athlete.kind === 'runner' ? 11 : 7) * scale
    const previous = athlete.lastMark
    if (previous && Math.hypot(point.x - previous.x, point.y - previous.y) < spacing) return
    athlete.lastMark = point
    if (!previous) return
    const angle = Math.atan2(point.y - previous.y, point.x - previous.x)
    const side = athlete.kind === 'runner' ? (athlete.stride % 2 ? 1 : -1) : 1
    const spread = athlete.kind === 'runner' ? 2 : 0
    tracks.insertAdjacentHTML(
      'beforeend',
      athlete.kind === 'runner'
        ? footprint(
            'boot',
            `${footprintTransform(point.x, point.y, angle, side, spread)} scale(${scale.toFixed(3)})`,
          )
        : `<path data-footprint="tyre" d="M-3-.5h6v1h-6Z" transform="${footprintTransform(point.x, point.y, angle, 1, 0)} scale(${scale.toFixed(3)})"/>`,
    )
    while (tracks.childElementCount > 80) tracks.firstElementChild?.remove()
  }
  const step = (athlete: Athlete, delta: number, bodies: ReturnType<typeof readActors>) => {
    if (!landscape) return
    if (athlete.phase !== 'out') {
      athlete.rest -= delta
      if (athlete.rest > 0) return
      if (athlete.phase === 'aid') {
        athlete.phase = 'out'
        athlete.leg = 1
        athlete.distance = 0
        athlete.element.dataset.phase = 'out'
        return
      }
      if (!fit(athlete.kind) || !plan(athlete)) {
        athlete.rest = 30_000
        return
      }
      athlete.phase = 'out'
      athlete.element.dataset.phase = 'out'
    }
    const leg = athlete.legs[athlete.leg]
    const here = at(leg, athlete.distance)
    const ahead = at(leg, Math.min(leg.route.length, athlete.distance + 6))
    const length = Math.hypot(ahead.x - here.x, ahead.y - here.y) || 1
    const tx = (ahead.x - here.x) / length
    const ty = (ahead.y - here.y) / length
    const scale = trailGroundScale(here.y, landscape)
    // Anyone ahead is passed with room to spare. The look ahead is about a second at full pace, so
    // a descending rider sees people in time to swing out. The lane is the offset from the trail,
    // within the verge, that clears every body in reach and is nearest the current one; until the
    // athlete is out in it they brake, down to a stop behind whoever they would hit. Held up
    // there, they widen the verge until they find a way round on the grass.
    const own = (athlete.kind === 'runner' ? 8 : 14) * scale
    const look = (24 + leg.pace * 0.9) * scale
    const verge = (60 + Math.min(100, athlete.held / 40)) * scale
    const blocks: { along: number; across: number; clear: number }[] = []
    for (const body of bodies) {
      const rx = body.x - here.x
      const ry = body.y - here.y
      const along = rx * tx + ry * ty
      const across = tx * ry - ty * rx
      const clear = body.radius + own + 4 * scale
      if (along < -clear || along > clear + look || Math.abs(across) > clear + verge) continue
      blocks.push({ along, across, clear })
    }
    const free = (offset: number) =>
      Math.abs(offset) <= verge &&
      blocks.every(block => Math.abs(offset - block.across) >= block.clear - 0.01)
    const cost = (offset: number) => Math.abs(offset - athlete.lane) + Math.abs(offset) * 0.25
    let lane: number | null = null
    for (const offset of [0, ...blocks.flatMap(b => [b.across - b.clear, b.across + b.clear])])
      if (free(offset) && (lane === null || cost(offset) < cost(lane))) lane = offset
    let pace = leg.pace
    if (blocks.length) {
      // The gap to the first body on the athlete's current line.
      let gap = Infinity
      for (const block of blocks)
        if (block.along > 0 && Math.abs(athlete.lane - block.across) < block.clear)
          gap = Math.min(gap, block.along - block.clear)
      pace =
        gap === Infinity
          ? leg.pace * 0.6
          : leg.pace * Math.max(0, Math.min(0.4, (gap - 4 * scale) / look))
    }
    athlete.held = blocks.length && pace < leg.pace * 0.05 ? athlete.held + delta : 0
    const seconds = delta / 1000
    const swing = 60 * scale * seconds
    athlete.lane += Math.max(-swing, Math.min(swing, (lane ?? athlete.lane) - athlete.lane))
    athlete.distance += pace * scale * seconds
    if (athlete.distance >= leg.route.length) {
      if (athlete.leg === 0 && athlete.kind === 'runner') {
        // Water and something to eat at the camp before the climb back.
        athlete.phase = 'aid'
        athlete.rest = 3000 + random() * 2000
        athlete.element.dataset.phase = 'aid'
      } else if (athlete.leg === 0) {
        athlete.leg = 1
        athlete.distance = 0
      } else return away(athlete)
    }
    const point = {
      x: Math.round(here.x - ty * athlete.lane),
      y: Math.round(here.y + tx * athlete.lane),
    }
    // A pedal stroke or a stride per beat; a rider descending stands on level cranks.
    athlete.cadence += delta
    const beat = athlete.kind === 'runner' ? 170 : 260
    if (athlete.phase === 'out' && pace > 0 && athlete.cadence >= beat) {
      athlete.cadence = 0
      if (athlete.kind === 'runner' || !leg.down) athlete.stride++
    }
    const element = athlete.element
    const transform = `translate(${point.x} ${point.y}) scale(${scale.toFixed(3)})`
    if (element.getAttribute('transform') !== transform)
      element.setAttribute('transform', transform)
    if (element.dataset.groundY !== String(point.y)) element.dataset.groundY = String(point.y)
    element.dataset.stride = String(athlete.kind === 'biker' && leg.down ? 0 : athlete.stride % 2)
    element.dataset.terrain = leg.down ? 'descent' : 'climb'
    athlete.facing.setAttribute('transform', `scale(${tx < 0 ? -1 : 1} 1)`)
    // Fade in from the mountain and back out on the way up.
    const edge = leg.down ? athlete.distance : leg.route.length - athlete.distance
    element.style.visibility = ''
    element.style.opacity = Math.min(1, edge / 24).toFixed(2)
    if (athlete.phase === 'out') mark(athlete, point, scale)
  }
  const tick = (time: number) => {
    frame = 0
    if (!running()) return
    // Pixel moves at about thirty a second.
    if (time - last >= 33) {
      const delta = last ? Math.min(100, time - last) : 33
      last = time
      const bodies = readActors(scene)
      for (const athlete of athletes) step(athlete, delta, bodies)
    }
    frame = view.requestAnimationFrame(tick)
  }
  const running = () =>
    sceneMotionReady(scene) &&
    !host.hidden &&
    !document.hidden &&
    !signal.aborted &&
    !reduced.matches &&
    !!landscape
  const pause = () => {
    view.cancelAnimationFrame(frame)
    frame = 0
    last = 0
  }
  const resume = () => {
    pause()
    if (!running()) {
      // Reduced motion keeps them in the mountains.
      if (reduced.matches) for (const athlete of athletes) away(athlete)
      return
    }
    if (athletes.length) frame = view.requestAnimationFrame(tick)
  }
  signal.addEventListener('abort', pause, { once: true })
  host.addEventListener('trail:hide', pause, { signal })
  host.addEventListener('trail:show', resume, { signal })
  host.addEventListener('trail:ready', resume, { signal })
  document.addEventListener('visibilitychange', () => (document.hidden ? pause() : resume()), {
    signal,
  })
  reduced.addEventListener('change', resume, { signal })
  return {
    pause,
    update(current: Landscape, seed: number, paths: JourneyRoute[]) {
      const next = `${seed}|${current.width}|${current.height}|${paths.map(path => `${path.id}:${path.length.toFixed(0)}`).join('|')}`
      landscape = current
      routes = paths
      if (next === signature) return resume()
      pause()
      signature = next
      random = trailRandom(seed ^ 0x2d1b5a7)
      for (const athlete of athletes) athlete.element.remove()
      tracks.replaceChildren()
      athletes = (['runner', 'biker'] as const).map(build)
      resume()
    },
  }
}
