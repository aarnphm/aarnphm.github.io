import {
  creekDistance,
  pondDistance,
  trailGroundScale,
  type Landscape,
  type Point,
} from './landscape'
import { pixels } from './pixels'
import { animateScene } from './scene-animation'

// One of the shore frogs lives: it hops along the near bank, croaks, takes dragonflies with its
// tongue through the summer, swims out and comes back. The other frogs keep their stylesheet
// hop. Nothing is stored; the frog decides each round from the scene's own flags (night, rain,
// season, ice) and from where the dragonflies are. Every frame faces right in the 1 px register
// the wetland art uses, bottom centre at the origin, the same as the drawn frog.
const sit = ['.....GEG.', '...GGGGGG', '.GGGGGGGG', 'GDDDGLLL.', 'DDDDLLLD.', 'DDDD..DD.']
const leap = ['......GEG', '....GGGGG', '.GGGGGLLL', 'GGGGLLL..', 'DD.DD....', 'D..D.....']
// Eyes and the top of the head over the water.
const swim = ['.....GEG.', '....GGGG.']
const tones = {
  G: 'trail-wetland-frog-back',
  D: 'trail-wetland-frog-leg',
  L: 'trail-wetland-belly',
  E: 'trail-animal-eye',
}

type Act = 'hop' | 'croak' | 'hunt' | 'dive'

export function mountFrog(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) return { refresh() {} }
  const motion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let frog: SVGGElement | null = null
  let facing: SVGGElement | null = null
  let tongue: SVGPathElement | null = null
  let landscape: Landscape | null = null
  let clear: (point: Point) => boolean = () => true
  let at: Point = { x: 0, y: 0 }
  let scale = 1
  let timer = 0
  let hop: Animation | null = null
  let placed = ''

  const place = (point: Point) => {
    if (!frog) return
    at = { x: Math.round(point.x * 2) / 2, y: Math.round(point.y * 2) / 2 }
    frog.setAttribute('transform', `translate(${at.x} ${at.y})`)
    frog.dataset.groundX = String(at.x)
    frog.dataset.groundY = String(at.y)
  }
  const face = (direction: number) => facing?.setAttribute('transform', `scale(${direction} 1)`)
  const pose = (name: string) => {
    if (frog) frog.dataset.pose = name
  }
  const stop = () => {
    view.clearTimeout(timer)
    hop?.cancel()
    hop = null
    if (frog) {
      pose('sitting')
      place(at)
    }
  }
  const wait = (ms: number) => new Promise<void>(resolve => (timer = view.setTimeout(resolve, ms)))
  const night = () => scene.dataset.lighting === 'moonlit'
  const active = () =>
    !signal.aborted &&
    !host.hidden &&
    !document.hidden &&
    !motion.matches &&
    !!frog?.isConnected &&
    scene.dataset.waterFrozen !== 'true' &&
    scene.dataset.weatherMotion !== 'paused'

  // Ground round the near shore the frog may sit on: on the bank, off the creek, off the heron's
  // spot and off the camp, and inside the scene.
  const heron = (): Point | null => {
    const bird = scene.querySelector<SVGGElement>('[data-wetland-animal="heron"]')
    if (!bird) return null
    return { x: Number(bird.dataset.groundX), y: Number(bird.dataset.groundY) }
  }
  const shoreSpot = (near: Point, reach: number): Point | null => {
    if (!landscape) return null
    const { pond } = landscape
    const wader = heron()
    const current = Math.atan2((near.y - pond.y) / pond.ry, (near.x - pond.x) / pond.rx)
    for (let attempt = 0; attempt < 12; attempt++) {
      // A step round the shore about as long as the reach, whichever side of the pond it sits.
      const angle = current + ((Math.random() - 0.5) * 2 * reach) / pond.rx
      const ratio = 1.12 + Math.random() * 0.16
      const point = {
        x: pond.x + Math.cos(angle) * pond.rx * ratio,
        y: pond.y + Math.sin(angle) * pond.ry * ratio,
      }
      const distance = Math.hypot(point.x - near.x, point.y - near.y)
      if (
        distance < 6 ||
        distance > reach ||
        point.x < 16 ||
        point.x > landscape.width - 16 ||
        point.y > landscape.height - 24 ||
        creekDistance(point, landscape) < 12 ||
        Math.hypot(point.x - landscape.camp.x, point.y - landscape.camp.y) < 100 ||
        (wader && Math.hypot(point.x - wader.x, point.y - wader.y) < 30) ||
        !clear(point)
      )
        continue
      return point
    }
    return null
  }
  // A spot in the water off the frog's stretch of bank.
  const waterSpot = (): Point => {
    const { pond } = landscape!
    const angle = Math.atan2((at.y - pond.y) / pond.ry, (at.x - pond.x) / pond.rx)
    const ratio = 0.86 + Math.random() * 0.06
    return {
      x: pond.x + Math.cos(angle) * pond.rx * ratio,
      y: pond.y + Math.sin(angle) * pond.ry * ratio,
    }
  }
  // One hop: an arc in half pixels, the leap frame through the air.
  const jump = async (to: Point) => {
    if (!frog) return
    face(to.x < at.x ? -1 : 1)
    pose('leaping')
    const from = at
    const distance = Math.hypot(to.x - from.x, to.y - from.y)
    const lift = (4 + distance * 0.3) * scale
    const steps = Math.max(4, Math.round(distance / 3))
    const frames = Array.from({ length: steps + 1 }, (_, index) => {
      const t = index / steps
      const x = Math.round((from.x + (to.x - from.x) * t) * 2) / 2
      const y = Math.round((from.y + (to.y - from.y) * t - Math.sin(t * Math.PI) * lift) * 2) / 2
      return { transform: `translate(${x}px, ${y}px)` }
    })
    hop = animateScene(frog, frames, { duration: 220 + distance * 9, fill: 'forwards' })
    await hop.finished
    hop.cancel()
    hop = null
    place(to)
    pose('sitting')
  }
  // Where each dragonfly is this instant, in scene space: its hover point and the dart offset
  // its stylesheet animation has reached, mirrored for the way it faces.
  const dragonflies = () => {
    const flies: { element: SVGGElement; x: number; y: number }[] = []
    if (scene.dataset.foliageSeason !== 'summer' || night()) return flies
    for (const element of scene.querySelectorAll<SVGGElement>('.trail-dragonfly')) {
      if (element.dataset.caught || !element.transform.baseVal.numberOfItems) continue
      if (view.getComputedStyle(element).display === 'none') continue
      const root = element.transform.baseVal.getItem(0).matrix
      const side =
        element.transform.baseVal.numberOfItems > 1
          ? element.transform.baseVal.getItem(1).matrix.a
          : 1
      const dart = element.querySelector<SVGGElement>('.trail-dragonfly-dart')
      const moved = dart
        ? new view.DOMMatrix(view.getComputedStyle(dart).transform)
        : new view.DOMMatrix()
      flies.push({ element, x: root.e + moved.e * side, y: root.f + moved.f })
    }
    return flies
  }
  // The tongue runs from the mouth to the fly in whole scene pixels, one pixel wide.
  const flick = async (target: Point) => {
    if (!frog || !tongue) return
    const direction = target.x < at.x ? -1 : 1
    face(direction)
    const mouth = { x: at.x + 4 * scale * direction, y: at.y - 4 * scale }
    const length = Math.hypot(target.x - mouth.x, target.y - mouth.y)
    const angle = (Math.atan2(target.y - mouth.y, target.x - mouth.x) * 180) / Math.PI
    tongue.setAttribute('d', `M0 -0.5h${length.toFixed(1)}v1H0Z`)
    tongue.setAttribute(
      'transform',
      `translate(${(mouth.x - at.x).toFixed(1)} ${(mouth.y - at.y).toFixed(1)}) rotate(${angle.toFixed(1)})`,
    )
    pose('tongue')
    await wait(140)
    pose('sitting')
  }
  const choose = (): Act => {
    const roll = Math.random()
    if (night()) return roll < 0.55 ? 'croak' : roll < 0.85 ? 'hop' : 'dive'
    const prey = dragonflies().some(fly => Math.hypot(fly.x - at.x, fly.y - at.y) < 70)
    if (prey && roll < 0.45) return 'hunt'
    if (scene.dataset.weatherMode === 'rain')
      return roll < 0.6 ? 'hop' : roll < 0.8 ? 'croak' : 'dive'
    return roll < 0.4 ? 'hop' : roll < 0.65 ? 'croak' : roll < 0.85 ? 'dive' : 'hunt'
  }
  const run = async () => {
    stop()
    if (!frog) return
    try {
      while (!signal.aborted) {
        await wait(
          (scene.dataset.weatherMode === 'rain' ? 2500 : 4000) +
            Math.random() * (night() ? 9000 : 6000),
        )
        if (!active() || !landscape) continue
        const act = choose()
        if (act === 'croak') {
          pose('croaking')
          await wait(1200 + Math.random() * 1800)
          pose('sitting')
        } else if (act === 'hop') {
          // One to three hops along the bank.
          const hops = 1 + Math.floor(Math.random() * 3)
          for (let index = 0; index < hops && active(); index++) {
            const to = shoreSpot(at, 10 + Math.random() * 14)
            if (!to) break
            await jump(to)
            await wait(200 + Math.random() * 500)
          }
        } else if (act === 'hunt') {
          // Close in on the nearest dragonfly, then take it if it darts within the tongue's reach.
          const flies = dragonflies()
            .map(fly => ({ ...fly, distance: Math.hypot(fly.x - at.x, fly.y - at.y) }))
            .sort((a, b) => a.distance - b.distance)
          const fly = flies[0]
          if (!fly) continue
          if (fly.distance > 22) {
            const toward = shoreSpot(
              { x: at.x + (fly.x - at.x) * 0.6, y: at.y + (fly.y - at.y) * 0.6 },
              16,
            )
            if (toward) await jump(toward)
          }
          pose('crouching')
          for (let tries = 0; tries < 4 && active(); tries++) {
            await wait(500 + Math.random() * 700)
            const now = dragonflies().find(candidate => candidate.element === fly.element)
            if (!now) break
            const reach = Math.hypot(now.x - at.x, now.y - at.y)
            if (reach > 24 * scale) continue
            await flick(now)
            if (reach <= 18 * scale) {
              // The fly is gone a while; the stylesheet hides it, and another takes the reeds.
              now.element.dataset.caught = 'true'
              const taken = now.element
              view.setTimeout(() => delete taken.dataset.caught, 25_000)
              pose('swallowing')
              await wait(700)
            }
            break
          }
          pose('sitting')
        } else {
          // Into the water, a slow swim along the bank, and out again onto a fresh spot.
          const shore = at
          await jump(waterSpot())
          frog.dataset.splash = 'true'
          pose('swimming')
          await wait(320)
          delete frog.dataset.splash
          const drift = (Math.random() - 0.5) * 24
          face(drift < 0 ? -1 : 1)
          const swimming = animateScene(
            frog,
            [
              { transform: `translate(${at.x}px, ${at.y}px)` },
              { transform: `translate(${at.x + drift}px, ${at.y + drift * 0.15}px)` },
            ],
            { duration: 3000 + Math.random() * 4000, fill: 'forwards' },
          )
          hop = swimming
          await swimming.finished
          swimming.cancel()
          hop = null
          place({ x: at.x + drift, y: at.y + drift * 0.15 })
          if (!active()) {
            place(shore)
            pose('sitting')
            continue
          }
          const out = shoreSpot(at, 40) ?? shore
          frog.dataset.splash = 'true'
          await jump(out)
          delete frog.dataset.splash
        }
      }
    } catch {
      // A cancelled hop ends the round; the frog sits where it was placed last.
    }
  }

  host.addEventListener('trail:hide', stop, { signal })
  host.addEventListener('trail:show', () => void run(), { signal })
  document.addEventListener('visibilitychange', () => void run(), { signal })
  signal.addEventListener('abort', stop, { once: true })

  return {
    // After each fauna render: the shore frog nearest the pond takes the life frames. `clear`
    // says whether a bank spot is free of the camp and the trees.
    refresh(next: Landscape, free: (point: Point) => boolean = () => true) {
      landscape = next
      clear = free
      let chosen: SVGGElement | null = null
      let best = Infinity
      for (const element of scene.querySelectorAll<SVGGElement>('[data-wetland-animal="frog"]')) {
        const point = { x: Number(element.dataset.groundX), y: Number(element.dataset.groundY) }
        const distance = pondDistance(point, next.pond)
        if (distance < best) {
          best = distance
          chosen = element
        }
      }
      const signature = chosen ? `${chosen.dataset.groundX},${chosen.dataset.groundY}` : ''
      if (chosen === frog && signature === placed && frog?.isConnected) return
      stop()
      placed = signature
      frog = chosen
      if (!frog) return
      at = { x: Number(frog.dataset.groundX), y: Number(frog.dataset.groundY) }
      scale = trailGroundScale(at.y, next)
      frog.dataset.driven = 'true'
      frog.dataset.pose = 'sitting'
      frog.innerHTML =
        `<g class="trail-frog-facing"><g transform="scale(${scale.toFixed(3)})">` +
        `<g class="trail-frog-sit" transform="translate(-4.5 -6)"><g class="trail-frog-body">${pixels(sit, tones, 1)}<path class="trail-frog-throat trail-wetland-belly" d="M4 5h3v1H4Z"/></g></g>` +
        `<g class="trail-frog-leap" transform="translate(-4.5 -6)">${pixels(leap, tones, 1)}</g>` +
        `<g class="trail-frog-swim" transform="translate(-4.5 -2)">${pixels(swim, tones, 1)}<path class="trail-frog-ring trail-pond-ripple" d="M-2 2h13v1H-2Z"/></g>` +
        `</g></g><path class="trail-frog-tongue"/>` +
        `<path class="trail-frog-splash trail-wetland-glint" d="M${(-5 * scale).toFixed(1)} ${(-3 * scale).toFixed(1)}h${(2 * scale).toFixed(1)}v1h-${(2 * scale).toFixed(1)}ZM${(3 * scale).toFixed(1)} ${(-4 * scale).toFixed(1)}h${(2 * scale).toFixed(1)}v1h-${(2 * scale).toFixed(1)}ZM-1 ${(-6 * scale).toFixed(1)}h2v1h-2Z"/>`
      facing = frog.querySelector('.trail-frog-facing')
      tongue = frog.querySelector('.trail-frog-tongue')
      face(at.x < next.pond.x ? 1 : -1)
      void run()
    },
  }
}
