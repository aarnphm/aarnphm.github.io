import { creekDistance, pondDistance, type Landscape, type Point } from './landscape'
import { pixels } from './pixels'
import { riverArt } from './river-art'

// The fishing bear works the creek instead of swiping on a loop. It reads the fish's own jump
// cycle and strikes as the fish breaks the surface, misses at first and lands more as it keeps
// at it, wades in when the bank is not working, eats what it takes, naps full, and walks to
// the other pool along the bank when a pool goes quiet. Autumn runs the salmon thicker; ice
// sends the bear to its den. Nothing is stored between visits.
const fishTones = {
  S: 'trail-river-fish-silver',
  F: 'trail-river-fish-flank',
  D: 'trail-river-fish-fin',
  E: 'trail-animal-eye',
}
const sleep = ['ZZZ', '..Z', '.Z.', 'Z..', 'ZZZ']
// The strike: the foreleg stretched from the shoulder down into the water, drawn in the bear's
// own 20 by 12 grid, in place of the resting paw. A moved paw alone floats off the body.
const strikeFrame = [
  '....................',
  '....................',
  '....................',
  '....................',
  '....................',
  '....................',
  '....................',
  '.............CCC....',
  '..............CCC...',
  '...............CCC..',
  '................CCC.',
  '.................BBB',
]
const coat = { C: 'trail-bear-coat', B: 'trail-bear-paw' }
const strikeAt = 0.7

type Site = { fish: SVGGElement; bank: Point; scale: number }

export function mountRiverBear(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) return { refresh() {} }
  const motion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let bear: SVGGElement | null = null
  let facing: SVGGElement | null = null
  let landscape: Landscape | null = null
  let sites: Site[] = []
  let site = 0
  let at: Point = { x: 0, y: 0 }
  let scale = 1
  let skill = 0.3
  let misses = 0
  let meals = 0
  let wading = false
  let timer = 0
  let walk: Animation | null = null
  let placed = ''
  let runSeason = ''

  const place = (point: Point) => {
    if (!bear) return
    at = { x: Math.round(point.x * 2) / 2, y: Math.round(point.y * 2) / 2 }
    bear.setAttribute('transform', `translate(${at.x} ${at.y})`)
    bear.dataset.groundX = String(at.x)
    bear.dataset.groundY = String(at.y)
  }
  const face = (direction: number) => facing?.setAttribute('transform', `scale(${direction} 1)`)
  const pose = (name: string) => {
    if (bear) bear.dataset.pose = name
  }
  const wade = (on: boolean) => {
    wading = on
    if (bear) bear.dataset.wading = String(on)
  }
  const wait = (ms: number) => new Promise<void>(resolve => (timer = view.setTimeout(resolve, ms)))
  const frozen = () => scene.dataset.waterFrozen === 'true'
  const active = () =>
    !signal.aborted &&
    !host.hidden &&
    !document.hidden &&
    !motion.matches &&
    !!bear?.isConnected &&
    !frozen() &&
    scene.dataset.weatherMotion !== 'paused'
  // Ice: the bear is in its den and nothing walks round it. Thaw brings it back to the bank.
  const den = () => {
    if (!bear) return
    const denned = frozen()
    if ((bear.dataset.denned === 'true') === denned) return
    bear.dataset.denned = String(denned)
    bear.dataset.enabled = String(!denned)
  }
  // Autumn is the run: the fish of every pool jump half again as often, and the bear knows it.
  const season = () => {
    const now = scene.dataset.foliageSeason ?? 'summer'
    if (now === runSeason) return
    runSeason = now
    for (const { fish } of sites) {
      const base = Number(fish.dataset.cycle)
      if (Number.isFinite(base))
        fish.style.setProperty(
          '--river-cycle',
          `${(now === 'autumn' ? base * 0.65 : base).toFixed(2)}s`,
        )
    }
  }
  const stop = () => {
    view.clearTimeout(timer)
    walk?.cancel()
    walk = null
    if (bear) {
      pose('watching')
      place(at)
    }
  }
  // Where the current pool's fish is in its jump: the stylesheet animation's progress.
  const phase = () => {
    const jump = sites[site]?.fish.querySelector('.trail-river-jump')
    const animation = jump?.getAnimations()[0]
    if (!animation) return null
    const timing = animation.effect?.getComputedTiming()
    if (!timing || typeof timing.duration !== 'number' || timing.progress == null) return null
    return { progress: timing.progress, duration: timing.duration }
  }
  const stand = (): Point => {
    const home = sites[site].bank
    return wading ? { x: home.x + 6 * scale, y: home.y + 3 * scale } : home
  }
  // A walk along the bank at a bear's pace, by whole pixels, with the lumber in the stylesheet.
  const go = async (to: Point) => {
    if (!bear) return
    face(to.x < at.x ? -1 : 1)
    pose('walking')
    const from = at
    const distance = Math.hypot(to.x - from.x, to.y - from.y)
    walk = bear.animate(
      [
        { transform: `translate(${from.x}px, ${from.y}px)` },
        { transform: `translate(${to.x}px, ${to.y}px)` },
      ],
      { duration: Math.max(300, (distance / (22 * scale)) * 1000), fill: 'forwards' },
    )
    const half = view.setTimeout(
      () => {
        // The depth sort reads the foot; the half-way foot keeps the bear among the right trees.
        if (bear) bear.dataset.groundY = String(Math.round((from.y + to.y) / 2))
      },
      (walk.effect?.getComputedTiming().duration as number) / 2,
    )
    try {
      await walk.finished
    } finally {
      view.clearTimeout(half)
    }
    walk.cancel()
    walk = null
    place(to)
    pose('watching')
  }
  // The bank between two pools, off the water: straight if it stays clear, else bent away from
  // the creek at the middle. Null when no such walk exists; the bear stays put.
  const route = (from: Point, to: Point): Point[] | null => {
    if (!landscape) return null
    const clear = (a: Point, b: Point) => {
      const steps = Math.max(2, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / 8))
      for (let index = 0; index <= steps; index++) {
        const point = {
          x: a.x + ((b.x - a.x) * index) / steps,
          y: a.y + ((b.y - a.y) * index) / steps,
        }
        if (creekDistance(point, landscape!) < 9 || pondDistance(point, landscape!.pond) < 1.3)
          return false
      }
      return true
    }
    if (clear(from, to)) return [to]
    const middle = { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 }
    for (const push of [36, 72, 110]) {
      for (const side of [-1, 1]) {
        const bend = { x: middle.x - 0.9 * push * side, y: middle.y - 0.45 * push * side }
        if (clear(from, bend) && clear(bend, to)) return [bend, to]
      }
    }
    return null
  }
  const strike = async () => {
    const fish = sites[site]?.fish
    const timing = phase()
    if (!fish || !timing || fish.dataset.caught) {
      await wait(3000)
      return
    }
    // Wait for the fish to show; the strike lands as it clears the water.
    const until = ((strikeAt - timing.progress + 1) % 1) * timing.duration
    await wait(until)
    if (!active()) return
    pose('striking')
    await wait(260)
    const odds = skill + (wading ? 0.18 : 0) + (runSeason === 'autumn' ? 0.12 : 0)
    if (Math.random() < odds) {
      misses = 0
      meals++
      skill = Math.min(0.75, skill + 0.04)
      fish.dataset.caught = 'true'
      view.setTimeout(() => delete fish.dataset.caught, 14_000 + Math.random() * 10_000)
      pose('caught')
      await wait(900)
      if (wading) {
        // Out of the water to eat, and a shake on the bank.
        wade(false)
        await go(stand())
        pose('shaking')
        await wait(800)
      }
      pose('eating')
      await wait(4500 + Math.random() * 3000)
      pose('watching')
      return
    }
    misses++
    skill = Math.min(0.75, skill + 0.07)
    pose('watching')
    await wait(400)
    if (misses === 3 && !wading) {
      wade(true)
      await go(stand())
    } else if (misses >= 6) {
      misses = 0
      const other = sites.length > 1 ? (site + 1) % sites.length : site
      if (other !== site) {
        if (wading) {
          wade(false)
          await go(stand())
          pose('shaking')
          await wait(800)
        }
        const legs = route(at, sites[other].bank)
        if (legs) {
          site = other
          for (const leg of legs) {
            if (!active()) return
            await go(leg)
          }
          scale = sites[site].scale
          bear!.dataset.collisionRadius = (16 * scale).toFixed(2)
          face(1)
        }
      }
    }
  }
  const run = async () => {
    stop()
    if (!bear) return
    try {
      while (!signal.aborted) {
        await wait(1500 + Math.random() * 2500)
        den()
        if (!active()) continue
        season()
        if (meals >= 2) {
          // Full: a nap on the bank, then back to the water with a fresh eye.
          meals = 0
          if (wading) {
            wade(false)
            await go(stand())
          }
          pose('napping')
          await wait(24_000 + Math.random() * 16_000)
          skill = Math.max(0.3, skill - 0.1)
          pose('watching')
          continue
        }
        await strike()
      }
    } catch {
      // A cancelled walk ends the round; the bear stands where it was placed last.
    }
  }

  host.addEventListener('trail:hide', stop, { signal })
  host.addEventListener('trail:show', () => void run(), { signal })
  document.addEventListener('visibilitychange', () => void run(), { signal })
  signal.addEventListener('abort', stop, { once: true })

  return {
    // After each fauna render: the drawn river bear takes the life frames, the fish of each
    // pool give the pools. The stylesheet's swipe loop stays for a reduced-motion visitor.
    refresh(next: Landscape) {
      landscape = next
      const drawn = scene.querySelector<SVGGElement>('.trail-river-bear')
      const fishes = [...scene.querySelectorAll<SVGGElement>('.trail-river-fish[data-site]')].sort(
        (a, b) => Number(a.dataset.site) - Number(b.dataset.site),
      )
      const signature = drawn
        ? `${drawn.dataset.groundX},${drawn.dataset.groundY}|${fishes.map(fish => fish.dataset.bankX).join(',')}`
        : ''
      if (drawn === bear && signature === placed && bear?.isConnected) return
      stop()
      placed = signature
      bear = drawn
      sites = fishes.map(fish => ({
        fish,
        bank: { x: Number(fish.dataset.bankX), y: Number(fish.dataset.bankY) },
        scale: Number(fish.dataset.scale) || 1,
      }))
      for (const { fish } of sites) {
        if (!fish.dataset.cycle) {
          const cycle = parseFloat(fish.style.getPropertyValue('--river-cycle'))
          if (Number.isFinite(cycle)) fish.dataset.cycle = cycle.toFixed(2)
        }
      }
      runSeason = ''
      if (!bear || motion.matches || !sites.length) return
      site = 0
      skill = 0.3
      misses = 0
      meals = 0
      wading = false
      at = { x: Number(bear.dataset.groundX), y: Number(bear.dataset.groundY) }
      scale = sites[0].scale
      facing = bear.querySelector('.trail-river-bear-facing')
      const body = bear.querySelector<SVGGElement>('.trail-river-bear-body')
      if (!facing || !body) return
      bear.dataset.driven = 'true'
      bear.dataset.pose = 'watching'
      bear.dataset.wading = 'false'
      // The stance wraps the drawn bear, so a pose can move it without touching the body's own
      // transform; the catch hangs from the muzzle, the sleep marks float over the head.
      body.innerHTML =
        `<g class="trail-river-bear-stance">${body.innerHTML}<g class="trail-river-strike">${pixels(strikeFrame, coat, 1.5)}</g><g class="trail-river-catch" transform="translate(28.5 10) rotate(78)"><g transform="translate(-4.5 -2.5)">${pixels(riverArt.fish, fishTones, 1)}</g></g>` +
        `<g class="trail-river-nap" transform="translate(21 -7)">${pixels(sleep, { Z: 'trail-bear-shade' }, 1)}</g>` +
        `<path class="trail-river-drops trail-ravine-foam" d="M-2 6h1.5v1.5H-2ZM30 4h1.5v1.5H30ZM12 -2h1.5v1.5H12Z"/></g>`
      const water = document.createElementNS('http://www.w3.org/2000/svg', 'path')
      water.setAttribute('class', 'trail-river-wade trail-ravine-water')
      water.setAttribute(
        'd',
        `M${(-16 * scale).toFixed(1)} ${(-3.5 * scale).toFixed(1)}h${(32 * scale).toFixed(1)}v${(3.5 * scale).toFixed(1)}h-${(32 * scale).toFixed(1)}Z`,
      )
      bear.append(water)
      face(1)
      place(at)
      den()
      void run()
    },
  }
}
