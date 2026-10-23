import type { TreePlacement } from './scenery'
import { pixels } from './pixels'
import { animateScene } from './scene-animation'

type Point = { x: number; y: number }

// A jay perches on the largest oak and caches acorns where the succession plan roots the next
// trees. Nothing is stored: a visitor who watched it dig finds the seedling there a week later.
// A blue jay: crest, white face, black necklace and barred blue wings, so it does not read as
// another brown songbird against the crowns. Flight frames use the grey belly tone for the face and
// underparts, so the bird keeps its shape against the pale sky. Each frame faces right.
const perched = [
  '.......B..',
  '......BBB.',
  '.....BWKWK',
  '....BBWWW.',
  '...BBBKKK.',
  '..BDWDBWW.',
  '.BBDBDBWG.',
  'BKBBBBGWG.',
  'K.B...K.K.',
]
const wingsUp = ['..WDB.....', '...WDB.B..', '....BBBBB.', 'BDBBBBGKGK', '.W..GGGK..']
const wingsDown = [
  '.......B..',
  '......BBB.',
  'BDBBBBGKGK',
  '.W.BBGGK..',
  '...BDW....',
  '..BDW.....',
  '..W.......',
]
// A dead limb over the crown, as on a stag-headed oak: the jay stands on the limb, and the limb
// runs down into the leaves, so the bird is not standing on foliage. The lit bark is on top.
const limb = [
  '..LLLLLLLLL',
  '..TTTTTTTTT',
  '.LT........',
  '.LT........',
  'LT.........',
  'LT.........',
  'LT.........',
  'LT.........',
]
const limbTones = { L: 'trail-tree-bark', T: 'trail-tree-trunk' }
// A downy woodpecker takes the dead limb while the jay is away: black and white, a red nape,
// clinging upright to the rising part of the limb. It leaves as the jay comes back.
const woodpecker = ['.R...', 'KKK..', 'WKK..', '.WKK.', '.WKK.', '.WK..', '..K..']
const woodpeckerTones = { K: 'trail-jay-ink', W: 'trail-jay-white', R: 'trail-woodpecker-nape' }
const tones = {
  B: 'trail-jay-blue',
  D: 'trail-jay-bar',
  K: 'trail-jay-ink',
  W: 'trail-jay-white',
  G: 'trail-jay-belly',
}
const cell = 2
// Forest years ahead the jay plants for: two months of real weeks.
const horizon = 8
const broadleaves = new Set(['oak', 'maple', 'birch', 'aspen', 'willow'])

export function mountJay(scene: HTMLElement, host: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const layer = scene.querySelector<SVGGElement>('.trail-birds')
  if (!view || !layer) return { refresh() {} }
  const motion = view.matchMedia('(prefers-reduced-motion: reduce)')
  const svg = 'http://www.w3.org/2000/svg'
  const jay = document.createElementNS(svg, 'g')
  jay.setAttribute('class', 'trail-jay')
  jay.dataset.pose = 'perched'
  const facing = document.createElementNS(svg, 'g')
  facing.innerHTML =
    `<g class="trail-jay-perched" transform="translate(0 ${-9 * cell})"><g class="trail-jay-peck">${pixels(perched, tones, cell)}</g></g>` +
    `<g class="trail-jay-up" transform="translate(0 ${-7 * cell})">${pixels(wingsUp, tones, cell)}</g>` +
    `<g class="trail-jay-down" transform="translate(0 ${-6 * cell})">${pixels(wingsDown, tones, cell)}</g>`
  jay.append(facing)
  const acorns = document.createElementNS(svg, 'g')
  acorns.setAttribute('class', 'trail-jay-acorns')
  const roost = document.createElementNS(svg, 'g')
  roost.setAttribute('class', 'trail-jay-roost')
  roost.setAttribute('aria-hidden', 'true')
  roost.innerHTML =
    pixels(limb, limbTones, cell) +
    `<g class="trail-woodpecker" transform="translate(${2 * cell} ${3 * cell})"><g class="trail-woodpecker-drum">${pixels(woodpecker, woodpeckerTones, cell)}</g></g>`
  let perch: Point | null = null
  let oak: Point | null = null
  let caches: Point[] = []
  let larder: Point[] = []
  let next = 0
  let timer = 0
  let flight: Animation | null = null
  let placed = ''

  const place = (point: Point) => {
    jay.setAttribute('transform', `translate(${Math.round(point.x)} ${Math.round(point.y)})`)
  }
  // The limb's top row is the jay's footing and its origin the sixth column. The limb turns with
  // the bird, rising behind it and reaching out the way it faces; it turns while the jay is away.
  const roostAt = (point: Point, direction: number) => {
    roost.setAttribute(
      'transform',
      `translate(${Math.round(point.x)} ${Math.round(point.y)}) scale(${direction} 1) translate(${-6 * cell} 0)`,
    )
  }
  const season = () => scene.dataset.foliageSeason ?? 'summer'
  const mark = (point: Point) => acorns.querySelector(`[data-cache="${point.x},${point.y}"]`)
  // Where the jay goes this trip. Autumn and early winter are for caching: the plan's sites, which
  // it forgets and which root, and a larder round the oak, which it remembers. Through winter and
  // spring it digs the larder up again. With the larder empty, and all summer, it forages in the
  // grass under the oak and leaves nothing.
  const trip = (): { to: Point; act: 'cache' | 'retrieve' | 'forage' } | null => {
    if (!oak) return null
    const now = season()
    const stored = larder.filter(point => mark(point))
    const stash = [...caches, ...larder]
    if (now === 'autumn' && stash.length) return { to: stash[next++ % stash.length], act: 'cache' }
    if ((now === 'winter' || now === 'spring') && stored.length)
      return { to: stored[next++ % stored.length], act: 'retrieve' }
    if (now === 'winter' && stash.length) return { to: stash[next++ % stash.length], act: 'cache' }
    return {
      to: {
        x: Math.round(oak.x + (Math.random() - 0.5) * 140),
        y: Math.round(oak.y + 6 + Math.random() * 40),
      },
      act: 'forage',
    }
  }
  const face = (direction: number) => {
    facing.setAttribute('transform', `scale(${direction} 1) translate(${-5 * cell} 0)`)
  }
  const stop = () => {
    view.clearTimeout(timer)
    flight?.cancel()
    flight = null
    jay.dataset.pose = 'perched'
    delete roost.dataset.guest
    if (perch) place(perch)
  }
  // An arc in whole pixels; the keyframes carry the hops, so the frames between are plain moves.
  const fly = (from: Point, to: Point) => {
    face(to.x < from.x ? -1 : 1)
    jay.dataset.pose = 'flying'
    const distance = Math.hypot(to.x - from.x, to.y - from.y)
    const lift = Math.min(60, 18 + distance * 0.2)
    const steps = Math.max(6, Math.round(distance / 14))
    const frames = Array.from({ length: steps + 1 }, (_, index) => {
      const t = index / steps
      const x = Math.round(from.x + (to.x - from.x) * t)
      const y = Math.round(from.y + (to.y - from.y) * t - Math.sin(t * Math.PI) * lift)
      return { transform: `translate(${x}px, ${y}px)` }
    })
    // About 300 px/s across the meadow; a long cache flight still lands within three seconds.
    flight = animateScene(jay, frames, {
      duration: Math.min(3000, 600 + distance * 3),
      fill: 'forwards',
    })
    return flight.finished
  }
  const wait = (ms: number) => new Promise<void>(resolve => (timer = view.setTimeout(resolve, ms)))
  const active = () =>
    !signal.aborted &&
    !host.hidden &&
    !document.hidden &&
    !motion.matches &&
    scene.dataset.lighting !== 'moonlit'
  const run = async () => {
    stop()
    if (!perch) return
    try {
      // Night, reduced motion or a hidden scene keep the jay on its perch until the next round.
      while (!signal.aborted) {
        await wait(5000 + Math.random() * 9000)
        if (!active() || !perch) continue
        const plan = trip()
        if (!plan) continue
        const { to, act } = plan
        await fly(perch, to)
        jay.dataset.pose = 'digging'
        // The woodpecker takes the empty limb on about half the trips.
        if (Math.random() < 0.5) roost.dataset.guest = 'woodpecker'
        await wait(act === 'forage' ? 1200 : 1600)
        if (act === 'cache' && !mark(to))
          acorns.insertAdjacentHTML(
            'beforeend',
            `<path class="trail-jay-acorn" data-cache="${to.x},${to.y}" d="M${to.x} ${to.y}h2v2h-2Z"/>`,
          )
        if (act === 'retrieve') mark(to)?.remove()
        delete roost.dataset.guest
        roostAt(perch, perch.x < to.x ? -1 : 1)
        await fly(to, perch)
        flight?.cancel()
        flight = null
        place(perch)
        jay.dataset.pose = 'perched'
      }
    } catch {
      // A cancelled flight ends the round; the jay is back on its perch.
    }
  }

  host.addEventListener('trail:hide', stop, { signal })
  host.addEventListener('trail:show', () => void run(), { signal })
  document.addEventListener('visibilitychange', () => void run(), { signal })
  signal.addEventListener('abort', stop, { once: true })

  return {
    // Placed once per forest layout: the perch from the drawn oak crowns, the caches from the plan.
    // `clear` says whether a larder spot may lie at a point: off the pond, the creek and the tents.
    refresh(trees: readonly TreePlacement[], clear: (point: Point) => boolean = () => true) {
      let found: Point | null = null
      let foot: Point | null = null
      let best = 0
      for (const tree of scene.querySelectorAll<SVGGElement>(
        '.trail-tree[data-tree="oak"][data-tree-depth="meadow"]',
      )) {
        const box = tree.getBBox()
        const size = box.width * box.height
        if (size <= best) continue
        best = size
        // The limb top clears the crown by 6 px and runs 16 px down into it.
        found = {
          x: Number(tree.dataset.groundX) + box.x + box.width * 0.5,
          y: Number(tree.dataset.groundY) + box.y - 6,
        }
        foot = { x: Number(tree.dataset.groundX), y: Number(tree.dataset.groundY) }
      }
      const soon = trees.filter(
        tree =>
          !tree.foothill &&
          tree.successor &&
          tree.succeeds !== undefined &&
          tree.succeeds > 0 &&
          tree.succeeds <= horizon,
      )
      // Oaks first; a jay caches other mast too, and nothing at all for the conifers.
      const planted = [
        ...soon.filter(tree => tree.successor === 'oak'),
        ...soon.filter(tree => tree.successor !== 'oak' && broadleaves.has(tree.successor!)),
      ]
      const chosen = planted
        .slice(0, 4)
        .map(tree => ({ x: Math.round(tree.x), y: Math.round(tree.y) }))
      // The forest refreshes often; a round in flight carries on unless the plan moved.
      const signature = JSON.stringify([found, chosen])
      if (signature === placed && jay.isConnected) return
      placed = signature
      stop()
      perch = found
      oak = foot
      caches = chosen
      // Three larder spots in the grass round the oak's foot, the ones the jay remembers: the
      // first three of a ring of candidates that lie clear of the water and the camp.
      larder = []
      if (foot) {
        const base = foot
        const ring = [-56, 30, 66, -84, 92, -28, 50, -110, 118].map((dx, index) => ({
          x: Math.round(base.x + dx),
          y: Math.round(base.y + 10 + ((index * 7) % 3) * 12),
        }))
        larder = ring.filter(spot => clear(spot)).slice(0, 3)
      }
      next = 0
      acorns.replaceChildren()
      if (!perch) {
        jay.remove()
        acorns.remove()
        roost.remove()
        return
      }
      face(1)
      place(perch)
      roostAt(perch, 1)
      // The roost sits under the birds layer, which hides at night; a dead limb stays in the dark.
      layer.before(roost)
      layer.append(acorns, jay)
      void run()
    },
  }
}
