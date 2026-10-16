import type { TreePlacement } from './scenery'
import { pixels } from './pixels'

type Point = { x: number; y: number }

// A jay perches on the largest oak and caches acorns where the succession plan roots the next
// trees. Nothing is stored: a visitor who watched it dig finds the seedling there a week later.
const perched = ['....PP..', '...PPPKE', 'KKPPPPP.', 'KKBWPPP.', '..KKPP..', '...K.K..']
const wingsUp = ['...KK...', '..KBW.PP', 'KKPPPPKE', '..PPPP..']
const wingsDown = ['......PP', 'KKPPPPKE', '..KBWP..', '...KK...']
const tones = { P: 'trail-jay-body', K: 'trail-jay-ink', B: 'trail-jay-blue', W: 'trail-jay-white' }
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
    `<g class="trail-jay-perched" transform="translate(0 ${-2 * cell})"><g class="trail-jay-peck">${pixels(perched, tones, cell)}</g></g>` +
    `<g class="trail-jay-up">${pixels(wingsUp, tones, cell)}</g>` +
    `<g class="trail-jay-down">${pixels(wingsDown, tones, cell)}</g>`
  jay.append(facing)
  const acorns = document.createElementNS(svg, 'g')
  acorns.setAttribute('class', 'trail-jay-acorns')
  let perch: Point | null = null
  let caches: Point[] = []
  let next = 0
  let timer = 0
  let flight: Animation | null = null
  let placed = ''

  const place = (point: Point) => {
    jay.setAttribute('transform', `translate(${Math.round(point.x)} ${Math.round(point.y)})`)
  }
  const face = (direction: number) => {
    facing.setAttribute('transform', `scale(${direction} 1) translate(${-4 * cell} ${-4 * cell})`)
  }
  const stop = () => {
    view.clearTimeout(timer)
    flight?.cancel()
    flight = null
    jay.dataset.pose = 'perched'
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
    flight = jay.animate(frames, { duration: Math.min(3000, 600 + distance * 3), fill: 'forwards' })
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
    if (!perch || !caches.length) return
    try {
      // Night, reduced motion or a hidden scene keep the jay on its perch until the next round.
      while (!signal.aborted) {
        await wait(5000 + Math.random() * 9000)
        if (!active() || !perch) continue
        const cache = caches[next++ % caches.length]
        await fly(perch, cache)
        jay.dataset.pose = 'digging'
        await wait(1600)
        if (!acorns.querySelector(`[data-cache="${cache.x},${cache.y}"]`))
          acorns.insertAdjacentHTML(
            'beforeend',
            `<path class="trail-jay-acorn" data-cache="${cache.x},${cache.y}" d="M${cache.x} ${cache.y}h2v2h-2Z"/>`,
          )
        await fly(cache, perch)
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
    refresh(trees: readonly TreePlacement[]) {
      let found: Point | null = null
      let best = 0
      for (const tree of scene.querySelectorAll<SVGGElement>(
        '.trail-tree[data-tree="oak"][data-tree-depth="meadow"]',
      )) {
        const box = tree.getBBox()
        const size = box.width * box.height
        if (size <= best) continue
        best = size
        found = {
          x: Number(tree.dataset.groundX) + box.x + box.width * 0.5,
          y: Number(tree.dataset.groundY) + box.y + 3,
        }
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
      caches = chosen
      next = 0
      acorns.replaceChildren()
      if (!perch || !caches.length) {
        jay.remove()
        acorns.remove()
        return
      }
      face(1)
      place(perch)
      layer.append(acorns, jay)
      void run()
    },
  }
}
