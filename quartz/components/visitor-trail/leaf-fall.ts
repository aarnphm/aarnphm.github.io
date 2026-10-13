import { trailRandom } from './random'

type Leaf = {
  x: number
  y: number
  ground: number
  vx: number
  vy: number
  sway: number
  phase: number
  tone: string
  shape: number
  cell: number
  rest: number | null
}
type Source = {
  x: number
  y: number
  width: number
  height: number
  ground: number
  tones: string[]
}

// Two-frame pixel leaves: a leaf shows its face, then turns edge-on as it tumbles.
const shapes = [
  [
    ['.##', '##.'],
    ['###', '...'],
  ],
  [
    ['#.', '##', '.#'],
    ['##', '##', '..'],
  ],
  [
    ['.#.', '###', '.#.'],
    ['...', '###', '...'],
  ],
]
// Each broadleaf drops its own autumn colour; conifers keep their needles.
const kinds: Record<string, string[]> = {
  maple: ['#c03e35', '#da702c', '#c03e35', '#d0a215'],
  oak: ['#bc5215', '#8b6a48', '#a0522d'],
  aspen: ['#d0a215', '#e5b72a', '#da702c'],
  birch: ['#d0a215', '#e5b72a', '#a0af54'],
  larch: ['#d0a215', '#bc8a2c'],
  willow: ['#a0af54', '#d0a215'],
}
const restLife = 24_000

// Leaves fall only in autumn. It peaks around 20 October in the north and 20 April in the south;
// the season the visitor sees decides which.
function leafRate(season: string | undefined, now: Date) {
  if (season !== 'autumn') return 0
  const day = (now.getTime() - Date.UTC(now.getUTCFullYear(), 0, 1)) / 86_400_000
  const peak = day > 180 ? 292 : 110
  return 0.02 + 0.06 * Math.exp(-((day - peak) ** 2) / (2 * 22 ** 2))
}

// Leaves drop from each broadleaf crown, sway and tumble in whole-pixel steps, race sideways as
// the scene's gust front passes the tree, and settle on the ground around its trunk before they
// fade. The pet walking through a drift kicks the leaves up again.
export function mountLeafFall(scene: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view || view.matchMedia('(prefers-reduced-motion: reduce)').matches) return
  const canvas = document.createElement('canvas')
  canvas.className = 'trail-leaf-fall'
  canvas.setAttribute('aria-hidden', 'true')
  scene.append(canvas)
  const context = canvas.getContext('2d')
  if (!context) return canvas.remove()
  const random = trailRandom(Math.floor(Math.random() * 4294967296))
  const leaves: Leaf[] = []
  let sources: Source[] = []
  let measured = -Infinity
  let frame = 0
  let last = 0
  let pet: DOMRect | null = null
  let petRead = 0

  // Trees move when the forest is redrawn, so the crowns are read again every few seconds.
  const measure = (time: number) => {
    measured = time
    const ratio = view.devicePixelRatio || 1
    const width = Math.round(scene.clientWidth * ratio)
    const height = Math.round(scene.clientHeight * ratio)
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
      context.imageSmoothingEnabled = false
    }
    sources = []
    for (const tree of scene.querySelectorAll<SVGGElement>('.trail-tree[data-tree]')) {
      const tones = kinds[tree.dataset.tree ?? '']
      if (!tones) continue
      const box = tree.getBBox()
      const x = Number(tree.dataset.groundX)
      const ground = Number(tree.dataset.groundY)
      if (!box.width || !Number.isFinite(x) || !Number.isFinite(ground)) continue
      // The crown is the upper part of the tree's box; the trunk takes the rest.
      sources.push({
        x: x + box.x,
        y: ground + box.y,
        width: box.width,
        height: box.height * 0.55,
        ground,
        tones,
      })
    }
  }

  const wind = () => {
    const number = (name: string) => Number.parseFloat(scene.style.getPropertyValue(name)) || 0
    return {
      pushX: number('--trail-leaf-push-x'),
      restX: number('--trail-leaf-rest-x'),
      gust: number('--trail-gust-strength'),
      period: number('--trail-wind-period') || 8,
      lagX: number('--trail-leaf-lag-x'),
      lagY: number('--trail-leaf-lag-y'),
    }
  }
  // The same gust front the crowns lean to: it reaches a point after a lag set by its position.
  const gustAt = (time: number, x: number, y: number, air: ReturnType<typeof wind>) => {
    const phase =
      (((time / 1000 - x * air.lagX - y * air.lagY) % air.period) + air.period) % air.period
    const u = phase / air.period
    return u < 0.35 ? Math.sin((Math.PI * u) / 0.35) : 0
  }

  const step = (time: number) => {
    frame = 0
    if (signal.aborted) return
    const dt = Math.min(0.05, last ? (time - last) / 1000 : 0)
    last = time
    const visible =
      scene.dataset.trailVisible === 'true' && scene.dataset.weatherMotion !== 'paused'
    if (scene.dataset.trailVisible !== 'true') leaves.length = 0
    const season = scene.dataset.season
    const rate = visible ? leafRate(season, new Date()) : 0
    if (rate > 0 && time - measured > 5000) measure(time)
    const air = wind()
    const bounds = scene.getBoundingClientRect()
    // Only trees in view drop leaves, and only that band of the canvas is redrawn.
    const top = -bounds.top - 40
    const bottom = top + view.innerHeight + 80
    if (rate > 0)
      for (const tree of sources) {
        if (tree.ground < top || tree.y > bottom) continue
        const gust = gustAt(time, tree.x, tree.y, air)
        if (random() > dt * rate * (0.6 + gust * air.gust * 6)) continue
        const cell = tree.width > 26 ? 2 : 1
        leaves.push({
          x: tree.x + random() * tree.width,
          y: tree.y + random() * tree.height,
          ground: tree.ground - 2 + random() * 10,
          vx: 0,
          vy: 6 + random() * 6,
          sway: (3 + random() * 6) * cell,
          phase: random() * Math.PI * 2,
          tone: tree.tones[Math.floor(random() * tree.tones.length)],
          shape: Math.floor(random() * shapes.length),
          cell,
          rest: null,
        })
      }
    if (time - petRead > 200) {
      petRead = time
      pet =
        document
          .querySelector('.garden-pet:not(.garden-pet-friend) .pet-body')
          ?.getBoundingClientRect() ?? null
    }
    for (let index = leaves.length - 1; index >= 0; index--) {
      const leaf = leaves[index]
      if (leaf.rest !== null) {
        const x = leaf.x + bounds.left
        const y = leaf.y + bounds.top
        if (
          pet &&
          x > pet.left - 2 &&
          x < pet.right + 2 &&
          y > pet.bottom - 10 &&
          y < pet.bottom + 6
        ) {
          // Kicked up: a short hop away from the pet, then down again a little further on.
          leaf.rest = null
          leaf.vy = -28 - random() * 20
          leaf.vx = Math.sign(x - (pet.left + pet.right) / 2 || 1) * 60
          leaf.ground = leaf.y + random() * 4
        } else if (time - leaf.rest > restLife) leaves.splice(index, 1)
        continue
      }
      const gust = gustAt(time, leaf.x, leaf.y, air)
      const drift = (air.restX * 6 + air.pushX * gust * 40) * (0.6 + leaf.sway / 24)
      leaf.vx += (drift - leaf.vx) * Math.min(1, dt * 1.5)
      leaf.vy += (10 + leaf.sway - gust * air.gust * 14 - leaf.vy) * Math.min(1, dt * 1.2)
      leaf.phase += dt * (1.6 + gust * 3)
      leaf.x += (leaf.vx + Math.cos(leaf.phase) * leaf.sway) * dt
      leaf.y += leaf.vy * dt
      if (leaf.y >= leaf.ground) {
        leaf.y = leaf.ground
        leaf.rest = time
      }
      if (leaf.x < -20 || leaf.x > scene.clientWidth + 20) leaves.splice(index, 1)
    }
    context.clearRect(0, top, scene.clientWidth, bottom - top)
    for (const leaf of leaves) {
      if (leaf.y < top || leaf.y > bottom) continue
      const frames = shapes[leaf.shape]
      // Resting leaves lie face up and fade; falling leaves turn edge-on for part of each sway.
      const rows = leaf.rest !== null ? frames[0] : frames[Math.sin(leaf.phase * 2) > 0.3 ? 1 : 0]
      const { cell } = leaf
      const ox = Math.round(leaf.x / cell) * cell
      const oy = Math.round((leaf.y - rows.length * cell) / cell) * cell
      context.globalAlpha =
        leaf.rest === null ? 1 : Math.max(0, Math.min(1, (restLife - (time - leaf.rest)) / 4000))
      context.fillStyle = leaf.tone
      rows.forEach((row, ry) =>
        [...row].forEach((mark, rx) => {
          if (mark === '#') context.fillRect(ox + rx * cell, oy + ry * cell, cell, cell)
        }),
      )
    }
    context.globalAlpha = 1
    if (rate > 0 || leaves.length) frame = view.requestAnimationFrame(step)
    else {
      last = 0
      context.clearRect(0, 0, scene.clientWidth, scene.clientHeight)
    }
  }
  // A hidden page never runs its frames, so the fall waits for the page to show again.
  const wake = () => {
    if (!frame && !signal.aborted && !document.hidden) frame = view.requestAnimationFrame(step)
  }
  document.addEventListener('visibilitychange', wake, { signal })

  // A change of season, a shown trail or resumed motion starts the fall again.
  const watch = new MutationObserver(wake)
  watch.observe(scene, {
    attributes: true,
    attributeFilter: ['data-season', 'data-trail-visible', 'data-weather-motion'],
  })
  view.addEventListener(
    'resize',
    () => {
      measured = -Infinity
      leaves.length = 0
    },
    { signal },
  )
  wake()
  signal.addEventListener(
    'abort',
    () => {
      view.cancelAnimationFrame(frame)
      watch.disconnect()
      canvas.remove()
    },
    { once: true },
  )
}
