// One leaf cluster of a crown, with what its motion needs read once from the markup.
type Cluster = {
  leaves: SVGGElement
  flip: SVGPathElement | null
  // Scene y of the tree's foot, to skip trees out of view.
  ground: number
  // Cells per unit of push: lift toward the crown top, the tree's own give, needles stiffer.
  reach: number
  cell: number
  // The cluster's centre in the tree, which sets when the gust front reaches it.
  x: number
  y: number
  // Seconds behind the scene's gust clock, from the tree's place along the wind.
  delay: number
  offset: string
  flipped: boolean
}

// Gust arrival, two flutters, release, and a one-cell rebound past rest, as
// [start of step in the period, share of the gust push downwind, share across the wind].
const gustSteps = [
  [0, 0, 0],
  [0.28, 0.45, 0],
  [0.34, 1, 0.5],
  [0.4, 0.7, -0.5],
  [0.47, 1, 0.4],
  [0.54, 0.5, -0.4],
  [0.62, 0.2, 0],
  [0.7, -0.2, 0],
  [0.78, 0, 0],
] as const
const needled = new Set(['pine', 'spruce', 'fir', 'hemlock', 'cypress', 'larch'])
const interval = 50
const reread = 3000

// Crown clusters shift by whole cells as the gust front crosses the scene. A CSS animation per
// cluster made the browser restyle a thousand clusters on every frame; this clock writes a
// cluster's transform only when its cell offset changes, and only for trees in view.
export function mountCrownWind(scene: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) return { refresh() {} }
  const motion = view.matchMedia('(prefers-reduced-motion: reduce)')
  const live = scene.getElementsByClassName('trail-tree-leaves')
  let clusters: Cluster[] = []
  let read = -Infinity
  let frame = 0
  let last = -Infinity
  let moving = false

  const number = (style: CSSStyleDeclaration, name: string) =>
    Number.parseFloat(style.getPropertyValue(name)) || 0
  const collect = (time: number) => {
    read = time
    clusters = []
    for (const leaves of scene.querySelectorAll<SVGGElement>('.trail-tree-leaves')) {
      const tree = leaves.closest<SVGGElement>('.trail-tree-motion')
      const root = leaves.closest<SVGGElement>(
        '.trail-tree, .trail-woodland-tree, .trail-understory',
      )
      const scale = tree
        ? Number.parseFloat(tree.style.getPropertyValue('--trail-tree-wind-scale'))
        : 1
      const lift = number(leaves.style, '--leaf-lift')
      const give = needled.has(root?.dataset.tree ?? '') ? 0.55 : 1
      clusters.push({
        leaves,
        flip: leaves.querySelector<SVGPathElement>('.trail-tree-flip'),
        ground: Number(root?.dataset.groundY ?? 0),
        reach: lift * (Number.isFinite(scale) ? scale : 1) * give,
        cell: number(leaves.style, '--leaf-cell') || 1,
        x: number(leaves.style, '--leaf-x'),
        y: number(leaves.style, '--leaf-y'),
        delay: tree ? number(tree.style, '--trail-weather-delay') : 0,
        offset: leaves.getAttribute('transform') ?? '',
        flipped: false,
      })
    }
  }
  const settle = () => {
    for (const cluster of clusters) {
      if (cluster.offset) cluster.leaves.removeAttribute('transform')
      if (cluster.flipped) cluster.flip?.style.removeProperty('opacity')
      cluster.offset = ''
      cluster.flipped = false
    }
  }
  const windy = () =>
    scene.dataset.weatherWindy === 'true' &&
    scene.dataset.trailVisible === 'true' &&
    !motion.matches
  const running = () =>
    windy() &&
    scene.dataset.weatherMotion !== 'paused' &&
    !document.hidden &&
    scene.dataset.trailRendered === 'true' &&
    (scene.dataset.trailRenderer === 'webgl' || !scene.querySelector('.trail-compositor'))

  const step = (time: number) => {
    frame = 0
    if (signal.aborted || !running()) return
    frame = view.requestAnimationFrame(step)
    if (time - last < interval) return
    last = time
    if (time - read > reread || live.length !== clusters.length) collect(time)
    const style = scene.style
    const pushX = number(style, '--trail-leaf-push-x')
    const pushY = number(style, '--trail-leaf-push-y')
    const restX = number(style, '--trail-leaf-rest-x')
    const restY = number(style, '--trail-leaf-rest-y')
    const period = number(style, '--trail-wind-period') || 9
    const lagX = number(style, '--trail-leaf-lag-x')
    const lagY = number(style, '--trail-leaf-lag-y')
    const flip = number(style, '--trail-leaf-flip')
    const top = -scene.getBoundingClientRect().top - 40
    const bottom = top + view.innerHeight + 200
    const now = time / 1000
    moving = true
    for (const cluster of clusters) {
      if (cluster.ground < top || cluster.ground > bottom) continue
      const { leaves, reach, cell } = cluster
      const local = now - cluster.delay - cluster.x * lagX - cluster.y * lagY
      const u = (((local % period) + period) % period) / period
      let index = gustSteps.length - 1
      while (gustSteps[index][0] > u) index--
      const [, along, across] = gustSteps[index]
      const dx = Math.round((restX + along * pushX - across * 0.45 * pushY) * reach) * cell
      const dy = Math.round((restY + along * pushY + across * 0.45 * pushX) * reach) * cell
      const offset = dx || dy ? `translate(${dx} ${dy})` : ''
      if (offset !== cluster.offset) {
        if (offset) leaves.setAttribute('transform', offset)
        else leaves.removeAttribute('transform')
        cluster.offset = offset
      }
      // Undersides show at the two gust peaks.
      const flipped = flip > 0 && ((u >= 0.34 && u < 0.37) || (u >= 0.47 && u < 0.5))
      if (cluster.flip && flipped !== cluster.flipped) {
        if (flipped) cluster.flip.style.opacity = flip.toFixed(3)
        else cluster.flip.style.removeProperty('opacity')
        cluster.flipped = flipped
      }
    }
  }
  // A paused scene holds its crowns where they are; calm air or reduced motion lets them rest.
  const wake = () => {
    if (!windy() && moving) {
      settle()
      moving = false
    }
    if (!frame && running()) frame = view.requestAnimationFrame(step)
  }
  const watch = new MutationObserver(wake)
  watch.observe(scene, {
    attributes: true,
    attributeFilter: [
      'data-weather-windy',
      'data-weather-motion',
      'data-trail-visible',
      'data-trail-rendered',
      'data-trail-renderer',
    ],
  })
  document.addEventListener('visibilitychange', wake, { signal })
  motion.addEventListener('change', wake, { signal })
  wake()
  signal.addEventListener(
    'abort',
    () => {
      view.cancelAnimationFrame(frame)
      watch.disconnect()
    },
    { once: true },
  )
  return {
    // The forest was drawn again: read the new clusters on the next step.
    refresh() {
      read = -Infinity
      wake()
    },
  }
}
