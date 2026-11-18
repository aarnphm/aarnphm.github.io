import type { MotionTuning } from './motion-tuning'
import {
  frontStrength,
  loopAlong,
  loopUp,
  sampleWind,
  windTables,
  type WindShape,
} from './wind-field'

// One piece the wind moves: a leaf clump, a conifer tier, a band of an evergreen's trunk, or a
// far crown in one piece.
type Clump = {
  leaves: SVGGElement
  flip: SVGPathElement | null
  // Share of the crown's lean it takes, growing toward the crown top.
  lift: number
  // Share of the leaf loop it walks: stiff at the middle of the crown, loose at its edge.
  loose: number
  cell: number
  // Seconds the gust front takes to reach it from the tree's foot.
  lag: number
  // Its place along the wind in wavelengths of the leaf wave, from its place in the scene.
  wave: number
  offset: string
  flipped: boolean
}
type Crown = {
  clumps: Clump[]
  // Scene y of the tree's foot, to skip trees out of view.
  ground: number
  // The tree's place across the wind, which sets how hard each front hits it.
  across: number
  shape: WindShape
  // The tree's own give, a conifer's stiffer leaves, and its longer lever.
  give: number
  flutter: number
  sway: number
  // Seconds behind the scene's gust clock, from the tree's place along the wind.
  delay: number
  // Pieces that have left rest hold a zero offset there, so the compositor keeps their textures.
  armed: boolean
  held: boolean
  resting: boolean
}

const needled = new Set(['pine', 'spruce', 'fir', 'hemlock', 'cypress', 'larch'])
// Poses change ten times a second. A slower clock reads as steps and a faster one as jitter.
const interval = 100
const reread = 3000
// Scene pixels between two crests of the leaf wave.
const wavelength = 60
// A crown narrower than this on screen moves in one piece in WebGL: its clumps are a few pixels
// wide, and each moving clump costs a texture and a draw.
const detail = 44
const turn = Math.PI * 2
const still = 'translate(0 0)'

// Leaf clumps walk a small loop in whole cells as a wave travelling downwind, and the crown leans
// into each gust and swings back like a spring. Between gusts the air is still and nothing is
// written. The clock writes only offsets that changed, for trees uncovered during the opening.
export function mountCrownWind(scene: HTMLElement, signal: AbortSignal, tuning: MotionTuning) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) return { refresh() {} }
  const motion = view.matchMedia('(prefers-reduced-motion: reduce)')
  const leaves = scene.getElementsByClassName('trail-tree-leaves')
  const canopies = scene.getElementsByClassName('trail-tree-canopy')
  // A crown keeps moving the way it first moved, in clumps or in one piece.
  const whole = new WeakMap<Element, boolean>()
  let crowns: Crown[] = []
  let counted = -1
  let cached = false
  let read = -Infinity
  let frame = 0
  let last = -Infinity
  let moving = false
  // The scene's document offset, read with the crowns. Each step derives its viewport position
  // from the scroll offset, so a step never forces layout after other writes.
  let sceneTop = 0

  const number = (style: CSSStyleDeclaration, name: string) =>
    Number.parseFloat(style.getPropertyValue(name)) || 0
  const settle = () => {
    for (const crown of crowns) {
      for (const clump of crown.clumps) {
        if (clump.offset) clump.leaves.removeAttribute('transform')
        if (clump.flipped) clump.flip?.style.removeProperty('opacity')
        clump.offset = ''
        clump.flipped = false
      }
      crown.armed = false
      crown.resting = true
    }
  }
  const collect = (time: number) => {
    const webgl = scene.dataset.trailRenderer === 'webgl'
    // The other renderer moves other elements; leave none of this one's behind.
    if (webgl !== cached) settle()
    cached = webgl
    read = time
    counted = leaves.length + canopies.length
    sceneTop = scene.getBoundingClientRect().top + view.scrollY
    const style = scene.style
    const lagX = number(style, '--trail-leaf-lag-x')
    const lagY = number(style, '--trail-leaf-lag-y')
    const pace = Math.hypot(lagX, lagY)
    const windX = pace ? lagX / pace : 1
    const windY = pace ? lagY / pace : 0
    crowns = []
    const places: number[] = []
    const site = (element: SVGGElement) => {
      const root = element.closest<SVGGElement>(
        '.trail-tree, .trail-woodland-tree, .trail-understory',
      )
      const tree = element.closest<SVGGElement>('.trail-tree-motion')
      const scale = Number(tree?.dataset.windScale)
      const conifer = needled.has(root?.dataset.tree ?? '')
      const x = Number(root?.dataset.groundX ?? 0)
      const y = Number(root?.dataset.groundY ?? 0)
      const crown: Crown = {
        clumps: [],
        ground: y,
        across: 0.5,
        shape: conifer ? 'conifer' : 'broadleaf',
        give: Math.sqrt(Number.isFinite(scale) && scale > 0 ? scale : 1),
        flutter: conifer ? 0.45 : 1,
        sway: conifer ? 1.8 : 1.3,
        delay: Number(tree?.dataset.windDelay) || 0,
        armed: false,
        held: false,
        resting: true,
      }
      crowns.push(crown)
      places.push(y * windX - x * windY)
      return { crown, tree, x, y, size: Number(root?.dataset.treeScale) || 1 }
    }
    // The bands of an evergreen's trunk slide with the lean and take no part in the leaf loop.
    // Holiday ornaments hang at mid-crown and ride the lean there.
    const stems = (at: ReturnType<typeof site>) => {
      if (!at.tree) return
      for (const stem of at.tree.getElementsByClassName('trail-tree-stem'))
        at.crown.clumps.push(clump(stem as SVGGElement, at, false))
      const ornament = at.tree.querySelector<SVGGElement>('[data-holiday-tree]')
      if (ornament)
        at.crown.clumps.push({
          ...clump(ornament, at, false),
          lift: 0.5,
          loose: 0,
          cell:
            Number(at.tree.querySelector<SVGGElement>('.trail-tree-leaves')?.dataset.leafCell) || 1,
        })
    }
    const clump = (group: SVGGElement, at: ReturnType<typeof site>, flips: boolean): Clump => {
      const x = (Number(group.dataset.leafX) || 0) * at.size
      const y = (Number(group.dataset.leafY) || 0) * at.size
      const flip = flips ? group.querySelector<SVGPathElement>('.trail-tree-flip') : null
      const loose = Number(group.dataset.leafLoose)
      return {
        leaves: group,
        flip,
        lift: Number(group.dataset.leafLift) || 0,
        loose: Number.isFinite(loose) ? loose : 0.7,
        cell: Number(group.dataset.leafCell) || 1,
        lag: x * lagX + y * lagY,
        wave: ((at.x + x) * windX + (at.y + y) * windY) / wavelength,
        offset: group.getAttribute('transform') ?? '',
        // A reread can land during a peak; retain its painted state so the next step releases it.
        flipped: flip !== null && flip.style.opacity !== '',
      }
    }
    if (webgl)
      for (const element of canopies) {
        const canopy = element as SVGGElement
        const at = site(canopy)
        let one = whole.get(canopy)
        if (one === undefined) {
          one = (Number(canopy.dataset.leafSpan) || 0) * at.size < detail
          whole.set(canopy, one)
        }
        if (one) {
          const sample = canopy.querySelector<SVGGElement>('.trail-tree-leaves')
          at.crown.clumps.push({
            leaves: canopy,
            flip: null,
            lift: 0.7,
            loose: 0.6,
            cell: Number(sample?.dataset.leafCell) || 1,
            lag: 0,
            wave: (at.x * windX + at.y * windY) / wavelength,
            offset: canopy.getAttribute('transform') ?? '',
            flipped: false,
          })
          continue
        }
        at.crown.held = true
        stems(at)
        for (const child of canopy.children)
          if (child.classList.contains('trail-tree-leaves'))
            at.crown.clumps.push(clump(child as SVGGElement, at, false))
      }
    else {
      const owners = new Map<Element, ReturnType<typeof site>>()
      for (const element of leaves) {
        const group = element as SVGGElement
        const parent = group.parentElement
        const owner = parent?.classList.contains('trail-tree-canopy') ? parent : group
        let at = owners.get(owner)
        if (!at) {
          at = site(group)
          owners.set(owner, at)
          stems(at)
        }
        at.crown.clumps.push(clump(group, at, true))
      }
    }
    let low = Infinity
    let high = -Infinity
    for (const place of places) {
      low = Math.min(low, place)
      high = Math.max(high, place)
    }
    for (let index = 0; index < crowns.length; index++) {
      const crown = crowns[index]
      if (high > low) crown.across = (places[index] - low) / (high - low)
      for (const piece of crown.clumps) {
        if (!piece.offset) continue
        if (crown.held) crown.armed = true
        if (piece.offset !== still) crown.resting = false
      }
    }
  }
  const windy = () => tuning.wind > 0 && scene.dataset.weatherWindy === 'true' && !motion.matches
  const running = () =>
    windy() &&
    scene.dataset.trailVisible === 'true' &&
    scene.dataset.weatherMotion !== 'paused' &&
    !document.hidden &&
    scene.dataset.trailOpening !== 'pending'

  const step = (time: number) => {
    frame = 0
    if (signal.aborted || !running()) return
    frame = view.requestAnimationFrame(step)
    if (time - last < interval) return
    last = time
    if (
      time - read > reread ||
      leaves.length + canopies.length !== counted ||
      (scene.dataset.trailRenderer === 'webgl') !== cached
    )
      collect(time)
    const style = scene.style
    const pushX = number(style, '--trail-leaf-push-x')
    // Cells a loose clump at the top of the crown moves at the gust's peak.
    const push = (number(style, '--trail-leaf-push') || Math.abs(pushX)) * (tuning.wind / 100)
    const period = number(style, '--trail-wind-period') || 9
    const lagX = number(style, '--trail-leaf-lag-x')
    const lagY = number(style, '--trail-leaf-lag-y')
    const pace = Math.hypot(lagX, lagY)
    // Crowns sway sideways on the page. A wind blowing up or down it still stirs them a little.
    const across = pace ? lagX / pace : pushX < 0 ? -1 : 1
    const heading = Math.abs(across) < 0.4 ? (across < 0 ? -0.4 : 0.4) : across
    const flip = number(style, '--trail-leaf-flip')
    // Harder wind shortens the leaf wave's period.
    const beat = 1.9 - 0.5 * Math.min(1, number(style, '--trail-gust-strength'))
    const tables = windTables(period)
    const top = view.scrollY - sceneTop - 40
    const bottom = top + view.innerHeight + 200
    const now = time / 1000
    const clock = now / beat
    moving = true
    for (const crown of crowns) {
      if (crown.ground < top || crown.ground > bottom) continue
      const passed = (now - crown.delay) / period
      const front = Math.floor(passed)
      const u = passed - front
      const lean = tables.lean[crown.shape]
      const reach = push * crown.give * frontStrength(front, crown.across)
      // Still air and a crown at rest: nothing to work out and nothing to write.
      if (
        crown.resting &&
        (Math.abs(sampleWind(lean, u)) * crown.sway +
          sampleWind(tables.breeze, u) * crown.flutter) *
          reach <
          0.5
      )
        continue
      let resting = true
      for (const clump of crown.clumps) {
        const at = u - clump.lag / period
        const gust = sampleWind(tables.gust, at)
        const breeze = sampleWind(tables.breeze, at)
        const angle = turn * (clock - clump.wave)
        const loop = (clump.loose * crown.flutter * reach) / 2.2
        const dx =
          Math.round(
            (sampleWind(lean, at) * crown.sway * clump.lift * reach +
              loopAlong(angle, breeze, gust) * loop) *
              heading,
          ) * clump.cell
        const dy = Math.round(loopUp(angle, breeze, gust) * loop) * -clump.cell
        const moved = dx !== 0 || dy !== 0
        if (moved) resting = false
        // The first piece to leave rest takes the rest of its crown with it in one write each, so
        // the compositor splits the crown's texture once.
        if (moved && crown.held && !crown.armed) {
          crown.armed = true
          for (const other of crown.clumps)
            if (!other.offset) {
              other.leaves.setAttribute('transform', still)
              other.offset = still
            }
        }
        const offset = moved ? `translate(${dx} ${dy || 0})` : crown.armed ? still : ''
        if (offset !== clump.offset) {
          if (offset) clump.leaves.setAttribute('transform', offset)
          else clump.leaves.removeAttribute('transform')
          clump.offset = offset
        }
        // Undersides show while a gust is at its hardest.
        if (clump.flip) {
          const flipped = flip > 0 && gust > 0.85
          if (flipped !== clump.flipped) {
            if (flipped) clump.flip.style.opacity = flip.toFixed(3)
            else clump.flip.style.removeProperty('opacity')
            clump.flipped = flipped
          }
        }
      }
      crown.resting = resting
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
      'data-trail-opening',
    ],
  })
  document.addEventListener('visibilitychange', wake, { signal })
  scene.addEventListener('trail:motion-tuning', wake, { signal })
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
    // The forest was drawn again: read the new crowns on the next step.
    refresh() {
      read = -Infinity
      wake()
    },
  }
}
