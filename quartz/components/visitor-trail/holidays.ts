import type { HolidaySelection } from '../../util/visitor-holiday'
import type { RouteSites } from './pasture'
import {
  readTrailHolidayChoice,
  subscribeTrailHolidayChoice,
} from '../../util/trail-holiday-choice'
import { trailHost } from '../../util/trail-scene'
import { subscribeVisitorHoliday, type TrailHoliday } from '../../util/visitor-holiday'
import { circleClearance, readActors, readTrunks } from './actor-spacing'
import { campSites } from './camp-layout'
import {
  holidayAnimalSprite,
  holidayArt,
  holidaySprite,
  type HolidaySpriteName,
} from './holiday-sprites'
import { onDryLand, pondDistance, type Landscape, type Point } from './landscape'
import { trailRandom } from './random'
import { sceneSettings, type CampSettings } from './scene-settings'

type Cell = { x: number; y: number; size: number }

// Tree paths are runs of `M x y h size …` cells in the tree's own coordinates, ground at y = 0.
function treeCells(path: Element | null | undefined) {
  const cells: Cell[] = []
  for (const [, x, y, size] of (path?.getAttribute('d') ?? '').matchAll(
    /M(-?[\d.]+) (-?[\d.]+)h([\d.]+)/g,
  ))
    cells.push({ x: Number(x), y: Number(y), size: Number(size) })
  return cells
}

function crownRows(cells: Cell[]) {
  const rows = new Map<number, { left: number; right: number }>()
  for (const { x, y, size } of cells) {
    const row = rows.get(y)
    rows.set(y, {
      left: Math.min(row?.left ?? x, x),
      right: Math.max(row?.right ?? x + size, x + size),
    })
  }
  return rows
}

// Crown width a quarter of the way down against the width near its foot: a cone is narrow above.
function taper(cells: Cell[]) {
  const rows = crownRows(cells)
  const ys = [...rows.keys()].sort((a, b) => a - b)
  if (ys.length < 6) return Infinity
  const width = (fraction: number) => {
    const row = rows.get(ys[Math.floor((ys.length - 1) * fraction)])!
    return row.right - row.left
  }
  return width(0.25) / Math.max(1, width(0.8))
}

const fixed = (value: number) => value.toFixed(1)

// Lights follow the crown's own cells: three swags that cross the tree on its pixel grid, a bulb in
// every third cell, and a star on the leader.
function christmasTree(cells: Cell[]) {
  const size = cells[0].size
  const filled = new Set(cells.map(({ x, y }) => `${x},${y}`))
  const rows = crownRows(cells)
  const ys = [...rows.keys()].sort((a, b) => a - b)
  const top = ys[0]
  const height = ys.at(-1)! + size - top
  const snap = (value: number, origin: number) =>
    Math.round((value - origin) / size) * size + origin
  const garland: string[] = []
  const bulbs: string[][] = [[], [], []]
  let lit = 0
  for (let swag = 0; swag < 3; swag++) {
    const fromY = snap(top + height * (0.3 + swag * 0.2), top)
    const toY = snap(top + height * (0.44 + swag * 0.2), top)
    const from = rows.get(fromY)
    const to = rows.get(toY)
    if (!from || !to) continue
    const rightward = swag % 2 === 0
    const x0 = rightward ? from.left + size : from.right - size * 2
    const x1 = rightward ? to.right - size * 2 : to.left + size
    const steps = Math.max(1, Math.round(Math.abs(x1 - x0) / size))
    for (let step = 0; step <= steps; step++) {
      const t = step / steps
      const x = snap(x0 + (x1 - x0) * t, cells[0].x)
      const y = snap(fromY + (toY - fromY) * t + Math.sin(Math.PI * t) * size, top)
      if (!filled.has(`${x},${y}`)) continue
      const cell = `M${x} ${y}h${size}v${size}h${-size}Z`
      if ((step + swag) % 3 === 1) bulbs[lit++ % 3].push(cell)
      else garland.push(cell)
    }
  }
  const apex = rows.get(top)!
  const leader = snap((apex.left + apex.right - size) / 2, cells[0].x)
  return (
    `<path class="trail-holiday-garland" d="${garland.join('')}"/>` +
    `<g class="trail-holiday-lights"><path class="trail-holiday-bulb-gold" d="${bulbs[0].join('')}"/><path class="trail-holiday-bulb-red" d="${bulbs[1].join('')}"/><path class="trail-holiday-bulb-white" d="${bulbs[2].join('')}"/></g>` +
    `<g transform="translate(${leader - size * 2} ${top - size * 3})">${holidaySprite('treeStar', size)}</g>`
  )
}

// A web spun in the angle between the trunk and the underside of the crown, where an orb weaver
// anchors one. Threads are lines, not cells, so the web stays fine at any tree scale.
function cobweb(cells: Cell[], trunk: Cell[], side: 1 | -1) {
  if (!cells.length || !trunk.length) return ''
  const size = cells[0].size
  const base = trunk.filter(cell => cell.y >= -size * 4)
  if (!base.length) return ''
  const left = Math.min(...base.map(cell => cell.x))
  const right = Math.max(...base.map(cell => cell.x + cell.size))
  const over = cells.filter(cell => cell.x + cell.size > left - size && cell.x < right + size)
  if (!over.length) return ''
  const under = Math.max(...over.map(cell => cell.y + cell.size))
  const collar = trunk.filter(cell => cell.y >= under && cell.y < under + size * 3)
  const edge = collar.length
    ? side > 0
      ? Math.max(...collar.map(cell => cell.x + cell.size))
      : Math.min(...collar.map(cell => cell.x))
    : side > 0
      ? right
      : left
  const rows = crownRows(cells)
  const crown = Math.max(...[...rows.values()].map(row => row.right - row.left))
  const reach = Math.min(-under * 0.75, size * 9, crown * 0.4)
  if (reach < size * 3) return ''
  const point = (radius: number, degrees: number) => {
    const angle = (degrees * Math.PI) / 180
    return `${fixed(edge + side * radius * Math.cos(angle))} ${fixed(under + radius * Math.sin(angle))}`
  }
  const spokes = [6, 30, 54, 78]
  let threads = spokes.map(degrees => `M${edge} ${under}L${point(reach, degrees)}`).join('')
  for (const ring of [0.34, 0.62, 0.9]) {
    threads += `M${point(reach * ring, spokes[0])}`
    for (let index = 1; index < spokes.length; index++)
      threads += `Q${point(reach * ring * 0.88, (spokes[index - 1] + spokes[index]) / 2)} ${point(reach * ring, spokes[index])}`
  }
  const angle = (42 * Math.PI) / 180
  const hangX = edge + side * reach * 0.62 * Math.cos(angle)
  const hangY = under + reach * 0.62 * Math.sin(angle)
  const drop = reach * 0.3
  const body = Math.max(0.5, size / 2)
  threads += `M${fixed(hangX)} ${fixed(hangY)}v${fixed(drop)}`
  return `<path class="trail-holiday-thread" d="${threads}"/><g transform="translate(${fixed(hangX - body * 2.5)} ${fixed(hangY + drop)})">${holidaySprite('spider', body)}</g>`
}

// A pool of candlelight on the ground, shown at night, in 2 px rows like the campfire's.
function lanternGlow(width: number, depth: number) {
  const pool = (radius: number, rise: number) => {
    let d = ''
    for (let y = -rise; y < rise; y += 2) {
      const half = Math.round((radius * Math.sqrt(1 - ((y + 1) / rise) ** 2)) / 2) * 2
      if (half) d += `M${-half} ${y}h${half * 2}v2h${-half * 2}Z`
    }
    return d
  }
  return `<path class="trail-holiday-glow" d="${pool(width, depth)}"/><path class="trail-holiday-glow-core" d="${pool(width * 0.55, depth * 0.6)}"/>`
}

export function mountTrailHolidays(scene: HTMLElement, signal: AbortSignal, changed: () => void) {
  let automatic: TrailHoliday | null = null
  // The visitor's choice on the boulder switch wins over the authored setting.
  let selection: HolidaySelection =
    readTrailHolidayChoice(scene.ownerDocument.defaultView) ?? sceneSettings.holiday
  let decorated: SVGGElement[] = []
  const current = () => (selection === 'auto' ? automatic : selection === 'none' ? null : selection)
  const state = () => {
    const holiday = current()
    scene.dataset.holiday = holiday ?? 'none'
    scene.dataset.holidaySelection = selection
    trailHost(scene)?.dispatchEvent(new CustomEvent('trail:holiday', { detail: holiday }))
  }
  subscribeVisitorHoliday(
    scene.ownerDocument,
    holiday => {
      if (automatic === holiday) return
      automatic = holiday
      state()
      if (selection === 'auto') changed()
    },
    signal,
  )
  subscribeTrailHolidayChoice(
    scene.ownerDocument,
    choice => {
      const next = choice ?? sceneSettings.holiday
      if (next === selection) return
      const before = current()
      selection = next
      state()
      if (current() !== before) changed()
    },
    signal,
  )
  state()

  const refreshTrees = () => {
    const previous = new Set(scene.querySelectorAll<SVGGElement>('[data-holiday-tree]'))
    decorated = []
    if (!scene.isConnected) return
    const holiday = current()
    if (holiday !== 'christmas' && holiday !== 'halloween') {
      for (const ornament of previous) ornament.remove()
      return
    }
    const trees = [
      ...scene.querySelectorAll<SVGGElement>('.trail-tree[data-tree-depth="meadow"]'),
    ].flatMap(tree => {
      const canopy = tree.querySelector<SVGGElement>('.trail-tree-canopy')
      const cells = treeCells(canopy?.querySelector('.trail-tree-hollow'))
      return canopy && cells.length >= 12 ? [{ tree, canopy, cells }] : []
    })
    trees.sort((a, b) => Number(b.tree.dataset.treeScale) - Number(a.tree.dataset.treeScale))
    const chosen: typeof trees = []
    if (holiday === 'halloween') chosen.push(...trees.slice(0, 6))
    // The largest true cones, spread across the meadow. Pine tufts and cypress columns are not
    // Christmas-tree shaped even when they taper.
    else
      for (const entry of trees)
        if (
          chosen.length < 3 &&
          ['spruce', 'fir', 'hemlock'].includes(entry.tree.dataset.tree ?? '') &&
          taper(entry.cells) < 0.5 &&
          chosen.every(
            other =>
              Math.abs(Number(other.tree.dataset.groundX) - Number(entry.tree.dataset.groundX)) >
              160,
          )
        )
          chosen.push(entry)
    for (const [index, { tree, canopy, cells }] of chosen.entries()) {
      const existing = canopy.parentElement?.querySelector<SVGGElement>('[data-holiday-tree]')
      if (existing?.dataset.holidayTree === holiday) {
        previous.delete(existing)
        decorated.push(canopy)
        continue
      }
      const markup =
        holiday === 'christmas'
          ? christmasTree(cells)
          : cobweb(
              cells,
              treeCells(tree.querySelector('.trail-tree-motion > .trail-tree-trunk')),
              index % 2 ? -1 : 1,
            )
      if (!markup) continue
      const ornament = scene.ownerDocument.createElementNS('http://www.w3.org/2000/svg', 'g')
      ornament.dataset.holidayTree = holiday
      ornament.innerHTML = markup
      canopy.after(ornament)
      decorated.push(canopy)
    }
    for (const ornament of previous) ornament.remove()
  }

  const draw = (landscape: Landscape, routes: RouteSites[], seed: number, camps: CampSettings) => {
    const layer = scene.querySelector<SVGGElement>('.trail-holidays')
    if (!layer) return
    const holiday = current()
    scene
      .querySelectorAll('[data-holiday-decoration], [data-holiday-animal]')
      .forEach(element => element.remove())
    layer.replaceChildren()
    refreshTrees()
    if (!holiday) return
    const random = trailRandom(seed ^ 0x68259d01)
    const occupied: (Point & { radius: number })[] = []
    const solids = readTrunks(scene)
    const bodies = readActors(scene)
    const road = routes.flatMap(route => route.samples)
    const lowerRoad = road.filter(point => point.y >= landscape.meadowTop + 80)
    const sites = campSites(landscape, seed, camps)
    const safeProp = (anchor: Point, radius: number) => {
      for (let attempt = 0; attempt < 96; attempt++) {
        const angle = attempt * 2.399963
        const reach = attempt ? 6 + Math.floor(attempt / 12) * 8 : 0
        const candidate = onDryLand(
          { x: anchor.x + Math.cos(angle) * reach, y: anchor.y + Math.sin(angle) * reach },
          landscape,
          1.8,
        )
        if (
          pondDistance(candidate, landscape.pond) >= 1.8 &&
          solids.every(solid => circleClearance(candidate, radius, solid) >= 0) &&
          bodies.every(body => circleClearance(candidate, radius, body) >= 0) &&
          occupied.every(
            point =>
              Math.hypot(candidate.x - point.x, candidate.y - point.y) > point.radius + radius + 8,
          ) &&
          road.every(
            point => Math.hypot(candidate.x - point.x, candidate.y - point.y) > radius + 12,
          )
        ) {
          occupied.push({ ...candidate, radius })
          return candidate
        }
      }
      return null
    }
    const props: string[] = []
    const prop = (
      name: HolidaySpriteName,
      anchor: Point,
      { lift = 0, under = '', variant = '', radius = 12 } = {},
    ) => {
      const point = safeProp(anchor, radius)
      if (!point) return false
      const rows = holidayArt[name]
      const x = Math.round(point.x)
      const y = Math.round(point.y)
      props.push(
        `<g class="trail-holiday-${name}${variant ? ` ${variant}` : ''}" data-holiday-decoration="${name}" data-ground-x="${x}" data-ground-y="${y}" transform="translate(${x} ${y})">${under}<g transform="translate(${-rows[0].length} ${-rows.length * 2 - lift})"><g${name === 'ghost' ? ' class="trail-holiday-float"' : ''}>${holidaySprite(name, 2)}</g></g></g>`,
      )
      return true
    }
    for (const [index, site] of sites.entries()) {
      if (holiday === 'thanksgiving') prop('harvest', { x: site.x + 58, y: site.y + 18 })
      if (holiday === 'columbus-day') prop('trailMap', { x: site.x + 58, y: site.y + 18 })
      if (holiday === 'halloween') {
        prop('lanterns', { x: site.x + 56, y: site.y + 18 }, { under: lanternGlow(26, 8) })
        // Tents and the camp spur crowd some sites; try the far side of the fire too.
        const figure = index % 2 ? 'ghost' : 'skeleton'
        for (const [dx, dy] of [
          [-66, -18],
          [-48, 34],
          [12, 44],
        ])
          if (
            prop(
              figure,
              { x: site.x + dx, y: site.y + dy },
              {
                lift: index % 2 ? 6 : 0,
                under: '<path class="trail-holiday-shadow" d="M-6-2h12v2h-12Z"/>',
              },
            )
          )
            break
      }
    }
    if (holiday === 'halloween') {
      // One wayside piece per stretch of the lower trail, so they do not bunch up.
      const wayside: [HolidaySpriteName, number][] = [
        ['gibbet', 18],
        ['pumpkin', 12],
      ]
      if (landscape.width >= 700) wayside.push(['graveyard', 30])
      for (const [index, [name, radius]] of wayside.entries()) {
        const anchor =
          lowerRoad[Math.floor(((index + random()) / wayside.length) * lowerRoad.length)]
        if (anchor)
          prop(
            name,
            { x: anchor.x + (index % 2 ? -1 : 1) * (radius + 20), y: anchor.y + 14 },
            { radius, under: name === 'pumpkin' ? lanternGlow(20, 6) : '' },
          )
      }
    }
    if (holiday === 'christmas')
      for (const [index, canopy] of decorated.entries()) {
        const tree = canopy.closest<SVGGElement>('.trail-tree')
        if (!tree) continue
        prop('gifts', {
          x: Number(tree.dataset.groundX) + (index % 2 ? -16 : 16),
          y: Number(tree.dataset.groundY) + 6,
        })
      }
    if (holiday === 'easter') {
      for (let index = 0; index < 3; index++)
        prop(
          'egg',
          {
            x: landscape.pond.x + landscape.pond.rx * (index % 2 ? -1.9 : 1.9),
            y: landscape.pond.y + (index - 1) * 42,
          },
          { variant: `trail-holiday-egg-${index}` },
        )
    }
    // Bats leave the webbed crowns at dusk. Placed once per layout; they do not fly.
    const bats: string[] = []
    const toLayer = holiday === 'halloween' ? layer.getScreenCTM()?.inverse() : null
    if (toLayer)
      for (const [index, canopy] of decorated.slice(0, 2).entries()) {
        const matrix = canopy.getScreenCTM()
        if (!matrix) continue
        const box = canopy.getBBox()
        // DOMMatrix.transformPoint is missing on SVG CTMs in some engines.
        const x = matrix.a * (box.x + box.width / 2) + matrix.c * box.y + matrix.e
        const y = matrix.b * (box.x + box.width / 2) + matrix.d * box.y + matrix.f
        const crown = {
          x: toLayer.a * x + toLayer.c * y + toLayer.e,
          y: toLayer.b * x + toLayer.d * y + toLayer.f,
        }
        for (const [name, dx, dy] of [
          ['bat', index ? 14 : -26, -22],
          ['batGlide', index ? -20 : 18, -38],
        ] as const)
          bats.push(
            `<g class="trail-holiday-bat" transform="translate(${Math.round(crown.x + dx)} ${Math.round(crown.y + dy)})">${holidaySprite(name, 2)}</g>`,
          )
      }
    const animals: string[] = []
    const kind = holiday === 'thanksgiving' ? 'turkey' : holiday === 'easter' ? 'rabbit' : null
    if (kind) {
      const count = landscape.width < 700 ? 1 : 2
      for (let index = 0; index < count; index++) {
        const anchor =
          lowerRoad[Math.floor(random() * lowerRoad.length)] ??
          landscape.groves[index % landscape.groves.length]
        const point = onDryLand(
          { x: anchor.x + (index % 2 ? 38 : -38), y: anchor.y + 24 },
          landscape,
          1.9,
        )
        const rows = holidayArt[kind]
        const size = index ? 1 : 1.5
        const x = Math.round(point.x)
        const y = Math.round(point.y)
        animals.push(
          `<g class="trail-${kind}" data-holiday-animal="${kind}" data-trail-wander="holiday-${kind}-${index}" data-trail-animal="${kind}" data-animal-home-x="${x}" data-animal-home-y="${y}" data-animal-size="${size}" data-phase="waiting" data-motion="paused" data-stride="0" transform="translate(${x} ${y})" style="--trail-animal-step:${size}px"><g class="trail-animal-facing"><g transform="translate(${(-rows[0].length * size) / 2} ${-(rows.length - 1) * size})">${holidayAnimalSprite(kind, size)}</g></g></g>`,
        )
      }
    }
    layer.innerHTML = `<g class="trail-holiday-bats">${bats.join('')}</g><g class="trail-holiday-props">${props.join('')}</g><g class="trail-holiday-animals">${animals.join('')}</g>`
  }

  signal.addEventListener(
    'abort',
    () => {
      scene
        .querySelectorAll('[data-holiday-tree], [data-holiday-decoration], [data-holiday-animal]')
        .forEach(element => element.remove())
      scene.querySelector('.trail-holidays')?.replaceChildren()
      scene.dataset.holiday = 'none'
      delete scene.dataset.holidaySelection
    },
    { once: true },
  )

  return { draw, refreshTrees }
}
