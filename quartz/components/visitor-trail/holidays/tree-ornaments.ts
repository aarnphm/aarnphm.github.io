import { holidaySprite } from '../holiday-sprites'

type Cell = { x: number; y: number; size: number }

// Tree paths are runs of `M x y h size …` cells in the tree's own coordinates, ground at y = 0.
export function treeCells(path: Element | null | undefined) {
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
export function taper(cells: Cell[]) {
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
export function christmasTree(cells: Cell[]) {
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
export function cobweb(cells: Cell[], trunk: Cell[], side: 1 | -1) {
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
