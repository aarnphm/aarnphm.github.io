import { nearCourse, PHOTO } from '../404-downhill'
import { mulberry, noise, type Pt } from '../404-pixel'
import { DOOR, type Kind } from './herd-state'

// The pasture on a grid of four units, about a pixel: grass clear of the line, the paths and the
// photographer, in one piece with the lane along the slope behind the cypress to the barn's doors.
// The lines cut off the grass across them, so the stock stay on the side of the barn. The cattle
// range over the middle turns; the ewes keep above them.
const CELL = 4
// A beast stands this far from the middle of a line or path, so its body stays off the tread.
const CLEAR = 16
const BOX = { x0: 244, y0: 548, x1: 468, y1: 744 }
const NX = Math.ceil((BOX.x1 - BOX.x0) / CELL)
const NY = Math.ceil((BOX.y1 - BOX.y0) / CELL)
const AREA: Record<Kind, [number, number, number, number]> = {
  cow: [256, 598, 376, 736],
  calf: [256, 598, 376, 736],
  sheep: [282, 556, 376, 628],
}
// The lane keeps in front of the barn's wall, so nobody is painted over the barn from behind it.
export const LANE = { x0: 370, y0: 676, x1: 466, y1: 700 }
const STEPS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
]
// Where a beast may set foot, and how many cells it is from there to the barn's doors.
export let walk: Uint8Array | null = null
export let home: Int16Array | null = null
export function survey() {
  const open = new Uint8Array(NX * NY)
  const inside = ([x0, y0, x1, y1]: number[], x: number, y: number) =>
    x >= x0 && x <= x1 && y >= y0 && y <= y1
  for (let j = 0; j < NY; j++)
    for (let i = 0; i < NX; i++) {
      const x = BOX.x0 + (i + 0.5) * CELL
      const y = BOX.y0 + (j + 0.5) * CELL
      const ground =
        inside(AREA.cow, x, y) ||
        inside(AREA.sheep, x, y) ||
        inside([LANE.x0, LANE.y0, LANE.x1, LANE.y1], x, y)
      open[j * NX + i] =
        ground && !nearCourse(x, y, CLEAR) && Math.hypot(x - PHOTO[0], y - PHOTO[1]) > 12 ? 1 : 0
    }
  home = new Int16Array(NX * NY).fill(-1)
  const start = cell(DOOR[0], DOOR[1])
  home[start] = 0
  const queue = [start]
  for (let n = 0; n < queue.length; n++) {
    const c = queue[n]
    const i = c % NX
    const j = (c - i) / NX
    for (const [di, dj] of STEPS) {
      const ni = i + di
      const nj = j + dj
      const nc = nj * NX + ni
      if (ni < 0 || nj < 0 || ni >= NX || nj >= NY || !open[nc] || home[nc] >= 0) continue
      home[nc] = home[c] + 1
      queue.push(nc)
    }
  }
  walk = Uint8Array.from(home, d => (d >= 0 ? 1 : 0))
}
export const cell = (x: number, y: number) => {
  const i = Math.floor((x - BOX.x0) / CELL)
  const j = Math.floor((y - BOX.y0) / CELL)
  return i >= 0 && j >= 0 && i < NX && j < NY ? j * NX + i : -1
}
const centreOf = (c: number): Pt => [
  BOX.x0 + ((c % NX) + 0.5) * CELL,
  BOX.y0 + (Math.floor(c / NX) + 0.5) * CELL,
]
const footing = (x: number, y: number) => {
  const c = cell(x, y)
  return c >= 0 && walk![c] === 1
}
export const onGrass = (k: Kind, x: number, y: number) => {
  const [x0, y0, x1, y1] = AREA[k]
  return x >= x0 && x <= x1 && y >= y0 && y <= y1 && footing(x, y)
}
// Whether a beast can walk straight from one point to another without leaving the grass.
export function sight(a: Pt, b: Pt) {
  const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]))
  for (let k = 1; k <= n; k++)
    if (!footing(a[0] + ((b[0] - a[0]) * k) / n, a[1] + ((b[1] - a[1]) * k) / n)) return false
  return true
}
// The way from a point to the barn's doors: down the count of cells to them, then pulled tight so
// that it runs straight wherever the grass allows.
export function route(x: number, y: number): Pt[] {
  const cells: Pt[] = []
  // Each step to the neighbour nearest the doors. A beast can stand a hair past the edge of the
  // grass where its way clipped the corner of a cell, and then the first step is back onto it.
  for (let c = cell(x, y); c >= 0 && home![c] !== 0; ) {
    const i = c % NX
    const j = (c - i) / NX
    let to = -1
    for (const [di, dj] of STEPS) {
      const nc = (j + dj) * NX + i + di
      if (i + di < 0 || i + di >= NX || j + dj < 0 || j + dj >= NY || home![nc] < 0) continue
      if (to < 0 || home![nc] < home![to]) to = nc
    }
    c = to
    if (to >= 0) cells.push(centreOf(to))
  }
  cells.push(DOOR)
  const out: Pt[] = []
  let from: Pt = [x, y]
  for (let k = 0; k < cells.length; ) {
    let far = k
    for (let m = cells.length - 1; m > k; m--)
      if (sight(from, cells[m])) {
        far = m
        break
      }
    out.push(cells[far])
    from = cells[far]
    k = far + 1
  }
  return out
}

// The terracettes: years of hooves working along the slope have trodden it into steps, a tread for
// every two metres or so of height, each wandering a little up and down and running out at
// the edge of the grazing. Where one wanders into the step below, the two run on as one. The stock
// graze along the treads.
type Terrace = { y: number; x0: number; x1: number; seed: number }
export const TERRACES: Terrace[] = (() => {
  const r = mulberry(0x7e44ac)
  const out: Terrace[] = []
  for (let y = 562 + r() * 6; y < 736; y += 14 + r() * 10) {
    const [x0, , x1] = y < AREA.cow[1] ? AREA.sheep : AREA.cow
    out.push({ y, x0: x0 + r() * 14, x1: x1 - r() * 14, seed: 90 + out.length })
  }
  return out
})()
// Where the nth terrace runs at a point across the slope, or null past its ends or where it has
// joined the one below.
export function onTerrace(n: number, x: number) {
  const at = (t: Terrace) =>
    x < t.x0 || x > t.x1 ? null : t.y + (noise(x * 0.02, 0, t.seed) - 0.5) * 16
  const y = at(TERRACES[n])
  const below = n + 1 < TERRACES.length ? at(TERRACES[n + 1]) : null
  return y === null || (below !== null && below - y < 6) ? null : y
}
// The tread within a couple of pixels of a point, if there is one.
export function tread(x: number, y: number) {
  let best: number | null = null
  for (let n = 0; n < TERRACES.length; n++) {
    const ty = onTerrace(n, x)
    if (
      ty !== null &&
      Math.abs(ty - y) < 9 &&
      (best === null || Math.abs(ty - y) < Math.abs(best - y))
    )
      best = ty
  }
  return best
}
