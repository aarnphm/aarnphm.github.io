// Packs a film collection into a steel archive rack: 35 mm cans stand on edge, rim to the aisle, on
// the shelves of one or more bays of a compactor carriage. World units are the bookcase's (1 unit ≈
// 1 mm) and so are the axes: +x along the run, +y away from the viewer, +z up; the viewer looks from
// (+x, -y, +z). Rims face -y and lids face +x.
import { type Point, clampOption, hashPath, project } from '../bookcase/geometry'

export const VAULT_DEFAULTS = {
  bayWidth: 1200,
  shelfRows: 5,
  dividers: 8,
  eraCuts: [1980, 2016] as readonly number[],
} as const

export const VAULT_LIMITS = { bayWidth: [600, 2400], shelfRows: [1, 8], dividers: [0, 40] } as const

/** Rack dimensions in world units. */
export const RACK = {
  post: 40,
  depth: 400,
  /** Shelf to shelf: a 2000 ft can, its clearance and a board. */
  pitch: 455,
  board: 25,
  deck: 20,
  rail: 20,
  /** Top of the bottom board above the carriage deck. */
  lift: 80,
  panel: 25,
  /** First can offset from the post, and the gap between cans. */
  start: 6,
  gap: 3,
  /** A divider sheet and the space it takes in the run. */
  divider: 2,
  dividerRoom: 10,
  /** A pulled can clears the shelf front by this much and rises by `lift` units. */
  pullClear: 12,
  pullLift: 10,
  /** Hover distance toward the aisle. */
  nudge: 18,
} as const

export const FINISHES = ['poly'] as const
export type CanFinish = (typeof FINISHES)[number]

/** Every print is in moulded polypropylene; an older print is a 1000 ft can, a newer one 2000 ft. */
const ERAS = [
  { finish: 'poly', footage: 1000, diameter: 270, thickness: 38 },
  { finish: 'poly', footage: 2000, diameter: 380, thickness: 50 },
  { finish: 'poly', footage: 2000, diameter: 380, thickness: 50 },
] as const

const unit = (path: string) => (hashPath(path) % 1000) / 1000
const half = (n: number) => Math.round(n * 2) / 2

export interface CanSize {
  finish: CanFinish
  footage: 1000 | 2000
  diameter: number
  thickness: number
  /** Distance of the rim behind the shelf front. */
  setback: number
}

export function canSize(
  path: string,
  year: number | undefined,
  cuts: readonly number[] = VAULT_DEFAULTS.eraCuts,
): CanSize {
  // A film without a year goes with the newest prints, where an unsorted year lands.
  const era = year === undefined ? 2 : cuts.filter(cut => year >= cut).length
  const { finish, footage, diameter, thickness } = ERAS[Math.min(era, 2)]
  return {
    finish,
    footage,
    // ±2 % so neighbouring lids do not align to the pixel.
    diameter: half(diameter * (0.98 + unit(`${path}#h`) * 0.04)),
    thickness,
    setback: half(unit(`${path}#y`) * 6),
  }
}

export interface VaultItem<T> {
  path: string
  year?: number
  /** Section label: the group, or the decade when the view has no groupBy. */
  section: string
  item: T
}

export interface PlacedCan<T> extends CanSize {
  item: T
  path: string
  /** Reading-order index; a film in several groups has one can per group. */
  index: number
  shelf: number
  /** World position of the can's left lid, its rim line and its foot. */
  x: number
  y: number
  z: number
}

export interface Divider {
  shelf: number
  x: number
  z: number
  height: number
}

export interface Holder {
  shelf: number
  /** World x of the holder's left edge and of the first x it may not reach. */
  x: number
  end: number
  z: number
  label: string
}

export interface VaultLayout<T> {
  cans: PlacedCan<T>[]
  dividers: Divider[]
  holders: Holder[]
  shelves: number
  bays: number
  shelfRows: number
  bayWidth: number
  /** Last can of each shelf, by shelf. */
  rowEnds: number[]
  bounds: readonly [number, number, number, number]
}

export interface VaultOptions {
  bayWidth?: unknown
  shelfRows?: unknown
  dividers?: unknown
  eraCuts?: unknown
}

export function vaultOptions(view: VaultOptions) {
  const cuts = Array.isArray(view.eraCuts)
    ? view.eraCuts.map(Number).filter(Number.isFinite).slice(0, 2)
    : []
  return {
    bayWidth: clampOption(view.bayWidth, VAULT_DEFAULTS.bayWidth, VAULT_LIMITS.bayWidth),
    shelfRows: Math.round(
      clampOption(view.shelfRows, VAULT_DEFAULTS.shelfRows, VAULT_LIMITS.shelfRows),
    ),
    dividers: Math.round(
      clampOption(view.dividers, VAULT_DEFAULTS.dividers, VAULT_LIMITS.dividers),
    ),
    eraCuts: cuts.length === 2 ? cuts.sort((a, b) => a - b) : VAULT_DEFAULTS.eraCuts,
  }
}

/** World x where bay `b`'s shelves begin; post `b` stands just before it. */
export const bayStart = (b: number, bayWidth: number) => b * (bayWidth + RACK.post) + RACK.post
/** Top of the board of `row` (0 is the top shelf). */
export const shelfTop = (row: number, shelfRows: number) =>
  RACK.lift + (shelfRows - 1 - row) * RACK.pitch
export const rackHeight = (shelfRows: number) => RACK.lift + shelfRows * RACK.pitch + 10

/** Width of a holder label character: 11 px Berkeley Mono at 0.6 em plus 1 px tracking. */
export const HOLDER_CHAR = 7.6
export const HOLDER_PAD = 22

/**
 * Packs the run in reading order. Shelves fill at `bayWidth`, bays at `shelfRows`. A divider goes
 * at every section change and after every `dividers` cans in a section; a section's card holder
 * sits on the lip under its first can.
 */
export function packVault<T>(items: VaultItem<T>[], view: VaultOptions = {}): VaultLayout<T> {
  const options = vaultOptions(view)
  const { bayWidth, shelfRows } = options
  let layout = packRun(items, options)
  // A last bay under half full reads as a mistake: widen the shelves by up to a quarter to fold it
  // into the bays before it.
  const stray = (l: VaultLayout<T>) =>
    l.bays > 1 && l.shelves - (l.bays - 1) * shelfRows < shelfRows / 2
  for (let width = bayWidth + 40; stray(layout) && width <= bayWidth * 1.25; width += 40) {
    const wider = packRun(items, { ...options, bayWidth: width })
    if (wider.bays < layout.bays) layout = wider
  }
  return layout
}

function packRun<T>(
  items: VaultItem<T>[],
  options: ReturnType<typeof vaultOptions>,
): VaultLayout<T> {
  const { bayWidth, shelfRows, dividers: every, eraCuts } = options
  const counts = new Map<string, number>()
  for (const { section } of items) counts.set(section, (counts.get(section) ?? 0) + 1)
  const cans: PlacedCan<T>[] = []
  const dividers: Divider[] = []
  const holders: Holder[] = []
  let shelf = 0
  let x: number = RACK.start
  let inSection = 0
  let previous: string | undefined
  let lastDiameter = 0
  const at = (s: number) => ({
    x0: bayStart(Math.floor(s / shelfRows), bayWidth),
    z: shelfTop(s % shelfRows, shelfRows),
  })
  for (const [index, { path, year, section, item }] of items.entries()) {
    const size = canSize(path, year, eraCuts)
    const opens = section !== previous
    const divide = index > 0 && (opens || (every > 0 && inSection > 0 && inSection % every === 0))
    let room = size.thickness + (divide ? RACK.dividerRoom : 0)
    if (x + room > bayWidth - RACK.start && x > RACK.start) {
      shelf++
      x = RACK.start
      room = size.thickness
    } else if (divide) {
      const { x0, z } = at(shelf)
      dividers.push({
        shelf,
        x: x0 + x + (RACK.dividerRoom - RACK.divider) / 2 - RACK.gap / 2,
        z,
        height: Math.round(Math.max(lastDiameter, size.diameter) * 0.72),
      })
      x += RACK.dividerRoom
    }
    if (opens) inSection = 0
    const { x0, z } = at(shelf)
    if (opens) {
      const label = `${section} · ${counts.get(section)}`.toUpperCase()
      const last = holders[holders.length - 1]
      if (last?.shelf === shelf) last.end = Math.min(last.end, x0 + x - 4)
      holders.push({ shelf, x: x0 + x, end: x0 + bayWidth, z, label })
    }
    cans.push({ ...size, item, path, index, shelf, x: x0 + x, y: size.setback, z })
    x = half(x + size.thickness + RACK.gap)
    inSection++
    previous = section
    lastDiameter = size.diameter
  }
  const shelves = Math.max(shelf + 1, Math.min(3, shelfRows))
  const bays = Math.ceil(shelves / shelfRows)
  const rowEnds: number[] = []
  for (const can of cans) rowEnds[can.shelf] = can.index
  const pull = RACK.start + 380 + RACK.pullClear
  const right = bayStart(bays, bayWidth) + RACK.panel + 140
  const top = rackHeight(shelfRows) + 40
  const corners: Point[] = []
  for (const cx of [-120, right])
    for (const cy of [-pull, RACK.depth + 20])
      for (const cz of [-40, top]) corners.push(project(cx, cy, cz))
  const xs = corners.map(([px]) => px)
  const ys = corners.map(([, py]) => py)
  return {
    cans,
    dividers,
    holders,
    shelves,
    bays,
    shelfRows,
    bayWidth,
    rowEnds,
    bounds: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
  }
}

// Can solids. A disc in the plane x = const projects to an ellipse with semi-axes √1.5·r and √0.5·r,
// major axis at −60° on screen; `discMatrix` draws it from a circle of radius r.

const r1 = (n: number) => Math.round(n * 10) / 10
const pt = ([px, py]: Point) => `${r1(px)} ${r1(py)}`

/** Screen point at angle θ (degrees) on the rim of the disc at x, about the can's axis. */
export function rimPoint(can: CanSize & { y: number; z: number }, x: number, deg: number): Point {
  const r = can.diameter / 2
  const t = (deg * Math.PI) / 180
  return project(x, can.y + r + r * Math.cos(t), can.z + r + r * Math.sin(t))
}

/** Elliptical arc command to `b`; the sweep follows the turn from a through m to b. */
function arcTo(r: number, a: Point, m: Point, b: Point): string {
  const cross = (m[0] - a[0]) * (b[1] - m[1]) - (m[1] - a[1]) * (b[0] - m[0])
  return `A${r1(1.2247 * r)} ${r1(0.7071 * r)} -60 0 ${cross > 0 ? 1 : 0} ${pt(b)}`
}

/**
 * Closed path over the visible side of the can between angles `from` and `to` (the visible band is
 * 45° to 225°) and between x0 and x1. Arcs are split at the middle angle so none spans 180°.
 */
export function bandPath(
  can: CanSize & { y: number; z: number },
  x0: number,
  x1: number,
  from: number,
  to: number,
): string {
  const r = can.diameter / 2
  const mid = (from + to) / 2
  const q = (x: number, deg: number) => rimPoint(can, x, deg)
  const half = (x: number, a: number, b: number) => arcTo(r, q(x, a), q(x, (a + b) / 2), q(x, b))
  return (
    `M${pt(q(x0, from))}${half(x0, from, mid)}${half(x0, mid, to)}` +
    `L${pt(q(x1, to))}${half(x1, to, mid)}${half(x1, mid, from)}Z`
  )
}

/** Shared label arc of radius r about the origin, from 150° (head) to 210° (foot). */
export function labelArc(r: number): string {
  const q = (deg: number): Point => {
    const t = (deg * Math.PI) / 180
    return project(0, r * Math.cos(t), r * Math.sin(t))
  }
  return `M${pt(q(150))}${arcTo(r, q(150), q(180), q(210))}`
}

/** Places a circle of radius r on the lid plane x, centred on the can's axis. */
export function discMatrix(can: CanSize & { y: number; z: number }, x: number): string {
  const r = can.diameter / 2
  const [e, f] = project(x, can.y + r, can.z + r)
  return `matrix(0.866 -0.5 0 1 ${r1(e)} ${r1(f)})`
}

/**
 * Moulded lid relief in the lid's own disc frame (radius r): four radial ribs from the hub to the
 * stacking ring, as one path.
 */
export function lidRibs(r: number): string {
  const [inner, outer, w] = [r * 0.24, r * 0.86, r * 0.05]
  return [0, 90, 180, 270]
    .map(deg => {
      const t = ((deg + 45) * Math.PI) / 180
      const [c, sn] = [Math.cos(t), Math.sin(t)]
      const p = (d: number, o: number) => `${r1(d * c - o * sn)} ${r1(d * sn + o * c)}`
      return `M${p(inner, -w)}L${p(outer, -w)}L${p(outer, w)}L${p(inner, w)}Z`
    })
    .join('')
}

/** Screen offset of a can pulled out of the rack along -y and lifted. */
export function pullOffset(can: CanSize): Point {
  return project(0, -(can.setback + can.diameter + RACK.pullClear), RACK.pullLift)
}

/** Screen centre of a can's rim label, for the camera. */
export function rimCentre(can: CanSize & { x: number; y: number; z: number }): Point {
  return rimPoint(can, can.x + can.thickness / 2, 180)
}
