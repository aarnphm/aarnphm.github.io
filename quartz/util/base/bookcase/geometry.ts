// Shared by the Quartz bookcase view and the Obsidian bookcase plugin, which imports this file by
// relative path. Keep it free of imports and side effects.
//
// World axes: +x right, +y away from the viewer, +z up. The viewer looks from (+x, -y, +z), so only
// faces whose normal is -y, +x or +z are visible. A straight shelf runs along +x with spines on -y
// and front covers on +x; styles.ts places runs for the other shelf styles.

export type Point = readonly [number, number]
export type Matrix = readonly [number, number, number, number, number, number]
export type Segment = readonly [Point, Point]

export const COS30 = Math.sqrt(3) / 2

export const BOOKCASE_DEFAULTS = {
  bookHeight: 220,
  scrollStep: 56,
  shelfWidth: 600,
  /** Median cover height / cover width of the antilibrary posters. */
  aspect: 1.53,
} as const

export const BOOKCASE_LIMITS = {
  bookHeight: [120, 320],
  scrollStep: [24, 120],
  shelfWidth: [320, 1200],
} as const

/** Case dimensions in world units, shared by every style. */
export const CASE = {
  board: 10,
  wall: 12,
  crown: 14,
  overhang: 5,
  plinth: 22,
  /** Gap between books in a run. */
  gap: 1.5,
  /** Offset of the first book in a run. */
  start: 4,
  setback: 5,
  headroom: 0.24,
  // A pulled book leaves the case entirely (past its setback and the label plate) and rises a
  // little, so it is nearer than every other part and is painted last without breaking depth order.
  pullClear: 12,
  lift: 10,
} as const

/** Flexoki 600 hues for books without a cover or a spine colour. */
export const FALLBACK_SPINES = [
  '#AF3029',
  '#BC5215',
  '#AD8301',
  '#66800B',
  '#24837B',
  '#205EA6',
  '#5E409D',
  '#A02F6F',
] as const

export const INK_DARK = '#100F0F'
export const INK_LIGHT = '#FFFCF0'

export function project(x: number, y: number, z: number): Point {
  return [(x + y) * COS30, (x - y) * 0.5 - z]
}

/** FNV-1a, 32 bit, over the vault-relative path (for example `library/Oblivion.md`). */
export function hashPath(path: string): number {
  let hash = 0x811c9dc5
  for (let i = 0; i < path.length; i++) hash = Math.imul(hash ^ path.charCodeAt(i), 0x01000193)
  return hash >>> 0
}

const unit = (path: string) => (hashPath(path) % 1000) / 1000

export function clampOption(
  value: unknown,
  fallback: number,
  [min, max]: readonly [number, number],
): number {
  const n = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback
}

const half = (n: number) => Math.round(n * 2) / 2

export interface BookSize {
  thickness: number
  height: number
  depth: number
  /** Distance of the spine behind the front plane. */
  setback: number
}

export function bookSize(
  path: string,
  bookHeight: number = BOOKCASE_DEFAULTS.bookHeight,
  aspect: number = BOOKCASE_DEFAULTS.aspect,
): BookSize {
  const thickness = half(16 + unit(path) * 18)
  const height = half(bookHeight * (0.86 + unit(`${path}#h`) * 0.14))
  return {
    thickness,
    height,
    depth: half(height / aspect),
    setback: half(unit(`${path}#y`) * CASE.setback),
  }
}

export function fallbackSpine(path: string): string {
  return FALLBACK_SPINES[hashPath(path) % FALLBACK_SPINES.length]
}

function luminance(hex: string): number | undefined {
  const match = /^#?([0-9a-f]{6})$/i.exec(hex.trim())
  if (!match) return undefined
  const value = parseInt(match[1], 16)
  const channel = (c: number) => {
    const s = c / 255
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return (
    0.2126 * channel((value >> 16) & 0xff) +
    0.7152 * channel((value >> 8) & 0xff) +
    0.0722 * channel(value & 0xff)
  )
}

/** The Flexoki ink (black or paper) with the higher contrast on a spine colour. */
export function spineInk(spine: string): string {
  const l = luminance(spine)
  if (l === undefined) return INK_LIGHT
  const contrast = (a: number, b: number) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)
  return contrast(l, luminance(INK_DARK)!) >= contrast(l, luminance(INK_LIGHT)!)
    ? INK_DARK
    : INK_LIGHT
}

export function rgbToHex(r: number, g: number, b: number): string {
  const part = (c: number) =>
    Math.round(Math.min(255, Math.max(0, c)))
      .toString(16)
      .padStart(2, '0')
  return `#${part(r)}${part(g)}${part(b)}`.toUpperCase()
}

export interface ShelfGroup<T> {
  label: string | null
  books: { path: string; item: T }[]
}

export interface PlacedBook<T> extends BookSize {
  item: T
  path: string
  /** Index in reading order (group order, then sort order). */
  index: number
  shelf: number
  /** Offset of the book's left edge along its run. */
  x: number
}

export interface Shelf {
  /** 0 is the first run in reading order (the top shelf of the straight case). */
  index: number
  /** Group label, set on the first run of each group. */
  label: string | null
  group: number
  /** Right end of the last book along the run. */
  end: number
}

export interface BookcaseLayout<T> {
  books: PlacedBook<T>[]
  shelves: Shelf[]
  bookHeight: number
  shelfWidth: number
  depth: number
  pitch: number
}

export interface PackOptions {
  bookHeight?: number
  shelfWidth?: number
  aspect?: number
  /** Groups follow each other on a run instead of each starting a new one. */
  continuous?: boolean
}

/**
 * Each group starts a new run, unless `continuous`; a group wraps onto the next run at
 * `shelfWidth`. A group's label goes on the run that holds its first book.
 */
export function packShelves<T>(
  groups: ShelfGroup<T>[],
  options: PackOptions = {},
): BookcaseLayout<T> {
  const bookHeight = options.bookHeight ?? BOOKCASE_DEFAULTS.bookHeight
  const shelfWidth = options.shelfWidth ?? BOOKCASE_DEFAULTS.shelfWidth
  const aspect = options.aspect ?? BOOKCASE_DEFAULTS.aspect
  const books: PlacedBook<T>[] = []
  const runs: Omit<Shelf, 'index'>[] = []
  let index = 0
  for (const [group, { label, books: members }] of groups.entries()) {
    if (members.length === 0) continue
    if (!options.continuous || runs.length === 0) runs.push({ label: null, group, end: CASE.start })
    for (const [i, { path, item }] of members.entries()) {
      const size = bookSize(path, bookHeight, aspect)
      let run = runs[runs.length - 1]
      if (run.end > CASE.start && run.end + size.thickness > shelfWidth) {
        runs.push({ label: null, group, end: CASE.start })
        run = runs[runs.length - 1]
      }
      if (i === 0 && label) run.label = run.label ? `${run.label} · ${label}` : label
      books.push({ ...size, item, path, index: index++, shelf: runs.length - 1, x: run.end })
      run.end = half(run.end + size.thickness + CASE.gap)
    }
  }
  // Empty query results still have a cabinet, with room for three rows of books.
  if (runs.length === 0)
    for (let shelf = 0; shelf < 3; shelf++) runs.push({ label: null, group: -1, end: CASE.start })
  return {
    books,
    shelves: runs.map((run, i) => ({ ...run, end: run.end - CASE.gap, index: i })),
    bookHeight,
    shelfWidth,
    depth: half(bookHeight / aspect + 10),
    pitch: half(bookHeight * (1 + CASE.headroom) + CASE.board),
  }
}

/** Faces and lines of a box solid in the anatomy idiom. */
export interface Solid {
  front: Point[]
  side: Point[]
  top: Point[]
  /** Folds from the near top corner. */
  creases: Segment[]
  /** Six-corner silhouette. */
  edge: Point[]
}

export function box(x: number, y: number, z: number, w: number, d: number, h: number): Solid {
  const A = project(x, y, z)
  const B = project(x + w, y, z)
  const C = project(x + w, y + d, z)
  const D = project(x + w, y + d, z + h)
  const E = project(x, y + d, z + h)
  const F = project(x, y, z + h)
  const N = project(x + w, y, z + h)
  return {
    front: [A, B, N, F],
    side: [B, C, D, N],
    top: [F, N, D, E],
    creases: [
      [N, B],
      [N, D],
      [N, F],
    ],
    edge: [A, B, C, D, E, F],
  }
}

/** Trims `text` with an ellipsis to fit `length` at `size`, given an average glyph width in em. */
export function fitText(text: string, length: number, size: number, em = 0.5): string {
  const max = Math.floor(length / (size * em))
  if (text.length <= max) return text
  return max <= 1 ? '' : `${text.slice(0, max - 1).trimEnd()}…`
}

const r1 = (n: number) => Math.round(n * 10) / 10

export function toPoints(points: readonly Point[]): string {
  return points.map(([x, y]) => `${r1(x)},${r1(y)}`).join(' ')
}

export function toPath(segments: readonly Segment[]): string {
  return segments.map(([a, b]) => `M${r1(a[0])} ${r1(a[1])}L${r1(b[0])} ${r1(b[1])}`).join('')
}

export function toMatrix(matrix: Matrix): string {
  return `matrix(${matrix.map(n => Math.round(n * 10000) / 10000).join(' ')})`
}
