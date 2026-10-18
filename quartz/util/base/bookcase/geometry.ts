// Shared by the Quartz bookcase view and the Obsidian bookcase plugin, which imports this file by
// relative path. Keep it free of imports and side effects.
//
// World axes: a shelf runs along +x, depth runs along +y (y = 0 is the front plane of the case),
// +z is up and z = 0 is the top of the lowest board. The viewer looks from (+x, -y, +z), so the
// visible faces of a book are the spine (-y), the front cover (+x) and the top (+z).

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

const BOARD = 10
const WALL = 12
const CROWN = 14
const OVERHANG = 5
const PLINTH = 22
const GAP = 1.5
const START = 4
const SETBACK = 5
const HEADROOM = 0.24
// A pulled book leaves the case entirely (past its setback and the label plate) and rises a
// little, so it is nearer than every other part and is painted last without breaking depth order.
const PULL_CLEAR = 12
const LIFT = 10

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
    setback: half(unit(`${path}#y`) * SETBACK),
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
  x: number
  /** Top of the board the book stands on. */
  z: number
}

export interface Shelf {
  /** 0 is the top shelf. */
  index: number
  z: number
  /** Group label, set on the first shelf of each group. */
  label: string | null
  group: number
  /** Right end of the last book. */
  end: number
}

export interface BookcaseLayout<T> {
  books: PlacedBook<T>[]
  shelves: Shelf[]
  shelfWidth: number
  depth: number
  pitch: number
  /** Screen-space bounds of the case and every pulled book: [minX, minY, maxX, maxY]. */
  bounds: readonly [number, number, number, number]
}

export interface PackOptions {
  bookHeight?: number
  shelfWidth?: number
  aspect?: number
}

/** Each group starts a new shelf; a group wraps onto the next shelf at `shelfWidth`. */
export function packShelves<T>(
  groups: ShelfGroup<T>[],
  options: PackOptions = {},
): BookcaseLayout<T> {
  const bookHeight = options.bookHeight ?? BOOKCASE_DEFAULTS.bookHeight
  const shelfWidth = options.shelfWidth ?? BOOKCASE_DEFAULTS.shelfWidth
  const aspect = options.aspect ?? BOOKCASE_DEFAULTS.aspect
  const pitch = half(bookHeight * (1 + HEADROOM) + BOARD)
  const depth = half(bookHeight / aspect + 10)

  type Pending = Omit<PlacedBook<T>, 'z'>
  const pending: Pending[] = []
  const runs: Omit<Shelf, 'z' | 'index'>[] = []
  let index = 0
  for (const [group, { label, books }] of groups.entries()) {
    if (books.length === 0) continue
    runs.push({ label, group, end: START })
    for (const { path, item } of books) {
      const size = bookSize(path, bookHeight, aspect)
      let run = runs[runs.length - 1]
      if (run.end > START && run.end + size.thickness > shelfWidth) {
        runs.push({ label: null, group, end: START })
        run = runs[runs.length - 1]
      }
      pending.push({ ...size, item, path, index: index++, shelf: runs.length - 1, x: run.end })
      run.end = half(run.end + size.thickness + GAP)
    }
  }

  const count = runs.length
  const shelves: Shelf[] = runs.map((run, i) => ({
    ...run,
    end: run.end - GAP,
    index: i,
    z: (count - 1 - i) * pitch,
  }))
  const books = pending.map(book => ({ ...book, z: shelves[book.shelf].z }))

  const layout: BookcaseLayout<T> = {
    books,
    shelves,
    shelfWidth,
    depth,
    pitch,
    bounds: [0, 0, 0, 0],
  }
  const corners: Point[] = []
  const parts = caseFaces(layout)
  corners.push(...parts.crown.edge, ...parts.halo, ...parts.rightWall.edge)
  for (const book of books) {
    const [dx, dy] = pullOffset(book)
    for (const [x, y] of bookSolid(book).edge) corners.push([x + dx, y + dy])
  }
  const xs = corners.map(p => p[0])
  const ys = corners.map(p => p[1])
  return { ...layout, bounds: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] }
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

export function bookSolid(book: PlacedBook<unknown>): Solid {
  return box(book.x, book.setback, book.z, book.thickness, book.depth, book.height)
}

export interface BookDetail {
  /** Page block on the top face, inset from the boards. */
  pages: Point[]
  pageCreases: Segment[]
  /** Rounded spine: the lit line, then the shaded one. */
  round: [Segment, Segment]
  /** Head and tail bands; empty on thin books. */
  bands: Segment[]
}

export function bookDetail(book: PlacedBook<unknown>): BookDetail {
  const { x, z, thickness: t, height: h, depth: d, setback: y } = book
  const board = 1.6
  const top = z + h
  const front = y + board * 2.4
  const back = y + d - board
  const along = (u: number): Segment => [project(x + u, front, top), project(x + u, back, top)]
  const across = (from: number): Segment => [
    project(x, y, top - from),
    project(x + t, y, top - from),
  ]
  const inner = t - 2 * board
  return {
    pages: [
      project(x + board, front, top),
      project(x + t - board, front, top),
      project(x + t - board, back, top),
      project(x + board, back, top),
    ],
    pageCreases: [along(board + inner / 3), along(board + (inner * 2) / 3)],
    round: [
      [project(x + 2.2, y, z), project(x + 2.2, y, top)],
      [project(x + t - 2.2, y, z), project(x + t - 2.2, y, top)],
    ],
    bands: t >= 14 ? [across(10), across(13), across(h - 13), across(h - 10)] : [],
  }
}

/** Screen offset that takes a book off its shelf: out along -y past the case, then up. */
export function pullOffset(book: BookSize): Point {
  return project(0, -(book.setback + book.depth + PULL_CLEAR), LIFT)
}

/** Screen centre of a book's spine where it stands. */
export function spineCentre(book: PlacedBook<unknown>): Point {
  return project(book.x + book.thickness / 2, book.setback, book.z + book.height / 2)
}

/** Trims `text` with an ellipsis to fit `length` at `size`, given an average glyph width in em. */
export function fitText(text: string, length: number, size: number, em = 0.5): string {
  const max = Math.floor(length / (size * em))
  if (text.length <= max) return text
  return max <= 1 ? '' : `${text.slice(0, max - 1).trimEnd()}…`
}

export interface SpineText {
  /** Frame 20 units below the spine's top centre; local x runs down the spine. */
  matrix: Matrix
  title: { text: string; size: number } | null
  /** Anchored at its end, 14 units above the foot. */
  author: { text: string; size: number; x: number } | null
}

/** Spine labels read top to bottom, letter tops toward the front cover. */
export function spineText(book: PlacedBook<unknown>, title: string, author?: string): SpineText {
  const { x, z, thickness: t, height: h, setback: y } = book
  const [e, f] = project(x + t / 2, y, z + h - 20)
  const size = Math.min(0.5 * t, 11.5)
  const authorSize = size * 0.8
  const foot = h - 34
  const authorText = author ? fitText(author, foot * 0.35, authorSize, 0.6) : ''
  const room = foot - (authorText ? authorText.length * authorSize * 0.6 + 10 : 0)
  const titleText = fitText(title, room, size, 0.42)
  return {
    matrix: [0, 1, -COS30, -0.5, e, f],
    title: titleText ? { text: titleText, size } : null,
    author: authorText ? { text: authorText, size: authorSize, x: foot } : null,
  }
}

/** Maps a `depth × height` image, top-left at the spine edge, onto the front cover. */
export function coverMatrix(book: PlacedBook<unknown>): Matrix {
  const [e, f] = project(book.x + book.thickness, book.setback, book.z + book.height)
  return [COS30, -0.5, 0, 1, e, f]
}

export interface ShelfFaces {
  board: Solid
  /** Line 2 units below the board's front-top edge. */
  bevel: Segment
  /** Floor of the blurred contact strip along the back of the board. */
  contact: Point[]
  /** Card holder on the first board of a group. */
  plate: { face: Point[]; ticks: Segment[]; matrix: Matrix; label: string } | null
}

export function shelfFaces(shelf: Shelf, layout: BookcaseLayout<unknown>): ShelfFaces {
  const { shelfWidth: w, depth: d } = layout
  const z = shelf.z
  let plate: ShelfFaces['plate'] = null
  if (shelf.label) {
    const label = shelf.label.toUpperCase()
    const x0 = 14
    const x1 = Math.min(w - 8, x0 + 7.2 * label.length + 18)
    const y = -1.2
    const z0 = z - BOARD + 1.5
    const z1 = z - 1.5
    const [e, f] = project(x0 + 5, y, (z0 + z1) / 2)
    plate = {
      face: [project(x0, y, z0), project(x1, y, z0), project(x1, y, z1), project(x0, y, z1)],
      ticks: [0, 1, 2, 3].map(i => {
        const tx = x1 - 4 - i * 2
        return [project(tx, y, z0 + 1.5), project(tx, y, z1 - 1.5)] as Segment
      }),
      matrix: [COS30, 0.5, 0, 1, e, f],
      label,
    }
  }
  const end = Math.min(w, shelf.end + 30)
  return {
    board: box(0, 0, z - BOARD, w, d, BOARD),
    bevel: [project(0, 0, z - 2), project(w, 0, z - 2)],
    contact: [project(2, d - 14, z), project(end, d - 14, z), project(end, d, z), project(2, d, z)],
    plate,
  }
}

export interface CaseFaces {
  halo: Point[]
  plinth: Solid
  back: Point[]
  /** Inner face of the left wall, painted behind the shelves. */
  leftInner: Point[]
  /** Front edge of the left wall, painted after the shelves. */
  leftFront: Point[]
  leftBevel: Segment
  rightWall: Solid
  rightBevel: Segment
  crown: Solid
  crownBevel: Segment
  crownInset: Point[]
  screws: Point[]
}

export function caseFaces(layout: BookcaseLayout<unknown>): CaseFaces {
  const { shelfWidth: w, depth: d, pitch, shelves } = layout
  const ceiling = shelves.length * pitch - BOARD
  const floor = -PLINTH
  const x0 = -WALL - OVERHANG
  const x1 = w + WALL + OVERHANG
  const top = ceiling + CROWN
  const inset = 8
  return {
    halo: [
      project(x0, -OVERHANG, floor),
      project(x1, -OVERHANG, floor),
      project(x1, d, floor),
      project(x0, d, floor),
    ],
    plinth: box(-WALL + 6, 6, floor, w + 2 * WALL - 12, d - 6, PLINTH - BOARD),
    back: [
      project(0, d, -BOARD),
      project(w, d, -BOARD),
      project(w, d, ceiling),
      project(0, d, ceiling),
    ],
    leftInner: [
      project(0, 0, floor),
      project(0, d, floor),
      project(0, d, ceiling),
      project(0, 0, ceiling),
    ],
    leftFront: [
      project(-WALL, 0, floor),
      project(0, 0, floor),
      project(0, 0, ceiling),
      project(-WALL, 0, ceiling),
    ],
    leftBevel: [project(-WALL + 2, 0, floor), project(-WALL + 2, 0, ceiling)],
    rightWall: box(w, 0, floor, WALL, d, ceiling - floor),
    rightBevel: [project(w + 2, 0, floor), project(w + 2, 0, ceiling)],
    crown: box(x0, -OVERHANG, ceiling, x1 - x0, d + OVERHANG, CROWN),
    crownBevel: [project(x0, -OVERHANG, top - 3), project(x1, -OVERHANG, top - 3)],
    crownInset: [
      project(x0 + inset, -OVERHANG + inset, top),
      project(x1 - inset, -OVERHANG + inset, top),
      project(x1 - inset, d - inset, top),
      project(x0 + inset, d - inset, top),
    ],
    screws: [
      project(x0 + 4, -OVERHANG + 4, top),
      project(x1 - 4, -OVERHANG + 4, top),
      project(x1 - 4, d - 4, top),
      project(x0 + 4, d - 4, top),
    ],
  }
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
