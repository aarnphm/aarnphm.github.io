import {
  box,
  CASE,
  COS30,
  fitText,
  project,
  type Matrix,
  type PlacedBook,
  type Point,
  type Segment,
} from '../geometry'
import { START, type ShelfPlacement, type World, type WorldBook } from './style-model'

export const at = ([x, y, z]: World): Point => project(x, y, z)
export const seg = (a: World, b: World): Segment => [at(a), at(b)]
export const chain = (points: readonly World[]): Segment[] =>
  points.slice(1).map((point, i) => seg(points[i], point))

export const toWorld = (p: ShelfPlacement, lx: number, ly: number, lz: number): World =>
  p.rot === 90
    ? [p.origin[0] - ly, p.origin[1] + lx, p.origin[2] + lz]
    : [p.origin[0] + lx, p.origin[1] + ly, p.origin[2] + lz]

export function worldBook(book: PlacedBook<unknown>, p: ShelfPlacement): WorldBook {
  const { thickness, height, depth, setback } = book
  const [ox, oy, oz] = p.origin
  const base = { thickness, height, depth, setback, rot: p.rot, pose: p.pose }
  if (p.pose === 'flat') {
    const x = p.align === 'tail' ? ox - height : ox
    return {
      ...base,
      x,
      y: oy + setback,
      z: oz + book.x - START,
      w: height,
      d: depth,
      h: thickness,
    }
  }
  if (p.rot === 90) {
    return {
      ...base,
      x: ox - setback - depth,
      y: oy + book.x,
      z: oz,
      w: depth,
      d: thickness,
      h: height,
    }
  }
  return { ...base, x: ox + book.x, y: oy + setback, z: oz, w: thickness, d: depth, h: height }
}

/** Book frame: u across the thickness, v from the spine into the book, a from the foot to the head. */
function bookFrame(b: WorldBook): (u: number, v: number, a: number) => Point {
  if (b.pose === 'flat') return (u, v, a) => project(b.x + b.height - a, b.y + v, b.z + u)
  if (b.rot === 90) return (u, v, a) => project(b.x + b.w - v, b.y + u, b.z + a)
  return (u, v, a) => project(b.x + u, b.y + v, b.z + a)
}

export interface BookFaces {
  spine: Point[]
  cover: Point[]
  /** Head boards when standing, the tail when flat. */
  end: Point[]
  creases: Segment[]
  edge: Point[]
}

export function bookFacesW(b: WorldBook): BookFaces {
  const s = box(b.x, b.y, b.z, b.w, b.d, b.h)
  if (b.pose === 'flat')
    return { spine: s.front, cover: s.top, end: s.side, creases: s.creases, edge: s.edge }
  if (b.rot === 90)
    return { spine: s.side, cover: s.front, end: s.top, creases: s.creases, edge: s.edge }
  return { spine: s.front, cover: s.side, end: s.top, creases: s.creases, edge: s.edge }
}

export interface BookDetail {
  /** Page block, inset from the boards. */
  pages: Point[]
  pageCreases: Segment[]
  /** Cover boards stand above the recessed page block. */
  boardRims: Point[][]
  /** The jacket turns over the head and tail of the spine. */
  jacketRims: Point[][]
  jacketCreases: Segment[]
  /** Rounded spine: the lit line, then the shaded one. */
  round: [Segment, Segment]
  /** Head and tail bands; empty on thin books. */
  bands: Segment[]
}

export function bookDetailW(b: WorldBook): BookDetail {
  const p = bookFrame(b)
  const { thickness: t, height: h, depth: d } = b
  const board = Math.min(2, t * 0.08)
  // Standing books show the page block on the head; a flat book shows it on the tail.
  const end = b.pose === 'flat' ? 0 : h
  const a = b.pose === 'flat' ? board * 1.5 : h - board * 1.5
  const front = board * 2.4
  const back = d - board
  const inner = t - 2 * board
  const along = (u: number): Segment => [p(u, front, a), p(u, back, a)]
  const lit = b.pose === 'flat' ? t - 2.2 : 2.2
  return {
    pages: [p(board, front, a), p(t - board, front, a), p(t - board, back, a), p(board, back, a)],
    pageCreases: [along(board + inner / 3), along(board + (inner * 2) / 3)],
    boardRims: [0, t - board].map(u => [
      p(u, 0, end),
      p(u + board, 0, end),
      p(u + board, d, end),
      p(u, d, end),
    ]),
    jacketRims: [0, h - board].map(a => [
      p(0, 0, a),
      p(t, 0, a),
      p(t, 0, a + board),
      p(0, 0, a + board),
    ]),
    jacketCreases: [board, t - board].map(u => [p(u, 0, board), p(u, 0, h - board)]),
    round: [
      [p(lit, 0, 0), p(lit, 0, h)],
      [p(t - lit, 0, 0), p(t - lit, 0, h)],
    ],
    bands:
      t >= 14 ? [10, 13, h - 13, h - 10].map(from => [p(0, 0, h - from), p(t, 0, h - from)]) : [],
  }
}

/** Screen offset that takes a book off its shelf: out along the shelf's normal past the case, then up. */
export function pullOffsetW(b: WorldBook): Point {
  const out = b.setback + b.depth + CASE.pullClear
  return b.rot === 90 ? project(out, 0, CASE.lift) : project(0, -out, CASE.lift)
}

/** Screen centre of a book's spine where it sits. */
export function spineCentreW(b: WorldBook): Point {
  return bookFrame(b)(b.thickness / 2, 0, b.height / 2)
}

/** Maps a thickness × height image from the head of the spine to its foot. */
export function spineMatrixW(b: WorldBook): Matrix {
  const p = bookFrame(b)
  const [e, f] = p(0, 0, b.height)
  const [x, y] = p(1, 0, b.height)
  const [dx, dy] = p(0, 0, b.height - 1)
  return [x - e, y - f, dx - e, dy - f, e, f]
}

export interface SpineText {
  /** Frame 20 units from the head on the spine's centre line; local x runs toward the foot. */
  matrix: Matrix
  title: { text: string; size: number } | null
  /** Anchored at its end, 14 units from the foot. */
  author: { text: string; size: number; x: number } | null
}

/** Spine labels read head to foot, letter tops toward the front cover. */
export function spineTextW(b: WorldBook, title: string, author?: string): SpineText {
  const { thickness: t, height: h } = b
  const [e, f] = bookFrame(b)(t / 2, 0, h - 20)
  const size = Math.min(0.5 * t, 11.5)
  const authorSize = size * 0.8
  const foot = h - 34
  const authorText = author ? fitText(author, foot * 0.35, authorSize, 0.6) : ''
  const room = foot - (authorText ? authorText.length * authorSize * 0.6 + 10 : 0)
  const titleText = fitText(title, room, size, 0.42)
  const axes: [number, number, number, number] =
    b.pose === 'flat'
      ? [COS30, 0.5, 0, 1]
      : b.rot === 90
        ? [0, 1, -COS30, 0.5]
        : [0, 1, -COS30, -0.5]
  return {
    matrix: [...axes, e, f],
    title: titleText ? { text: titleText, size } : null,
    author: authorText ? { text: authorText, size: authorSize, x: foot } : null,
  }
}

/**
 * Maps a `depth × height` image, top-left at the spine edge of the head, onto the front cover.
 * Null for a rot-90 book, whose visible side is the back cover.
 */
export function coverMatrixW(b: WorldBook): Matrix | null {
  if (b.rot === 90) return null
  const [e, f] = bookFrame(b)(b.thickness, 0, b.height)
  return b.pose === 'flat' ? [COS30, -0.5, COS30, 0.5, e, f] : [COS30, -0.5, 0, 1, e, f]
}

/** Reference for the sheen that rounds a book's spine, lit along its lit edge. */
export function sheenFill(b: WorldBook): string {
  return b.pose === 'flat' ? 'url(#bc-g-sheen-v)' : 'url(#bc-g-sheen)'
}
