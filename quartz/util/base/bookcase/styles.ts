// Shelf styles and materials for the bookcase view, shared with the Obsidian plugin. A style maps
// each packed run to a world placement and adds the solids around it; packShelves is unchanged.
// Keep it free of side effects and of imports other than ./geometry.
import {
  type BookSize,
  type BookcaseLayout,
  type Matrix,
  type PackOptions,
  type PlacedBook,
  type Point,
  type Segment,
  BOOKCASE_DEFAULTS,
  CASE,
  COS30,
  box,
  fitText,
  project,
} from './geometry'

const {
  board: BOARD,
  wall: WALL,
  crown: CROWN,
  overhang: OVERHANG,
  plinth: PLINTH,
  start: START,
} = CASE

export const BOOKCASE_STYLES = ['straight', 'corner', 'spine', 'wall', 'zigzag', 'ladder'] as const
export type BookcaseStyle = (typeof BOOKCASE_STYLES)[number]

export const SHELF_MATERIALS = ['paper', 'walnut', 'birch-ply', 'black-lacquer', 'steel'] as const
export type ShelfMaterial = (typeof SHELF_MATERIALS)[number]

/**
 * A topic is a value of the view's groupBy property or a `topicShelves` key. A book belongs to a
 * topic when its groupBy value is the topic or when it carries the topic as a tag.
 */
export interface TopicShelf {
  style: BookcaseStyle
  material?: ShelfMaterial
  /** Overrides the view's `showTags` for this topic. */
  showTags?: boolean
}

/** Used when the view has no `topicShelves` entry for a topic. */
export const TOPIC_SHELVES: Readonly<Record<string, TopicShelf>> = {
  philosophy: { style: 'wall', material: 'walnut' },
  'p/fiction': { style: 'corner', material: 'paper' },
  'p/nf': { style: 'ladder', material: 'birch-ply' },
  'p/scifi': { style: 'spine', material: 'steel' },
}

const TOPIC_WORDS: Readonly<Record<string, string>> = { nf: 'non-fiction', scifi: 'sci-fi' }

const entryOf = (shelves: unknown, topic: string): Record<string, unknown> | undefined => {
  const entry =
    shelves && typeof shelves === 'object' ? (shelves as Record<string, unknown>)[topic] : null
  return entry && typeof entry === 'object' ? (entry as Record<string, unknown>) : undefined
}

/** Keys of the view's `topicShelves`. */
export function topicKeys(shelves?: unknown): string[] {
  return shelves && typeof shelves === 'object' ? Object.keys(shelves) : []
}

/** The entry's `displayNames`, else the value without its `p/` prefix. */
export function topicLabel(topic: string, shelves?: unknown): string {
  const own = entryOf(shelves, topic)?.displayNames
  if (typeof own === 'string' && own.trim()) return own.trim()
  const bare = topic.replace(/^p\//, '')
  return TOPIC_WORDS[bare] ?? bare
}

/**
 * The view's `topicShelves` entry when its style is known, else the default for the topic. The
 * entry's `showTags` applies either way.
 */
export function topicShelf(topic: string, shelves?: unknown): TopicShelf | undefined {
  const own = entryOf(shelves, topic)
  const style = BOOKCASE_STYLES.find(name => name === own?.style)
  const shelf: TopicShelf | undefined = style
    ? { style, material: SHELF_MATERIALS.find(name => name === own?.material) }
    : TOPIC_SHELVES[topic]
  return shelf && typeof own?.showTags === 'boolean' ? { ...shelf, showTags: own.showTags } : shelf
}

/** Shortest unique uppercase prefix of each label, shown on a collapsed topic tab. */
export function topicShortcuts(labels: readonly string[]): string[] {
  const words = labels.map(label => label.replace(/[^\p{L}\p{N}]/gu, '').toUpperCase() || '?')
  return words.map((word, i) => {
    for (let n = 1; n < word.length; n++) {
      const head = word.slice(0, n)
      if (words.every((other, j) => j === i || !other.startsWith(head))) return head
    }
    return word
  })
}

/** True when a tag list holds the topic itself or a tag nested under it (`p/nf/essays`). */
export function hasTopicTag(tags: unknown, topic: string): boolean {
  if (!Array.isArray(tags)) return false
  return tags.some(tag => {
    if (typeof tag !== 'string') return false
    const bare = tag.replace(/^#/, '')
    return bare === topic || bare.startsWith(`${topic}/`)
  })
}

export type StylePart =
  | 'board'
  | 'wall'
  | 'crown'
  | 'plinth'
  | 'panel'
  | 'pole'
  | 'collar'
  | 'leaf'
  | 'blade'
  | 'upright'
  | 'return'
  | 'foot'
  | 'rail'
  | 'stretcher'
  | 'base'

export type Tone = 'top' | 'face' | 's3' | 's2' | 's1' | 's0'

/**
 * Run length for a style. The spine style packs a stack height and the wall style short blades;
 * the ladder needs longer boards so it stays within six tiers. These bypass BOOKCASE_LIMITS, which
 * only clamp user input.
 */
export function stylePackOptions(
  style: BookcaseStyle,
  options: PackOptions,
  bookCount = 0,
): PackOptions {
  switch (style) {
    case 'spine':
      return { ...options, shelfWidth: 160 }
    case 'wall':
      return { ...options, shelfWidth: 320 }
    case 'ladder':
      return {
        ...options,
        shelfWidth: Math.max(
          options.shelfWidth ?? BOOKCASE_DEFAULTS.shelfWidth,
          Math.ceil((bookCount * 26.5) / 6),
        ),
      }
    default:
      return options
  }
}

type World = readonly [number, number, number]
type Plan = readonly (readonly [number, number])[]

const at = ([x, y, z]: World): Point => project(x, y, z)
const seg = (a: World, b: World): Segment => [at(a), at(b)]
const chain = (points: readonly World[]): Segment[] =>
  points.slice(1).map((point, i) => seg(points[i], point))

/**
 * Where a run goes. `rot: 0` runs along +x with spines on -y; `rot: 90` runs along +y at x ≤ 0
 * with spines on +x, and shows the back cover. `pose: 'flat'` stacks books: the run offset is a
 * height, heads point to -x, spines face -y and the front cover is the +z face.
 */
export interface ShelfPlacement {
  shelf: number
  /** World point of the run's origin: offset 0, the board's front plane, the board's top. */
  origin: World
  rot: 0 | 90
  pose: 'standing' | 'flat'
  /** Flat stacks align heads or tails at origin x. */
  align?: 'head' | 'tail'
  /** Board depth when it differs from the layout's (ladder). */
  depth?: number
  /** Wing, column or side. */
  lane: number
}

const toWorld = (p: ShelfPlacement, lx: number, ly: number, lz: number): World =>
  p.rot === 90
    ? [p.origin[0] - ly, p.origin[1] + lx, p.origin[2] + lz]
    : [p.origin[0] + lx, p.origin[1] + ly, p.origin[2] + lz]

/** A book's world box (w along x, d along y, h along z) under its placement. */
export interface WorldBook extends BookSize {
  x: number
  y: number
  z: number
  w: number
  d: number
  h: number
  rot: 0 | 90
  pose: 'standing' | 'flat'
}

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

export interface Ladder {
  top: string
  face: string
  s3: string
  s2: string
  s1: string
  s0: string
  mid: string
  faint: string
  lo: string
  hi: string
  halo: string
  /** Stripe on thin plywood edges. */
  ply?: string
}

const ladderOf = (row: string, halo: string, ply?: string): Ladder => {
  const [top, face, s3, s2, s1, s0, mid, faint, lo, hi] = row.split(' ')
  return { top, face, s3, s2, s1, s0, mid, faint, lo, hi, halo, ply }
}

const LADDERS: Record<'light' | 'dark', Record<ShelfMaterial, Ladder>> = {
  light: {
    paper: ladderOf(
      '#FFFCF0 #FFFCF0 #F9F6EA #F1EEE2 #E8E5DA #DFDCD1 #6F6E69 #D2D0C6 #9F9D96 #282726',
      '#403E3C',
    ),
    walnut: ladderOf(
      '#8F6D54 #7A5236 #6F4B32 #63432D #563B29 #493324 #3A2A1F #63432D #4D3626 #100F0F',
      '#2A1A10',
    ),
    'birch-ply': ladderOf(
      '#F3EAD6 #EADBC0 #DFCFB4 #D3C3A8 #C6B69C #BBAB90 #6E5B45 #C5B59B #AC9B83 #3A2E22',
      '#403E3C',
      '#B2A189',
    ),
    'black-lacquer': ladderOf(
      '#3C3B38 #2C2B29 #1C1B1A #181716 #151413 #100F0F #8E8C85 #4E4D49 #696863 #FFFCF0',
      '#000000',
    ),
    steel: ladderOf(
      '#D5D8DC #AEB4BA #9EA4AA #8D949A #7C8389 #6C7379 #3C454D #8C9399 #757D84 #1F262B',
      '#2A3238',
    ),
  },
  dark: {
    paper: ladderOf(
      '#403E3C #343331 #2F2E2C #292827 #242322 #1E1D1D #9F9D96 #52514D #6F6E69 #CECDC3',
      '#000000',
    ),
    walnut: ladderOf(
      '#70543F #5C3D27 #533724 #483121 #3E2B1D #33241A #9D8977 #765C47 #8A725F #FFFCF0',
      '#000000',
    ),
    'birch-ply': ladderOf(
      '#C6B599 #B8A383 #A49175 #8E7E66 #786B57 #645949 #3A2E22 #928066 #796953 #FFFCF0',
      '#000000',
      '#7F6E57',
    ),
    'black-lacquer': ladderOf(
      '#454441 #33322F #252423 #1C1B1A #171616 #100F0F #A4A29A #575652 #777570 #FFFCF0',
      '#000000',
    ),
    steel: ladderOf(
      '#AEB3B7 #7F868C #72787E #63696F #545A5F #474D51 #D5DADF #999FA5 #AAB0B6 #F4F6F8',
      '#000000',
    ),
  },
}

/** Flat tone ladder of a material. Book colours never change with the material. */
export function materialLadder(material: ShelfMaterial, theme: 'light' | 'dark'): Ladder {
  return LADDERS[theme][material]
}

/** A filled part in the anatomy idiom: faces, then lines, creases, silhouette and screws. */
export interface StyleSolid {
  part: StylePart
  material: ShelfMaterial
  faces: { points: Point[]; tone: Tone }[]
  lines: { kind: 'bevel' | 'ply' | 'highlight'; segments: Segment[] }[]
  creases: Segment[]
  /** Closed silhouette (boxes and panels). */
  outline: Point[] | null
  /** Open silhouette segments (prisms). */
  edges: Segment[]
  screws: Point[]
}

export interface StyleHalo {
  kind: 'case' | 'contact' | 'shadow'
  material: ShelfMaterial
  points: Point[]
}

export interface Plate {
  face: Point[]
  ticks: Segment[]
  matrix: Matrix
  label: string
}

/**
 * Painter's order. A shelf layer paints its label plate, its contact halo and its books in the
 * order given. The pulled book is painted after every layer, in its own layer.
 */
export type StyleLayer =
  | { kind: 'solids'; halos: StyleHalo[]; solids: StyleSolid[] }
  | { kind: 'shelf'; shelf: number; plate: Plate | null; contact: StyleHalo; books: number[] }

export interface StyledLayout<T> extends BookcaseLayout<T> {
  style: BookcaseStyle
  placements: ShelfPlacement[]
  /** World box of each book, by reading index. */
  worldBooks: WorldBook[]
  layers: StyleLayer[]
  /** Screen bounds of every part and every pulled book: [minX, minY, maxX, maxY]. */
  bounds: readonly [number, number, number, number]
  /** Screen width the camera fits to the stage. */
  fit: number
}

type Tones = Partial<Record<'front' | 'side' | 'top', Tone>>

function solidOf(part: StylePart, material: ShelfMaterial): StyleSolid {
  return { part, material, faces: [], lines: [], creases: [], outline: null, edges: [], screws: [] }
}

function boxPart(
  part: StylePart,
  material: ShelfMaterial,
  [x, y, z]: World,
  [w, d, h]: World,
  tones: Tones = {},
): StyleSolid {
  const s = box(x, y, z, w, d, h)
  const solid = solidOf(part, material)
  solid.faces = [
    { points: s.front, tone: tones.front ?? 'face' },
    { points: s.side, tone: tones.side ?? 's2' },
    { points: s.top, tone: tones.top ?? 'top' },
  ]
  if (material === 'birch-ply' && h <= 16) {
    const stripes = [1 / 3, 2 / 3].flatMap(k =>
      chain([
        [x, y, z + h * k],
        [x + w, y, z + h * k],
        [x + w, y + d, z + h * k],
      ]),
    )
    solid.lines.push({ kind: 'ply', segments: stripes })
  }
  if (material === 'steel') {
    const tall = h >= w && h >= d
    solid.lines.push({
      kind: 'highlight',
      segments: [
        tall
          ? seg([x + 2, y, z + 2], [x + 2, y, z + h - 2])
          : seg([x + 2, y, z + h - 2], [x + w - 2, y, z + h - 2]),
      ],
    })
  }
  solid.creases = s.creases
  solid.outline = s.edge
  return solid
}

/**
 * Right prism over an xy polygon (L-shaped boards and crowns). Side faces whose outward normal
 * points to +x or -y are visible; a corner between two visible faces gets a crease, a corner
 * between a visible and a hidden face an edge.
 */
function prismPart(
  part: StylePart,
  material: ShelfMaterial,
  plan: Plan,
  z0: number,
  h: number,
  tones: Tones = {},
): StyleSolid {
  let area = 0
  for (const [i, [ax, ay]] of plan.entries()) {
    const [bx, by] = plan[(i + 1) % plan.length]
    area += ax * by - bx * ay
  }
  const q = area < 0 ? [...plan].reverse() : [...plan]
  const n = q.length
  const z1 = z0 + h
  const next = (i: number) => q[(i + 1) % n]
  const visible = q.map(([ax, ay], i) => {
    const [bx, by] = next(i)
    return by - ay + (bx - ax) > 0
  })
  const solid = solidOf(part, material)
  for (const [i, [ax, ay]] of q.entries()) {
    if (!visible[i]) continue
    const [bx, by] = next(i)
    solid.faces.push({
      points: [project(ax, ay, z0), project(bx, by, z0), project(bx, by, z1), project(ax, ay, z1)],
      tone: Math.abs(bx - ax) > Math.abs(by - ay) ? (tones.front ?? 'face') : (tones.side ?? 's2'),
    })
  }
  solid.faces.push({ points: q.map(([x, y]) => project(x, y, z1)), tone: tones.top ?? 'top' })
  for (const [i, [x, y]] of q.entries()) {
    const before = visible[(i - 1 + n) % n]
    const [nx, ny] = next(i)
    const riser = seg([x, y, z0], [x, y, z1])
    if (before && visible[i]) solid.creases.push(riser)
    else if (before !== visible[i]) solid.edges.push(riser)
    if (visible[i]) {
      solid.creases.push(seg([x, y, z1], [nx, ny, z1]))
      solid.edges.push(seg([x, y, z0], [nx, ny, z0]))
    } else solid.edges.push(seg([x, y, z1], [nx, ny, z1]))
  }
  return solid
}

function panelPart(
  part: StylePart,
  material: ShelfMaterial,
  points: readonly World[],
  tone: Tone,
  outline = false,
): StyleSolid {
  const solid = solidOf(part, material)
  solid.faces = [{ points: points.map(at), tone }]
  if (outline) solid.outline = points.map(at)
  return solid
}

const withLines = (
  solid: StyleSolid,
  kind: StyleSolid['lines'][number]['kind'],
  segments: Segment[],
): StyleSolid => ({ ...solid, lines: [...solid.lines, { kind, segments }] })

const halo = (
  kind: StyleHalo['kind'],
  material: ShelfMaterial,
  points: readonly World[],
): StyleHalo => ({ kind, material, points: points.map(at) })

const solids = (parts: StyleSolid[], halos: StyleHalo[] = []): StyleLayer => ({
  kind: 'solids',
  halos,
  solids: parts,
})

interface StyleContext {
  layout: BookcaseLayout<unknown>
  pick: (part: StylePart) => ShelfMaterial
  /** Book indices of each run, in run order. */
  runs: number[][]
  showTags: boolean
}

interface ShelfOptions {
  material: ShelfMaterial
  /** Board span in run coordinates. The plate starts at its left edge and stops at its end. */
  board: readonly [number, number]
  /** Contact strip in world points, when it is not the strip along the back of the board. */
  contact?: World[]
  /** Plate bottom and top relative to the board top; thin blades hang it below the edge. */
  plateZ?: readonly [number, number]
}

/** Advance of one plate character: 5.4px Berkeley Mono at 0.6em plus 0.9px tracking. */
const PLATE_CHAR = 4.2
/** Plate room outside its label: 5 before the text, 13 after it for the ticks. */
const PLATE_PAD = 18

function shelfLayer(ctx: StyleContext, p: ShelfPlacement, options: ShelfOptions): StyleLayer {
  const shelf = ctx.layout.shelves[p.shelf]
  const depth = ctx.layout.depth
  const w = (lx: number, ly: number): World => toWorld(p, lx, ly, 0)
  // The contact shadow lies behind the books, not at the back of a deeper board. It stops short of
  // the last book by its blur, so none of it shows on the empty board past the run.
  const end = Math.max(2, shelf.end - 6)
  const contact = options.contact ?? [
    w(2, depth - 14),
    w(end, depth - 14),
    w(end, depth),
    w(2, depth),
  ]
  let plate: Plate | null = null
  if (shelf.label && ctx.showTags) {
    const [x0, end] = options.board
    const full = shelf.label.toUpperCase()
    const x1 = Math.min(end, x0 + PLATE_CHAR * full.length + PLATE_PAD)
    // One unit of slack keeps a label that fits exactly from losing a character to rounding.
    const label = fitText(full, x1 - x0 - PLATE_PAD + 1, PLATE_CHAR, 1)
    const [z0, z1] = options.plateZ ?? [-BOARD + 1.5, -1.5]
    const q = (lx: number, lz: number) => at(toWorld(p, lx, -1.2, lz))
    const [e, f] = q(x0 + 5, (z0 + z1) / 2)
    if (label) {
      plate = {
        face: [q(x0, z0), q(x1, z0), q(x1, z1), q(x0, z1)],
        ticks: [0, 1, 2, 3].map(
          i => [q(x1 - 4 - i * 2, z0 + 1.5), q(x1 - 4 - i * 2, z1 - 1.5)] as Segment,
        ),
        matrix: [COS30, p.rot === 90 ? -0.5 : 0.5, 0, 1, e, f],
        label,
      }
    }
  }
  // A rot-90 run is painted from its far end to the near end; a stack from the bottom up.
  const books = p.rot === 90 ? [...ctx.runs[p.shelf]].reverse() : ctx.runs[p.shelf]
  return {
    kind: 'shelf',
    shelf: p.shelf,
    plate,
    contact: halo('contact', options.material, contact),
    books,
  }
}

interface Built {
  placements: ShelfPlacement[]
  layers: StyleLayer[]
}

const standing = (
  shelf: number,
  origin: World,
  extra: Partial<ShelfPlacement> = {},
): ShelfPlacement => ({ shelf, origin, rot: 0, pose: 'standing', lane: 0, ...extra })

/** Bottom run first: the order in which stacked tiers are painted. */
const bottomUp = (count: number) => Array.from({ length: count }, (_, i) => count - 1 - i)

function straight(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const n = shelves.length
  const board = ctx.pick('board')
  const wall = ctx.pick('wall')
  const crown = ctx.pick('crown')
  const ceiling = n * pitch - BOARD
  const floor = -PLINTH
  const x0 = -WALL - OVERHANG
  const x1 = w + WALL + OVERHANG
  const top = ceiling + CROWN
  const inset = 8
  const placements = shelves.map((_, i) => standing(i, [0, 0, (n - 1 - i) * pitch]))
  const layers: StyleLayer[] = [
    solids(
      [
        boxPart(
          'plinth',
          ctx.pick('plinth'),
          [-WALL + 6, 6, floor],
          [w + 2 * WALL - 12, d - 6, PLINTH - BOARD],
          { front: 's1', side: 's0' },
        ),
        panelPart(
          'panel',
          ctx.pick('panel'),
          [
            [0, d, -BOARD],
            [w, d, -BOARD],
            [w, d, ceiling],
            [0, d, ceiling],
          ],
          's0',
        ),
        panelPart(
          'wall',
          wall,
          [
            [0, 0, floor],
            [0, d, floor],
            [0, d, ceiling],
            [0, 0, ceiling],
          ],
          's1',
        ),
      ],
      [
        halo('case', board, [
          [x0, -OVERHANG, floor],
          [x1, -OVERHANG, floor],
          [x1, d, floor],
          [x0, d, floor],
        ]),
      ],
    ),
  ]
  for (const i of bottomUp(n)) {
    const z = placements[i].origin[2]
    layers.push(
      solids([
        withLines(
          boxPart('board', board, [0, 0, z - BOARD], [w, d, BOARD], { top: 's3' }),
          'bevel',
          [seg([0, 0, z - 2], [w, 0, z - 2])],
        ),
      ]),
      shelfLayer(ctx, placements[i], { material: board, board: [0, w] }),
    )
  }
  const loop: World[] = [
    [x0 + inset, -OVERHANG + inset, top],
    [x1 - inset, -OVERHANG + inset, top],
    [x1 - inset, d - inset, top],
    [x0 + inset, d - inset, top],
    [x0 + inset, -OVERHANG + inset, top],
  ]
  layers.push(
    solids([
      withLines(boxPart('wall', wall, [w, 0, floor], [WALL, d, ceiling - floor]), 'bevel', [
        seg([w + 2, 0, floor], [w + 2, 0, ceiling]),
      ]),
      withLines(
        panelPart(
          'wall',
          wall,
          [
            [-WALL, 0, floor],
            [0, 0, floor],
            [0, 0, ceiling],
            [-WALL, 0, ceiling],
          ],
          'face',
          true,
        ),
        'bevel',
        [seg([-WALL + 2, 0, floor], [-WALL + 2, 0, ceiling])],
      ),
      {
        ...withLines(
          boxPart('crown', crown, [x0, -OVERHANG, ceiling], [x1 - x0, d + OVERHANG, CROWN]),
          'bevel',
          [seg([x0, -OVERHANG, top - 3], [x1, -OVERHANG, top - 3]), ...chain(loop)],
        ),
        screws: [
          at([x0 + 4, -OVERHANG + 4, top]),
          at([x1 - 4, -OVERHANG + 4, top]),
          at([x1 - 4, d - 4, top]),
          at([x0 + 4, d - 4, top]),
        ],
      },
    ]),
  )
  return { placements, layers }
}

/**
 * An L-shaped case. Wing A is the straight case along +x; wing B turns 90° and runs along +y at
 * x ≤ 0. The inner corner opens toward the viewer, so neither wing hides the other: a view ray from
 * wing A keeps x > 0, one from wing B keeps y < 0.
 */
function corner(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const levels = Math.ceil(shelves.length / 2)
  const L = w + 8
  const top = levels * pitch
  const board = ctx.pick('board')
  const wall = ctx.pick('wall')
  const panel = ctx.pick('panel')
  const placements = shelves.map((_, i) => {
    const z = (levels - 1 - Math.floor(i / 2)) * pitch
    return i % 2 === 0 ? standing(i, [0, -L, z], { rot: 90, lane: 1 }) : standing(i, [0, 0, z])
  })
  const crownPlan: Plan = [
    [-d - 5, d],
    [-d - 5, -L - WALL - 5],
    [5, -L - WALL - 5],
    [5, -5],
    [L + WALL + 5, -5],
    [L + WALL + 5, d],
  ]
  const layers: StyleLayer[] = [
    solids(
      [
        prismPart(
          'plinth',
          ctx.pick('plinth'),
          [
            [-d, d],
            [-d, -L - WALL + 6],
            [-6, -L - WALL + 6],
            [-6, 6],
            [L + WALL - 6, 6],
            [L + WALL - 6, d],
          ],
          -PLINTH,
          PLINTH - BOARD,
          { front: 's1', side: 's0', top: 's0' },
        ),
        panelPart(
          'panel',
          panel,
          [
            [-d, d, -BOARD],
            [L, d, -BOARD],
            [L, d, top - BOARD],
            [-d, d, top - BOARD],
          ],
          's0',
        ),
        panelPart(
          'panel',
          panel,
          [
            [-d, -L, -BOARD],
            [-d, d, -BOARD],
            [-d, d, top - BOARD],
            [-d, -L, top - BOARD],
          ],
          's1',
        ),
      ],
      [
        halo(
          'case',
          board,
          crownPlan.map(([x, y]) => [x, y, -PLINTH] as World),
        ),
      ],
    ),
  ]
  for (const level of bottomUp(levels)) {
    const z = (levels - 1 - level) * pitch
    layers.push(
      solids([
        withLines(
          prismPart(
            'board',
            board,
            [
              [-d, d],
              [-d, -L],
              [0, -L],
              [0, 0],
              [L, 0],
              [L, d],
            ],
            z - BOARD,
            BOARD,
            { top: 's3' },
          ),
          'bevel',
          // One bevel along both fronts, through the inner corner.
          chain([
            [0, -L, z - 2],
            [0, 0, z - 2],
            [L, 0, z - 2],
          ]),
        ),
      ]),
    )
    for (const i of [2 * level + 1, 2 * level]) {
      if (i < shelves.length) {
        // Each wing's plate stops 4 short of the inner corner, clear of the other wing's plate.
        layers.push(
          shelfLayer(ctx, placements[i], {
            material: board,
            board: placements[i].rot === 90 ? [0, L - 4] : [4, L],
          }),
        )
      }
    }
  }
  const ct = top - BOARD + CROWN
  layers.push(
    solids([
      withLines(boxPart('wall', wall, [L, 0, -BOARD], [WALL, d, top]), 'bevel', [
        seg([L + 2, 0, -BOARD + 2], [L + 2, 0, top - BOARD - 2]),
      ]),
      withLines(boxPart('wall', wall, [-d, -L - WALL, -BOARD], [d, WALL, top]), 'bevel', [
        seg([-d + 2, -L - WALL, -BOARD + 2], [-d + 2, -L - WALL, top - BOARD - 2]),
      ]),
      {
        ...withLines(
          prismPart('crown', ctx.pick('crown'), crownPlan, top - BOARD, CROWN),
          'bevel',
          [
            ...chain([
              [5, -L - WALL - 5, ct - 3],
              [5, -5, ct - 3],
              [L + WALL + 5, -5, ct - 3],
            ]),
            ...chain([
              [-d + 3, d - 8, ct],
              [-d + 3, -L - WALL + 3, ct],
              [-3, -L - WALL + 3, ct],
              [-3, 3, ct],
              [L + WALL - 3, 3, ct],
              [L + WALL - 3, d - 8, ct],
              [-d + 3, d - 8, ct],
            ]),
          ],
        ),
        screws: (
          [
            [-d - 1, d - 4],
            [-d - 1, -L - WALL - 1],
            [1, -L - WALL - 1],
            [L + WALL + 1, -1],
            [L + WALL + 1, d - 4],
          ] as const
        ).map(([x, y]) => project(x, y, ct)),
      },
    ]),
  )
  return { placements, layers }
}

/**
 * One steel pole on a base plate with small leaves up it; books lie flat with spines out. Leaves
 * alternate right and left of the pole without rotating, since a turned leaf would show a pulled
 * cover upside down. The pole stands near the back of the leaves, between the two lanes, and each
 * leaf ends against a collar on the pole. A plane through the pole separates the lanes, so the left
 * lane is painted first, then the pole, then the right lane.
 */
function spine(ctx: StyleContext): Built {
  const { shelves, shelfWidth: run, depth: d, bookHeight } = ctx.layout
  const n = shelves.length
  const POLE = 14
  const LEAF = 6
  const COLLAR = 10
  const leafW = bookHeight + 16
  const step = run + LEAF + 14
  const baseZ = -30
  const topZ = (n - 1) * step + run + 24
  const py = d - POLE - 6
  const half = Math.round(leafW * 0.6)
  const leaf = ctx.pick('leaf')
  const pole = ctx.pick('pole')
  const collar = ctx.pick('collar')
  const placements = shelves.map((_, i) =>
    i % 2 === 0
      ? standing(i, [POLE / 2 + 8, 0, (n - 1 - i) * step], { pose: 'flat', align: 'head' })
      : standing(i, [-POLE / 2 - 8, 0, (n - 1 - i) * step], {
          pose: 'flat',
          align: 'tail',
          lane: 1,
        }),
  )
  const layers: StyleLayer[] = [
    solids(
      [
        boxPart('base', ctx.pick('base'), [-half, -16, baseZ - 12], [2 * half, d + 32, 12], {
          top: 's3',
        }),
      ],
      [
        halo('case', leaf, [
          [-half - 10, -26, baseZ - 12],
          [half + 10, -26, baseZ - 12],
          [half + 10, d + 26, baseZ - 12],
          [-half - 10, d + 26, baseZ - 12],
        ]),
      ],
    ),
  ]
  const leafLayers = (i: number): StyleLayer[] => {
    const p = placements[i]
    const [ox, , z] = p.origin
    const right = p.lane === 0
    const x0 = right ? POLE / 2 : -POLE / 2 - leafW
    // An empty leaf keeps a contact as long as a full-height book.
    const span = Math.max(
      ...ctx.runs[i].map(index => ctx.layout.books[index].height),
      ctx.runs[i].length > 0 ? 0 : bookHeight,
    )
    const sx = right ? ox - 4 : ox - span - 6
    const ex = right ? ox + span + 6 : ox + 4
    return [
      solids([boxPart('leaf', leaf, [x0, 0, z - LEAF], [leafW, d, LEAF], { top: 's3' })]),
      shelfLayer(ctx, p, {
        material: leaf,
        board: [x0 - ox, x0 + leafW - ox],
        contact: [
          [sx, d - 14, z],
          [ex, d - 14, z],
          [ex, d, z],
          [sx, d, z],
        ],
        plateZ: [-LEAF - 7, -LEAF + 0.5],
      }),
    ]
  }
  const order = bottomUp(n)
  for (const i of order) if (placements[i].lane === 1) layers.push(...leafLayers(i))
  // The pole runs in segments between collars, bottom up, so no two parts of it overlap.
  const column: StyleSolid[] = []
  let from = baseZ
  for (const i of order) {
    const z = placements[i].origin[2]
    column.push(
      boxPart('pole', pole, [-POLE / 2, py, from], [POLE, POLE, z - LEAF - COLLAR - from]),
      boxPart(
        'collar',
        collar,
        [-POLE / 2, py - 4, z - LEAF - COLLAR],
        [POLE, POLE + 8, COLLAR + LEAF],
        { front: 's2', side: 's1', top: 's3' },
      ),
    )
    from = z
  }
  column.push(
    boxPart('pole', pole, [-POLE / 2, py, from], [POLE, POLE, topZ - from]),
    boxPart('pole', pole, [-POLE / 2 - 3, py - 3, topZ], [POLE + 6, POLE + 6, 6]),
  )
  layers.push(solids(column))
  for (const i of order) if (placements[i].lane === 0) layers.push(...leafLayers(i))
  return { placements, layers }
}

/**
 * A tall panel with short blades in two columns, staggered by half a pitch. The column gap is at
 * least the depth, so the right column never overlaps the left one on screen and is painted after it.
 */
function wallHung(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const n = shelves.length
  const BLADE = 6
  const gap = Math.max(160, d + 6)
  const topZ = ((n - 1) * pitch) / 2 + pitch + 10
  const px0 = -30
  const px1 = 2 * w + gap + 30
  const blade = ctx.pick('blade')
  const panel = ctx.pick('panel')
  const placements = shelves.map((_, i) =>
    standing(i, [(i % 2) * (w + gap), 0, ((n - 1 - i) * pitch) / 2], { lane: i % 2 }),
  )
  const layers: StyleLayer[] = [
    solids(
      [
        boxPart('panel', panel, [px0, d, -60], [px1 - px0, 6, topZ + 60], {
          front: 's1',
          side: 's0',
          top: 's2',
        }),
      ],
      [
        halo('case', panel, [
          [px0 + 6, d - 2, -70],
          [px1 + 6, d - 2, -70],
          [px1 + 6, d - 2, -40],
          [px0 + 6, d - 2, -40],
        ]),
      ],
    ),
  ]
  for (const lane of [0, 1]) {
    for (const i of bottomUp(n).filter(i => i % 2 === lane)) {
      const p = placements[i]
      const z = p.origin[2]
      const x0 = p.origin[0] - 6
      const len = shelves[i].end + 14
      layers.push(
        solids(
          [boxPart('blade', blade, [x0, 0, z - BLADE], [len, d, BLADE], { top: 's3' })],
          [
            halo('shadow', panel, [
              [x0, d - 0.5, z - BLADE - 18],
              [x0 + len, d - 0.5, z - BLADE - 18],
              [x0 + len, d - 0.5, z - BLADE],
              [x0, d - 0.5, z - BLADE],
            ]),
          ],
        ),
        shelfLayer(ctx, p, {
          material: blade,
          board: [x0 - p.origin[0], x0 + len - p.origin[0]],
          plateZ: [-BLADE - 7, -BLADE + 0.5],
        }),
      )
    }
  }
  return { placements, layers }
}

/**
 * The S shelf: boards at the straight pitch, an upright in each gap alternating left and right, a
 * short return under each board opposite its upright. A tier under a left upright starts its run
 * clear of it.
 */
function zigzag(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const n = shelves.length
  const UP = 44
  const RET = 12
  const RET_H = 44
  const L = w + UP + 12
  const board = ctx.pick('board')
  const upright = ctx.pick('upright')
  // Gap k lies under tier k; its upright is on the left for even k.
  const leftGap = (k: number) => k % 2 === 0
  const placements = shelves.map((_, i) =>
    standing(i, [i > 0 && leftGap(i - 1) ? UP + 2 : 0, 0, (n - 1 - i) * pitch]),
  )
  const layers: StyleLayer[] = [
    solids(
      [8, L - 38].map(x =>
        boxPart('foot', ctx.pick('foot'), [x, 10, -BOARD - 14], [30, d - 20, 14]),
      ),
      [
        halo('case', board, [
          [-4, -4, -BOARD - 14],
          [L + 4, -4, -BOARD - 14],
          [L + 4, d + 4, -BOARD - 14],
          [-4, d + 4, -BOARD - 14],
        ]),
      ],
    ),
  ]
  for (const i of bottomUp(n)) {
    const p = placements[i]
    const z = p.origin[2]
    const parts: StyleSolid[] = []
    // Every inner board meets an upright at both ends, so only the top board has a free end; its
    // return hangs there. A return under an inner board would stub out below the upright above it.
    // The return is painted before its board, which covers the return's top face.
    if (i === 0 && n > 1) {
      parts.push(
        boxPart(
          'return',
          ctx.pick('return'),
          [leftGap(0) ? L - RET : 0, 0, z - BOARD - RET_H],
          [RET, d, RET_H],
        ),
      )
    }
    parts.push(
      withLines(boxPart('board', board, [0, 0, z - BOARD], [L, d, BOARD]), 'bevel', [
        seg([2, 0, z - 2], [L - 2, 0, z - 2]),
      ]),
    )
    const above = i > 0 ? (leftGap(i - 1) ? 'left' : 'right') : null
    const rise = i > 0 ? placements[i - 1].origin[2] - BOARD - z : 0
    // Uprights are flush with the board ends and front, so each corner closes on one edge.
    if (above === 'left') parts.push(boxPart('upright', upright, [0, 0, z], [UP, d, rise]))
    layers.push(
      solids(parts),
      shelfLayer(ctx, p, { material: board, board: [-p.origin[0], L - p.origin[0]] }),
    )
    // The right upright stands past the last book, so it is painted after the books.
    if (above === 'right')
      layers.push(solids([boxPart('upright', upright, [L - UP, 0, z], [UP, d, rise])]))
  }
  return { placements, layers }
}

/**
 * A leaning ladder: boards share a back plane and grow deeper toward the floor, so their fronts
 * step forward without tilting a face. The rails are the only slanted solids; they meet the board
 * ends with no gap.
 */
function ladder(ctx: StyleContext): Built {
  const { shelves, shelfWidth: w, depth: d, pitch } = ctx.layout
  const n = shelves.length
  // 22 units per tier up to six tiers; more tiers share the same total lean.
  const STEP = n > 6 ? 110 / (n - 1) : 22
  const RW = 12
  const RT = 16
  const FLOOR = -30
  const L = w + 8
  const yBack = d + (n - 1) * STEP
  // The stretcher clears the tallest book on the top tier.
  const zTop = (n - 1) * pitch + ctx.layout.bookHeight + 40
  const board = ctx.pick('board')
  const railMaterial = ctx.pick('rail')
  // Rail front: through every board's front edge, 2 units proud.
  const yAt = (z: number) => yBack - d - 2 - (STEP / pitch) * ((n - 1) * pitch - z)
  const placements = shelves.map((_, i) =>
    standing(i, [0, yBack - (d + i * STEP), (n - 1 - i) * pitch], { depth: d + i * STEP }),
  )
  const rail = (x: number): StyleSolid => {
    const y0 = yAt(FLOOR)
    const y1 = yAt(zTop)
    const [A, B, C, D, E, F, N] = (
      [
        [x, y0, FLOOR],
        [x + RW, y0, FLOOR],
        [x + RW, y0 + RT, FLOOR],
        [x + RW, y1 + RT, zTop],
        [x, y1 + RT, zTop],
        [x, y1, zTop],
        [x + RW, y1, zTop],
      ] as const
    ).map(at)
    const solid = solidOf('rail', railMaterial)
    solid.faces = [
      { points: [A, B, N, F], tone: 'face' },
      { points: [B, C, D, N], tone: 's2' },
      { points: [F, N, D, E], tone: 'top' },
    ]
    solid.lines = [
      { kind: 'highlight', segments: [seg([x + 2, y0, FLOOR + 2], [x + 2, y1, zTop - 2])] },
    ]
    solid.creases = [
      [N, B],
      [N, D],
      [N, F],
    ]
    solid.outline = [A, B, C, D, E, F]
    return solid
  }
  const layers: StyleLayer[] = [
    solids(
      [rail(-RW)],
      [
        halo('case', railMaterial, [
          [-RW - 4, yAt(FLOOR) - 4, FLOOR],
          [L + RW + 4, yAt(FLOOR) - 4, FLOOR],
          [L + RW + 4, yBack + 4, FLOOR],
          [-RW - 4, yBack + 4, FLOOR],
        ]),
      ],
    ),
  ]
  for (const i of bottomUp(n)) {
    const p = placements[i]
    const [, y, z] = p.origin
    layers.push(
      solids([
        withLines(
          boxPart('board', board, [0, y, z - BOARD], [L, p.depth!, BOARD], { top: 's3' }),
          'bevel',
          [seg([2, y, z - 2], [L - 2, y, z - 2])],
        ),
      ]),
      shelfLayer(ctx, p, { material: board, board: [0, L] }),
    )
  }
  // The stretcher spans between the rails, flush with their tops and fronts; the right rail is
  // nearer than its end, so it is painted after it.
  layers.push(
    solids([
      boxPart('stretcher', ctx.pick('stretcher'), [0, yAt(zTop), zTop - 12], [L, RT, 12]),
      rail(L),
    ]),
  )
  return { placements, layers }
}

const STYLE_BUILDERS: Record<BookcaseStyle, (ctx: StyleContext) => Built> = {
  straight,
  corner,
  spine,
  wall: wallHung,
  zigzag,
  ladder,
}

/** Each style's own materials: a base for every part, then per-part exceptions. */
const STYLE_MATERIALS: Record<
  BookcaseStyle,
  [ShelfMaterial, Partial<Record<StylePart, ShelfMaterial>>]
> = {
  straight: ['paper', {}],
  corner: ['walnut', {}],
  spine: ['steel', {}],
  wall: ['birch-ply', {}],
  zigzag: ['black-lacquer', { upright: 'walnut', foot: 'walnut' }],
  ladder: ['birch-ply', { rail: 'steel', stretcher: 'steel' }],
}

export interface PlaceOptions {
  /** False drops every shelf's label plate (the `showTags` view key). */
  showTags?: boolean
}

/**
 * Places each packed run for a style and lists every part in painter's order. Pure. `materials`
 * replaces the style's main material and keeps its per-part exceptions (SPEC 15.3), or gives
 * per-part overrides.
 */
export function placeShelves<T>(
  layout: BookcaseLayout<T>,
  style: BookcaseStyle,
  materials?: ShelfMaterial | Partial<Record<StylePart, ShelfMaterial>>,
  options: PlaceOptions = {},
): StyledLayout<T> {
  const [main, own] = STYLE_MATERIALS[style]
  const base = typeof materials === 'string' ? materials : main
  const parts = typeof materials === 'string' ? undefined : materials
  const pick = (part: StylePart): ShelfMaterial => parts?.[part] ?? own[part] ?? base
  // Runs hold positions in layout.books; callers may renumber `index` (the browser repack does).
  const runs: number[][] = layout.shelves.map(() => [])
  layout.books.forEach((book, i) => runs[book.shelf].push(i))
  const showTags = options.showTags ?? true
  const { placements, layers } = STYLE_BUILDERS[style]({ layout, pick, runs, showTags })
  const worldBooks = layout.books.map(book => worldBook(book, placements[book.shelf]))

  const corners: Point[] = []
  for (const layer of layers) {
    if (layer.kind === 'solids') {
      for (const solid of layer.solids) for (const face of solid.faces) corners.push(...face.points)
      for (const h of layer.halos) corners.push(...h.points)
    } else {
      corners.push(...layer.contact.points, ...(layer.plate?.face ?? []))
    }
  }
  for (const b of worldBooks) {
    const [dx, dy] = pullOffsetW(b)
    for (const [x, y] of bookFacesW(b).edge) corners.push([x, y], [x + dx, y + dy])
  }
  const xs = corners.map(p => p[0])
  const ys = corners.map(p => p[1])
  const bounds = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] as const
  const caseWidth =
    (Math.max(layout.shelfWidth, BOOKCASE_DEFAULTS.shelfWidth) + 2 * WALL + 10 + layout.depth) *
    COS30
  return {
    ...layout,
    style,
    placements,
    worldBooks,
    layers,
    bounds,
    fit: Math.min(bounds[2] - bounds[0], caseWidth),
  }
}
