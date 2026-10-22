import {
  CASE,
  type BookcaseLayout,
  type BookSize,
  type Matrix,
  type Point,
  type Segment,
} from '../geometry'

export const {
  board: BOARD,
  wall: WALL,
  crown: CROWN,
  overhang: OVERHANG,
  plinth: PLINTH,
  start: START,
} = CASE

export const BOOKCASE_STYLES = ['straight', 'corner', 'spine', 'wall', 'zigzag', 'ladder'] as const
export type BookcaseStyle = (typeof BOOKCASE_STYLES)[number]

export const SHELF_MATERIALS = [
  'paper',
  'walnut',
  'birch-ply',
  'black-lacquer',
  'steel',
  'copper',
  'concrete',
] as const
export type ShelfMaterial = (typeof SHELF_MATERIALS)[number]

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

export type Orientation = 'front' | 'side' | 'top'

export interface Face {
  points: Point[]
  tone: Tone
  /** Which way the face looks; the finish shades it from its lit edge. */
  o?: Orientation
  /** The long axis is up (or along the depth for a top), not along the run. */
  tall?: boolean
}

export type World = readonly [number, number, number]
export type Plan = readonly (readonly [number, number])[]

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

/** A filled part in the anatomy idiom: faces, then lines, creases, silhouette and screws. */
export interface StyleSolid {
  part: StylePart
  material: ShelfMaterial
  faces: Face[]
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

export type Tones = Partial<Record<'front' | 'side' | 'top', Tone>>

export interface StyleContext {
  layout: BookcaseLayout<unknown>
  pick: (part: StylePart) => ShelfMaterial
  /** Book indices of each run, in run order. */
  runs: number[][]
  showTags: boolean
}

export interface ShelfOptions {
  material: ShelfMaterial
  /** Board span in run coordinates. The plate starts at its left edge and stops at its end. */
  board: readonly [number, number]
  /** Contact strip in world points, when it is not the strip along the back of the board. */
  contact?: World[]
  /** Plate bottom and top relative to the board top; thin blades hang it below the edge. */
  plateZ?: readonly [number, number]
}

export interface Built {
  placements: ShelfPlacement[]
  layers: StyleLayer[]
}
