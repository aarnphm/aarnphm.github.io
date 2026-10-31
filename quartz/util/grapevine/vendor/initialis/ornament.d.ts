import type { Field } from './field.js'

export type Pt = [number, number]
export interface Stem {
  pts: Pt[]
  ws: number[]
  dirs: number[]
  ks: number[]
  sign: number
  end: 'len' | 'blocked' | 'curl'
  depth: number
  len: number
  id: number
}
/** Leaf and flower geometry: closed polygons to fill, detail polylines, and dots [x, y, r]. */
export interface Shape {
  polys: Pt[][]
  lines: Pt[][]
  dots: [number, number, number][]
}
export interface Items {
  stems: Stem[]
  leaves: Shape[]
  flowers: Shape[]
  /** [x, y, r, t0] */
  dots: number[][]
  beads: number[][]
}
export interface OrnOptions {
  margin: number
  stemW: number
  taper: number
  curl: number
  turns: number
  undulate: number
  wander: number
  thin: number
  branch: number
  maxDepth: number
  branchAngle: number
  length: number
  leaf: string
  leafSize: number
  leafEvery: number
  leafW: number
  lobes: number
  lobeDepth: number
  serr: number
  leafCurl: number
  leafAngle: number
  leafSide: 'alt' | 'outer'
  flower: string
  flowerChance: number
  flowerSize: number
  petals: number
  berries: number
  berrySize: number
  dots: number
  beads: number
  terminal: 'curl' | 'ball' | 'leaf' | 'flower' | 'point'
  density: number
  sprout: number
  mains: number
  fillLeaves: boolean
  detail: 'none' | 'vein' | 'veins' | 'hatch' | 'inline'
  budget: number
  /** Longest a single stem may grow, in field units (garden addition; upstream fixes 4000). */
  maxLen?: number
  settings: Record<string, unknown>
}
export interface Seed {
  x: number
  y: number
  dir: number
  w?: number
  len?: number
  sign?: number
  grace?: number
}
export function ornSizes(o: OrnOptions): {
  leafMax: number
  flowerMax: number
  fillMin: number
  sproutMin: number
  cap: number
}
export function grow(
  field: Field,
  o: OrnOptions,
  R: () => number,
  noise: (x: number, y: number) => number,
  seeds: Seed[],
): Items
export function stemPoly(
  st: Pick<Stem, 'pts' | 'ws' | 'dirs'>,
  contrast?: number,
  penAngle?: number,
  endKind?: 'round' | 'flat',
  minW?: number,
): Pt[]
export function stemDetail(st: Stem, kind: string): Pt[][]
