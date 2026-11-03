export function rng(seed: number): () => number
export function hash(...xs: unknown[]): number
export function makeNoise(seed: number): (x: number, y: number) => number
export const clamp: (x: number, a: number, b: number) => number
export const lerp: (a: number, b: number, t: number) => number
export const TAU: number
export function resample(pts: number[][], step: number, closed?: boolean): number[][]
export function signedArea(pts: number[][]): number
export function offsetPolygon(pts: number[][], d: number): number[][]
export function bbox(contours: number[][][]): {
  x0: number
  y0: number
  x1: number
  y1: number
  w: number
  h: number
}
export function edt(f: Float64Array, nx: number, ny: number): Float64Array
