/**
 * Seeded randomness. Every ornament is a pure function of its params, so the
 * same seed always grows the same flourish — that's what lets a document store
 * a handful of numbers instead of a blob of SVG.
 */

export interface Rng {
  /** Uniform in [0, 1). */
  next(): number
  range(min: number, max: number): number
  int(min: number, max: number): number
  chance(p: number): boolean
  sign(): 1 | -1
  pick<T>(items: readonly T[]): T
  /** Independent stream, so one part's draws don't reshuffle another's. */
  fork(label: string | number): Rng
}

function hashString(input: string): number {
  let h = 1779033703 ^ input.length
  for (let i = 0; i < input.length; i++) {
    h = Math.imul(h ^ input.charCodeAt(i), 3432918353)
    h = (h << 13) | (h >>> 19)
  }
  h = Math.imul(h ^ (h >>> 16), 2246822507)
  h = Math.imul(h ^ (h >>> 13), 3266489909)
  return (h ^= h >>> 16) >>> 0
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function createRng(seed: number | string): Rng {
  const key = String(seed)
  const next = mulberry32(hashString(key))

  const rng: Rng = {
    next,
    range: (min, max) => min + (max - min) * next(),
    int: (min, max) => Math.floor(min + (max - min + 1) * next()),
    chance: p => next() < p,
    sign: () => (next() < 0.5 ? -1 : 1),
    pick: items => items[Math.floor(next() * items.length)],
    fork: label => createRng(`${key}/${label}`),
  }
  return rng
}
