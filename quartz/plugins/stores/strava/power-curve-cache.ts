import type { PowerCurvePoint } from './efforts-and-curves'
import { releasedOnReload } from '../../../util/module-caches'

export const MEAN_MAX_CURVE_ALGORITHM_VERSION = 'best-power-windows-v1'
export const MEAN_MAX_CURVE_MAX_ENTRY_POINTS = 86_400
export const MEAN_MAX_CURVE_MAX_TOTAL_POINTS = 2_000_000
export const MEAN_MAX_CURVE_MAX_ENTRIES = 512

export interface MeanMaxCurveCacheOptions {
  algorithmVersion?: string
  maxEntries?: number
  maxTotalPoints?: number
  maxEntryPoints?: number
}

export interface MeanMaxCurveCacheStats {
  algorithmVersion: string
  entries: number
  totalPoints: number
  hits: number
  misses: number
  evictions: number
  skippedOversized: number
  maxEntries: number
  maxTotalPoints: number
  maxEntryPoints: number
}

interface CachedCurveEntry {
  watts: Float64Array
  values: Float64Array
}

export interface MeanMaxCurveCache {
  computeKey(watts: Float64Array): string
  getOrCompute(watts: Float64Array, compute: () => PowerCurvePoint[]): PowerCurvePoint[]
  clear(): void
  stats(): MeanMaxCurveCacheStats
}

function wattsContentEqual(left: Float64Array, right: Float64Array): boolean {
  if (left.length !== right.length) return false
  const a = new Uint32Array(left.buffer, left.byteOffset, left.byteLength / 4)
  const b = new Uint32Array(right.buffer, right.byteOffset, right.byteLength / 4)
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function curvePoints(values: Float64Array): PowerCurvePoint[] {
  // meanMaxCurve has one value for each consecutive duration, starting at one second.
  return Array.from(values, (w, index) => ({ s: index + 1, w }))
}

export function createMeanMaxCurveCache(options?: MeanMaxCurveCacheOptions): MeanMaxCurveCache {
  const algorithmVersion = options?.algorithmVersion ?? MEAN_MAX_CURVE_ALGORITHM_VERSION
  const maxEntries = Math.max(1, options?.maxEntries ?? MEAN_MAX_CURVE_MAX_ENTRIES)
  const maxTotalPoints = Math.max(1, options?.maxTotalPoints ?? MEAN_MAX_CURVE_MAX_TOTAL_POINTS)
  const maxEntryPoints = Math.min(
    maxTotalPoints,
    Math.max(1, options?.maxEntryPoints ?? MEAN_MAX_CURVE_MAX_ENTRY_POINTS),
  )

  const entries = new Map<string, CachedCurveEntry>()
  let totalPoints = 0
  let hits = 0
  let misses = 0
  let evictions = 0
  let skippedOversized = 0

  const computeKey = (watts: Float64Array): string => {
    const words = new Uint32Array(watts.buffer, watts.byteOffset, watts.byteLength / 4)
    let hash = 2_166_136_261
    for (const word of words) hash = Math.imul(hash ^ word, 16_777_619) >>> 0
    return `${algorithmVersion}:${watts.length}:${hash}`
  }

  const removeEntry = (key: string, entry: CachedCurveEntry): void => {
    entries.delete(key)
    totalPoints = totalPoints - entry.values.length
  }

  return {
    computeKey,
    getOrCompute(watts: Float64Array, compute: () => PowerCurvePoint[]): PowerCurvePoint[] {
      if (watts.length === 0) return []
      if (watts.length > maxEntryPoints || watts.length > maxTotalPoints) {
        misses++
        skippedOversized++
        return compute()
      }

      const key = computeKey(watts)
      const existing = entries.get(key)
      // The fingerprint chooses a bucket; exact input bytes decide whether reuse is valid.
      if (existing && wattsContentEqual(existing.watts, watts)) {
        hits++
        entries.delete(key)
        entries.set(key, existing)
        return curvePoints(existing.values)
      }

      misses++
      const computed = compute()
      if (computed.length !== watts.length) return computed

      if (existing) removeEntry(key, existing)

      while (
        entries.size > 0 &&
        (entries.size >= maxEntries || totalPoints + computed.length > maxTotalPoints)
      ) {
        const oldestKey = entries.keys().next().value
        if (oldestKey === undefined) break
        const oldestEntry = entries.get(oldestKey)
        if (oldestEntry) removeEntry(oldestKey, oldestEntry)
        else entries.delete(oldestKey)
        evictions++
      }

      entries.set(key, {
        watts: Float64Array.from(watts),
        values: Float64Array.from(computed, point => point.w),
      })
      totalPoints += computed.length
      return computed
    },
    clear(): void {
      entries.clear()
      totalPoints = 0
      hits = 0
      misses = 0
      evictions = 0
      skippedOversized = 0
    },
    stats(): MeanMaxCurveCacheStats {
      return {
        algorithmVersion,
        entries: entries.size,
        totalPoints,
        hits,
        misses,
        evictions,
        skippedOversized,
        maxEntries,
        maxTotalPoints,
        maxEntryPoints,
      }
    },
  }
}

let sharedCache: MeanMaxCurveCache | undefined

export function getMeanMaxCurveCache(): MeanMaxCurveCache {
  return (sharedCache ??= releasedOnReload(createMeanMaxCurveCache()))
}
