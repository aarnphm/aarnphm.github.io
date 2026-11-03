import type { LatentKind, LatentTrackDoc } from './types'
import { yearOfDay } from './types'

/**
 * The rail is a time-ordered document track: every dated document gets one slot in date order
 * (rank, not linear time, so a 1928 paper does not squash a decade of notes into a pixel), the
 * undated documents share a short band at the bottom. The open document expands into chunk rows
 * at its slot; leaders run from there to the neighbours' slots. Geometry is in viewBox units and
 * shared by the build-time SVG and the client rebuild.
 */

export const RAIL_WIDTH = 200
export const RAIL_HEIGHT = 920
export const RAIL_TOP = 18
export const RAIL_BOTTOM = 16
export const UNDATED_BAND = 72
export const LABEL_ROW = 13
export const LANE_X: Record<LatentKind, number> = { note: 30, pdf: 42, image: 54, base: 54 }
export const TRACK_X = 24
export const LEADER_X = 70
export const LABEL_X = 76
export const OPEN_WIDTH = 150

export interface RailLayout {
  /** y of each document slot, indexed like docs. */
  y: Float32Array
  datedCount: number
  undatedTop: number
  years: { year: number; y: number }[]
}

export function layoutRail(docs: LatentTrackDoc[]): RailLayout {
  const y = new Float32Array(docs.length)
  const dated = docs.filter(d => d.day > 0).length
  const undated = docs.length - dated
  const undatedTop = RAIL_HEIGHT - RAIL_BOTTOM - (undated > 0 ? UNDATED_BAND : 0)
  const datedSpan = undatedTop - RAIL_TOP - (undated > 0 ? 10 : 0)
  const years: { year: number; y: number }[] = []
  let di = 0
  let ui = 0
  let lastYear = 0
  for (let i = 0; i < docs.length; i++) {
    const doc = docs[i]
    if (doc.day > 0) {
      y[i] = RAIL_TOP + (dated > 1 ? (di / (dated - 1)) * datedSpan : datedSpan / 2)
      const year = yearOfDay(doc.day)
      if (year !== lastYear) {
        years.push({ year, y: y[i] })
        lastYear = year
      }
      di++
    } else {
      y[i] = undatedTop + 8 + (undated > 1 ? (ui / (undated - 1)) * (UNDATED_BAND - 16) : 0)
      ui++
    }
  }
  return { y, datedCount: dated, undatedTop, years }
}

/** Keep year ticks at least `gap` apart: the first of each run of close years wins. */
export function thinYears(
  years: { year: number; y: number }[],
  gap = 22,
): { year: number; y: number }[] {
  const out: { year: number; y: number }[] = []
  for (const tick of years) {
    const last = out[out.length - 1]
    if (!last || tick.y - last.y >= gap) out.push(tick)
  }
  return out
}

/**
 * 1-d label de-overlap by block averaging (the mythos-map pass): labels keep their order, runs
 * that would overlap are centred on the mean of their sources. Returns the label centres.
 */
export function placeLabels(sources: number[], row: number, top: number, bottom: number): number[] {
  const blocks: { start: number; count: number; sum: number }[] = []
  sources.forEach((source, index) => {
    blocks.push({ start: index, count: 1, sum: source - index * row })
    while (blocks.length > 1) {
      const right = blocks[blocks.length - 1]
      const left = blocks[blocks.length - 2]
      if (left.sum / left.count <= right.sum / right.count) break
      blocks.splice(-2, 2, {
        start: left.start,
        count: left.count + right.count,
        sum: left.sum + right.sum,
      })
    }
  })
  const centres: number[] = []
  const lowest = bottom - (sources.length - 1) * row
  for (const block of blocks) {
    const base = Math.max(top, Math.min(lowest, block.sum / block.count))
    for (let offset = 0; offset < block.count; offset++) {
      const index = block.start + offset
      centres[index] = base + index * row
    }
  }
  return centres
}

export function truncate(text: string, max: number): string {
  const clean = text.replace(/\s+/g, ' ').trim()
  return clean.length <= max ? clean : `${clean.slice(0, max - 1).trimEnd()}…`
}
