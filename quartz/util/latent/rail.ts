import type { LatentKind, LatentTrackDoc } from './types'
import { KIND_ORDER, yearOfDay } from './types'

/**
 * The rail is a time-ordered document track: every dated document gets one slot in date order
 * (rank, not linear time, so a 1928 paper does not squash a decade of notes into a pixel), the
 * undated documents share a short band at the bottom. The open document expands into chunk rows
 * at its slot; leaders run from there to the neighbours' slots. Undated documents are not ranked:
 * they sit on one summary row per kind. Geometry is in viewBox units and shared by the build-time
 * SVG and the client rebuild.
 */

export const RAIL_WIDTH = 200
/** viewBox height at build time; the client re-lays the rail out to its pane's aspect ratio */
export const RAIL_HEIGHT = 920
export const RAIL_TOP = 18
export const RAIL_BOTTOM = 16
/** y of a kind's row inside the undated band, from the band's top. */
export const UNDATED_ROW = 11
/** The undated band: its header line, then one row per kind that has undated documents. */
export const UNDATED_HEAD = 16
export const UNDATED_FIRST = 14
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
  /** top of the undated band; height - RAIL_BOTTOM when every document is dated */
  undatedTop: number
  /** kinds with undated documents, in KIND_ORDER; one band row each */
  undatedKinds: LatentKind[]
  years: { year: number; y: number }[]
}

export function layoutRail(docs: LatentTrackDoc[], height = RAIL_HEIGHT): RailLayout {
  const y = new Float32Array(docs.length)
  const dated = docs.filter(d => d.day > 0).length
  const undated = docs.length - dated
  const present = new Set<LatentKind>()
  for (const doc of docs) if (doc.day === 0) present.add(doc.kind)
  const undatedKinds = KIND_ORDER.filter(kind => present.has(kind))
  const band = undated > 0 ? UNDATED_HEAD + undatedKinds.length * UNDATED_ROW : 0
  const undatedTop = height - RAIL_BOTTOM - band
  const datedSpan = undatedTop - RAIL_TOP - (undated > 0 ? 10 : 0)
  const years: { year: number; y: number }[] = []
  let di = 0
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
      y[i] = undatedTop + UNDATED_FIRST + undatedKinds.indexOf(doc.kind) * UNDATED_ROW
    }
  }
  return { y, datedCount: dated, undatedTop, undatedKinds, years }
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
