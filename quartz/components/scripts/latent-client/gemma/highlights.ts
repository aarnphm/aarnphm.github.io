import type { Runtime } from '../runtime'
import type { BakeRow } from './bake'
import { align, chunkText, domRange, type Aligned } from './text'

// ---- highlights -------------------------------------------------------------------------------

const HIGHLIGHTS = ['lat-gemma-1', 'lat-gemma-2', 'lat-gemma-3']

export function clearHighlights(): void {
  if (!('highlights' in CSS)) return
  for (const name of HIGHLIGHTS) CSS.highlights.delete(name)
}

/** three strength buckets: the Highlight API has no per-range opacity */
export function highlight(al: Aligned, data: BakeRow, items: { t: number; a: number }[]): void {
  clearHighlights()
  if (!('highlights' in CSS) || !items.length) return
  const max = Math.max(...items.map(i => i.a))
  const buckets: Range[][] = [[], [], []]
  for (const { t, a } of items) {
    const range = domRange(al, data.spans[2 * t], data.spans[2 * t + 1])
    if (!range) continue
    const x = a / max
    buckets[x > 0.66 ? 2 : x > 0.33 ? 1 : 0].push(range)
  }
  buckets.forEach((ranges, i) => {
    if (ranges.length) CSS.highlights.set(HIGHLIGHTS[i], new Highlight(...ranges))
  })
}

const MARKS = new WeakMap<HTMLElement, Aligned>()

/**
 * Highlight raw-text spans (UTF-16 offsets into the chunk's markdown) of a reader passage, each
 * in one of the three strengths (0 faint .. 2 strong). For the circuit tab: its context window,
 * traced token, and hovered node's token. Resolves false when the passage is not on screen.
 */
export async function markPassage(
  rt: Runtime,
  row: number,
  spans: [number, number, number][],
): Promise<boolean> {
  const chunk = rt.reader.querySelector<HTMLElement>(`.lat-chunk[data-row="${row}"]`)
  if (!chunk || !('highlights' in CSS)) return false
  let al = MARKS.get(chunk)
  if (!al || !al.nodes[0]?.node.isConnected) {
    const raw = await chunkText(rt, row)
    if (raw === null) return false
    al = align(chunk, row, raw)
    MARKS.set(chunk, al)
  }
  clearHighlights()
  const buckets: Range[][] = [[], [], []]
  for (const [s, e, level] of spans) {
    const range = domRange(al, s, e)
    if (range) buckets[Math.max(0, Math.min(2, level))].push(range)
  }
  buckets.forEach((ranges, i) => {
    if (ranges.length) CSS.highlights.set(HIGHLIGHTS[i], new Highlight(...ranges))
  })
  return true
}
