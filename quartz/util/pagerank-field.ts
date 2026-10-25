import type { PageRankModel } from './pagerank-view'

/** Monospace tick labels at 11px: Berkeley Mono advances 0.6em. */
export const tickCharWidth = 6.6
const tickValues = [0.4, 1, 3, 10, 30, 70]
/** Ticks keep their label in this order when two would collide. */
const tickPriority = [1, 10, 3, 30, 0.4, 70]
const hitBucket = 8

export interface FieldTick {
  readonly x: number
  readonly label: string
}

/** Every note placed once: linked notes in a swarm on the log score axis, unlinked notes in a block. */
export interface FieldLayout {
  readonly width: number
  readonly height: number
  readonly radius: number
  readonly cell: number
  /** Slots in one column of the swarm, and in one column of the block. */
  readonly rows: number
  /** Dot centres by node id. */
  readonly x: Float32Array
  readonly y: Float32Array
  readonly axisY: number
  readonly axisX0: number
  readonly axisX1: number
  /** Unlinked block extent (dot centres); `blockCount` 0 means there is no block. */
  readonly blockX0: number
  readonly blockX1: number
  readonly blockTop: number
  readonly blockCount: number
  /** "no inbound link", wrapped to the block width. */
  readonly blockLabel: readonly string[]
  readonly ticks: readonly FieldTick[]
  /** Largest distance, in columns, that the packer moved a dot from its score column. */
  readonly spill: number
  /** Dots that found no free slot and were placed on the axis line; 0 for any real corpus. */
  readonly overflow: number
  /** Nearest dot within `reach` px of a point, or -1. */
  nearest(px: number, py: number, reach: number): number
  /** x of a score on the axis, in ×mean. */
  scale(times: number): number
}

const wrap = (text: string, chars: number) => {
  const lines: string[] = []
  for (const word of text.split(' ')) {
    const last = lines.length - 1
    if (last >= 0 && lines[last].length + 1 + word.length <= chars) lines[last] += ` ${word}`
    else lines.push(word)
  }
  return lines
}

export const tickLabel = (value: number) => `${value}×`

export function fieldLayout(model: PageRankModel, width: number): FieldLayout {
  const narrow = width < 600
  const height = narrow ? 380 : 430
  const radius = width < 360 ? 1.6 : narrow ? 1.9 : 2.5
  const cell = radius * 2.3
  const n = model.nodes.length
  const x = new Float32Array(n)
  const y = new Float32Array(n)

  // Bands: annotations above, the swarm in the middle, the axis and its labels below.
  const axisY = height - 36
  const top = narrow ? 56 : 60
  const bottom = axisY - 12
  const mid = (top + bottom) / 2
  const half = Math.max(1, Math.floor((bottom - top) / 2 / cell))
  const rows = 2 * half + 1
  const left = 8

  const unlinked = model.byPosition.filter(node => node.inbound === 0)
  const linked = model.byPosition.filter(node => node.inbound > 0)

  const blockCols = Math.ceil(unlinked.length / rows)
  const blockX0 = left + radius
  const blockX1 = blockX0 + Math.max(0, blockCols - 1) * cell
  for (const [q, node] of unlinked.entries()) {
    x[node.id] = blockX0 + Math.floor(q / rows) * cell
    y[node.id] = mid + ((q % rows) - half) * cell
  }
  const blockWidth = blockX1 - blockX0 + 2 * radius
  const blockLabel = unlinked.length
    ? wrap('no inbound link', Math.max(10, Math.floor(blockWidth / tickCharWidth)))
    : []
  const labelWidth = Math.max(0, ...blockLabel.map(line => line.length * tickCharWidth))
  const axisX0 = unlinked.length
    ? left + Math.max(blockWidth, labelWidth) + (narrow ? 20 : 28)
    : left + radius

  // The axis ends on a column, so the highest score sits exactly on its tick.
  const cols = Math.max(1, Math.floor((width - 16 - axisX0) / cell) + 1)
  const axisX1 = axisX0 + (cols - 1) * cell
  const low = Math.log10(linked.at(-1)?.times ?? 1)
  const high = Math.log10(linked[0]?.times ?? 1)
  const span = high - low
  const scale = (times: number) =>
    span > 0
      ? axisX0 + ((Math.log10(times) - low) / span) * (axisX1 - axisX0)
      : (axisX0 + axisX1) / 2

  // Grid packer: columns one cell apart along x, each filled from the middle row outwards in
  // score order, so the highest score of a column sits on the line. A full column spills its
  // dot to the nearest column with room, alternating right and left.
  const filled = new Uint16Array(cols)
  let spill = 0
  let overflow = 0
  for (const node of linked) {
    const ideal = scale(node.times)
    const c0 = Math.min(cols - 1, Math.max(0, Math.round((ideal - axisX0) / cell)))
    let placed = false
    for (let k = 0; k < cols * 2 && !placed; k++) {
      const c = c0 + (k % 2 ? -(k + 1) / 2 : k / 2)
      if (c < 0 || c >= cols || filled[c] >= rows) continue
      const slot = filled[c]++
      x[node.id] = axisX0 + c * cell
      y[node.id] = mid + (slot % 2 ? (slot + 1) / 2 : -slot / 2) * cell
      spill = Math.max(spill, Math.abs(c - c0))
      placed = true
    }
    if (!placed) {
      x[node.id] = ideal
      y[node.id] = mid
      overflow++
    }
  }

  const ticks: FieldTick[] = []
  // The block's label shares the tick row.
  const spans: [number, number][] = unlinked.length
    ? [[left, left + Math.max(blockWidth, labelWidth) + 12]]
    : []
  for (const value of tickPriority) {
    if (!tickValues.includes(value) || value < 10 ** low * 0.999 || value > 10 ** high * 1.001)
      continue
    const tx = scale(value)
    const w = tickLabel(value).length * tickCharWidth
    const [a, b] = [tx - w / 2 - 4, tx + w / 2 + 4]
    if (b > width || spans.some(([s0, s1]) => a < s1 && b > s0)) continue
    spans.push([a, b])
    ticks.push({ x: tx, label: tickLabel(value) })
  }
  ticks.sort((a, b) => a.x - b.x)

  // Bucketed linked lists for hit testing: each bucket is one reach wide.
  const bx = Math.ceil(width / hitBucket) + 1
  const by = Math.ceil(height / hitBucket) + 1
  const head = new Int32Array(bx * by).fill(-1)
  const next = new Int32Array(n).fill(-1)
  const bucket = (px: number, py: number) =>
    Math.min(by - 1, Math.max(0, Math.floor(py / hitBucket))) * bx +
    Math.min(bx - 1, Math.max(0, Math.floor(px / hitBucket)))
  for (let id = 0; id < n; id++) {
    const b = bucket(x[id], y[id])
    next[id] = head[b]
    head[b] = id
  }
  const nearest = (px: number, py: number, reach: number) => {
    let best = -1
    let bestDistance = reach * reach
    const steps = Math.ceil(reach / hitBucket)
    const cx = Math.floor(px / hitBucket)
    const cy = Math.floor(py / hitBucket)
    for (let j = cy - steps; j <= cy + steps; j++) {
      if (j < 0 || j >= by) continue
      for (let i = cx - steps; i <= cx + steps; i++) {
        if (i < 0 || i >= bx) continue
        for (let id = head[j * bx + i]; id >= 0; id = next[id]) {
          const d = (x[id] - px) ** 2 + (y[id] - py) ** 2
          if (d <= bestDistance) [best, bestDistance] = [id, d]
        }
      }
    }
    return best
  }

  return {
    width,
    height,
    radius,
    cell,
    rows,
    x,
    y,
    axisY,
    axisX0,
    axisX1,
    blockX0,
    blockX1,
    blockTop: mid - half * cell,
    blockCount: unlinked.length,
    blockLabel,
    ticks,
    spill,
    overflow,
    nearest,
    scale,
  }
}
