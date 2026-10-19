import { type FunctionalComponent } from 'preact'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/kv-compress-blocks.inline'
import style from '../styles/kvCompressBlocks.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type View = 'evict' | 'vllm' | 'kvcompress'
type Props = { caption?: string; view?: View; step?: number }

// Eviction example read from the KV-Compress paper figure: two KV heads, block size two, one
// sequence that must evict two blocks. Physical ids are illustrative; the figure leaves them out.
const BLOCK = 2
const EVICT = 2
type PoolBlock = { phys: number; h: number; lb: number; m: (number | null)[] }
const POOL: PoolBlock[] = [
  { phys: 1, h: 0, lb: 2, m: [0.1, null] },
  { phys: 3, h: 1, lb: 0, m: [1.2, 2.3] },
  { phys: 5, h: 0, lb: 0, m: [0.4, 2.5] },
  { phys: 7, h: 0, lb: 1, m: [0.6, 1.7] },
]

type KV = { id: number; h: number; li: number; m: number | null }
type Mark = 'kv' | 'empty' | 'evict' | 'free' | 'moved' | 'vacated'
type Cell = { h: number; li: number; m: number | null; mark: Mark; key?: boolean }
type Frame =
  | { kind: 'pool'; blocks: { phys: number; h: number; cells: Cell[] }[] }
  | { kind: 'grid'; cols: Cell[][] }
  | { kind: 'row'; cells: Cell[]; range?: [number, number]; move?: [number, number] }

type Eviction = {
  frames: Frame[]
  perHead: number[]
  evictedKVs: number[]
  moves: number
  freed: number[]
  kept: number[]
}

// The empty slot of a partly filled block counts as metric zero, so it sorts first in its head.
const metric = (kv: KV) => kv.m ?? 0
const heads = (kvs: KV[]) => [...new Set(kvs.map(kv => kv.h))].sort((a, b) => a - b)

const evictBlocks = (pool: PoolBlock[], B: number, E: number): Eviction => {
  let next = 0
  const kvs: KV[] = pool.flatMap(b =>
    b.m.map((m, s) => ({ id: next++, h: b.h, li: b.lb * B + s, m })),
  )
  const H = heads(kvs)

  const byMetric = [...kvs].sort((a, b) => a.h - b.h || metric(a) - metric(b) || a.li - b.li)
  const blocks: KV[][] = []
  for (const h of H) {
    const run = byMetric.filter(kv => kv.h === h)
    for (let i = 0; i < run.length; i += B) blocks.push(run.slice(i, i + B))
  }
  // A block's metric is its largest entry: the last one after the per-head sort.
  const blockKey = (blk: KV[]) => metric(blk[blk.length - 1])
  const ranked = [...blocks].sort((a, b) => blockKey(a) - blockKey(b) || a[0].h - b[0].h)
  const chosen = ranked.slice(0, E)
  const evicted = new Set(chosen.flat().map(kv => kv.id))

  const markOf = (kv: KV): Mark => (kv.m === null ? 'empty' : evicted.has(kv.id) ? 'evict' : 'kv')
  const cell = (kv: KV, mark: Mark = markOf(kv)): Cell => ({ h: kv.h, li: kv.li, m: kv.m, mark })

  const byIndex = [...kvs].sort((a, b) => a.h - b.h || a.li - b.li)
  const perHead = H.map(h => chosen.filter(blk => blk[0].h === h).length)
  const rangeStart = H.map((h, i) => kvs.filter(kv => kv.h === h).length - perHead[i] * B)
  const inRange = (kv: KV) => kv.li >= rangeStart[H.indexOf(kv.h)]

  // Kept entries inside the range fill evicted slots before it, in logical order.
  const contents = new Map<string, KV | null>(byIndex.map(kv => [`${kv.h}:${kv.li}`, kv]))
  const moved: { from: KV; to: KV }[] = []
  for (const h of H) {
    const run = byIndex.filter(kv => kv.h === h)
    const keep = run.filter(kv => inRange(kv) && kv.m !== null && !evicted.has(kv.id))
    const holes = run.filter(kv => !inRange(kv) && (kv.m === null || evicted.has(kv.id)))
    keep.forEach((from, i) => {
      const to = holes[i]
      contents.set(`${h}:${to.li}`, from)
      contents.set(`${h}:${from.li}`, null)
      moved.push({ from, to })
    })
  }
  const after = (slot: KV, moving: boolean): Cell => {
    const kv = contents.get(`${slot.h}:${slot.li}`)
    if (kv === null || kv === undefined)
      return { h: slot.h, li: slot.li, m: null, mark: moving ? 'vacated' : 'empty' }
    if (kv.id !== slot.id) return { h: slot.h, li: slot.li, m: kv.m, mark: moving ? 'moved' : 'kv' }
    return cell(slot)
  }
  const rangeOf = (h: number): [number, number] => {
    const run = byIndex.filter(kv => kv.h === h)
    const first = byIndex.indexOf(run[0])
    return [first + rangeStart[H.indexOf(h)], first + run.length]
  }
  const evictedHead = H[perHead.findIndex(n => n > 0)]
  const range = evictedHead === undefined ? undefined : rangeOf(evictedHead)
  const move = moved[0]
    ? ([byIndex.indexOf(moved[0].from), byIndex.indexOf(moved[0].to)] as [number, number])
    : undefined

  const poolFrame = (end: boolean): Frame => ({
    kind: 'pool',
    blocks: pool.map(b => ({
      phys: b.phys,
      h: b.h,
      cells: b.m.map((_, s) => {
        const slot = kvs.find(kv => kv.h === b.h && kv.li === b.lb * B + s)!
        if (!end) return cell(slot, slot.m === null ? 'empty' : 'kv')
        if (inRange(slot)) return { h: slot.h, li: slot.li, m: null, mark: 'free' }
        return after(slot, false)
      }),
    })),
  })

  const plain = (kv: KV) => cell(kv, kv.m === null ? 'empty' : 'kv')
  const keyed = (blk: KV[], mark: (kv: KV) => Cell) =>
    blk.map((kv, i) => ({ ...mark(kv), key: i === blk.length - 1 }))

  const frames: Frame[] = [
    poolFrame(false),
    { kind: 'row', cells: byMetric.map(plain) },
    { kind: 'grid', cols: blocks.map(blk => blk.map(plain)) },
    { kind: 'grid', cols: ranked.map(blk => keyed(blk, plain)) },
    { kind: 'grid', cols: ranked.map(blk => blk.map(kv => cell(kv))) },
    { kind: 'row', cells: ranked.flat().map(kv => cell(kv)) },
    { kind: 'row', cells: byIndex.map(kv => cell(kv)), range },
    { kind: 'row', cells: byIndex.map(kv => after(kv, true)), range, move },
    {
      kind: 'row',
      cells: byIndex.map(kv =>
        inRange(kv) ? { h: kv.h, li: kv.li, m: null, mark: 'free' } : after(kv, false),
      ),
      range,
    },
    poolFrame(true),
  ]

  return {
    frames,
    perHead,
    evictedKVs: kvs.filter(kv => kv.m !== null && evicted.has(kv.id)).map(kv => kv.m!),
    moves: moved.length,
    freed: pool.filter(b => b.lb * B >= rangeStart[H.indexOf(b.h)]).map(b => b.phys),
    kept: H.map(
      h =>
        byIndex.filter(kv => kv.h === h && !inRange(kv) && contents.get(`${h}:${kv.li}`)?.m != null)
          .length,
    ),
  }
}

const STEPS = [
  'Inputs: the KV metrics and logical indices of one sequence, scattered over the paged cache. The sequence must evict $E = 2$ blocks.',
  'Sort by (sequence, head, metric). The empty slot counts as metric $0$.',
  "Reshape each head's run to [blocks $\\times$ block size]. Each column is one candidate block.",
  'Sort blocks by (sequence, block metric). The block metric is its largest entry, the outlined bottom cell.',
  'Select the first $E = 2$ blocks for eviction. Both belong to head 0.',
  'Flatten the blocks back to one row.',
  "Sort by (sequence, head, logical index). A head's eviction range is its last $e_h$ blocks: $e_0 = 2$, $e_1 = 0$.",
  'Move kept KVs out of the eviction range into evicted slots before it.',
  'Mark the blocks in the eviction range to be freed.',
  'Free those blocks of the KV metrics and the KV cache.',
]

const EVICTION = evictBlocks(POOL, BLOCK, EVICT)

// Block layout comparison: layer 0 holds the eviction example's head lengths, layer 1 is
// illustrative. Free lists stand in for whatever order the allocator hands out ids.
const LAYERS = 2
const HEADS = 2
const LENGTHS = [
  [5, 2],
  [3, 4],
]
const VLLM_POOL = 6
const VLLM_FREE = [1, 4, 2, 0, 3, 5]
const KVC_POOL = 12
const KVC_FREE = [3, 9, 0, 5, 7, 1, 10, 4, 2, 6, 8, 11]

type TableRow = { label: string; entries: number[]; ctx: number }
type Layout = {
  rows: TableRow[]
  // owner[phys] is the (layer, head) the block serves, or null for other sequences.
  owner: (null | { l: number; h: number; lb: number })[]
  slots: number
  stored: number
  blocks: number
}

const ceilDiv = (a: number, b: number) => Math.ceil(a / b)

const vllmLayout = (): Layout => {
  // One block id covers every head of every layer, so all heads share the longest length.
  const ctx = Math.max(...LENGTHS.flat())
  const entries = VLLM_FREE.slice(0, ceilDiv(ctx, BLOCK))
  const owner: Layout['owner'] = Array(VLLM_POOL).fill(null)
  entries.forEach((p, lb) => (owner[p] = { l: -1, h: -1, lb }))
  return {
    rows: [{ label: 'seq', entries, ctx }],
    owner,
    slots: entries.length * BLOCK * LAYERS * HEADS,
    stored: LENGTHS.flat().reduce((a, b) => a + b, 0),
    blocks: entries.length,
  }
}

const kvcLayout = (): Layout => {
  const owner: Layout['owner'] = Array(KVC_POOL).fill(null)
  const rows: TableRow[] = []
  let cursor = 0
  for (let l = 0; l < LAYERS; l++)
    for (let h = 0; h < HEADS; h++) {
      const entries = KVC_FREE.slice(cursor, cursor + ceilDiv(LENGTHS[l][h], BLOCK))
      cursor += entries.length
      entries.forEach((p, lb) => (owner[p] = { l, h, lb }))
      rows.push({ label: `L${l} h${h}`, entries, ctx: LENGTHS[l][h] })
    }
  return {
    rows,
    owner,
    slots: cursor * BLOCK,
    stored: LENGTHS.flat().reduce((a, b) => a + b, 0),
    blocks: cursor,
  }
}

const VLLM = vllmLayout()
const KVC = kvcLayout()
const MAX_ENTRIES = Math.max(...[...VLLM.rows, ...KVC.rows].map(r => r.entries.length))
// The lit block: logical block 0 of layer 0, head 0.
const LIT_VLLM = VLLM.rows[0].entries[0]
const LIT_KVC = KVC.rows[0].entries[0]

const TABS: { id: View; label: string }[] = [
  { id: 'evict', label: 'eviction' },
  { id: 'vllm', label: 'vLLM blocks' },
  { id: 'kvcompress', label: 'KV-Compress blocks' },
]

// Eviction drawing geometry.
const CW = 32
const CH = 34
const GAP = 6
const BAND = 14
const OTHER_W = 14
const TOP = 6
const VIEW_W = 362
const VIEW_H = 96
const fmt = (m: number) => m.toFixed(1)

const SlotCell: FunctionalComponent<{ c: Cell; x: number; y: number }> = ({ c, x, y }) => (
  <g class="kvc-slot" data-kvc-mark={c.mark} data-kvc-key={c.key ? 'true' : undefined}>
    <rect class="kvc-cell" x={x} y={y} width={CW} height={CH} />
    {c.m !== null ? (
      <text class="kvc-metric" x={x + CW / 2} y={y + 16} text-anchor="middle">
        {fmt(c.m)}
      </text>
    ) : null}
    <text class="kvc-index" x={x + CW / 2} y={y + CH - 5} text-anchor="middle">
      {String(c.li)}
    </text>
  </g>
)

const Band: FunctionalComponent<{ x: number; w: number; h: number }> = ({ x, w, h }) => (
  <g>
    <rect class="kvc-band" x={x} y={TOP} width={w} height={BAND} />
    <text class="kvc-band-label" x={x + w / 2} y={TOP + 10} text-anchor="middle">
      {w >= 70 ? `head ${h}` : `h${h}`}
    </text>
  </g>
)

const runs = (cells: Cell[]) => {
  const out: { h: number; start: number; end: number }[] = []
  cells.forEach((c, i) => {
    const last = out[out.length - 1]
    if (last && last.h === c.h) last.end = i + 1
    else out.push({ h: c.h, start: i, end: i + 1 })
  })
  return out
}

const FrameView: FunctionalComponent<{ f: Frame }> = ({ f }) => {
  const y = TOP + BAND
  if (f.kind === 'pool') {
    const width = f.blocks.length * (OTHER_W + BLOCK * CW) + (2 * f.blocks.length - 1) * GAP
    let x = (VIEW_W - width) / 2
    const parts = f.blocks.map(b => {
      const ox = x
      const bx = ox + OTHER_W + GAP
      x = bx + BLOCK * CW + GAP
      return (
        <g>
          <rect class="kvc-other" x={ox} y={y} width={OTHER_W} height={CH} />
          <Band x={bx} w={BLOCK * CW} h={b.h} />
          {b.cells.map((c, s) => (
            <SlotCell c={c} x={bx + s * CW} y={y} />
          ))}
          <text class="kvc-phys" x={bx + CW} y={y + CH + 12} text-anchor="middle">
            {`p${b.phys}`}
          </text>
        </g>
      )
    })
    return <g>{parts}</g>
  }
  if (f.kind === 'grid') {
    const width = f.cols.length * CW + (f.cols.length - 1) * GAP
    const x0 = (VIEW_W - width) / 2
    return (
      <g>
        {f.cols.map((col, i) => {
          const x = x0 + i * (CW + GAP)
          return (
            <g>
              <Band x={x} w={CW} h={col[0].h} />
              {col.map((c, r) => (
                <SlotCell c={c} x={x} y={y + r * CH} />
              ))}
            </g>
          )
        })}
      </g>
    )
  }
  const groups = runs(f.cells)
  const width = f.cells.length * CW + (groups.length - 1) * GAP
  const x0 = (VIEW_W - width) / 2
  const xs = f.cells.map((_, i) => x0 + i * CW + groups.findIndex(g => i < g.end) * GAP)
  const bottom = y + CH
  const barY = bottom + 2
  return (
    <g>
      {groups.map(g => (
        <Band x={xs[g.start]} w={(g.end - g.start) * CW} h={g.h} />
      ))}
      {f.cells.map((c, i) => (
        <SlotCell c={c} x={xs[i]} y={y} />
      ))}
      {f.range ? (
        <g>
          <rect
            class="kvc-range"
            x={xs[f.range[0]]}
            y={barY}
            width={xs[f.range[1] - 1] + CW - xs[f.range[0]]}
            height={3}
          />
          <text class="kvc-range-label" x={xs[f.range[1] - 1] + CW} y={barY + 14} text-anchor="end">
            eviction range
          </text>
        </g>
      ) : null}
      {f.move
        ? (() => {
            const sx = xs[f.move[0]] + CW / 2
            const dx = xs[f.move[1]] + CW / 2
            return (
              <g>
                <path
                  class="kvc-move"
                  d={`M ${sx} ${bottom + 5} C ${sx} ${bottom + 20}, ${dx} ${bottom + 20}, ${dx} ${bottom + 5}`}
                />
                <path
                  class="kvc-move-head"
                  d={`M ${dx - 3.5} ${bottom + 6} L ${dx} ${bottom} L ${dx + 3.5} ${bottom + 6} Z`}
                />
              </g>
            )
          })()
        : null}
    </g>
  )
}

const LayoutTable: FunctionalComponent<{ layout: Layout; lit: number; head: string }> = ({
  layout,
  lit,
  head,
}) => (
  <table class="kvc-table">
    <caption class="kvc-table-caption">
      Block tables ({layout.rows.length} {layout.rows.length === 1 ? 'row' : 'rows'}) and context
      lengths
    </caption>
    <thead>
      <tr>
        <th scope="col">{head}</th>
        {Array.from({ length: MAX_ENTRIES }, (_, i) => (
          <th scope="col">{`b${i}`}</th>
        ))}
        <th scope="col">ctx</th>
      </tr>
    </thead>
    <tbody>
      {layout.rows.map((r, ri) => (
        <tr>
          <th scope="row">{r.label}</th>
          {Array.from({ length: MAX_ENTRIES }, (_, i) => (
            <td data-kvc-lit={ri === 0 && i === 0 && r.entries[i] === lit ? 'true' : undefined}>
              {r.entries[i] === undefined ? '–' : `p${r.entries[i]}`}
            </td>
          ))}
          <td>{String(r.ctx)}</td>
        </tr>
      ))}
    </tbody>
  </table>
)

// Layout drawing geometry.
const SW = 16
const SH = 16
const BGAP = 4
const LABEL_W = 46
const LAYOUT_SCALE = 1.4
const scaled = (w: number) => ({ maxWidth: `${(w * LAYOUT_SCALE).toFixed(0)}px` })

const VllmCache: FunctionalComponent = () => {
  const width = LABEL_W + VLLM_POOL * (BLOCK * SW) + (VLLM_POOL - 1) * BGAP
  const layerH = HEADS * SH
  const y0 = 18
  const height = y0 + LAYERS * layerH + (LAYERS - 1) * 10 + 4
  return (
    <svg
      class="kvc-layout-svg"
      viewBox={`0 0 ${width} ${height}`}
      style={scaled(width)}
      role="img"
      aria-label={`vLLM KV cache: ${LAYERS} layers by ${HEADS} heads by ${VLLM_POOL} blocks of ${BLOCK} slots. One block id spans every head of every layer.`}
    >
      {VLLM.owner.map((_, p) => (
        <text
          class="kvc-phys"
          x={LABEL_W + p * (BLOCK * SW + BGAP) + SW}
          y={12}
          text-anchor="middle"
        >{`p${p}`}</text>
      ))}
      {Array.from({ length: LAYERS }, (_, l) => {
        const ly = y0 + l * (layerH + 10)
        return (
          <g>
            {Array.from({ length: HEADS }, (_, h) => (
              <text class="kvc-axis" x={LABEL_W - 6} y={ly + h * SH + 11.5} text-anchor="end">
                {`L${l} h${h}`}
              </text>
            ))}
            {VLLM.owner.map((o, p) =>
              Array.from({ length: HEADS }, (_, h) =>
                Array.from({ length: BLOCK }, (_, s) => {
                  const mark = !o
                    ? 'other'
                    : p === LIT_VLLM
                      ? 'lit'
                      : o.lb * BLOCK + s < LENGTHS[l][h]
                        ? 'kv'
                        : 'pad'
                  return (
                    <rect
                      class="kvc-lslot"
                      data-kvc-mark={mark}
                      x={LABEL_W + p * (BLOCK * SW + BGAP) + s * SW}
                      y={ly + h * SH}
                      width={SW}
                      height={SH}
                    />
                  )
                }),
              ),
            )}
          </g>
        )
      })}
    </svg>
  )
}

const KvcCache: FunctionalComponent = () => {
  const width = KVC_POOL * (BLOCK * SW) + (KVC_POOL - 1) * BGAP
  const y = 28
  return (
    <svg
      class="kvc-layout-svg"
      viewBox={`0 0 ${width} ${y + SH + 4}`}
      style={scaled(width)}
      role="img"
      aria-label={`KV-Compress KV cache: ${KVC_POOL} blocks of ${BLOCK} slots, each block holding one head of one layer.`}
    >
      {KVC.owner.map((o, p) => {
        const x = p * (BLOCK * SW + BGAP)
        return (
          <g>
            <text class="kvc-phys" x={x + SW} y={10} text-anchor="middle">{`p${p}`}</text>
            {o ? (
              <text class="kvc-axis" x={x + SW} y={22} text-anchor="middle">
                {`${o.l}·${o.h}`}
              </text>
            ) : null}
            {Array.from({ length: BLOCK }, (_, s) => {
              const mark = !o
                ? 'other'
                : p === LIT_KVC
                  ? 'lit'
                  : o.lb * BLOCK + s < LENGTHS[o.l][o.h]
                    ? 'kv'
                    : 'pad'
              return (
                <rect
                  class="kvc-lslot"
                  data-kvc-mark={mark}
                  x={x + s * SW}
                  y={y}
                  width={SW}
                  height={SH}
                />
              )
            })}
          </g>
        )
      })}
    </svg>
  )
}

const LayoutCompare: FunctionalComponent = () => (
  <table class="kvc-compare">
    <caption class="kvc-table-caption">Same KVs, both layouts</caption>
    <thead>
      <tr>
        <th scope="col">per sequence</th>
        <th scope="col">vLLM</th>
        <th scope="col">KV-Compress</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">block-table rows</th>
        <td>{String(VLLM.rows.length)}</td>
        <td>{`${LAYERS} × ${HEADS} = ${KVC.rows.length}`}</td>
      </tr>
      <tr>
        <th scope="row">slots per block</th>
        <td>{`${LAYERS} × ${HEADS} × ${BLOCK} = ${LAYERS * HEADS * BLOCK}`}</td>
        <td>{String(BLOCK)}</td>
      </tr>
      <tr>
        <th scope="row">blocks allocated</th>
        <td>{String(VLLM.blocks)}</td>
        <td>{String(KVC.blocks)}</td>
      </tr>
      <tr>
        <th scope="row">KV slots allocated</th>
        <td>{String(VLLM.slots)}</td>
        <td>{String(KVC.slots)}</td>
      </tr>
      <tr>
        <th scope="row">KVs stored</th>
        <td>{String(VLLM.stored)}</td>
        <td>{String(KVC.stored)}</td>
      </tr>
      <tr>
        <th scope="row">empty slots</th>
        <td>{String(VLLM.slots - VLLM.stored)}</td>
        <td>{String(KVC.slots - KVC.stored)}</td>
      </tr>
    </tbody>
  </table>
)

const LEGEND_EVICT: { mark: Mark | 'other'; label: string }[] = [
  { mark: 'kv', label: 'KV' },
  { mark: 'empty', label: 'empty slot' },
  { mark: 'evict', label: 'marked for eviction' },
  { mark: 'free', label: 'freed' },
  { mark: 'other', label: 'other sequences' },
]

const legendLayout = (lit: string): { mark: string; label: string }[] => [
  { mark: 'kv', label: 'KV' },
  { mark: 'empty', label: 'allocated, unused' },
  { mark: 'lit', label: lit },
  { mark: 'other', label: 'other sequences' },
]

const Legend: FunctionalComponent<{ items: { mark: string; label: string }[] }> = ({ items }) => (
  <ul class="kvc-legend-list">
    {items.map(l => (
      <li>
        <span class="kvc-swatch" data-kvc-mark={l.mark} aria-hidden="true" />
        {l.label}
      </li>
    ))}
  </ul>
)

let instance = 0

const KVCompressBlocksImpl: QuartzMdxComponent<Props> = ({ caption, view = 'evict', step = 1 }) => {
  const id = `kvc-${++instance}`
  const v0: View = TABS.some(t => t.id === view) ? view : 'evict'
  const s0 = Math.min(STEPS.length, Math.max(1, Math.round(step)))
  const ev = EVICTION
  return (
    <figure
      class="kv-compress-blocks"
      data-kv-compress-blocks
      data-kvc-view={v0}
      data-kvc-step={s0}
    >
      <div class="kvc-tabs" role="tablist" aria-label="KV-Compress view">
        {TABS.map(t => (
          <button
            type="button"
            role="tab"
            class="kvc-tab"
            id={`${id}-tab-${t.id}`}
            aria-controls={`${id}-panel-${t.id}`}
            aria-selected={t.id === v0 ? 'true' : 'false'}
            tabIndex={t.id === v0 ? 0 : -1}
            data-kvc-tab={t.id}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div
        class="kvc-panel"
        role="tabpanel"
        id={`${id}-panel-evict`}
        aria-labelledby={`${id}-tab-evict`}
        hidden={v0 !== 'evict'}
        data-kvc-panel="evict"
      >
        <div class="kvc-stepper">
          <button type="button" class="kvc-nav" data-kvc-nav="-1" disabled={s0 === 1}>
            prev
          </button>
          <fieldset class="kvc-steps">
            <legend class="kvc-legend">Step</legend>
            {STEPS.map((_, i) => (
              <label class="kvc-step">
                <input
                  type="radio"
                  name={`${id}-step`}
                  value={String(i + 1)}
                  checked={i + 1 === s0}
                  data-kvc-step-input
                />
                <span>{String(i + 1)}</span>
              </label>
            ))}
          </fieldset>
          <button type="button" class="kvc-nav" data-kvc-nav="1" disabled={s0 === STEPS.length}>
            next
          </button>
        </div>

        <div class="kvc-step-texts" aria-live="polite">
          {STEPS.map((text, i) => (
            <p class="kvc-step-text" data-kvc-step-text={String(i + 1)}>
              <span class="kvc-step-num">{`${i + 1}.`}</span>{' '}
              <MathText text={text} mathClass="kvc-math" />
            </p>
          ))}
        </div>

        <svg
          class="kvc-svg"
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          role="img"
          aria-label="KV metrics of one sequence across the ten eviction steps; the selected step is drawn."
        >
          {ev.frames.map((f, i) => (
            <g class="kvc-frame" data-kvc-frame={String(i + 1)}>
              <FrameView f={f} />
            </g>
          ))}
        </svg>

        <Legend items={LEGEND_EVICT} />

        <dl class="kvc-readout">
          <div>
            <dt>Blocks to evict, block size</dt>
            <dd>{`${EVICT}, ${BLOCK}`}</dd>
          </div>
          <div>
            <dt>Blocks evicted per head</dt>
            <dd data-kvc-out="per-head">{ev.perHead.map((n, h) => `h${h}: ${n}`).join(' · ')}</dd>
          </div>
          <div>
            <dt>KVs evicted</dt>
            <dd data-kvc-out="evicted">{ev.evictedKVs.map(fmt).join(', ')}</dd>
          </div>
          <div>
            <dt>KVs moved</dt>
            <dd data-kvc-out="moves">{String(ev.moves)}</dd>
          </div>
          <div>
            <dt>Physical blocks freed</dt>
            <dd data-kvc-out="freed">{ev.freed.map(p => `p${p}`).join(', ')}</dd>
          </div>
          <div>
            <dt>KVs kept per head</dt>
            <dd data-kvc-out="kept">{ev.kept.map((n, h) => `h${h}: ${n}`).join(' · ')}</dd>
          </div>
        </dl>
      </div>

      <div
        class="kvc-panel"
        role="tabpanel"
        id={`${id}-panel-vllm`}
        aria-labelledby={`${id}-tab-vllm`}
        hidden={v0 !== 'vllm'}
        data-kvc-panel="vllm"
      >
        <p class="kvc-step-text kvc-panel-note">
          One block id names the same slots in every layer and every head. One block table per
          sequence, one context length, so every head keeps the longest head's blocks.
        </p>
        <VllmCache />
        <Legend items={legendLayout(`block b0 = p${LIT_VLLM}, every layer and head`)} />
        <LayoutTable layout={VLLM} lit={LIT_VLLM} head="sequence" />
        <LayoutCompare />
      </div>

      <div
        class="kvc-panel"
        role="tabpanel"
        id={`${id}-panel-kvcompress`}
        aria-labelledby={`${id}-tab-kvcompress`}
        hidden={v0 !== 'kvcompress'}
        data-kvc-panel="kvcompress"
      >
        <p class="kvc-step-text kvc-panel-note">
          Each block holds one head of one layer, tagged layer·head. Block tables expand to{' '}
          <MathText text="$l \times H$" mathClass="kvc-math" /> rows, each with its own context
          length.
        </p>
        <KvcCache />
        <Legend items={legendLayout(`block b0 of L0 h0 = p${LIT_KVC}`)} />
        <LayoutTable layout={KVC} lit={LIT_KVC} head="layer, head" />
        <LayoutCompare />
      </div>

      {caption ? (
        <figcaption class="kvc-caption">
          <MathText text={caption} mathClass="kvc-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const KVCompressBlocksComponent = KVCompressBlocksImpl as QuartzMdxComponent<Props>
KVCompressBlocksComponent.css = style
KVCompressBlocksComponent.afterDOMLoaded = script

export const KVCompressBlocks = registerMdxComponent('KVCompressBlocks', KVCompressBlocksComponent)

export default (() => KVCompressBlocks) satisfies (opts: undefined) => QuartzMdxComponent<Props>
