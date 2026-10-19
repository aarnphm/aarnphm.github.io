import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/kv-budget-allocation.inline'
import style from '../styles/kvBudgetAllocation.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type PolicyId = 'full' | 'streaming' | 'snapkv' | 'pyramidkv' | 'adakv'
type Props = { caption?: string; policy?: PolicyId; budget?: number }

const LAYERS = 4
const HEADS = 3
const PREFIX = 16
const WINDOW = 2
const ROWS = LAYERS * HEADS
const BUDGETS = [2, 4, 6] as const
// PyramidKV sets the top layer to k/beta and mirrors it at the bottom; beta = 4 keeps integers at k = 4.
const PYRAMID_BETA = 4

// Synthetic window votes shaped after the PyramidKV funnel: low layers spread attention, high
// layers concentrate it. One head per layer is dispersed so Ada-KV has budget to move.
const LAYER_TEMP = [3.2, 1.1, 0.4, 0.12] as const
const HEAD_SPREAD = [1.9, 1.0, 0.55] as const
const HEAVY_TOKENS = new Set([3, 7, 8, 12])
const SINK_BONUS = 2.2
const HEAVY_BONUS = 1.4
const RECENCY_BONUS = 0.8
const NOISE = 1.2
const SEED = 7

const POLICIES: readonly { id: PolicyId; label: string; rule: string }[] = [
  { id: 'full', label: 'Full', rule: 'No eviction. Every head keeps all 16 prefix entries.' },
  {
    id: 'streaming',
    label: 'StreamingLLM',
    rule: 'Keep the sink token and the newest $k-1$ prefix entries. Position alone decides.',
  },
  {
    id: 'snapkv',
    label: 'SnapKV',
    rule: 'Each head keeps its top-$k$ entries by the window vote $C$. Every head gets the same $k$.',
  },
  {
    id: 'pyramidkv',
    label: 'PyramidKV',
    rule: 'Same vote, but the budget falls linearly with depth, from $2k - k/\\beta$ at layer 0 to $k/\\beta$ at the top ($\\beta = 4$).',
  },
  {
    id: 'adakv',
    label: 'Ada-SnapKV',
    rule: "Each layer keeps $Hk$ entries: one per head, then the top votes pooled across the layer's heads.",
  },
]

type State = {
  mask: string
  rows: number[]
  rowMass: number[]
  kept: number
  layerBudget: string
  mean: number
  worst: number
}

const mulberry32 = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

const buildVotes = (): number[][] => {
  const rand = mulberry32(SEED)
  const votes: number[][] = []
  for (let l = 0; l < LAYERS; l++) {
    for (let h = 0; h < HEADS; h++) {
      const tau = LAYER_TEMP[l] * HEAD_SPREAD[(h + l) % HEADS]
      const logits = Array.from(
        { length: PREFIX },
        (_, t) =>
          (t === 0 ? SINK_BONUS : 0) +
          (HEAVY_TOKENS.has(t) ? HEAVY_BONUS : 0) +
          (RECENCY_BONUS * t) / (PREFIX - 1) +
          NOISE * rand(),
      )
      const max = Math.max(...logits)
      const exp = logits.map(x => Math.exp((x - max) / tau))
      const sum = exp.reduce((a, b) => a + b, 0)
      votes.push(exp.map(x => x / sum))
    }
  }
  return votes
}

const topK = (vote: number[], k: number): number[] =>
  vote
    .map((v, t) => [v, t] as const)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .slice(0, k)
    .map(([, t]) => t)

// Largest-remainder rounding, so integer layer budgets still sum to L * k.
const apportion = (values: number[], total: number): number[] => {
  const out = values.map(v => Math.min(PREFIX, Math.max(1, Math.floor(v))))
  let rest = total - out.reduce((a, b) => a + b, 0)
  const order = values
    .map((v, i) => [v - Math.floor(v), i] as const)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
  for (let j = 0; rest > 0; j++) {
    const i = order[j % values.length][1]
    if (out[i] < PREFIX) {
      out[i] += 1
      rest -= 1
    }
  }
  return out
}

const pyramidBudgets = (k: number): number[] => {
  const top = k / PYRAMID_BETA
  const bottom = 2 * k - top
  return apportion(
    Array.from({ length: LAYERS }, (_, l) => bottom - ((bottom - top) * l) / (LAYERS - 1)),
    LAYERS * k,
  )
}

const evict = (votes: number[][], policy: PolicyId, k: number): State => {
  const keep = votes.map(() => new Set<number>())
  const pyramid = pyramidBudgets(k)
  for (let l = 0; l < LAYERS; l++) {
    const rows = Array.from({ length: HEADS }, (_, h) => l * HEADS + h)
    if (policy === 'adakv') {
      for (const r of rows) keep[r].add(topK(votes[r], 1)[0])
      rows
        .flatMap(r => votes[r].map((v, t) => [v, r, t] as const))
        .filter(([, r, t]) => !keep[r].has(t))
        .sort((a, b) => b[0] - a[0] || a[1] - b[1] || a[2] - b[2])
        .slice(0, HEADS * k - HEADS)
        .forEach(([, r, t]) => keep[r].add(t))
      continue
    }
    for (const r of rows) {
      if (policy === 'full') for (let t = 0; t < PREFIX; t++) keep[r].add(t)
      else if (policy === 'streaming') {
        keep[r].add(0)
        for (let t = PREFIX - (k - 1); t < PREFIX; t++) keep[r].add(t)
      } else {
        const budget = policy === 'pyramidkv' ? pyramid[l] : k
        for (const t of topK(votes[r], budget)) keep[r].add(t)
      }
    }
  }

  const rowMass = keep.map((set, r) => [...set].reduce((a, t) => a + votes[r][t], 0))
  const perLayer = (b: number[]) => b.join(' · ')
  const layerBudget =
    policy === 'full'
      ? perLayer(Array(LAYERS).fill(PREFIX))
      : policy === 'pyramidkv'
        ? perLayer(pyramid)
        : perLayer(Array(LAYERS).fill(k))
  return {
    mask: keep
      .map(set => Array.from({ length: PREFIX }, (_, t) => (set.has(t) ? 1 : 0)).join(''))
      .join(''),
    rows: keep.map(set => set.size),
    rowMass,
    kept: keep.reduce((a, set) => a + set.size, 0),
    layerBudget,
    mean: rowMass.reduce((a, b) => a + b, 0) / ROWS,
    worst: Math.min(...rowMass),
  }
}

const VOTES = buildVotes()
const VOTE_MAX = Math.max(...VOTES.flat())
const STATES: Record<string, State> = {}
for (const p of POLICIES) for (const k of BUDGETS) STATES[`${p.id}:${k}`] = evict(VOTES, p.id, k)

const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`
const readout = (s: State) => ({
  kept: `${s.kept} / ${ROWS * PREFIX}`,
  layers: s.layerBudget,
  mean: pct(s.mean, 1),
  worst: pct(s.worst, 1),
})
// The inline script only applies precomputed states; the model above is the single source.
const CLIENT_STATES = Object.fromEntries(
  Object.entries(STATES).map(([key, s]) => [
    key,
    { mask: s.mask, rows: s.rows, rowMass: s.rowMass.map(m => pct(m)), ...readout(s) },
  ]),
)

const PITCH = 18
const CELL = 16
const X0 = 52
const Y0 = 22
const LAYER_GAP = 8
const WINDOW_X = X0 + PREFIX * PITCH + 8
const READ_X = WINDOW_X + WINDOW * PITCH + 14
const VIEW_W = READ_X + 78
const rowY = (r: number) => Y0 + r * PITCH + Math.floor(r / HEADS) * LAYER_GAP
const VIEW_H = rowY(ROWS - 1) + PITCH + 6
const markSide = (v: number) => 2 + 11 * Math.sqrt(v / VOTE_MAX)

let instance = 0

const KVBudgetAllocationImpl: QuartzMdxComponent<Props> = ({
  caption,
  policy = 'pyramidkv',
  budget = 4,
}) => {
  const id = `kvb-${++instance}`
  const p0 = POLICIES.some(p => p.id === policy) ? policy : 'pyramidkv'
  const k0 = (BUDGETS as readonly number[]).includes(budget) ? budget : 4
  const s0 = STATES[`${p0}:${k0}`]
  const r0 = readout(s0)

  return (
    <figure
      class="kv-budget-allocation"
      data-kv-budget-allocation
      data-kvb-policy={p0}
      data-kvb-states={JSON.stringify(CLIENT_STATES)}
    >
      <div class="kvb-controls">
        <fieldset class="kvb-segments">
          <legend class="kvb-legend">Policy</legend>
          {POLICIES.map(p => (
            <label class="kvb-segment">
              <input
                type="radio"
                name={`${id}-policy`}
                value={p.id}
                checked={p.id === p0}
                data-kvb-policy-input
              />
              <span>{p.label}</span>
            </label>
          ))}
        </fieldset>
        <fieldset class="kvb-segments">
          <legend class="kvb-legend">
            <MathText text="Budget $k$ per head" mathClass="kvb-math" />
          </legend>
          {BUDGETS.map(k => (
            <label class="kvb-segment">
              <input
                type="radio"
                name={`${id}-budget`}
                value={String(k)}
                checked={k === k0}
                data-kvb-budget-input
              />
              <span>{`${k} / ${PREFIX}`}</span>
            </label>
          ))}
        </fieldset>
      </div>

      {POLICIES.map(p => (
        <p class="kvb-rule" data-kvb-rule={p.id}>
          <MathText text={p.rule} mathClass="kvb-math" />
        </p>
      ))}

      <svg
        class="kvb-svg"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`KV cache of ${LAYERS} layers by ${HEADS} heads by ${PREFIX} prefix tokens, with retained entries filled.`}
      >
        {Array.from({ length: PREFIX }, (_, t) =>
          t % 4 === 0 || t === PREFIX - 1 ? (
            <text class="kvb-axis" x={X0 + t * PITCH + CELL / 2} y={Y0 - 7} text-anchor="middle">
              {t}
            </text>
          ) : null,
        )}
        <text class="kvb-axis" x={WINDOW_X + PITCH - 1} y={Y0 - 7} text-anchor="middle">
          window
        </text>
        <text class="kvb-axis" x={READ_X} y={Y0 - 7}>
          kept
        </text>
        <text class="kvb-axis" x={VIEW_W - 2} y={Y0 - 7} text-anchor="end">
          mass
        </text>

        {Array.from({ length: ROWS }, (_, r) => {
          const y = rowY(r)
          const l = Math.floor(r / HEADS)
          const h = r % HEADS
          return (
            <g class="kvb-row">
              {h === 0 ? (
                <text class="kvb-label kvb-label--layer" x={4} y={y + 12}>
                  {`L${l}`}
                </text>
              ) : null}
              <text class="kvb-label" x={X0 - 6} y={y + 12} text-anchor="end">
                {`h${h}`}
              </text>
              {VOTES[r].map((v, t) => {
                const x = X0 + t * PITCH
                const side = markSide(v)
                return (
                  <g>
                    <rect
                      class="kvb-cell"
                      data-kvb-cell={String(r * PREFIX + t)}
                      data-kvb-kept={s0.mask[r * PREFIX + t] === '1' ? 'true' : 'false'}
                      x={x}
                      y={y}
                      width={CELL}
                      height={CELL}
                    />
                    <rect
                      class="kvb-mark"
                      x={x + (CELL - side) / 2}
                      y={y + (CELL - side) / 2}
                      width={side}
                      height={side}
                    />
                  </g>
                )
              })}
              {Array.from({ length: WINDOW }, (_, w) => (
                <rect
                  class="kvb-cell kvb-cell--window"
                  x={WINDOW_X + w * PITCH}
                  y={y}
                  width={CELL}
                  height={CELL}
                />
              ))}
              <text
                class="kvb-value"
                x={READ_X + 16}
                y={y + 12}
                text-anchor="end"
                data-kvb-row={String(r)}
              >
                {String(s0.rows[r])}
              </text>
              <text
                class="kvb-value"
                x={VIEW_W - 2}
                y={y + 12}
                text-anchor="end"
                data-kvb-row-mass={String(r)}
              >
                {pct(s0.rowMass[r])}
              </text>
            </g>
          )
        })}
      </svg>

      <dl class="kvb-readout">
        <div>
          <dt>Prefix entries kept</dt>
          <dd data-kvb-out="kept">{r0.kept}</dd>
        </div>
        <div>
          <dt>Per-head budget by layer, L0 to L3</dt>
          <dd data-kvb-out="layers">{r0.layers}</dd>
        </div>
        <div>
          <dt>Mean vote mass kept per head</dt>
          <dd data-kvb-out="mean">{r0.mean}</dd>
        </div>
        <div>
          <dt>Worst head</dt>
          <dd data-kvb-out="worst">{r0.worst}</dd>
        </div>
      </dl>

      {caption ? (
        <figcaption class="kvb-caption">
          <MathText text={caption} mathClass="kvb-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const KVBudgetAllocationComponent = KVBudgetAllocationImpl as QuartzMdxComponent<Props>
KVBudgetAllocationComponent.css = style
KVBudgetAllocationComponent.afterDOMLoaded = script

export const KVBudgetAllocation = registerMdxComponent(
  'KVBudgetAllocation',
  KVBudgetAllocationComponent,
)

export default (() => KVBudgetAllocation) satisfies (opts: undefined) => QuartzMdxComponent<Props>
