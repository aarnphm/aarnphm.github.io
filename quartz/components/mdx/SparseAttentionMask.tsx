import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/sparse-attention-mask.inline'
import style from '../styles/sparseAttentionMask.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type ModeId = 'cmp' | 'slc' | 'win' | 'nsa' | 'dsa'
type Props = { caption?: string; mode?: ModeId; query?: number }

type NsaParams = { l: number; d: number; ls: number; n: number; w: number }

// Paper values: NSA section 4 (l = 32, d = 16, l' = 64, n = 16 with 1 initial and 2 local
// blocks, w = 512); DeepSeek-V3.2-Exp inference config (64 indexer heads of dim 128, top-k 2048).
const PAPER_NSA: NsaParams = { l: 32, d: 16, ls: 64, n: 16, w: 512 }
const PAPER_DSA = { k: 2048, heads: 64, dim: 128 }
const TOY_NSA: NsaParams = { l: 4, d: 2, ls: 8, n: 5, w: 24 }
const TOY_DSA = { k: 16, heads: 4, dim: 8 }
const N = 64
const PAPER_T = [32768, 131072] as const

const MODES: readonly { id: ModeId; label: string; rule: string }[] = [
  {
    id: 'cmp',
    label: 'compressed',
    rule: 'Compression: $\\tilde k_i=\\varphi(k_{id:id+l})$ summarises $l$ keys and slides by $d$, so neighbouring blocks overlap. Query $t$ reads every complete block, $\\lfloor (t+1-l)/d\\rfloor+1$ compressed keys. Each wide cell is one compressed key, drawn over its first stride; the strip shows its full span.',
  },
  {
    id: 'slc',
    label: 'selected',
    rule: "Selection: the score of a block of $l'$ keys sums the compressed scores $p^{\\mathrm{cmp}}_t$ of the compression blocks that cover it (Eq. 9). Keep the top-$n$ blocks. The initial block and the 2 local blocks are always kept.",
  },
  {
    id: 'win',
    label: 'sliding',
    rule: 'Sliding window: the last $w$ keys, $k_{t-w+1:t}$, with their own key and value projections.',
  },
  {
    id: 'nsa',
    label: 'NSA union',
    rule: 'NSA mixes the three branches with learned gates: $o_t=\\sum_{c}g^c_t\\,\\mathrm{Attn}(q_t,\\tilde K^c_t,\\tilde V^c_t)$, $g^c_t\\in[0,1]$. The grid shows the raw keys that selection or the window reads; compressed keys add to $N_t$ in the table.',
  },
  {
    id: 'dsa',
    label: 'DSA top-k',
    rule: 'DSA: the lightning indexer scores every earlier key, $I_{t,s}=\\sum_j w^I_{t,j}\\,\\mathrm{ReLU}(q^I_{t,j}\\cdot k^I_s)$, from FP8 $q^I$ and $k^I$. Core attention reads the top-$k$ keys of each query, one token at a time. Bars under the zero line come from the head with $w^I<0$.',
  },
]

// --- Synthetic inputs ------------------------------------------------------------------------
// Keys and queries come from four topic directions that recur in short runs, so the keys that
// matter to a query are scattered across blocks. A fixed seed keeps every render identical.
const DK = 8
const TOPICS = 4
const SEED = 11
const INDEXER_W = [1.0, 0.8, 0.6, -0.3] as const

const mulberry32 = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0)
const add = (a: number[], b: number[]) => a.map((x, i) => x + b[i])

// E4M3: 3 mantissa bits, smallest normal exponent -6, largest finite value 448.
const e4m3 = (v: number): number => {
  if (v === 0) return 0
  const a = Math.min(Math.abs(v), 448)
  const step = 2 ** (Math.max(Math.floor(Math.log2(a)), -6) - 3)
  return Math.sign(v) * Math.min(448, Math.round(a / step) * step)
}

// One FP32 scale per vector, as act_quant does with block size = indexer head dim.
const quantFp8 = (v: number[]): number[] => {
  const scale = Math.max(...v.map(Math.abs)) / 448
  return scale === 0 ? v : v.map(x => e4m3(x / scale) * scale)
}

const buildInputs = () => {
  const rand = mulberry32(SEED)
  const gauss = () =>
    Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-9))) * Math.cos(2 * Math.PI * rand())
  const vec = (n: number, s: number) => Array.from({ length: n }, () => s * gauss())
  const unit = (v: number[]) => {
    const m = Math.hypot(...v)
    return v.map(x => x / m)
  }
  const dirs = Array.from({ length: TOPICS }, () => unit(vec(DK, 1)))
  const topic: number[] = []
  while (topic.length < N) {
    const t = Math.floor(rand() * TOPICS)
    const len = 1 + Math.floor(rand() * 4)
    for (let i = 0; i < len && topic.length < N; i++) topic.push(t)
  }
  const keys = topic.map(t =>
    add(
      dirs[t].map(x => 2 * x),
      vec(DK, 0.5),
    ),
  )
  const queries = topic.map(t =>
    add(
      dirs[t].map(x => 2 * x),
      vec(DK, 0.5),
    ),
  )
  const idxK = keys.map(quantFp8)
  const idxQ = queries.map(q => INDEXER_W.map(() => quantFp8(add(q, vec(DK, 0.6)))))
  return { keys, queries, idxK, idxQ }
}

// --- Model -----------------------------------------------------------------------------------
// Counts of keys read by the query at 0-based position t (t + 1 causal keys).
const nsaCounts = (t: number, p: NsaParams) => {
  const T = t + 1
  const cmp = T >= p.l ? Math.floor((T - p.l) / p.d) + 1 : 0
  const blocks = Math.floor(t / p.ls) + 1
  const current = T - (blocks - 1) * p.ls
  // Every kept block before the current one is full, whichever blocks the scores pick.
  const slc = (Math.min(p.n, blocks) - 1) * p.ls + current
  const win = Math.min(p.w, T)
  return { dense: T, cmp, slc, win, nsa: cmp + slc + win }
}

type QueryState = {
  cmp: number
  pslc: number[]
  chosen: number[]
  forced: number[]
  winStart: number
  index: number[]
  top: number[]
}

const queryState = (t: number, x: ReturnType<typeof buildInputs>): QueryState => {
  const { l, d, ls, n, w } = TOY_NSA
  const { cmp } = nsaCounts(t, TOY_NSA)
  // phi is a learned MLP in NSA; the toy uses the block mean.
  const kc = Array.from({ length: cmp }, (_, i) => {
    const blk = x.keys.slice(i * d, i * d + l)
    return blk[0].map((_, c) => blk.reduce((s, k) => s + k[c], 0) / l)
  })
  const logits = kc.map(k => dot(x.queries[t], k) / Math.sqrt(DK))
  const max = Math.max(...logits)
  const exp = logits.map(v => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  const pcmp = exp.map(v => v / sum)
  const blocks = Math.floor(t / ls) + 1
  // Stride chunk c of block j is covered by compression blocks c - m, m < l/d.
  const pslc = Array.from({ length: blocks }, (_, j) => {
    let s = 0
    for (let a = 0; a < ls / d; a++)
      for (let m = 0; m < l / d; m++) {
        const i = (ls / d) * j + a - m
        if (i >= 0 && i < cmp) s += pcmp[i]
      }
    return s
  })
  const forced = [...new Set([0, blocks - 1, blocks - 2].filter(j => j >= 0))]
  const chosen = new Set(forced)
  pslc
    .map((p, j) => [p, j] as const)
    .filter(([, j]) => !chosen.has(j))
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .forEach(([, j]) => chosen.size < n && chosen.add(j))

  const index = Array.from({ length: t + 1 }, (_, s) =>
    INDEXER_W.reduce((acc, wj, j) => acc + wj * Math.max(0, dot(x.idxQ[t][j], x.idxK[s])), 0),
  )
  const top = index
    .map((v, s) => [v, s] as const)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .slice(0, TOY_DSA.k)
    .map(([, s]) => s)
  return {
    cmp,
    pslc,
    chosen: [...chosen].sort((a, b) => a - b),
    forced,
    winStart: Math.max(0, t - w + 1),
    index,
    top: top.sort((a, b) => a - b),
  }
}

const INPUTS = buildInputs()
const STATES = Array.from({ length: N }, (_, t) => queryState(t, INPUTS))

const attends = (mode: 'slc' | 'win' | 'nsa' | 'dsa', t: number, s: number): boolean => {
  const q = STATES[t]
  const slc = q.chosen.includes(Math.floor(s / TOY_NSA.ls))
  const win = s >= q.winStart
  if (mode === 'slc') return slc
  if (mode === 'win') return win
  if (mode === 'nsa') return slc || win
  return q.top.includes(s)
}

const fmt = (n: number) => n.toLocaleString('en-US')
const pct = (a: number, b: number) => `${((100 * a) / b).toFixed(a / b < 0.1 ? 1 : 0)}%`

type Readout = Record<string, string>
const readout = (t: number, nsa: NsaParams, k: number): Readout => {
  const c = nsaCounts(t, nsa)
  const top = Math.min(k, c.dense)
  return {
    dense: fmt(c.dense),
    cmp: fmt(c.cmp),
    slc: fmt(c.slc),
    win: fmt(c.win),
    nsa: fmt(c.nsa),
    nsaPct: pct(c.nsa, c.dense),
    dsa: fmt(top),
    dsaPct: pct(top, c.dense),
    idx: fmt(c.dense),
  }
}

const ROWS: readonly { key: string; label: string; modes: ModeId[] }[] = [
  { key: 'dense', label: 'dense causal, $t+1$', modes: [] },
  { key: 'cmp', label: 'compressed', modes: ['cmp', 'nsa'] },
  { key: 'slc', label: 'selected', modes: ['slc', 'nsa'] },
  { key: 'win', label: 'sliding', modes: ['win', 'nsa'] },
  { key: 'nsa', label: 'NSA $N_t$', modes: ['nsa'] },
  { key: 'nsaPct', label: 'NSA vs dense', modes: ['cmp', 'slc', 'win', 'nsa'] },
  { key: 'dsa', label: 'DSA top-$k$', modes: ['dsa'] },
  { key: 'dsaPct', label: 'DSA vs dense', modes: ['dsa'] },
  { key: 'idx', label: 'indexer scores, FP8', modes: ['dsa'] },
]

const PARAMS: readonly [string, number, number][] = [
  ['l', PAPER_NSA.l, TOY_NSA.l],
  ['d', PAPER_NSA.d, TOY_NSA.d],
  ["l'", PAPER_NSA.ls, TOY_NSA.ls],
  ['n', PAPER_NSA.n, TOY_NSA.n],
  ['w', PAPER_NSA.w, TOY_NSA.w],
  ['k', PAPER_DSA.k, TOY_DSA.k],
  ['H^I', PAPER_DSA.heads, TOY_DSA.heads],
  ['d^I', PAPER_DSA.dim, TOY_DSA.dim],
]

// --- Geometry --------------------------------------------------------------------------------
const P = 6
const CELL = 5
const GX0 = 40
const ST = 8
const SH = 56
const SB = ST + SH
const GY0 = SB + 18
const VIEW_W = GX0 + N * P + 2
const VIEW_H = GY0 + N * P + 4
const BAR_MAX = SH - 16
const ZERO = ST + 44
const NEG_MAX = SB - ZERO
const kx = (s: number) => GX0 + s * P
const r1 = (v: number) => Math.round(v * 10) / 10

const stripState = (t: number) => {
  const q = STATES[t]
  const pmax = Math.max(...q.pslc, 1e-9)
  const imax = Math.max(...q.index.map(Math.abs), 1e-9)
  return {
    t,
    cmp: q.cmp,
    blocks: q.pslc.length,
    slcH: q.pslc.map(p => r1((BAR_MAX * p) / pmax)),
    slcY: q.pslc.map(p => r1(SB - (BAR_MAX * p) / pmax)),
    chosen: q.chosen,
    tags: q.pslc.map((_, j) => (j === 0 ? 'init' : q.forced.includes(j) ? 'local' : '')),
    frameY: GY0 + t * P - 1,
    frameW: (t + 1) * P + 1,
    winX: kx(q.winStart),
    winW: (t - q.winStart + 1) * P - 1,
    unionW: Array.from({ length: N / TOY_NSA.ls }, (_, j) =>
      Math.max(0, Math.min(TOY_NSA.ls, t + 1 - j * TOY_NSA.ls) * P - 1),
    ),
    // Signed bars around one zero line; negative bars clip at the strip floor.
    dsaY: q.index.map(v => r1(v >= 0 ? ZERO - ((ZERO - ST - 4) * v) / imax : ZERO)),
    dsaH: q.index.map(v =>
      r1(v >= 0 ? ((ZERO - ST - 4) * v) / imax : Math.min(NEG_MAX, ((ZERO - ST - 4) * -v) / imax)),
    ),
    top: q.top,
    out: readout(t, TOY_NSA, TOY_DSA.k),
  }
}

const CLIENT = Array.from({ length: N }, (_, t) => stripState(t))
const PAPER_OUT = PAPER_T.map(T => readout(T - 1, PAPER_NSA, PAPER_DSA.k))

let instance = 0

const SparseAttentionMaskImpl: QuartzMdxComponent<Props> = ({ caption, mode, query }) => {
  const id = `sam-${++instance}`
  const m0: ModeId = MODES.some(m => m.id === mode) ? (mode as ModeId) : 'nsa'
  const t0 = Math.min(N - 1, Math.max(0, Math.round(query ?? 55)))
  const s0 = CLIENT[t0]
  const ticks = [0, 16, 32, 48, 63]

  return (
    <figure
      class="sparse-attention-mask"
      data-sparse-attention-mask
      data-sam-mode={m0}
      data-sam-states={JSON.stringify(CLIENT)}
    >
      <div class="sam-controls">
        <fieldset class="sam-segments">
          <legend class="sam-legend">Pattern</legend>
          {MODES.map(m => (
            <label class="sam-segment">
              <input
                type="radio"
                name={`${id}-mode`}
                value={m.id}
                checked={m.id === m0}
                data-sam-mode-input
              />
              <span>{m.label}</span>
            </label>
          ))}
        </fieldset>
        <div class="sam-slider">
          <label class="sam-legend" for={`${id}-query`}>
            <MathText text="query $t$" mathClass="sam-math" />
          </label>
          <input
            id={`${id}-query`}
            type="range"
            min="0"
            max={String(N - 1)}
            step="1"
            value={String(t0)}
            data-sam-query-input
            aria-valuetext={`query ${t0}`}
          />
          <output class="sam-slider-value" for={`${id}-query`} data-sam-query-value>
            {String(t0)}
          </output>
        </div>
      </div>

      {MODES.map(m => (
        <p class="sam-rule" data-sam-rule={m.id}>
          <MathText text={m.rule} mathClass="sam-math" />
        </p>
      ))}

      <div class="sam-stage">
        <svg
          class="sam-svg"
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`Causal attention mask over ${N} tokens. Rows are queries, columns are keys. The strip above shows how the selected query picks its keys.`}
        >
          {/* Strip: how query t picks keys, per mode. Columns align with the grid's keys. */}
          <g class="sam-strip" data-sam-strip="cmp">
            <text class="sam-lane" x={2} y={ST + 23}>
              spans
            </text>
            {Array.from({ length: Math.floor((N - TOY_NSA.l) / TOY_NSA.d) + 1 }, (_, i) => (
              <rect
                class="sam-span"
                data-sam-cmp-span={String(i)}
                data-on={i < s0.cmp ? 'true' : 'false'}
                x={kx(i * TOY_NSA.d)}
                y={i % 2 === 0 ? ST + 12 : ST + 32}
                width={TOY_NSA.l * P - 1}
                height={16}
              />
            ))}
          </g>

          <g class="sam-strip" data-sam-strip="slc">
            <text class="sam-lane" x={2} y={SB - 4}>
              score
            </text>
            {Array.from({ length: N / TOY_NSA.ls }, (_, j) => (
              <g
                data-sam-block={String(j)}
                data-on={s0.chosen.includes(j) ? 'true' : 'false'}
                data-shown={j < s0.blocks ? 'true' : 'false'}
              >
                <rect
                  class="sam-block"
                  x={kx(j * TOY_NSA.ls)}
                  y={ST + 2}
                  width={TOY_NSA.ls * P - 2}
                  height={SB - ST - 2}
                />
                <rect
                  class="sam-score"
                  data-sam-score={String(j)}
                  x={kx(j * TOY_NSA.ls) + 4}
                  y={s0.slcY[j] ?? SB}
                  width={TOY_NSA.ls * P - 10}
                  height={s0.slcH[j] ?? 0}
                />
                <text
                  class="sam-tag"
                  data-sam-tag={String(j)}
                  x={kx(j * TOY_NSA.ls) + (TOY_NSA.ls * P - 2) / 2}
                  y={ST + 12}
                  text-anchor="middle"
                >
                  {s0.tags[j] ?? ''}
                </text>
              </g>
            ))}
          </g>

          <g class="sam-strip" data-sam-strip="win">
            <text class="sam-lane" x={2} y={ST + 35}>
              window
            </text>
            <rect
              class="sam-span"
              data-on="true"
              data-sam-win
              x={s0.winX}
              y={ST + 24}
              width={s0.winW}
              height={16}
            />
          </g>

          <g class="sam-strip" data-sam-strip="nsa">
            <text class="sam-lane" x={2} y={ST + 13}>
              cmp
            </text>
            <text class="sam-lane" x={2} y={ST + 31}>
              slc
            </text>
            <text class="sam-lane" x={2} y={ST + 49}>
              win
            </text>
            {Array.from({ length: Math.floor((N - TOY_NSA.l) / TOY_NSA.d) + 1 }, (_, i) => (
              <rect
                class="sam-span"
                data-sam-cmp-span={String(i)}
                data-on={i < s0.cmp ? 'true' : 'false'}
                x={kx(i * TOY_NSA.d)}
                y={i % 2 === 0 ? ST + 2 : ST + 10}
                width={TOY_NSA.l * P - 1}
                height={6}
              />
            ))}
            {s0.unionW.map((w, j) => (
              <rect
                class="sam-span"
                data-sam-union-block={String(j)}
                data-on={s0.chosen.includes(j) ? 'true' : 'false'}
                x={kx(j * TOY_NSA.ls)}
                y={ST + 21}
                width={w}
                height={14}
              />
            ))}
            <rect
              class="sam-span"
              data-on="true"
              data-sam-win
              x={s0.winX}
              y={ST + 39}
              width={s0.winW}
              height={14}
            />
          </g>

          <g class="sam-strip" data-sam-strip="dsa">
            <text class="sam-lane" x={2} y={ST + 23}>
              index
            </text>
            <line class="sam-zero" x1={GX0} x2={kx(N) - 1} y1={ZERO} y2={ZERO} />
            {Array.from({ length: N }, (_, s) => (
              <rect
                class="sam-bar"
                data-sam-bar={String(s)}
                data-on={s0.top.includes(s) ? 'true' : 'false'}
                data-shown={s <= t0 ? 'true' : 'false'}
                x={kx(s)}
                y={s0.dsaY[s] ?? ZERO}
                width={CELL}
                height={s0.dsaH[s] ?? 0}
              />
            ))}
          </g>

          {ticks.map(v => (
            <>
              <text class="sam-axis" x={kx(v) + CELL / 2} y={GY0 - 5} text-anchor="middle">
                {String(v)}
              </text>
              <text
                class="sam-axis sam-axis--qtick"
                x={GX0 - 4}
                y={GY0 + v * P + CELL}
                text-anchor="end"
              >
                {String(v)}
              </text>
            </>
          ))}
          <text class="sam-axis sam-axis--title" x={kx(40)} y={GY0 - 5} text-anchor="middle">
            key s
          </text>
          <text
            class="sam-axis sam-axis--title sam-axis--query"
            x={GX0 - 4}
            y={GY0 + 40 * P + CELL}
            text-anchor="end"
          >
            query t
          </text>

          <g class="sam-grid">
            {Array.from({ length: N }, (_, t) => (
              <g data-sam-row={String(t)} data-lit={t === t0 ? 'true' : 'false'}>
                {Array.from({ length: t + 1 }, (_, s) => {
                  const cls = (['slc', 'win', 'nsa', 'dsa'] as const)
                    .filter(m => attends(m, t, s))
                    .join(' ')
                  return (
                    <rect
                      class={`sam-cell ${cls}`.trim()}
                      x={kx(s)}
                      y={GY0 + t * P}
                      width={CELL}
                      height={CELL}
                    />
                  )
                })}
                {Array.from({ length: STATES[t].cmp }, (_, i) => (
                  <rect
                    class="sam-cmp"
                    x={kx(i * TOY_NSA.d)}
                    y={GY0 + t * P}
                    width={TOY_NSA.d * P - 1}
                    height={CELL}
                  />
                ))}
              </g>
            ))}
          </g>
          <rect
            class="sam-row-frame"
            data-sam-row-frame
            x={GX0 - 1}
            y={s0.frameY}
            width={s0.frameW}
            height={P + 1}
          />
        </svg>

        <div class="sam-side">
          <div class="sam-table-wrap">
            <table class="sam-table" aria-label="Keys read by one query">
              <thead>
                <tr>
                  <th scope="col">keys read</th>
                  <th scope="col">
                    toy <span data-sam-head>{`t=${t0}`}</span>
                  </th>
                  <th scope="col">
                    <MathText text="$T=32\mathrm{k}$" mathClass="sam-math" />
                  </th>
                  <th scope="col">
                    <MathText text="$T=128\mathrm{k}$" mathClass="sam-math" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {ROWS.map(r => (
                  <tr data-sam-out-row={r.key} data-modes={r.modes.join(' ')}>
                    <th scope="row">
                      <MathText text={r.label} mathClass="sam-math" />
                    </th>
                    <td data-sam-out={r.key}>{s0.out[r.key]}</td>
                    {PAPER_OUT.map(o => (
                      <td>{o[r.key]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div class="sam-table-wrap">
            <table class="sam-table" aria-label="Paper and toy parameters">
              <thead>
                <tr>
                  <th scope="col">parameter</th>
                  <th scope="col">paper</th>
                  <th scope="col">toy</th>
                </tr>
              </thead>
              <tbody>
                {PARAMS.map(([sym, paper, toy]) => (
                  <tr>
                    <th scope="row">
                      <MathText text={`$${sym}$`} mathClass="sam-math" />
                    </th>
                    <td>{fmt(paper)}</td>
                    <td>{fmt(toy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {caption ? (
        <figcaption class="sam-caption">
          <MathText text={caption} mathClass="sam-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const SparseAttentionMaskComponent = SparseAttentionMaskImpl as QuartzMdxComponent<Props>
SparseAttentionMaskComponent.css = style
SparseAttentionMaskComponent.afterDOMLoaded = script

export const SparseAttentionMask = registerMdxComponent(
  'SparseAttentionMask',
  SparseAttentionMaskComponent,
)

export default (() => SparseAttentionMask) satisfies (opts: undefined) => QuartzMdxComponent<Props>
