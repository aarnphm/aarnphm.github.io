import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { renderFigureLabel } from '../../util/figure-label'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/mla-attention-modes.inline'
import style from '../styles/mlaAttentionModes.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type ModeId = 'mha' | 'mqa'
type Props = { caption?: string; mode?: ModeId }

// DeepSeek-V3 attention dims: latent d_c, decoupled RoPE d_h^R, heads n_h, head dim d_h.
const DIMS = { dc: 512, dr: 64, nh: 128, dh: 128 }
const BYTES = 2 // BF16

type Kind = 'cache' | 'built' | 'plain'
type Seg = { dims: number; kind: Kind }
type Row = { tex: string; segs: Seg[] } | { gap: string }
type Group = { title: string; rows: Row[] }

const headRowsMha = (i: string): Row[] => [
  {
    tex: `\\mathbf{q}_{t,${i}}=[\\mathbf{q}^C_{t,${i}};\\mathbf{q}^R_{t,${i}}]`,
    segs: [
      { dims: DIMS.dh, kind: 'plain' },
      { dims: DIMS.dr, kind: 'plain' },
    ],
  },
  {
    tex: `\\mathbf{k}_{t,${i}}=[W^{UK}_{${i}}\\mathbf{c}^{KV}_t;\\mathbf{k}^R_t]`,
    segs: [
      { dims: DIMS.dh, kind: 'built' },
      { dims: DIMS.dr, kind: 'cache' },
    ],
  },
  {
    tex: `\\mathbf{v}_{t,${i}}=W^{UV}_{${i}}\\mathbf{c}^{KV}_t`,
    segs: [{ dims: DIMS.dh, kind: 'built' }],
  },
]

const headRowMqa = (i: string): Row => ({
  tex: `\\mathbf{q}^A_{t,${i}}=[W^{UK\\top}_{${i}}\\mathbf{q}^C_{t,${i}};\\mathbf{q}^R_{t,${i}}]`,
  segs: [
    { dims: DIMS.dc, kind: 'built' },
    { dims: DIMS.dr, kind: 'plain' },
  ],
})

const CACHE: Group = {
  title: 'cached per token, both modes',
  rows: [
    { tex: '\\mathbf{c}^{KV}_t', segs: [{ dims: DIMS.dc, kind: 'cache' }] },
    { tex: '\\mathbf{k}^R_t', segs: [{ dims: DIMS.dr, kind: 'cache' }] },
  ],
}

const MODES: readonly { id: ModeId; label: string; rule: string; groups: Group[] }[] = [
  {
    id: 'mha',
    label: 'MHA mode',
    rule: 'Training and prefill: each head reconstructs $\\mathbf{k}^C_{t,i}=W^{UK}_i\\mathbf{c}^{KV}_t$ and $\\mathbf{v}_{t,i}=W^{UV}_i\\mathbf{c}^{KV}_t$. Core attention uses width $d_h+d_h^R$.',
    groups: [
      CACHE,
      { title: 'head 1', rows: headRowsMha('1') },
      { title: 'head 2', rows: [...headRowsMha('2'), { gap: `heads 3 to ${DIMS.nh - 1}` }] },
      { title: `head ${DIMS.nh}`, rows: headRowsMha(String(DIMS.nh)) },
    ],
  },
  {
    id: 'mqa',
    label: 'MQA mode',
    rule: 'Decode: absorb $W^{UK}_i$ into the query, so each head attends to cached $[\\mathbf{c}^{KV}_s;\\mathbf{k}^R_s]$. $W^{UV}_i$ maps the $d_c$ output to $d_h$.',
    groups: [
      CACHE,
      {
        title: 'per head',
        rows: [
          headRowMqa('1'),
          headRowMqa('2'),
          { gap: `heads 3 to ${DIMS.nh - 1}` },
          headRowMqa(String(DIMS.nh)),
        ],
      },
      {
        title: `shared by all ${DIMS.nh} heads: the cache itself`,
        rows: [
          {
            tex: '\\mathbf{k}_s=[\\mathbf{c}^{KV}_s;\\mathbf{k}^R_s]',
            segs: [
              { dims: DIMS.dc, kind: 'cache' },
              { dims: DIMS.dr, kind: 'cache' },
            ],
          },
          { tex: '\\mathbf{v}_s=\\mathbf{c}^{KV}_s', segs: [{ dims: DIMS.dc, kind: 'cache' }] },
        ],
      },
    ],
  },
]

// --- Model -----------------------------------------------------------------------------------
const cached = DIMS.dc + DIMS.dr
const mode = {
  mha: { kv: DIMS.nh * (DIMS.dh + DIMS.dr + DIMS.dh), qWidth: DIMS.dh + DIMS.dr, vWidth: DIMS.dh },
  mqa: { kv: cached, qWidth: DIMS.dc + DIMS.dr, vWidth: DIMS.dc },
}
// One multiply-add for each score dim and each value dim; 2 FLOPs per multiply-add.
const flopsPerKey = (m: ModeId) => 2 * (mode[m].qWidth + mode[m].vWidth)

const fmt = (n: number) => n.toLocaleString('en-US')
const READOUT: readonly { label: string; mha: string; mqa: string }[] = [
  { label: 'used for', mha: 'training, prefill', mqa: 'decode' },
  {
    label: 'cached per token per layer, values · bytes',
    mha: `${fmt(cached)} · ${fmt(cached * BYTES)} B`,
    mqa: `${fmt(cached)} · ${fmt(cached * BYTES)} B`,
  },
  {
    label: 'K and V per token seen by attention, values · bytes',
    mha: `${fmt(mode.mha.kv)} · ${fmt(mode.mha.kv * BYTES)} B`,
    mqa: `${fmt(mode.mqa.kv)} · ${fmt(mode.mqa.kv * BYTES)} B`,
  },
  { label: 'score width per head', mha: fmt(mode.mha.qWidth), mqa: fmt(mode.mqa.qWidth) },
  { label: 'value width per head', mha: fmt(mode.mha.vWidth), mqa: fmt(mode.mqa.vWidth) },
  { label: 'FLOPs per key per head', mha: fmt(flopsPerKey('mha')), mqa: fmt(flopsPerKey('mqa')) },
]

// Bars are drawn to scale against the widest vector, d_c + d_h^R.
const WIDEST = DIMS.dc + DIMS.dr
const share = (dims: number) => `${((100 * dims) / WIDEST).toFixed(3)}%`

const tex = (t: string): string =>
  renderFigureLabel(t, m =>
    katex.renderToString(m, {
      ...katexOptions,
      displayMode: false,
      output: 'html',
      macros: customMacros,
      strict: false,
      throwOnError: false,
    }),
  )

const BarRow: FunctionalComponent<{ row: Extract<Row, { tex: string }> }> = ({ row }) => {
  const total = row.segs.reduce((a, s) => a + s.dims, 0)
  return (
    <div class="mam-row">
      <span class="mam-label" dangerouslySetInnerHTML={{ __html: tex(row.tex) }} />
      <span class="mam-track">
        {row.segs.map(s => (
          <span
            class={`mam-seg mam-seg--${s.kind}`}
            data-dims={String(s.dims)}
            style={{ width: share(s.dims) }}
          />
        ))}
      </span>
      <span class="mam-dims">
        {row.segs.length > 1 ? `${row.segs.map(s => s.dims).join('+')} = ${total}` : total}
      </span>
    </div>
  )
}

let instance = 0

const MLAAttentionModesImpl: QuartzMdxComponent<Props> = ({ caption, mode: initial }) => {
  const id = `mam-${++instance}`
  const m0: ModeId = initial === 'mqa' ? 'mqa' : 'mha'

  return (
    <figure class="mla-attention-modes" data-mla-attention-modes data-mam-mode={m0}>
      <fieldset class="mam-segments">
        <legend class="mam-legend">Mode</legend>
        {MODES.map(m => (
          <label class="mam-segment">
            <input
              type="radio"
              name={`${id}-mode`}
              value={m.id}
              checked={m.id === m0}
              data-mam-mode-input
            />
            <span>{m.label}</span>
          </label>
        ))}
      </fieldset>

      {MODES.map(m => (
        <p class="mam-rule" data-mam-rule={m.id}>
          <MathText text={m.rule} mathClass="mam-math" />
        </p>
      ))}

      {MODES.map(m => (
        <div
          class="mam-panel"
          data-mam-panel={m.id}
          role="group"
          aria-label={`${m.label} of MLA: vectors drawn to scale in dimensions, DeepSeek-V3 sizes.`}
        >
          {m.groups.map(g => (
            <div class="mam-group">
              <p class="mam-title">{g.title}</p>
              {g.rows.map(r =>
                'gap' in r ? <p class="mam-gap">{`⋮ ${r.gap}`}</p> : <BarRow row={r} />,
              )}
            </div>
          ))}
        </div>
      ))}

      <ul class="mam-key" aria-label="Legend">
        <li>
          <span class="mam-swatch mam-swatch--cache" aria-hidden="true" />
          sage: read from the latent cache
        </li>
        <li>
          <span class="mam-swatch mam-swatch--built" aria-hidden="true" />
          salmon: built per head in this mode
        </li>
        <li>
          <span class="mam-swatch mam-swatch--plain" aria-hidden="true" />
          query projection
        </li>
      </ul>

      <div class="mam-table-wrap">
        <table class="mam-table" aria-label="Per-token cost of each MLA mode">
          <thead>
            <tr>
              <th scope="col">DeepSeek-V3, BF16</th>
              <th scope="col" data-col="mha">
                MHA mode
              </th>
              <th scope="col" data-col="mqa">
                MQA mode
              </th>
            </tr>
          </thead>
          <tbody>
            {READOUT.map(r => (
              <tr>
                <th scope="row">{r.label}</th>
                <td data-col="mha">{r.mha}</td>
                <td data-col="mqa">{r.mqa}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {caption ? (
        <figcaption class="mam-caption">
          <MathText text={caption} mathClass="mam-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MLAAttentionModesComponent = MLAAttentionModesImpl as QuartzMdxComponent<Props>
MLAAttentionModesComponent.css = style
MLAAttentionModesComponent.afterDOMLoaded = script

export const MLAAttentionModes = registerMdxComponent(
  'MLAAttentionModes',
  MLAAttentionModesComponent,
)

export default (() => MLAAttentionModes) satisfies (opts: undefined) => QuartzMdxComponent<Props>
