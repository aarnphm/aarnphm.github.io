import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/attention-cost-calculator.inline'
import style from '../styles/attentionCostCalculator.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

type DType = 'bf16' | 'fp16' | 'fp32' | 'int8' | 'int4'

type SliderSpec = { k: string; tex: string; lbl: string; values: number[]; def: number }

const DM_VALUES = [128, 256, 512, 1024, 2048, 4096, 8192]
const NL_VALUES = [1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96]
const NH_VALUES = [1, 2, 4, 8, 16, 32, 64]
const L_VALUES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
const B_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

const DTYPE_BYTES: Record<DType, number> = { bf16: 2, fp16: 2, fp32: 4, int8: 1, int4: 0.5 }

const SLIDERS: SliderSpec[] = [
  { k: 'dm', tex: 'd_{\\text{model}}', lbl: 'model dim', values: DM_VALUES, def: 4096 },
  { k: 'nl', tex: 'N', lbl: 'layers', values: NL_VALUES, def: 32 },
  { k: 'nh', tex: 'h', lbl: 'heads', values: NH_VALUES, def: 32 },
  { k: 'sl', tex: 'L', lbl: 'seq length', values: L_VALUES, def: 4096 },
  { k: 'bs', tex: 'B', lbl: 'batch', values: B_VALUES, def: 1 },
]

const DTYPES: { v: DType; lbl: string; bytes: number }[] = [
  { v: 'bf16', lbl: 'bf16', bytes: 2 },
  { v: 'fp16', lbl: 'fp16', bytes: 2 },
  { v: 'fp32', lbl: 'fp32', bytes: 4 },
  { v: 'int8', lbl: 'int8', bytes: 1 },
  { v: 'int4', lbl: 'int4', bytes: 0.5 },
]

type ReadoutCard = {
  k: string
  label: string
  formula: string
  unit?: string
  accent?: 'salmon' | 'sage'
}

const READOUTS: ReadoutCard[] = [
  { k: 'dh', label: 'per-head dim', formula: 'd_h = d_{\\text{model}} / h' },
  { k: 'params', label: 'attention params', formula: '4\\,N\\,d_{\\text{model}}^{2}' },
  { k: 'flops', label: 'attention FLOPs / decoded token', formula: '4\\,L\\,d_{\\text{model}}' },
  { k: 'kvtok', label: 'KV / token / layer', formula: '2\\,d_{\\text{model}}\\cdot\\text{bytes}' },
  {
    k: 'kvtotal',
    label: 'KV cache total',
    formula: 'B\\,L\\,N\\cdot 2\\,d_{\\text{model}}\\cdot\\text{bytes}',
    accent: 'salmon',
  },
  {
    k: 'ratio',
    label: 'cache / param weight ratio',
    formula: '\\frac{B\\,L\\cdot 2\\,\\text{bytes}}{4\\,d_{\\text{model}}}',
    accent: 'sage',
  },
]

const tex = (t: string, d = false): string =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: d,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const M: FunctionalComponent<{ t: string; d?: boolean }> = ({ t, d }) => (
  <span
    class={`acc-math${d ? ' acc-math--display' : ''}`}
    dangerouslySetInnerHTML={{ __html: tex(t, d) }}
  />
)

const FormulaFO: FunctionalComponent<{ t: string }> = ({ t }) => (
  <span class="acc-card-eq" dangerouslySetInnerHTML={{ __html: tex(t) }} />
)

const fmtBytes = (n: number): string => {
  if (n < 1024) return `${n.toFixed(0)} B`
  const units = ['KB', 'MB', 'GB', 'TB', 'PB']
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} ${units[i]}`
}

const fmtCount = (n: number): string => {
  if (n < 1000) return n.toString()
  const units = ['K', 'M', 'B', 'T', 'P']
  let v = n / 1000
  let i = 0
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000
    i++
  }
  return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)}${units[i]}`
}

const initial = { dm: 4096, nl: 32, nh: 32, sl: 4096, bs: 1, dt: 'bf16' as DType }

const computeInitial = () => {
  const dh = initial.dm / initial.nh
  const bytes = DTYPE_BYTES[initial.dt]
  const params = initial.nl * 4 * initial.dm * initial.dm
  const flops = 4 * initial.sl * initial.dm
  const kvtok = 2 * initial.dm * bytes
  const ffnParams = initial.nl * 8 * initial.dm * initial.dm
  const kvtotal = initial.bs * initial.sl * initial.nl * 2 * initial.dm * bytes
  const paramBytes = params * bytes
  const ratio = paramBytes > 0 ? kvtotal / paramBytes : 0
  return { dh, params, flops, kvtok, kvtotal, ratio, ffnParams, paramBytes }
}

const init = computeInitial()

const readoutValue = (k: string): string => {
  switch (k) {
    case 'dh':
      return init.dh.toString()
    case 'params':
      return fmtCount(init.params)
    case 'flops':
      return fmtCount(init.flops)
    case 'kvtok':
      return fmtBytes(init.kvtok)
    case 'kvtotal':
      return fmtBytes(init.kvtotal)
    case 'ratio':
      return `${init.ratio.toFixed(init.ratio >= 10 ? 0 : 2)}x`
    default:
      return '-'
  }
}

const AttentionCostCalculatorImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const totalParams = Math.max(1, init.params + init.ffnParams)
  const totalBytes = Math.max(1, init.paramBytes + init.kvtotal)
  const attnFrac = init.params / totalParams
  const ffnFrac = init.ffnParams / totalParams
  const kvFracVsParams = init.kvtotal / totalBytes
  const paramFracVsKv = init.paramBytes / totalBytes

  return (
    <figure
      class="attention-cost-calculator"
      data-acc-root
      data-acc-dm={String(initial.dm)}
      data-acc-nl={String(initial.nl)}
      data-acc-nh={String(initial.nh)}
      data-acc-sl={String(initial.sl)}
      data-acc-bs={String(initial.bs)}
      data-acc-dt={initial.dt}
    >
      <div class="acc-stage">
        <aside class="acc-controls" role="group" aria-label="Transformer cost controls">
          {SLIDERS.map(s => (
            <div class="acc-control">
              <label class="acc-label" for={`acc-${s.k}`}>
                <M t={s.tex} /> <span class="acc-label-lbl">{s.lbl}</span>
              </label>
              <input
                id={`acc-${s.k}`}
                class="acc-slider"
                type="range"
                min={0}
                max={s.values.length - 1}
                step={1}
                value={s.values.indexOf(s.def)}
                data-acc-input={s.k}
                aria-valuemin={s.values[0]}
                aria-valuemax={s.values[s.values.length - 1]}
                aria-valuenow={s.def}
                aria-valuetext={`${s.lbl} ${s.def}`}
              />
              <span class="acc-value" data-acc-value={s.k}>
                {s.def}
              </span>
            </div>
          ))}
          <div class="acc-control acc-control--dropdown">
            <label class="acc-label" for="acc-dt">
              <span class="acc-label-lbl">dtype</span>
            </label>
            <select
              id="acc-dt"
              class="acc-select"
              data-acc-input="dt"
              aria-label="numeric precision (bytes per element)"
            >
              {DTYPES.map(dt => (
                <option value={dt.v} selected={dt.v === initial.dt}>
                  {dt.lbl} ({dt.bytes}B)
                </option>
              ))}
            </select>
          </div>
          <p class="acc-warning" data-acc-warning hidden>
            <M t="d_{\text{model}} / h" /> not integer, choose <M t="h" /> that divides{' '}
            <M t="d_{\text{model}}" />.
          </p>
        </aside>

        <div class="acc-cards" role="list">
          {READOUTS.map(r => (
            <article
              class={`acc-card${r.accent ? ` acc-card--${r.accent}` : ''}`}
              data-acc-card={r.k}
              data-metric={r.k}
              role="listitem"
            >
              <header class="acc-card-head">
                <span class="acc-card-label">{r.label}</span>
                <FormulaFO t={r.formula} />
              </header>
              <div class="acc-card-body">
                <span class="acc-card-num" data-acc-num={r.k}>
                  {readoutValue(r.k)}
                </span>
                <span class="acc-card-delta" data-acc-delta={r.k} aria-live="polite" />
              </div>
            </article>
          ))}
        </div>
      </div>

      <div
        class="acc-bar"
        aria-label="relative cost: attention params vs FFN params vs KV cache bytes"
      >
        <div class="acc-bar-row">
          <span class="acc-bar-tag">attn params</span>
          <div class="acc-bar-track">
            <span
              class="acc-bar-fill acc-bar-fill--attn"
              data-acc-bar="attn"
              style={`width:${(attnFrac * 100).toFixed(1)}%`}
            />
          </div>
          <span class="acc-bar-val" data-acc-barval="attn">
            {fmtCount(init.params)}
          </span>
        </div>
        <div class="acc-bar-row">
          <span class="acc-bar-tag">FFN params</span>
          <div class="acc-bar-track">
            <span
              class="acc-bar-fill acc-bar-fill--ffn"
              data-acc-bar="ffn"
              style={`width:${(ffnFrac * 100).toFixed(1)}%`}
            />
          </div>
          <span class="acc-bar-val" data-acc-barval="ffn">
            {fmtCount(init.ffnParams)}
          </span>
        </div>
        <div class="acc-bar-row">
          <span class="acc-bar-tag">KV cache</span>
          <div class="acc-bar-track acc-bar-track--bytes">
            <span
              class="acc-bar-fill acc-bar-fill--kv"
              data-acc-bar="kv"
              style={`width:${(kvFracVsParams * 100).toFixed(1)}%`}
            />
            <span
              class="acc-bar-fill acc-bar-fill--params"
              data-acc-bar="paramBytes"
              style={`width:${(paramFracVsKv * 100).toFixed(1)}%;left:auto;right:0;opacity:.25`}
            />
          </div>
          <span class="acc-bar-val" data-acc-barval="kv">
            {fmtBytes(init.kvtotal)}
          </span>
        </div>
      </div>

      <p class="acc-summary" data-acc-summary>
        At current config: <strong data-acc-sumkv>{fmtBytes(init.kvtotal)}</strong> KV vs{' '}
        <strong data-acc-sumparams>{fmtBytes(init.paramBytes)}</strong> attention weights, ratio{' '}
        <strong data-acc-sumratio>{init.ratio.toFixed(init.ratio >= 10 ? 0 : 2)}x</strong>.{' '}
        <span class="acc-summary-hint">
          Long-context inference is cache-bound (linear in <M t="L \cdot B \cdot N" />
          ); short-context is param-bound. Sliding <M t="h" /> alone is free in both.
        </span>
      </p>

      {caption ? (
        <figcaption class="acc-caption">
          <MathText text={caption} mathClass="acc-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const AttentionCostCalculatorComponent = AttentionCostCalculatorImpl as QuartzMdxComponent<Props>
AttentionCostCalculatorComponent.css = style
AttentionCostCalculatorComponent.afterDOMLoaded = script

export const AttentionCostCalculator = registerMdxComponent(
  'AttentionCostCalculator',
  AttentionCostCalculatorComponent,
)

export default (() => AttentionCostCalculator) satisfies (
  opts: undefined,
) => QuartzMdxComponent<Props>
