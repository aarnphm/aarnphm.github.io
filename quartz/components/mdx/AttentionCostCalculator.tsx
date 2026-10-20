import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { renderFigureLabel } from '../../util/figure-label'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/attention-cost-calculator.inline'
import {
  ACC_DTYPE_BYTES,
  ACC_INITIAL,
  ACC_VALUES,
  accBytesTex,
  accCompute,
  accMetricTex,
  accRatioTex,
  type AccDType,
  type AccNumKey,
} from '../scripts/attention-cost-model'
import style from '../styles/attentionCostCalculator.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

type SliderSpec = { k: AccNumKey; tex: string; lbl: string }

const SLIDERS: SliderSpec[] = [
  { k: 'dm', tex: 'd_{\\text{model}}', lbl: 'model dim' },
  { k: 'nl', tex: 'N', lbl: 'layers' },
  { k: 'nh', tex: 'h', lbl: 'heads' },
  { k: 'sl', tex: 'L', lbl: 'seq length' },
  { k: 'bs', tex: 'B', lbl: 'batch' },
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
  {
    k: 'flops',
    label: 'attention FLOPs / decoded token',
    formula: '8\\,N\\,d_{\\text{model}}^{2} + 4\\,N\\,L\\,d_{\\text{model}}',
  },
  { k: 'kvtok', label: 'KV / token / layer', formula: '2\\,d_{\\text{model}}\\cdot\\text{bytes}' },
  {
    k: 'kvtotal',
    label: 'KV cache total',
    formula: 'B\\,L\\,N\\cdot 2\\,d_{\\text{model}}\\cdot\\text{bytes}',
    accent: 'salmon',
  },
  {
    k: 'ratio',
    label: 'KV cache / attention weights (BF16)',
    formula: '\\frac{B\\,L\\,\\text{bytes}}{4\\,d_{\\text{model}}}',
    accent: 'sage',
  },
]

const katexHtml = (m: string, d: boolean): string =>
  katex.renderToString(m, {
    ...katexOptions,
    displayMode: d,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const tex = (t: string, d = false, whole = false): string =>
  whole ? katexHtml(t, d) : renderFigureLabel(t, m => katexHtml(m, d), d)

const M: FunctionalComponent<{ t: string; d?: boolean }> = ({ t, d }) => (
  <span
    class={`acc-math${d ? ' acc-math--display' : ''}`}
    dangerouslySetInnerHTML={{ __html: tex(t, d) }}
  />
)

// Numbers are math: KaTeX renders them whole, with no word splitting.
const Num: FunctionalComponent<{ t: string; class: string; attr: Record<string, string> }> = ({
  t,
  class: cls,
  attr,
}) => <span class={cls} {...attr} dangerouslySetInnerHTML={{ __html: tex(t, false, true) }} />

const FormulaFO: FunctionalComponent<{ t: string }> = ({ t }) => (
  <span class="acc-card-eq" dangerouslySetInnerHTML={{ __html: tex(t) }} />
)

const init = accCompute(ACC_INITIAL)

const DTYPE_LABEL = (dt: AccDType): string => `${dt} (${ACC_DTYPE_BYTES[dt]} B)`

const AttentionCostCalculatorImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const totalMem = init.paramBytes + init.ffnBytes + init.kvtotal
  const bars = [
    { k: 'attn', tag: 'attn weights', bytes: init.paramBytes, track: '' },
    { k: 'ffn', tag: 'FFN weights', bytes: init.ffnBytes, track: '' },
    { k: 'kv', tag: 'KV cache', bytes: init.kvtotal, track: ' acc-bar-track--bytes' },
  ]

  return (
    <figure
      class="attention-cost-calculator"
      data-acc-root
      data-acc-dm={String(ACC_INITIAL.dm)}
      data-acc-nl={String(ACC_INITIAL.nl)}
      data-acc-nh={String(ACC_INITIAL.nh)}
      data-acc-sl={String(ACC_INITIAL.sl)}
      data-acc-bs={String(ACC_INITIAL.bs)}
      data-acc-dt={ACC_INITIAL.dt}
    >
      <div class="acc-stage">
        <aside class="acc-controls" role="group" aria-label="Transformer cost controls">
          {SLIDERS.map(s => {
            const values = ACC_VALUES[s.k]
            const def = ACC_INITIAL[s.k]
            return (
              <div class="acc-control">
                <label class="acc-label" for={`acc-${s.k}`}>
                  <M t={s.tex} /> <span class="acc-label-lbl">{s.lbl}</span>
                </label>
                <input
                  id={`acc-${s.k}`}
                  class="acc-slider"
                  type="range"
                  min={0}
                  max={values.length - 1}
                  step={1}
                  value={values.indexOf(def)}
                  data-acc-input={s.k}
                  aria-valuemin={values[0]}
                  aria-valuemax={values[values.length - 1]}
                  aria-valuenow={def}
                  aria-valuetext={`${s.lbl} ${def}`}
                />
                <button
                  type="button"
                  class="acc-value"
                  data-acc-value={s.k}
                  aria-label={`${s.lbl} ${def}, type a value`}
                  dangerouslySetInnerHTML={{ __html: tex(String(def), false, true) }}
                />
                <input
                  class="acc-value-edit"
                  type="text"
                  inputmode="numeric"
                  data-acc-edit={s.k}
                  aria-label={`${s.lbl} value`}
                  hidden
                />
              </div>
            )
          })}
          <div class="acc-control acc-control--dropdown">
            <span class="acc-label" id="acc-dt-label">
              <span class="acc-label-lbl">kv dtype</span>
            </span>
            {/* The inline script mounts the shared select here; this trigger holds its place. */}
            <div class="acc-dtype" data-acc-dtype>
              <div class="g-select">
                <button type="button" class="g-select-trigger" aria-labelledby="acc-dt-label">
                  <span class="g-select-value">{DTYPE_LABEL(ACC_INITIAL.dt)}</span>
                </button>
              </div>
            </div>
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
              <span class="acc-card-label">{r.label}</span>
              <Num
                t={accMetricTex(r.k, init)}
                class="acc-card-num"
                attr={{ 'data-acc-num': r.k, 'aria-live': 'polite' }}
              />
              <FormulaFO t={r.formula} />
            </article>
          ))}
        </div>
      </div>

      <div
        class="acc-bar"
        aria-label="memory footprint: attention weights vs FFN weights vs KV cache"
      >
        {bars.map(b => (
          <div class="acc-bar-row">
            <span class="acc-bar-tag">{b.tag}</span>
            <div class={`acc-bar-track${b.track}`}>
              <span
                class={`acc-bar-fill acc-bar-fill--${b.k}`}
                data-acc-bar={b.k}
                style={`width:${((b.bytes / totalMem) * 100).toFixed(1)}%`}
              />
            </div>
            <Num t={accBytesTex(b.bytes)} class="acc-bar-val" attr={{ 'data-acc-barval': b.k }} />
          </div>
        ))}
      </div>

      <div class="acc-summary" data-acc-summary>
        <dl class="acc-summary-table">
          <div class="acc-summary-row">
            <dt>KV</dt>
            <dd
              data-acc-sumkv
              dangerouslySetInnerHTML={{ __html: tex(accBytesTex(init.kvtotal), false, true) }}
            />
          </div>
          <div class="acc-summary-row">
            <dt>attn weight</dt>
            <dd
              data-acc-sumparams
              dangerouslySetInnerHTML={{ __html: tex(accBytesTex(init.paramBytes), false, true) }}
            />
          </div>
          <div class="acc-summary-row">
            <dt>ratio</dt>
            <dd
              data-acc-sumratio
              dangerouslySetInnerHTML={{ __html: tex(accRatioTex(init.ratio), false, true) }}
            />
          </div>
        </dl>
        <p class="acc-summary-hint">
          Long-context inference is cache-bound (linear in <M t="L \cdot B \cdot N" />
          ); short-context is param-bound. Sliding <M t="h" /> alone is free in both.
        </p>
      </div>

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
