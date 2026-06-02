import katex from 'katex'
import { type ComponentChildren, type FunctionalComponent, type VNode } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/positional-encoding-comparison.inline'
import style from '../styles/positionalEncodingComparison.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; length?: number }

const PEC_LENGTHS = [8, 16, 32, 64] as const
type PecLength = (typeof PEC_LENGTHS)[number]
const DEFAULT_LENGTH: PecLength = 16
const ABS_DIM = 16
const ROPE_PAIRS = 4
const ROPE_POSITIONS = 8

const clampLength = (raw: number): PecLength => {
  const rounded = Math.round(raw)
  for (const candidate of PEC_LENGTHS) {
    if (rounded <= candidate) return candidate
  }
  return PEC_LENGTHS[PEC_LENGTHS.length - 1]
}

const renderMath = (tex: string, display = false): string =>
  katex.renderToString(tex, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const MathLabel: FunctionalComponent<{ tex: string; display?: boolean }> = ({
  tex,
  display = false,
}) => {
  const Tag = display ? 'div' : 'span'
  const cls = `pec-math ${display ? 'pec-math--display' : 'pec-math--inline'}`
  return <Tag class={cls} dangerouslySetInnerHTML={{ __html: renderMath(tex, display) }} />
}

const absValue = (pos: number, dim: number, totalDim: number): number => {
  const halfIndex = Math.floor(dim / 2)
  const denom = Math.pow(10000, (2 * halfIndex) / totalDim)
  const phase = pos / denom
  return dim % 2 === 0 ? Math.sin(phase) : Math.cos(phase)
}

const lerpFill = (value: number): string => {
  const v = Math.max(-1, Math.min(1, value))
  if (v >= 0) {
    const mix = Math.round(v * 78)
    return `color-mix(in srgb, var(--pec-pos) ${mix}%, var(--pec-neutral))`
  }
  const mix = Math.round(-v * 65)
  return `color-mix(in srgb, var(--pec-neg) ${mix}%, var(--pec-neutral))`
}

const magFill = (mag: number): string => {
  const v = Math.max(0, Math.min(1, mag))
  const mix = Math.round(v * 82)
  return `color-mix(in srgb, var(--pec-pos) ${mix}%, var(--pec-neutral))`
}

const penaltyFill = (mag: number): string => {
  const v = Math.max(0, Math.min(1, mag))
  const mix = Math.round(v * 78)
  return `color-mix(in srgb, var(--pec-neg) ${mix}%, var(--pec-neutral))`
}

const relBias = (delta: number, length: number): number => {
  if (length <= 1) return 0
  const abs = Math.abs(delta)
  const norm = Math.sqrt(abs) / Math.sqrt(length - 1)
  return Math.max(0, 1 - norm)
}

const alibiSlope = (length: number): number => 4 / Math.max(1, length - 1)

const ropeAngle = (pos: number, pairIndex: number, totalDim: number): number => {
  const denom = Math.pow(10000, (2 * pairIndex) / totalDim)
  return pos / denom
}

const AbsolutePanel: FunctionalComponent<{ length: number }> = ({ length }) => {
  const rows = Array.from({ length }, (_, p) => p)
  const cols = Array.from({ length: ABS_DIM }, (_, d) => d)
  return (
    <svg
      viewBox={`0 0 ${ABS_DIM} ${length}`}
      preserveAspectRatio="none"
      shape-rendering="crispEdges"
      data-pec-vis="absolute"
      role="img"
      aria-label="Sinusoidal absolute positional encoding heatmap: rows are token positions, columns are embedding dimensions, salmon is positive, neutral is zero"
    >
      {rows.map(p =>
        cols.map(d => (
          <rect
            class="pec-cell"
            x={d}
            y={p}
            width={1}
            height={1}
            fill={lerpFill(absValue(p, d, ABS_DIM))}
          />
        )),
      )}
    </svg>
  )
}

const RelativePanel: FunctionalComponent<{ length: number }> = ({ length }) => {
  const idx = Array.from({ length }, (_, i) => i)
  return (
    <svg
      viewBox={`0 0 ${length} ${length}`}
      preserveAspectRatio="none"
      shape-rendering="crispEdges"
      data-pec-vis="relative"
      role="img"
      aria-label="Relative positional bias matrix: cell (i, j) brightness scales with proximity along the diagonal"
    >
      {idx.map(i =>
        idx.map(j => (
          <rect
            class="pec-cell"
            x={j}
            y={i}
            width={1}
            height={1}
            fill={magFill(relBias(j - i, length))}
          />
        )),
      )}
    </svg>
  )
}

const RopePanel: FunctionalComponent<{ length: number }> = ({ length }) => {
  const rows = Math.min(ROPE_POSITIONS, length)
  const step = Math.max(1, Math.floor(length / rows))
  const totalDim = ABS_DIM
  const radius = 0.36
  const handLen = radius * 0.6
  const cells: VNode[] = []
  for (let r = 0; r < rows; r++) {
    const p = r * step
    for (let c = 0; c < ROPE_PAIRS; c++) {
      const angle = ropeAngle(p, c, totalDim)
      const cx = c + 0.5
      const cy = r + 0.5
      cells.push(
        <g>
          <rect class="pec-cell" x={c} y={r} width={1} height={1} fill="transparent" />
          <circle class="pec-clock-bg" cx={cx} cy={cy} r={radius} />
          <circle class="pec-clock" cx={cx} cy={cy} r={radius} />
          <line
            class="pec-clock-hand"
            x1={cx}
            y1={cy}
            x2={cx + handLen * Math.cos(angle)}
            y2={cy + handLen * Math.sin(angle)}
          />
        </g>,
      )
    }
  }
  return (
    <svg
      viewBox={`0 0 ${ROPE_PAIRS} ${rows}`}
      preserveAspectRatio="xMidYMid meet"
      data-pec-vis="rope"
      role="img"
      aria-label="Rotary position encoding dial grid: rows are sampled positions, columns are dimension pairs, hand angle equals the per-position rotation"
    >
      {cells}
    </svg>
  )
}

const AlibiPanel: FunctionalComponent<{ length: number }> = ({ length }) => {
  const idx = Array.from({ length }, (_, i) => i)
  const m = alibiSlope(length)
  return (
    <svg
      viewBox={`0 0 ${length} ${length}`}
      preserveAspectRatio="none"
      shape-rendering="crispEdges"
      data-pec-vis="alibi"
      role="img"
      aria-label="ALiBi penalty matrix: cell (i, j) darkens linearly with distance from the diagonal"
    >
      {idx.map(i =>
        idx.map(j => (
          <rect
            class="pec-cell"
            x={j}
            y={i}
            width={1}
            height={1}
            fill={penaltyFill(Math.min(1, m * Math.abs(j - i)))}
          />
        )),
      )}
    </svg>
  )
}

const Panel: FunctionalComponent<{
  title: string
  tag: string
  caption: string
  math: string
  logit: string
  children: ComponentChildren
}> = ({ title, tag, caption, math, logit, children }) => (
  <section class="pec-panel" role="group" aria-label={`${title} positional encoding`}>
    <header class="pec-panel-head">
      <span class="pec-panel-title">{title}</span>
      <span class="pec-panel-tag">{tag}</span>
    </header>
    <div class="pec-panel-vis">
      {children}
      <div class="pec-logit-mode" aria-hidden="true">
        <span dangerouslySetInnerHTML={{ __html: logit }} />
      </div>
    </div>
    <div class="pec-panel-math" dangerouslySetInnerHTML={{ __html: renderMath(math, true) }} />
    <p class="pec-panel-caption">{caption}</p>
  </section>
)

const PositionalEncodingComparisonImpl: QuartzMdxComponent<Props> = ({
  caption,
  length = DEFAULT_LENGTH,
}) => {
  const L = clampLength(length)

  return (
    <figure
      class="positional-encoding-comparison"
      data-positional-encoding-comparison
      data-pec-length={String(L)}
      data-pec-logit="false"
    >
      <div class="pec-grid">
        <Panel
          title="Absolute"
          tag="input embed"
          caption="Added to input embeddings; weak extrapolation beyond training length."
          math="\mathrm{PE}(p, 2i) = \sin\!\left(\tfrac{p}{10000^{2i/d}}\right)"
          logit={renderMath('x_p \\leftarrow E_p + \\mathrm{PE}_p', false)}
        >
          <AbsolutePanel length={L} />
        </Panel>
        <Panel
          title="Relative bias"
          tag="logit add"
          caption="Added to attention logits; bias as a learned monotonic function of pairwise distance."
          math="A_{ij} \mathrel{+}= b_{\,j-i}"
          logit={renderMath('A_{ij} \\mathrel{+}= b_{\\,j-i}', false)}
        >
          <RelativePanel length={L} />
        </Panel>
        <Panel
          title="RoPE"
          tag="rotate Q/K"
          caption="Rotates Q and K in the complex plane; inner products encode relative offsets."
          math="q' = R_p\, q,\ \ \theta_p^{(i)} = \tfrac{p}{10000^{2i/d}}"
          logit={renderMath('A_{ij} = (R_i q_i)^{\\top} (R_j k_j)', false)}
        >
          <RopePanel length={L} />
        </Panel>
        <Panel
          title="ALiBi"
          tag="logit add"
          caption="Additive linear penalty; extrapolates without re-training, slope chosen per head."
          math="A_{ij} \mathrel{+}= -\,m\,\lvert\,j - i\,\rvert"
          logit={renderMath('A_{ij} \\mathrel{+}= -m\\,|j-i|', false)}
        >
          <AlibiPanel length={L} />
        </Panel>
      </div>

      <section class="pec-extrap" aria-label="Length extrapolation summary">
        <div class="pec-extrap-title">length extrapolation</div>
        <dl>
          <dt>absolute</dt>
          <dd>poor, fixed table; off-distribution at longer L.</dd>
          <dt>relative</dt>
          <dd>decent, bias table caps at training distance.</dd>
          <dt>RoPE</dt>
          <dd>good with NTK / YaRN / LongRoPE rescaling of frequency base.</dd>
          <dt>ALiBi</dt>
          <dd>best by construction, penalty is unbounded in L.</dd>
        </dl>
      </section>

      <div class="pec-controls" role="group" aria-label="Positional encoding controls">
        <div class="pec-control">
          <span class="pec-label">
            sequence <MathLabel tex="L" />
          </span>
          <div class="pec-tablist" role="tablist" aria-label="Sequence length">
            {PEC_LENGTHS.map(len => (
              <button
                type="button"
                class={`pec-tab${len === L ? ' is-active' : ''}`}
                data-pec-length-btn={String(len)}
                role="tab"
                aria-selected={len === L ? 'true' : 'false'}
                tabIndex={len === L ? 0 : -1}
                aria-label={`length ${len}`}
              >
                {len}
              </button>
            ))}
          </div>
        </div>

        <label class="pec-switch">
          <input
            type="checkbox"
            data-pec-logit-toggle
            aria-label="Show as attention logit modifier"
          />
          <span class="pec-switch-track" aria-hidden="true">
            <span class="pec-switch-thumb" />
          </span>
          show as attention logit modifier
        </label>
      </div>

      <p class="pec-intuition">
        Three orthogonal axes separate these schemes: where positional information enters (input
        embedding for absolute; attention logits for relative and ALiBi; Q/K rotation for RoPE), how
        the effect scales with distance (constant table, monotonic falloff, periodic dial, linear
        penalty), and how well it extrapolates beyond the training length. RoPE wins for length
        extrapolation when combined with NTK-aware scaling, YaRN, or LongRoPE; ALiBi extrapolates by
        construction because the penalty is a closed-form function of <MathLabel tex="|j-i|" />.
      </p>

      {caption ? (
        <figcaption class="pec-caption">
          <MathText text={caption} mathClass="pec-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const PositionalEncodingComparisonComponent =
  PositionalEncodingComparisonImpl as QuartzMdxComponent<Props>
PositionalEncodingComparisonComponent.css = style
PositionalEncodingComparisonComponent.afterDOMLoaded = script

export const PositionalEncodingComparison = registerMdxComponent(
  'PositionalEncodingComparison',
  PositionalEncodingComparisonComponent,
)

export default (() => PositionalEncodingComparison) satisfies (
  opts: undefined,
) => QuartzMdxComponent<Props>
