import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/attention-circuits.inline'
import style from '../styles/attentionCircuits.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const tex = (t: string, d = false) =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: d,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const math = String.raw

type FoProps = { x: number; y: number; w: number; h: number; t: string; cls?: string }
const Fo: FunctionalComponent<FoProps> = ({ x, y, w, h, t, cls }) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div class={`acir-fo ${cls ?? ''}`.trim()} dangerouslySetInnerHTML={{ __html: tex(t) }} />
  </foreignObject>
)

const STD_ARROW = 'acir-std-arrow'
const CIR_ARROW = 'acir-cir-arrow'

const ArrowDefs: FunctionalComponent<{ id: string }> = ({ id }) => (
  <defs>
    <marker
      id={id}
      viewBox="0 0 10 10"
      refX="9"
      refY="5"
      markerWidth="6"
      markerHeight="6"
      orient="auto-start-reverse"
    >
      <path class="acir-arrowhead" d="M0,0 L10,5 L0,10 z" />
    </marker>
  </defs>
)

const Weight: FunctionalComponent<{
  x: number
  y: number
  t: string
  circuit: 'qk' | 'ov'
  k: string
}> = ({ x, y, t, circuit, k }) => (
  <g class="acir-weight" data-acir-weight={k} data-acir-circuit={circuit}>
    <rect class="acir-box acir-box--weight" x={x} y={y} width={56} height={40} rx={4} />
    <Fo x={x} y={y} w={56} h={40} t={t} />
  </g>
)

const StandardPanel: FunctionalComponent = () => {
  const arr = `url(#${STD_ARROW})`
  return (
    <svg
      class="acir-graph acir-graph--standard"
      viewBox="0 0 320 460"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Standard attention flow: x branches into Q, K, V projections; Q and K feed softmax; weights apply to V; result passes through W_O."
    >
      <ArrowDefs id={STD_ARROW} />
      <rect class="acir-box acir-box--input" x={6} y={210} width={44} height={40} rx={4} />
      <Fo x={6} y={210} w={44} h={40} t="x" cls="acir-fo--big" />
      <path class="acir-line" d="M 50 230 L 60 230" />
      <path class="acir-line" d="M 60 230 L 60 100 L 88 100" marker-end={arr} />
      <path class="acir-line" d="M 60 230 L 88 230" marker-end={arr} />
      <path class="acir-line" d="M 60 230 L 60 360 L 88 360" marker-end={arr} />
      <Weight x={92} y={80} t="W_Q" circuit="qk" k="WQ" />
      <Weight x={92} y={210} t="W_K" circuit="qk" k="WK" />
      <Weight x={92} y={340} t="W_V" circuit="ov" k="WV" />
      <g class="acir-circuit-mark" data-acir-circuit="qk">
        <path class="acir-line acir-line--qk" d="M 148 100 L 180 100" marker-end={arr} />
        <path
          class="acir-line acir-line--qk"
          d="M 148 230 L 166 230 L 166 140 L 180 140"
          marker-end={arr}
        />
      </g>
      <rect class="acir-box acir-box--softmax" x={184} y={78} width={64} height={98} rx={4} />
      <Fo x={184} y={80} w={64} h={30} t={math`\frac{QK^\top}{\sqrt{d_h}}`} cls="acir-fo--sm" />
      <Fo x={184} y={110} w={64} h={24} t={math`\operatorname{softmax}`} cls="acir-fo--xs" />
      <g class="acir-attn-grid" data-acir-attn-grid>
        {Array.from({ length: 4 }, (_, r) =>
          Array.from({ length: 4 }, (_, c) => (
            <rect
              class="acir-attn-cell"
              x={192 + c * 13}
              y={140 + r * 7}
              width={11}
              height={5}
              rx={1}
              style={{ opacity: r >= c ? Math.max(0.15, 1 - (r - c) * 0.22) : 0 }}
            />
          )),
        )}
      </g>
      <g class="acir-circuit-mark" data-acir-circuit="ov">
        <path class="acir-line acir-line--ov" d="M 148 360 L 186 360" marker-end={arr} />
      </g>
      <rect class="acir-box acir-box--value" x={190} y={340} width={52} height={40} rx={4} />
      <Fo x={190} y={340} w={52} h={40} t="V" />
      <circle class="acir-node" cx={216} cy={250} r={9} />
      <Fo x={208} y={242} w={16} h={16} t={math`\times`} cls="acir-fo--sm" />
      <path class="acir-line acir-line--qk" d="M 216 176 L 216 241" marker-end={arr} />
      <path class="acir-line acir-line--ov" d="M 216 340 L 216 259" marker-end={arr} />
      <path class="acir-line acir-line--ov" d="M 225 250 L 252 250" marker-end={arr} />
      <Weight x={256} y={230} t="W_O" circuit="ov" k="WO" />
      <path class="acir-line acir-line--ov" d="M 284 270 L 284 412" marker-end={arr} />
      <rect class="acir-box acir-box--output" x={262} y={412} width={44} height={36} rx={4} />
      <Fo x={262} y={412} w={44} h={36} t={math`\mathrm{out}`} cls="acir-fo--sm" />
    </svg>
  )
}

const CircuitPanel: FunctionalComponent = () => {
  const arr = `url(#${CIR_ARROW})`
  return (
    <svg
      class="acir-graph acir-graph--circuit"
      viewBox="0 0 320 460"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Circuit decomposition: x_j weighted by a_ij (QK circuit) then transformed by W_V W_O (OV circuit)."
    >
      <ArrowDefs id={CIR_ARROW} />
      {Array.from({ length: 4 }, (_, idx) => {
        const y = 90 + idx * 60
        const label = idx === 3 ? 'i' : String(idx)
        return (
          <g>
            <rect class="acir-box acir-box--input" x={18} y={y - 14} width={44} height={28} rx={4} />
            <Fo x={18} y={y - 14} w={44} h={28} t={`x_{${label}}`} cls="acir-fo--sm" />
            <path class="acir-line" d={`M 62 ${y} L 106 ${y}`} marker-end={arr} />
          </g>
        )
      })}
      <g class="acir-aij" data-acir-circuit="qk">
        <rect class="acir-box acir-box--qk-pattern" x={110} y={80} width={56} height={250} rx={4} />
        <Fo x={110} y={82} w={56} h={26} t="a^{l,h}_{ij}" cls="acir-fo--sm" />
        {Array.from({ length: 4 }, (_, idx) => (
          <rect
            class="acir-aij-bar"
            x={122}
            y={120 + idx * 50}
            width={32}
            height={28}
            rx={2}
            style={{ opacity: 0.85 - idx * 0.18 }}
          />
        ))}
        <Fo x={110} y={303} w={56} h={22} t={math`\text{QK}`} cls="acir-fo--label acir-fo--qk-label" />
      </g>
      <g data-acir-circuit="ov">
        <g data-acir-ov-split>
          <rect class="acir-box acir-box--weight" x={200} y={165} width={56} height={40} rx={4} />
          <Fo x={200} y={165} w={56} h={40} t="W_V" />
          <rect class="acir-box acir-box--weight" x={200} y={205} width={56} height={40} rx={4} />
          <Fo x={200} y={205} w={56} h={40} t="W_O" />
        </g>
        <g data-acir-ov-rank hidden>
          <rect class="acir-box acir-box--ov-rank" x={200} y={170} width={56} height={70} rx={4} />
          <Fo x={200} y={170} w={56} h={34} t="W_V W_O" cls="acir-fo--xs" />
          <Fo
            x={200}
            y={206}
            w={56}
            h={30}
            t={math`\mathrm{rank}\le d_h`}
            cls="acir-fo--xs acir-fo--rank"
          />
        </g>
        <Fo x={196} y={252} w={64} h={22} t={math`\text{OV}`} cls="acir-fo--label acir-fo--ov-label" />
      </g>
      <path class="acir-line" d="M 166 205 L 196 205" marker-end={arr} />
      <path class="acir-line" d="M 256 205 L 276 205" marker-end={arr} />
      <rect class="acir-box acir-box--output" x={280} y={185} width={36} height={40} rx={4} />
      <Fo x={280} y={185} w={36} h={40} t={math`\mathrm{out}`} cls="acir-fo--sm" />
      <rect class="acir-sum-bg" x={30} y={380} width={260} height={56} rx={4} />
      <foreignObject x={30} y={380} width={260} height={56}>
        <div
          class="acir-fo acir-fo--display"
          dangerouslySetInnerHTML={{
            __html: tex(math`\textstyle \sum_{j \le i} a^{l,h}_{ij}\, x_j\, W^{l,h}_V W^{l,h}_O`),
          }}
        />
      </foreignObject>
    </svg>
  )
}

const PILLS: { id: 'qk' | 'ov' | 'none'; label: string; pressed: boolean; aria: string }[] = [
  {
    id: 'qk',
    label: 'QK circuit',
    pressed: false,
    aria: 'Highlight QK circuit: W_Q, W_K, attention pattern',
  },
  {
    id: 'ov',
    label: 'OV circuit',
    pressed: false,
    aria: 'Highlight OV circuit: W_V, W_O, value path',
  },
  { id: 'none', label: 'clear', pressed: true, aria: 'Clear circuit highlight' },
]

const Controls: FunctionalComponent = () => (
  <div class="acir-controls">
    <fieldset class="acir-circuit-toggle" aria-label="highlight a circuit">
      <legend>circuit highlight</legend>
      <div class="acir-pill-track" role="tablist">
        {PILLS.map(p => (
          <button
            type="button"
            class={`acir-pill acir-pill--${p.id}`}
            data-acir-pill={p.id}
            role="tab"
            aria-pressed={p.pressed ? 'true' : 'false'}
            aria-label={p.aria}
          >
            {p.label}
          </button>
        ))}
      </div>
    </fieldset>
    <label class="acir-toggle">
      <input type="checkbox" data-acir-rank-toggle aria-label="show OV as one low-rank rectangle" />
      <span>show OV as low-rank rectangle</span>
    </label>
  </div>
)

const AttentionCircuitsImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="attention-circuits" data-attention-circuits>
    <Controls />
    <div class="acir-panels">
      <section class="acir-panel" data-acir-panel="standard">
        <h4 class="acir-panel-title">standard view</h4>
        <StandardPanel />
      </section>
      <section class="acir-panel" data-acir-panel="circuit">
        <h4 class="acir-panel-title">circuit decomposition</h4>
        <CircuitPanel />
      </section>
    </div>
    {caption ? (
      <figcaption class="acir-caption">
        <MathText text={caption} mathClass="acir-math" />
      </figcaption>
    ) : null}
  </figure>
)

const AttentionCircuitsComponent = AttentionCircuitsImpl as QuartzMdxComponent<Props>
AttentionCircuitsComponent.css = style
AttentionCircuitsComponent.afterDOMLoaded = script

export const AttentionCircuits = registerMdxComponent(
  'AttentionCircuits',
  AttentionCircuitsComponent,
)

export default (() => AttentionCircuits) satisfies (opts: undefined) => QuartzMdxComponent<Props>
