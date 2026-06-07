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

const T: FunctionalComponent<{ t: string; d?: boolean; cls?: string }> = ({ t, d, cls }) => (
  <span class={cls} dangerouslySetInnerHTML={{ __html: tex(t, d) }} />
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
  const inX = 30
  const splitX = 95
  const projX = 150
  const smX = 220
  const outX = 290
  const mulX = smX + 33
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
      <Fo x={70} y={16} w={180} h={24} t={math`\text{softmax view}`} cls="acir-fo--axis" />
      <rect class="acir-box acir-box--input" x={inX - 22} y={210} width={44} height={40} rx={4} />
      <Fo x={inX - 22} y={210} w={44} h={40} t="x" cls="acir-fo--big" />
      <path class="acir-line" d={`M ${inX + 22} 230 L ${splitX} 230`} />
      <path
        class="acir-line"
        d={`M ${splitX} 230 L ${splitX} 100 L ${projX - 4} 100`}
        marker-end={arr}
      />
      <path class="acir-line" d={`M ${splitX} 230 L ${projX - 4} 230`} marker-end={arr} />
      <path
        class="acir-line"
        d={`M ${splitX} 230 L ${splitX} 360 L ${projX - 4} 360`}
        marker-end={arr}
      />
      <Weight x={projX} y={80} t="W_Q" circuit="qk" k="WQ" />
      <Weight x={projX} y={210} t="W_K" circuit="qk" k="WK" />
      <Weight x={projX} y={340} t="W_V" circuit="ov" k="WV" />
      <g class="acir-circuit-mark" data-acir-circuit="qk">
        <path
          class="acir-line acir-line--qk"
          d={`M ${projX + 56} 100 L ${smX - 4} 100`}
          marker-end={arr}
        />
        <path
          class="acir-line acir-line--qk"
          d={`M ${projX + 56} 230 L ${smX - 30} 230 L ${smX - 30} 150 L ${smX - 4} 150`}
          marker-end={arr}
        />
      </g>
      <rect class="acir-box acir-box--softmax" x={smX} y={75} width={66} height={100} rx={4} />
      <Fo x={smX} y={75} w={66} h={32} t={math`\frac{QK^\top}{\sqrt{d_h}}`} cls="acir-fo--sm" />
      <Fo x={smX} y={108} w={66} h={28} t={math`\operatorname{softmax}`} cls="acir-fo--xs" />
      <g class="acir-attn-grid" data-acir-attn-grid>
        {Array.from({ length: 4 }, (_, r) =>
          Array.from({ length: 4 }, (_, c) => (
            <rect
              class="acir-attn-cell"
              x={smX + 6 + c * 13}
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
        <path
          class="acir-line acir-line--ov"
          d={`M ${projX + 56} 360 L ${smX - 4} 360`}
          marker-end={arr}
        />
      </g>
      <rect class="acir-box acir-box--value" x={smX} y={340} width={56} height={40} rx={4} />
      <Fo x={smX} y={340} w={56} h={40} t="V" />
      <circle class="acir-node" cx={mulX} cy={250} r={9} />
      <Fo x={mulX - 8} y={242} w={16} h={16} t={math`\times`} cls="acir-fo--sm" />
      <path class="acir-line acir-line--qk" d={`M ${mulX} 175 L ${mulX} 241`} marker-end={arr} />
      <path
        class="acir-line acir-line--ov"
        d={`M ${smX + 28} 340 L ${smX + 28} 260 L ${mulX + 9} 250`}
        marker-end={arr}
      />
      <Weight x={outX - 28} y={230} t="W_O" circuit="ov" k="WO" />
      <path
        class="acir-line acir-line--ov"
        d={`M ${mulX + 9} 250 L ${outX - 32} 250`}
        marker-end={arr}
      />
      <path class="acir-line" d={`M ${outX} 270 L ${outX} 410`} marker-end={arr} />
      <rect class="acir-box acir-box--output" x={outX - 22} y={410} width={44} height={36} rx={4} />
      <Fo x={outX - 22} y={410} w={44} h={36} t={math`\mathrm{out}`} cls="acir-fo--sm" />
    </svg>
  )
}

const CircuitPanel: FunctionalComponent = () => {
  const tokX = 40
  const aijX = 110
  const ovX = 200
  const outX = 280
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
      <Fo x={70} y={16} w={180} h={24} t={math`\text{circuit decomposition}`} cls="acir-fo--axis" />
      {Array.from({ length: 4 }, (_, idx) => {
        const y = 90 + idx * 60
        const label = idx === 3 ? 'i' : String(idx)
        return (
          <g>
            <rect
              class="acir-box acir-box--input"
              x={tokX - 22}
              y={y - 14}
              width={44}
              height={28}
              rx={4}
            />
            <Fo x={tokX - 22} y={y - 14} w={44} h={28} t={`x_{${label}}`} cls="acir-fo--sm" />
            <path class="acir-line" d={`M ${tokX + 22} ${y} L ${aijX - 4} ${y}`} marker-end={arr} />
          </g>
        )
      })}
      <g class="acir-aij" data-acir-circuit="qk">
        <rect
          class="acir-box acir-box--qk-pattern"
          x={aijX}
          y={80}
          width={56}
          height={250}
          rx={4}
        />
        <Fo x={aijX} y={80} w={56} h={28} t="a^{l,h}_{ij}" cls="acir-fo--sm" />
        {Array.from({ length: 4 }, (_, idx) => (
          <rect
            class="acir-aij-bar"
            x={aijX + 12}
            y={120 + idx * 50}
            width={32}
            height={28}
            rx={2}
            style={{ opacity: 0.85 - idx * 0.18 }}
          />
        ))}
        <Fo
          x={aijX}
          y={310}
          w={56}
          h={24}
          t={math`\text{QK}`}
          cls="acir-fo--label acir-fo--qk-label"
        />
      </g>
      <g data-acir-circuit="ov">
        <g data-acir-ov-split>
          <rect class="acir-box acir-box--weight" x={ovX} y={155} width={56} height={40} rx={4} />
          <Fo x={ovX} y={155} w={56} h={40} t="W_V" />
          <Fo x={ovX + 20} y={200} w={16} h={16} t={math`\cdot`} cls="acir-fo--sm" />
          <rect class="acir-box acir-box--weight" x={ovX} y={215} width={56} height={40} rx={4} />
          <Fo x={ovX} y={215} w={56} h={40} t="W_O" />
        </g>
        <g data-acir-ov-rank hidden>
          <rect
            class="acir-box acir-box--ov-rank"
            x={ovX - 8}
            y={170}
            width={72}
            height={70}
            rx={4}
          />
          <Fo x={ovX - 8} y={170} w={72} h={32} t="W_V W_O" cls="acir-fo--sm" />
          <Fo
            x={ovX - 8}
            y={204}
            w={72}
            h={32}
            t={math`\mathrm{rank}\le d_h`}
            cls="acir-fo--xs acir-fo--rank"
          />
        </g>
        <Fo
          x={ovX - 4}
          y={270}
          w={72}
          h={24}
          t={math`\text{OV}`}
          cls="acir-fo--label acir-fo--ov-label"
        />
      </g>
      <path class="acir-line" d={`M ${aijX + 56} 205 L ${ovX - 4} 205`} marker-end={arr} />
      <path class="acir-line" d={`M ${ovX + 64} 205 L ${outX - 4} 205`} marker-end={arr} />
      <rect class="acir-box acir-box--output" x={outX} y={185} width={36} height={40} rx={4} />
      <Fo x={outX} y={185} w={36} h={40} t={math`\mathrm{out}`} cls="acir-fo--sm" />
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
      {PILLS.map(p => (
        <button
          type="button"
          class={`acir-pill acir-pill--${p.id}`}
          data-acir-pill={p.id}
          aria-pressed={p.pressed ? 'true' : 'false'}
          aria-label={p.aria}
        >
          {p.label}
        </button>
      ))}
    </fieldset>
    <label class="acir-toggle">
      <input type="checkbox" data-acir-rank-toggle aria-label="show OV as one low-rank rectangle" />
      <span>show OV as low-rank rectangle</span>
    </label>
  </div>
)

const Props_: FunctionalComponent = () => (
  <dl class="acir-props">
    <div class="acir-prop">
      <dt>what moves</dt>
      <dd>
        OV circuit: <T t={math`W_V W_O \in \mathbb{R}^{d \times d}`} /> with{' '}
        <T t={math`\mathrm{rank} \le d_h`} />.
      </dd>
    </div>
    <div class="acir-prop">
      <dt>where to read</dt>
      <dd>
        QK circuit: <T t={math`W_Q^{\!\top} W_K \in \mathbb{R}^{d \times d}`} /> with{' '}
        <T t={math`\mathrm{rank} \le d_h`} />.
      </dd>
    </div>
    <div class="acir-prop">
      <dt>full head</dt>
      <dd>
        <T
          d
          t={math`\mathrm{Attn}^{l,h}(X_{\le i}) = \sum_{j \le i} a^{l,h}_{ij}\, x_j\, W^{l,h}_V W^{l,h}_O`}
        />
        <T
          d
          t={math`a^{l,h}_{ij} = \operatorname{softmax}_j\!\Big(\tfrac{(xW_Q)(xW_K)^{\!\top}}{\sqrt{d_h}}\Big)`}
        />
      </dd>
    </div>
  </dl>
)

const AttentionCircuitsImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="attention-circuits" data-attention-circuits>
    <Controls />
    <div class="acir-panels">
      <section class="acir-panel" data-acir-panel="standard">
        <h4 class="acir-panel-title">standard view</h4>
        <p class="acir-panel-sub">one head, split into three projections, recombined.</p>
        <StandardPanel />
      </section>
      <section class="acir-panel" data-acir-panel="circuit">
        <h4 class="acir-panel-title">circuit decomposition</h4>
        <p class="acir-panel-sub">
          multiply through; <T t="W_V W_O" /> and the attention pattern fall out as independent
          low-rank channels.
        </p>
        <CircuitPanel />
      </section>
    </div>
    <Props_ />
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
