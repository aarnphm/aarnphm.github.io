import { MathText, renderInlineMath } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/flash-data-flow.inline'
import style from '../styles/flashDataFlow.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

type Mode = { id: string; label: string; hbmTex: string; chipTex: string; bullets: string[] }

const MODES: readonly Mode[] = [
  {
    id: 'fa1',
    label: 'FA-1',
    hbmTex: '\\text{HBM2e}\\,{\\cdot}\\,\\text{1.55 TB/s}\\,{\\cdot}\\,\\text{40 GB}',
    chipTex: '\\text{A100}\\,{\\cdot}\\,\\text{SRAM 192 KB/SM}',
    bullets: [
      '$K,V$ outer, $Q$ inner',
      '$O_i$ read-modify-written $\\times T_c$',
      'serial: no overlap',
    ],
  },
  {
    id: 'fa2',
    label: 'FA-2',
    hbmTex: '\\text{HBM2e}\\,{\\cdot}\\,\\text{1.55 TB/s}\\,{\\cdot}\\,\\text{40 GB}',
    chipTex: '\\text{A100}\\,{\\cdot}\\,\\text{SRAM 192 KB/SM}',
    bullets: ['$Q$ outer, $K,V$ inner', '$O_i$ written once', 'split-$Q$, parallel across SMs'],
  },
  {
    id: 'fa3',
    label: 'FA-3',
    hbmTex: '\\text{HBM3}\\,{\\cdot}\\,\\text{3.35 TB/s}\\,{\\cdot}\\,\\text{80 GB}',
    chipTex: '\\text{H100}\\,{\\cdot}\\,\\text{SMEM 228 KB/SM}',
    bullets: [
      'TMA $\\to$ SMEM ring $\\to$ WGMMA',
      'load $t{+}1$ overlaps compute $t$',
      'Hopper warp-specialised',
    ],
  },
  {
    id: 'fa4',
    label: 'FA-4',
    hbmTex: '\\text{HBM3e}\\,{\\cdot}\\,\\text{8.0 TB/s}\\,{\\cdot}\\,\\text{192 GB}',
    chipTex: '\\text{B200}\\,{\\cdot}\\,\\text{SMEM 228 + TMEM 256 KB/SM}',
    bullets: [
      'async $\\texttt{tcgen05}$ MMA, TMEM accum.',
      'MMA $t$ overlaps softmax $t{+}1$',
      '$\\exp$ = 3-FMA poly on FMA units',
    ],
  },
] as const

const INITIAL = 'fa1'
const CELLS = ['Q', 'K', 'V', 'O'] as const
const CELL_CY = [100, 146, 192, 238] as const

const Fo = ({
  cx,
  cy,
  w,
  h = 16,
  tex,
  cls = '',
}: {
  cx: number
  cy: number
  w: number
  h?: number
  tex: string
  cls?: string
}) => (
  <foreignObject x={cx - w / 2} y={cy - h / 2} width={w} height={h}>
    <div class={`fdf-fo ${cls}`} dangerouslySetInnerHTML={{ __html: renderInlineMath(tex) }} />
  </foreignObject>
)

const FoL = ({
  x,
  cy,
  w,
  h = 16,
  tex,
  cls = '',
}: {
  x: number
  cy: number
  w: number
  h?: number
  tex: string
  cls?: string
}) => (
  <foreignObject x={x} y={cy - h / 2} width={w} height={h}>
    <div
      class={`fdf-fo fdf-fo--start ${cls}`}
      dangerouslySetInnerHTML={{ __html: renderInlineMath(tex) }}
    />
  </foreignObject>
)

const HBMTier = () => (
  <g class="fdf-hbm">
    <rect class="fdf-tier fdf-tier--hbm" x={28} y={64} width={130} height={212} rx={7} />
    <Fo cx={93} cy={50} w={70} tex="\mathrm{HBM}" cls="fdf-fo--big" />
    {CELL_CY.map(cy => (
      <rect class="fdf-cell" x={42} y={cy - 16} width={102} height={32} rx={3} />
    ))}
    {CELLS.map((c, i) => (
      <Fo cx={93} cy={CELL_CY[i]} w={40} tex={c} cls="fdf-fo--cell" />
    ))}
  </g>
)

const Kernel = ({ x, y }: { x: number; y: number }) => (
  <g class="fdf-kernel">
    <rect class="fdf-chip fdf-chip--mma" x={x} y={y} width={84} height={30} rx={4} />
    <Fo cx={x + 42} cy={y + 15} w={80} tex="S = QK^\top" cls="fdf-fo--chip" />
    <line
      class="fdf-flow-line"
      x1={x + 84}
      y1={y + 15}
      x2={x + 104}
      y2={y + 15}
      marker-end="url(#fdf-ah)"
    />
    <rect class="fdf-chip fdf-chip--sm" x={x + 106} y={y} width={44} height={30} rx={4} />
    <Fo cx={x + 128} cy={y + 15} w={40} tex="P" cls="fdf-fo--chip" />
    <line
      class="fdf-flow-line"
      x1={x + 150}
      y1={y + 15}
      x2={x + 170}
      y2={y + 15}
      marker-end="url(#fdf-ah)"
    />
    <rect class="fdf-chip fdf-chip--mma" x={x + 172} y={y} width={96} height={30} rx={4} />
    <Fo cx={x + 220} cy={y + 15} w={92} tex="O \mathrel{{+}{=}} PV" cls="fdf-fo--chip" />
    <rect class="fdf-chip fdf-chip--stat" x={x} y={y + 42} width={84} height={26} rx={4} />
    <Fo cx={x + 42} cy={y + 55} w={80} tex="m_i,\,\ell_i" cls="fdf-fo--chip fdf-fo--sm" />
    <Fo cx={x + 150} cy={y + 55} w={70} tex="\mathrm{softmax}" cls="fdf-fo--tiny" />
  </g>
)

const Spec = ({ mode }: { mode: Mode }) => (
  <g class="fdf-spec">
    <Fo cx={93} cy={298} w={172} tex={mode.hbmTex} cls="fdf-fo--spec" />
    <Fo cx={500} cy={298} w={330} tex={mode.chipTex} cls="fdf-fo--spec" />
  </g>
)

const SceneFA1 = () => (
  <g class="fdf-scene fdf-scene--fa1">
    <title>FlashAttention-1 serial dataflow</title>
    <HBMTier />
    <rect class="fdf-tier fdf-tier--chip" x={312} y={132} width={296} height={96} rx={7} />
    <FoL
      x={318}
      cy={122}
      w={190}
      tex="\text{on-chip}\,{\cdot}\,\text{one SM}"
      cls="fdf-fo--label"
    />
    <Fo cx={232} cy={150} w={96} tex="Q_i\,K_j\,V_j" cls="fdf-fo--flow" />
    <line class="fdf-flow-line" x1={158} y1={165} x2={322} y2={165} marker-end="url(#fdf-ah)" />
    <g class="fdf-lane fdf-lane--s-load">
      <rect class="fdf-pkt fdf-pkt--load" x={158} y={158} width={18} height={14} />
    </g>
    <Kernel x={324} y={150} />
    <line class="fdf-roundtrip" x1={158} y1={210} x2={312} y2={210} marker-end="url(#fdf-ah)" />
    <line
      class="fdf-roundtrip fdf-roundtrip--out"
      x1={312}
      y1={222}
      x2={158}
      y2={222}
      marker-end="url(#fdf-ah)"
    />
    <Fo
      cx={250}
      cy={244}
      w={260}
      tex="\text{read-modify-write }O_i, m_i, \ell_i\ {\times}T_c"
      cls="fdf-fo--flow fdf-fo--warn"
    />
    <g class="fdf-lane fdf-lane--s-store">
      <rect class="fdf-pkt fdf-pkt--store" x={282} y={215} width={18} height={14} />
    </g>
    <Spec mode={MODES[0]} />
  </g>
)

const FA2_LANES = [110, 164, 218] as const

const SceneFA2 = () => (
  <g class="fdf-scene fdf-scene--fa2">
    <title>FlashAttention-2 parallel dataflow</title>
    <HBMTier />
    <Fo cx={246} cy={84} w={130} tex="K_j,V_j\ \text{stream in}" cls="fdf-fo--flow" />
    {FA2_LANES.map((cy, i) => (
      <g class={`fdf-sm fdf-sm--${i}`}>
        <line class="fdf-flow-line" x1={158} y1={cy} x2={332} y2={cy} marker-end="url(#fdf-ah)" />
        <rect
          class="fdf-chip fdf-chip--sm fdf-sm-box"
          x={334}
          y={cy - 19}
          width={252}
          height={38}
          rx={5}
        />
        <Fo cx={398} cy={cy} w={110} tex="Q_i\ \text{resident}" cls="fdf-fo--chip" />
        <rect
          class="fdf-chip fdf-chip--mma fdf-sm-glyph"
          x={474}
          y={cy - 13}
          width={104}
          height={26}
          rx={4}
        />
        <Fo cx={526} cy={cy} w={96} tex="S\,{\cdot}\,P\,{\cdot}\,O" cls="fdf-fo--chip fdf-fo--sm" />
        <g class="fdf-lane fdf-lane--stream">
          <rect class="fdf-pkt fdf-pkt--load" x={158} y={cy - 7} width={16} height={14} />
        </g>
        <g class="fdf-lane fdf-lane--once">
          <rect class="fdf-pkt fdf-pkt--store" x={314} y={cy - 7} width={16} height={14} />
        </g>
      </g>
    ))}
    <Fo
      cx={262}
      cy={250}
      w={300}
      tex="O_i\ \text{written once}\,{\cdot}\,\text{SMs in parallel}"
      cls="fdf-fo--flow fdf-fo--ok"
    />
    <Spec mode={MODES[1]} />
  </g>
)

const RING_SLOTS = [128, 158, 188] as const

const SceneFA3 = () => (
  <g class="fdf-scene fdf-scene--fa3">
    <title>FlashAttention-3 asynchronous dataflow</title>
    <HBMTier />
    <rect class="fdf-tier fdf-tier--chip" x={296} y={90} width={420} height={156} rx={7} />
    <Fo
      cx={236}
      cy={108}
      w={124}
      tex="\text{producer}\,{\cdot}\,\text{TMA}"
      cls="fdf-fo--flow fdf-fo--prod"
    />
    <line class="fdf-flow-line" x1={158} y1={122} x2={330} y2={122} marker-end="url(#fdf-ah)" />
    <g class="fdf-lane fdf-lane--tma">
      <rect class="fdf-pkt fdf-pkt--load" x={158} y={115} width={16} height={14} />
    </g>
    <g class="fdf-lane fdf-lane--tma fdf-lane--tma2">
      <rect class="fdf-pkt fdf-pkt--load" x={158} y={115} width={16} height={14} />
    </g>
    <rect class="fdf-tier fdf-tier--ring" x={332} y={106} width={70} height={112} rx={6} />
    <Fo cx={367} cy={98} w={64} tex="\mathrm{SMEM}" cls="fdf-fo--big fdf-fo--ring" />
    {RING_SLOTS.map((cy, i) => (
      <rect class={`fdf-slot fdf-slot--${i}`} x={344} y={cy - 11} width={46} height={22} rx={3} />
    ))}
    <line class="fdf-flow-line" x1={402} y1={162} x2={424} y2={162} marker-end="url(#fdf-ah)" />
    <Fo
      cx={540}
      cy={108}
      w={156}
      tex="\text{consumer}\,{\cdot}\,\text{WGMMA}"
      cls="fdf-fo--flow fdf-fo--cons"
    />
    <Kernel x={426} y={148} />
    <Spec mode={MODES[2]} />
  </g>
)

const SceneFA4 = () => (
  <g class="fdf-scene fdf-scene--fa4">
    <title>FlashAttention-4 asynchronous-MMA dataflow</title>
    <HBMTier />
    <rect class="fdf-tier fdf-tier--chip" x={296} y={90} width={400} height={170} rx={7} />
    <rect class="fdf-chip fdf-chip--sched" x={300} y={98} width={132} height={30} rx={5} />
    <Fo cx={366} cy={113} w={124} tex="\text{scheduler}" cls="fdf-fo--chip" />
    <line class="fdf-flow-line" x1={158} y1={113} x2={298} y2={113} marker-end="url(#fdf-ah)" />
    <g class="fdf-lane fdf-lane--sched">
      <rect class="fdf-pkt fdf-pkt--load" x={158} y={106} width={16} height={14} />
    </g>
    <line
      class="fdf-flow-line fdf-flow-line--dispatch"
      x1={366}
      y1={128}
      x2={366}
      y2={160}
      marker-end="url(#fdf-ah)"
    />
    <rect
      class="fdf-chip fdf-chip--mma fdf-pulse--mma"
      x={300}
      y={162}
      width={150}
      height={40}
      rx={5}
    />
    <Fo cx={375} cy={176} w={140} tex="\texttt{tcgen05}\ \text{MMA}" cls="fdf-fo--chip" />
    <Fo cx={375} cy={192} w={120} tex="\text{tile }t" cls="fdf-fo--tiny" />
    <rect
      class="fdf-chip fdf-chip--sm fdf-pulse--softmax"
      x={300}
      y={214}
      width={150}
      height={40}
      rx={5}
    />
    <Fo cx={362} cy={228} w={130} tex="\text{CUDA cores}" cls="fdf-fo--chip" />
    <Fo
      cx={362}
      cy={244}
      w={150}
      tex="\mathrm{softmax}\,{\cdot}\,\text{tile }t{+}1"
      cls="fdf-fo--tiny"
    />
    <rect
      class="fdf-chip fdf-chip--poly fdf-pulse--softmax"
      x={470}
      y={220}
      width={112}
      height={28}
      rx={4}
    />
    <Fo
      cx={526}
      cy={234}
      w={108}
      tex="\text{poly-exp}\,{\cdot}\,\text{3 FMA}"
      cls="fdf-fo--chip fdf-fo--sm"
    />
    <rect class="fdf-tier fdf-tier--tmem" x={470} y={150} width={214} height={52} rx={6} />
    <FoL x={478} cy={142} w={90} tex="\mathrm{TMEM}" cls="fdf-fo--label" />
    <Fo
      cx={577}
      cy={176}
      w={184}
      tex="S, P, O\ \text{accumulators}"
      cls="fdf-fo--chip fdf-fo--sm"
    />
    <line class="fdf-flow-line" x1={450} y1={178} x2={468} y2={178} marker-end="url(#fdf-ah)" />
    <line
      class="fdf-flow-line fdf-flow-line--fb"
      x1={520}
      y1={202}
      x2={452}
      y2={218}
      marker-end="url(#fdf-ah)"
    />
    <line class="fdf-flow-line" x1={450} y1={234} x2={468} y2={234} marker-end="url(#fdf-ah)" />
    <Spec mode={MODES[3]} />
  </g>
)

const FlashDataFlowImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  return (
    <figure class="flash-data-flow" data-flash-data-flow data-fdf-mode={INITIAL}>
      <header class="fdf-head">
        <div class="fdf-tablist" role="tablist" aria-label="FlashAttention generation">
          {MODES.map(m => (
            <button
              type="button"
              class="fdf-tab"
              data-fdf-tab={m.id}
              role="tab"
              aria-selected={m.id === INITIAL ? 'true' : 'false'}
              tabindex={m.id === INITIAL ? 0 : -1}
            >
              {m.label}
            </button>
          ))}
        </div>
      </header>

      {MODES.map(m => (
        <div class={`fdf-sub fdf-sub--${m.id}`}>
          <ul class="fdf-sub-list">
            {m.bullets.map(b => (
              <li>
                <MathText text={b} mathClass="fdf-math" />
              </li>
            ))}
          </ul>
        </div>
      ))}

      <svg
        class="fdf-svg"
        viewBox="0 0 720 320"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="HBM to on-chip data movement across FlashAttention generations"
      >
        <defs>
          <marker
            id="fdf-ah"
            viewBox="0 0 8 8"
            refX={6}
            refY={4}
            markerWidth={5.5}
            markerHeight={5.5}
            orient="auto-start-reverse"
          >
            <path class="fdf-ah" d="M0 0 L8 4 L0 8 Z" />
          </marker>
        </defs>
        <SceneFA1 />
        <SceneFA2 />
        <SceneFA3 />
        <SceneFA4 />
      </svg>

      {caption ? (
        <figcaption class="fdf-caption">
          <MathText text={caption} mathClass="fdf-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const FlashDataFlowComponent = FlashDataFlowImpl as QuartzMdxComponent<Props>
FlashDataFlowComponent.css = style
FlashDataFlowComponent.afterDOMLoaded = script

export const FlashDataFlow = registerMdxComponent('FlashDataFlow', FlashDataFlowComponent)

export default (() => FlashDataFlow) satisfies (opts: undefined) => QuartzMdxComponent<Props>
