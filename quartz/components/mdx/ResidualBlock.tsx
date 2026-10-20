import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { renderFigureLabel } from '../../util/figure-label'
import { MathText } from '../../util/math-text'
import style from '../styles/residualBlock.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const renderMath = (tex: string): string =>
  renderFigureLabel(tex, m =>
    katex.renderToString(m, {
      ...katexOptions,
      displayMode: false,
      output: 'html',
      macros: customMacros,
      strict: false,
      throwOnError: false,
    }),
  )

const MathFO: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  tex: string
  align?: 'center' | 'start' | 'end'
}> = ({ x, y, w, h, tex, align = 'center' }) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div class={`rb-fo rb-fo--${align}`} dangerouslySetInnerHTML={{ __html: renderMath(tex) }} />
  </foreignObject>
)

// One block of He et al.'s Figure 2: two weight layers with a ReLU between them form F,
// the identity shortcut joins at the sum, and a second ReLU follows the sum.
const CX = 120
const BOX_W = 128
const RELU_W = 72
const H = 28
const NODES = [
  { y: 8, w: 48, tex: 'x', kind: 'io' },
  { y: 62, w: BOX_W, tex: '\\text{weight layer}', kind: 'op' },
  { y: 116, w: RELU_W, tex: '\\operatorname{ReLU}', kind: 'act' },
  { y: 170, w: BOX_W, tex: '\\text{weight layer}', kind: 'op' },
] as const
const SUM_Y = 248
const SUM_R = 11
const RELU2_Y = 290
const OUT_Y = 344
const SKIP_X = 228
const VIEW_W = 300
const VIEW_H = OUT_Y + H + 8

let instance = 0

const ResidualBlockImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const arrow = `rb-arrow-${++instance}`
  const bottom = (i: number) => NODES[i].y + H
  const edges = [
    [bottom(0), NODES[1].y],
    [bottom(1), NODES[2].y],
    [bottom(2), NODES[3].y],
    [bottom(3), SUM_Y - SUM_R],
    [SUM_Y + SUM_R, RELU2_Y],
    [RELU2_Y + H, OUT_Y],
  ]
  const inMid = NODES[0].y + H / 2

  return (
    <figure class="residual-block">
      <svg
        class="rb-svg"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        role="img"
        aria-label="Residual block: F(x) is added to x, then passed through ReLU."
      >
        <defs>
          <marker
            id={arrow}
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path class="rb-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        {edges.map(([y1, y2]) => (
          <line class="rb-edge" x1={CX} x2={CX} y1={y1} y2={y2} marker-end={`url(#${arrow})`} />
        ))}
        <path
          class="rb-skip"
          d={`M${CX + NODES[0].w / 2} ${inMid}H${SKIP_X}V${SUM_Y}H${CX + SUM_R}`}
          marker-end={`url(#${arrow})`}
        />

        {NODES.map(n => (
          <g>
            <rect
              class={`rb-node rb-node--${n.kind}`}
              x={CX - n.w / 2}
              y={n.y}
              width={n.w}
              height={H}
            />
            <MathFO x={CX - n.w / 2} y={n.y} w={n.w} h={H} tex={n.tex} />
          </g>
        ))}

        <MathFO x={CX - 92} y={bottom(3) + 8} w={84} h={20} tex="F(x)" align="end" />
        <MathFO x={SKIP_X + 8} y={(inMid + SUM_Y) / 2 - 20} w={64} h={20} tex="x" align="start" />
        <MathFO
          x={SKIP_X + 8}
          y={(inMid + SUM_Y) / 2}
          w={64}
          h={20}
          tex="\text{identity}"
          align="start"
        />

        <circle class="rb-sum" cx={CX} cy={SUM_Y} r={SUM_R} />
        <line class="rb-glyph" x1={CX - 5} x2={CX + 5} y1={SUM_Y} y2={SUM_Y} />
        <line class="rb-glyph" x1={CX} x2={CX} y1={SUM_Y - 5} y2={SUM_Y + 5} />
        <MathFO x={CX + 8} y={SUM_Y + SUM_R + 2} w={96} h={20} tex="F(x)+x" align="start" />

        <rect
          class="rb-node rb-node--act"
          x={CX - RELU_W / 2}
          y={RELU2_Y}
          width={RELU_W}
          height={H}
        />
        <MathFO x={CX - RELU_W / 2} y={RELU2_Y} w={RELU_W} h={H} tex="\operatorname{ReLU}" />
        <rect class="rb-node rb-node--io" x={CX - 24} y={OUT_Y} width={48} height={H} />
        <MathFO x={CX - 24} y={OUT_Y} w={48} h={H} tex="y" />
      </svg>
      {caption ? (
        <figcaption class="rb-caption">
          <MathText text={caption} mathClass="rb-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const ResidualBlockComponent = ResidualBlockImpl as QuartzMdxComponent<Props>
ResidualBlockComponent.css = style

export const ResidualBlock = registerMdxComponent('ResidualBlock', ResidualBlockComponent)

export default (() => ResidualBlock) satisfies (opts: undefined) => QuartzMdxComponent<Props>
