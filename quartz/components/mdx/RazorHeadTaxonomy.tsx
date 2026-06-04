import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
import style from '../styles/razorHeadTaxonomy.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const VIEW_W = 620
const VIEW_H = 248
const CELL_W = 44
const CELL_H = 44
const ROW_Y = 132
const math = String.raw

const tex = (t: string): string =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: false,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const Fo: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  t: string
  cls?: string
}> = ({ x, y, w, h, t, cls }) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div class={`rht-fo ${cls ?? ''}`.trim()} dangerouslySetInnerHTML={{ __html: tex(t) }} />
  </foreignObject>
)

type Kind = 'ctx' | 'echo' | 'induction' | 'query' | 'predict'
type Cell = { x: number; label: string; kind: Kind; sub?: string; subAlign?: 'start' | 'end' }

const CELLS: Cell[] = [
  { x: 40, label: 'A', kind: 'ctx' },
  { x: 96, label: 'B', kind: 'echo', sub: '\\text{echo token}', subAlign: 'end' },
  { x: 152, label: 'C', kind: 'induction', sub: '\\text{induction token}', subAlign: 'start' },
  { x: 264, label: 'A', kind: 'ctx' },
  { x: 320, label: 'B', kind: 'query', sub: '\\text{query } q_m' },
  { x: 432, label: 'C', kind: 'predict', sub: '\\text{copied out}' },
]

const cx = (c: Cell): number => c.x + CELL_W / 2
const QUERY = CELLS[4]
const ECHO = CELLS[1]
const INDUCTION = CELLS[2]
const PREDICT = CELLS[5]

const ARC_TOP = ROW_Y - 4
const SUB_W = 112
const echoArc = `M ${cx(QUERY)} ${ARC_TOP} Q ${(cx(QUERY) + cx(ECHO)) / 2} 34 ${cx(ECHO)} ${ARC_TOP}`
const inductionArc = `M ${cx(QUERY)} ${ARC_TOP} Q ${(cx(QUERY) + cx(INDUCTION)) / 2} 72 ${cx(INDUCTION)} ${ARC_TOP}`
const copyArc = `M ${QUERY.x + CELL_W} ${ROW_Y + CELL_H / 2} L ${PREDICT.x - 6} ${ROW_Y + CELL_H / 2}`

const subX = (c: Cell): number => {
  if (c.subAlign === 'end') return c.x + CELL_W - SUB_W
  if (c.subAlign === 'start') return c.x
  return cx(c) - SUB_W / 2
}
const subCls = (c: Cell): string => `rht-fo--sub${c.subAlign ? ` rht-fo--sub-${c.subAlign}` : ''}`

const RazorHeadTaxonomyImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="razor-head-taxonomy" data-razor-head-taxonomy>
    <div class="rht-stage">
      <svg
        class="rht-graph"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Token strip A B C ellipsis A B with the second B as the current query. An echo head attends back to the earlier identical B; an induction head attends back to the C that followed the earlier B, copying it out as the next token."
      >
        <defs>
          <marker
            id="rht-arrow-echo"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6.5"
            markerHeight="6.5"
            orient="auto-start-reverse"
          >
            <path class="rht-arrowhead rht-arrowhead--echo" d="M0,0 L10,5 L0,10 z" />
          </marker>
          <marker
            id="rht-arrow-induction"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6.5"
            markerHeight="6.5"
            orient="auto-start-reverse"
          >
            <path class="rht-arrowhead rht-arrowhead--induction" d="M0,0 L10,5 L0,10 z" />
          </marker>
          <marker
            id="rht-arrow-copy"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path class="rht-arrowhead rht-arrowhead--copy" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        <Fo x={40} y={20} w={260} h={20} t={math`\text{repeated context}`} cls="rht-fo--axis" />

        <g
          class="rht-arc-group rht-arc-group--echo"
          tabindex={0}
          role="img"
          aria-label="echo head: attends to the previous identical token, detecting the repeat"
        >
          <path class="rht-arc-hit" d={echoArc} />
          <path class="rht-arc rht-arc--echo" d={echoArc} marker-end="url(#rht-arrow-echo)" />
        </g>
        <g
          class="rht-arc-group rht-arc-group--induction"
          tabindex={0}
          role="img"
          aria-label="induction head: attends to the token after the repeat, then copies it out as the prediction"
        >
          <path class="rht-arc-hit" d={inductionArc} />
          <path
            class="rht-arc rht-arc--induction"
            d={inductionArc}
            marker-end="url(#rht-arrow-induction)"
          />
        </g>

        <path class="rht-arc rht-arc--copy" d={copyArc} marker-end="url(#rht-arrow-copy)" />

        <Fo x={208} y={ROW_Y + 4} w={48} h={CELL_H - 8} t={math`\cdots`} cls="rht-fo--ellipsis" />

        {CELLS.map(c => (
          <g class={`rht-cell rht-cell--${c.kind}`}>
            <rect class="rht-cell-box" x={c.x} y={ROW_Y} width={CELL_W} height={CELL_H} rx={4} />
            <Fo x={c.x} y={ROW_Y} w={CELL_W} h={CELL_H} t={c.label} cls="rht-fo--token" />
            {c.sub ? (
              <Fo x={subX(c)} y={ROW_Y + CELL_H + 8} w={SUB_W} h={16} t={c.sub} cls={subCls(c)} />
            ) : null}
          </g>
        ))}
      </svg>

      <div class="rht-tip rht-tip--echo" role="tooltip">
        attends to the previous identical token. detects the repeat.
      </div>
      <div class="rht-tip rht-tip--induction" role="tooltip">
        attends to the token after that repeat, then copies it out as the prediction.
      </div>
    </div>

    <div class="rht-legend">
      <span class="rht-legend-item rht-legend-item--echo">
        <span class="rht-swatch rht-swatch--echo" aria-hidden="true" />
        <span
          class="rht-legend-name"
          dangerouslySetInnerHTML={{ __html: tex(math`\text{echo head}`) }}
        />
      </span>
      <span class="rht-legend-item rht-legend-item--induction">
        <span class="rht-swatch rht-swatch--induction" aria-hidden="true" />
        <span
          class="rht-legend-name"
          dangerouslySetInnerHTML={{ __html: tex(math`\text{induction head}`) }}
        />
      </span>
    </div>

    {caption ? (
      <figcaption class="rht-caption">
        <MathText text={caption} mathClass="rht-math" />
      </figcaption>
    ) : null}
  </figure>
)

const RazorHeadTaxonomyComponent = RazorHeadTaxonomyImpl as QuartzMdxComponent<Props>
RazorHeadTaxonomyComponent.css = style

export const RazorHeadTaxonomy = registerMdxComponent(
  'RazorHeadTaxonomy',
  RazorHeadTaxonomyComponent,
)

export default (() => RazorHeadTaxonomy) satisfies (opts: undefined) => QuartzMdxComponent<Props>
