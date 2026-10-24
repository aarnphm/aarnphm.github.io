import {
  AG_BAR_SPAN,
  AG_EDGES,
  AG_NODE_W,
  AG_NODES,
  AG_REST,
  type AgNode,
  edgeKey,
  evaluate,
  fixed,
  pruned,
  strokeFor,
} from '../../util/attribution-graph'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/attribution-graph.inline'
import style from '../styles/attributionGraph.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

// columns 0 and 1 hold only prompt tokens, so the row labels sit above them instead of in a gutter
const PAD = 4
const NODE_W = AG_NODE_W
const X0 = PAD + NODE_W / 2
const COL_W = 88
const ROW_Y = [20, 88, 160, 232, 304, 380]
const NODE_H = 30
const TOKEN_H = 24
const VIEW_W = X0 + 6 * COL_W + NODE_W / 2 + PAD
const VIEW_H = ROW_Y[5] + TOKEN_H / 2 + PAD
// the horizontal run of an edge sits this far under its target row; incoming edges stagger by 3px
const BUS_DROP = 18
const BAR_H = 3

const DEFAULT_THRESHOLD = 0
const MAX_THRESHOLD = 0.5
const STEP = 0.05

const byId = new Map(AG_NODES.map(n => [n.id, n]))
const node = (id: string) => byId.get(id)!
const x = (n: AgNode) => X0 + n.col * COL_W
const y = (n: AgNode) => ROW_Y[n.row]
const halfH = (n: AgNode) => (n.kind === 'token' ? TOKEN_H : NODE_H) / 2

const BASE = evaluate(new Set())

/** Each edge's orthogonal path: up from the source, along the bus under the target, up into it. */
const paths = () => {
  const incoming = new Map<string, number>()
  return AG_EDGES.map(e => {
    const s = node(e.from)
    const t = node(e.to)
    const i = incoming.get(e.to) ?? 0
    incoming.set(e.to, i + 1)
    const sx = x(s) + e.lane
    const tx = x(t) + e.land
    const top = y(s) - halfH(s)
    const bottom = y(t) + halfH(t)
    const bus = bottom + BUS_DROP + 3 * i
    const d = sx === tx ? `M${sx} ${top}V${bottom}` : `M${sx} ${top}V${bus}H${tx}V${bottom}`
    return { e, d }
  })
}

const PATHS = paths()

const inputs = (id: string) => {
  const rows = AG_EDGES.filter(e => e.to === id).map(e => ({
    label: node(e.from).label,
    value: BASE.attr.get(edgeKey(e)) ?? 0,
  }))
  const rest = AG_REST[id]
  if (rest != null) rows.push({ label: id === 'logit' ? 'other paths' : 'error node', value: rest })
  return rows
}

let instance = 0

const AttributionGraphImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const id = `ag-${instance++}`
  const arrow = `${id}-arrow`
  const logit = node('logit')
  const stats = pruned(BASE, DEFAULT_THRESHOLD)
  return (
    <figure class="attribution-graph" data-attribution-graph data-ag-selected="logit">
      <div class="ag-controls" role="group" aria-label="pruning controls">
        <div class="ag-control ag-control--slider">
          <label class="ag-prune-label" for={`${id}-prune`}>
            prune <MathText text="$\lvert A_{s\to t}\rvert$" mathClass="ag-math" /> below
          </label>
          <input
            id={`${id}-prune`}
            class="ag-slider"
            type="range"
            min="0"
            max={String(MAX_THRESHOLD)}
            step={String(STEP)}
            value={String(DEFAULT_THRESHOLD)}
            data-ag-prune
            aria-valuetext={`threshold ${fixed(DEFAULT_THRESHOLD)}`}
          />
        </div>
        <output class="ag-prune-value" for={`${id}-prune`} data-ag-prune-value>
          {`${fixed(DEFAULT_THRESHOLD)} · ${stats.count}/${AG_EDGES.length} edges · ${Math.round(stats.share * 100)}% kept`}
        </output>
        <button type="button" class="ag-restore" data-ag-restore hidden>
          restore ablated nodes
        </button>
      </div>

      <div class="ag-stage">
        <div class="ag-scroll" tabindex={0} role="region" aria-label="attribution graph">
          <svg
            class="ag-svg"
            viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
            preserveAspectRatio="xMidYMid meet"
            role="group"
            aria-label={`Attribution graph: ${AG_NODES.length} nodes in six rows from the prompt tokens to the DAG logit, ${AG_EDGES.length} edges. Hover or focus a node to read its inputs; press it to ablate it.`}
          >
            <defs>
              <marker
                id={arrow}
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                markerUnits="userSpaceOnUse"
                orient="auto"
              >
                <path class="ag-arrowhead" d="M0,0 L10,5 L0,10 z" />
              </marker>
            </defs>

            <text class="ag-axis" x={PAD} y={ROW_Y[0] + 3}>
              logit
            </text>
            <text class="ag-axis" x={PAD} y={ROW_Y[2] + 3}>
              features
            </text>
            <text class="ag-axis" x={PAD} y={ROW_Y[3] + 3}>
              depth ↑
            </text>

            {PATHS.map(({ e, d }) => {
              const a = BASE.attr.get(edgeKey(e)) ?? 0
              return (
                <path
                  class="ag-edge"
                  data-ag-edge={edgeKey(e)}
                  data-ag-from={e.from}
                  data-ag-to={e.to}
                  d={d}
                  stroke-width={strokeFor(a)}
                  marker-end={`url(#${arrow})`}
                />
              )
            })}

            {AG_NODES.map(n => {
              const w = NODE_W
              const h = n.kind === 'token' ? TOKEN_H : NODE_H
              const left = x(n) - w / 2
              const top = y(n) - h / 2
              const a = BASE.act.get(n.id) ?? 0
              const ablatable = n.kind !== 'logit'
              // a long label breaks at its space; the box keeps one width
              const lines = n.label.length > 10 ? n.label.split(' ') : [n.label]
              return (
                <g
                  class={`ag-node ag-node--${n.kind}`}
                  data-ag-node={n.id}
                  role="button"
                  tabindex={0}
                  aria-pressed={ablatable ? 'false' : undefined}
                  aria-disabled={ablatable ? undefined : 'true'}
                  aria-label={`${n.label}, ${n.note}, activation ${fixed(a)}`}
                >
                  <rect class="ag-box" x={left} y={top} width={w} height={h} />
                  {n.kind === 'token' ? null : (
                    <rect
                      class="ag-bar"
                      data-ag-bar={n.id}
                      x={left + 2}
                      y={top + h - BAR_H - 2}
                      width={AG_BAR_SPAN * a}
                      height={BAR_H}
                    />
                  )}
                  {lines.map((line, i) => (
                    <text
                      class="ag-label"
                      x={x(n)}
                      y={y(n) + (n.kind === 'token' ? 3.5 : 1) + (i - (lines.length - 1) / 2) * 12}
                      text-anchor="middle"
                    >
                      {line}
                    </text>
                  ))}
                </g>
              )
            })}
          </svg>
        </div>

        <dl class="ag-readout" aria-live="polite">
          <div>
            <dt>node</dt>
            <dd data-ag-read-node>{`${logit.label} ${logit.note}`}</dd>
          </div>
          <div>
            <dt>activation</dt>
            <dd data-ag-read-act>{fixed(BASE.act.get('logit') ?? 0)}</dd>
          </div>
          <div class="ag-readout-head">
            <dt>
              input <MathText text="$s$" mathClass="ag-math" />
            </dt>
            <dd>
              <MathText text="$A_{s\to t}$" mathClass="ag-math" />
            </dd>
          </div>
          <div class="ag-readout-rows" data-ag-rows>
            {inputs('logit').map(row => (
              <div>
                <dt>{row.label}</dt>
                <dd>{fixed(row.value)}</dd>
              </div>
            ))}
          </div>
        </dl>
      </div>

      {caption ? (
        <figcaption class="ag-caption">
          <MathText text={caption} mathClass="ag-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const AttributionGraphComponent = AttributionGraphImpl as QuartzMdxComponent<Props>
AttributionGraphComponent.css = style
AttributionGraphComponent.afterDOMLoaded = script

export const AttributionGraph = registerMdxComponent('AttributionGraph', AttributionGraphComponent)
