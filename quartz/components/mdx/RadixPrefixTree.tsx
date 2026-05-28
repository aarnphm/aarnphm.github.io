import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/radix-prefix-tree.inline'
import style from '../styles/radixPrefixTree.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }
type Branch = 'chat_a' | 'chat_b' | 'few_shot' | 'root' | 'new'

type TreeNode = {
  id: string
  label: string
  tokens: number
  x: number
  y: number
  branch: Branch
  parent?: string
}

type Seg = { id: string; label: string; tokens: number; x: number; y: number }

type Prompt = {
  id: string
  label: string
  branch: Branch
  matchPath: string[]
  newSegments: Seg[]
  newTokens: number
  evictTarget?: 'chat_a' | 'chat_b' | 'few_shot'
}

const tex = (t: string, d = false): string =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: d,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const M: FunctionalComponent<{ t: string; d?: boolean; cls?: string }> = ({ t, d, cls }) => (
  <span class={cls} dangerouslySetInnerHTML={{ __html: tex(t, d) }} />
)

const NODES: TreeNode[] = [
  { id: 'root', label: 'root', tokens: 0, x: 90, y: 220, branch: 'root' },
  { id: 'a', label: 'chat_a', tokens: 5, x: 240, y: 90, branch: 'chat_a', parent: 'root' },
  { id: 'a1', label: 'turn1', tokens: 3, x: 410, y: 60, branch: 'chat_a', parent: 'a' },
  { id: 'a2', label: 'turn2', tokens: 3, x: 410, y: 130, branch: 'chat_a', parent: 'a' },
  { id: 'b', label: 'chat_b', tokens: 4, x: 240, y: 220, branch: 'chat_b', parent: 'root' },
  { id: 'b1', label: 'turn1', tokens: 2, x: 410, y: 220, branch: 'chat_b', parent: 'b' },
  { id: 'f', label: 'few_shot', tokens: 6, x: 240, y: 350, branch: 'few_shot', parent: 'root' },
  { id: 'f1', label: 'ex1', tokens: 4, x: 410, y: 350, branch: 'few_shot', parent: 'f' },
]

const mkSeg = (id: string, label: string, tokens: number, x: number, y: number): Seg => ({
  id,
  label,
  tokens,
  x,
  y,
})

const PROMPTS: Prompt[] = [
  {
    id: 'p1',
    label: '#1  chat_a -> turn1 + "explain"',
    branch: 'chat_a',
    matchPath: ['root', 'a', 'a1'],
    newSegments: [mkSeg('a1n', '+explain', 3, 560, 60)],
    newTokens: 3,
  },
  {
    id: 'p2',
    label: '#2  chat_b + "how are you"',
    branch: 'chat_b',
    matchPath: ['root', 'b'],
    newSegments: [mkSeg('b2', '+how', 3, 410, 270)],
    newTokens: 3,
  },
  {
    id: 'p3',
    label: '#3  few_shot + new example',
    branch: 'few_shot',
    matchPath: ['root', 'f'],
    newSegments: [mkSeg('f2', '+ex2', 5, 410, 395)],
    newTokens: 5,
  },
  {
    id: 'p4',
    label: '#4  fresh "summarise..."',
    branch: 'new',
    matchPath: ['root'],
    newSegments: [mkSeg('n1', '+summ', 4, 240, 395)],
    newTokens: 4,
    evictTarget: 'few_shot',
  },
]

const NW = 78
const NH = 28
const VW = 680
const VH = 440
const HALF_W = NW / 2
const HALF_H = NH / 2

const EDGES = NODES.filter(n => n.parent).map(n => ({ from: n.parent!, to: n.id }))
const ek = (from: string, to: string) => `${from}->${to}`
const byId = (id: string) => NODES.find(n => n.id === id)!

const curve = (fx: number, fy: number, tx: number, ty: number): string => {
  const mx = (fx + tx) / 2
  return `M ${fx} ${fy} C ${mx} ${fy}, ${mx} ${ty}, ${tx} ${ty}`
}

const LEGEND = [
  { x: 20, cls: 'match', text: '\\text{matched prefix}' },
  { x: 148, cls: 'new', text: '\\text{newly cached}' },
  { x: 268, cls: 'evict', text: '\\text{evicted}' },
] as const

const STATS = [
  { key: 'last', label: 'last request', initial: '-' },
  { key: 'cached', label: 'cached prefill', initial: '0' },
  { key: 'new', label: 'new prefill', initial: '0' },
  { key: 'hit', label: 'hit rate', initial: '-' },
  { key: 'cum', label: 'cumulative H', initial: '-' },
  { key: 'resident', label: 'resident / evicted', initial: '8 / 0' },
] as const

const labelTex = (label: string): string =>
  `\\mathrm{${label.replaceAll('_', '\\_').replaceAll('+', '{+}')}}`

const SvgMath: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  t: string
  cls?: string
  dataEdgeLabel?: string
}> = ({ x, y, w, h, t, cls, dataEdgeLabel }) => (
  <foreignObject class={cls} x={x} y={y} width={w} height={h} data-rpt-edge-label={dataEdgeLabel}>
    <div class="rpt-fo" dangerouslySetInnerHTML={{ __html: tex(t) }} />
  </foreignObject>
)

const NodeBox: FunctionalComponent<{ label: string }> = ({ label }) => (
  <>
    <rect class="rpt-node-shape" width={NW} height={NH} rx={5} />
    <SvgMath x={0} y={0} w={NW} h={NH} t={labelTex(label)} cls="rpt-node-label" />
  </>
)

const EdgeArc: FunctionalComponent<{
  from: TreeNode
  to: { x: number; y: number; tokens: number }
  edgeKey?: string
  isNew?: boolean
}> = ({ from, to, edgeKey, isNew }) => {
  const fx = from.x + HALF_W
  const tx = to.x - HALF_W
  return (
    <>
      <path
        class={isNew ? 'rpt-edge rpt-edge--new' : 'rpt-edge'}
        data-rpt-edge={edgeKey}
        d={curve(fx, from.y, tx, to.y)}
        marker-end={isNew ? 'url(#rpt-arrow)' : undefined}
      />
      <SvgMath
        cls={isNew ? 'rpt-edge-label rpt-edge-label--new' : 'rpt-edge-label'}
        x={(fx + tx) / 2 - 24}
        y={(from.y + to.y) / 2 - 17}
        w={48}
        h={18}
        t={`${isNew ? '+' : ''}${to.tokens}\\,t`}
        dataEdgeLabel={edgeKey}
      />
    </>
  )
}

const RadixPrefixTreeImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure
    class="radix-prefix-tree"
    data-radix-prefix-tree
    data-rpt-prompts={JSON.stringify(PROMPTS)}
  >
    <div class="rpt-stage">
      <aside class="rpt-prompts" aria-label="Request panel">
        <h4 class="rpt-card-title">incoming requests</h4>
        <ol class="rpt-prompt-list" role="list">
          {PROMPTS.map(p => (
            <li key={p.id}>
              <button
                type="button"
                class="rpt-prompt"
                data-rpt-prompt={p.id}
                aria-label={`Route request ${p.label} through the radix tree`}
              >
                <span class="rpt-prompt-label">{p.label}</span>
                <span class="rpt-prompt-meta">
                  branch <code>{p.branch}</code>
                </span>
              </button>
            </li>
          ))}
        </ol>
        <button type="button" class="rpt-reset" data-rpt-reset aria-label="Reset radix tree state">
          reset state
        </button>
      </aside>

      <div class="rpt-canvas">
        <svg
          class="rpt-svg"
          viewBox={`0 0 ${VW} ${VH}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="Radix prefix tree with root, three named branches (chat_a, chat_b, few_shot), and turn or example nodes; matched paths highlight in coral and newly cached nodes in olive when a prompt is routed."
        >
          <defs>
            <marker
              id="rpt-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path class="rpt-arrowhead" d="M0,0 L10,5 L0,10 z" />
            </marker>
          </defs>

          {EDGES.map(e => (
            <g key={ek(e.from, e.to)} class="rpt-edge-group">
              <EdgeArc from={byId(e.from)} to={byId(e.to)} edgeKey={ek(e.from, e.to)} />
            </g>
          ))}

          {NODES.map(n => (
            <g
              key={n.id}
              class={`rpt-node rpt-node--${n.branch}`}
              data-rpt-node={n.id}
              data-rpt-branch={n.branch}
              transform={`translate(${n.x - HALF_W}, ${n.y - HALF_H})`}
              aria-label={`tree node ${n.label}, ${n.tokens} tokens, branch ${n.branch}`}
            >
              <NodeBox label={n.label} />
            </g>
          ))}

          {PROMPTS.flatMap(p =>
            p.newSegments.map(s => (
              <g
                key={s.id}
                class="rpt-new-group"
                data-rpt-new-group={p.id}
                data-rpt-new-segment={s.id}
              >
                <EdgeArc from={byId(p.matchPath[p.matchPath.length - 1])} to={s} isNew />
                <g
                  class="rpt-node rpt-node--new"
                  transform={`translate(${s.x - HALF_W}, ${s.y - HALF_H})`}
                  aria-label={`new node ${s.label}, ${s.tokens} tokens, cached after routing`}
                >
                  <NodeBox label={s.label} />
                </g>
              </g>
            )),
          )}

          <g class="rpt-legend">
            {LEGEND.map(l => (
              <g key={l.cls}>
                <rect
                  class={`rpt-legend-chip rpt-legend-chip--${l.cls}`}
                  x={l.x}
                  y={VH - 32}
                  width={12}
                  height={12}
                  rx={2}
                />
                <SvgMath
                  x={l.x + 18}
                  y={VH - 35}
                  w={112}
                  h={18}
                  t={l.text}
                  cls="rpt-legend-label"
                />
              </g>
            ))}
          </g>

          <foreignObject x={VW - 220} y={VH - 38} width={210} height={28}>
            <div
              class="rpt-fo rpt-fo--axis"
              dangerouslySetInnerHTML={{ __html: tex('H = 1 - \\tfrac{C}{\\sum_{r}|r|}') }}
            />
          </foreignObject>
        </svg>
      </div>

      <aside class="rpt-side">
        <div class="rpt-card">
          <h4>live readout</h4>
          <dl class="rpt-stats">
            {STATS.map(s => (
              <div key={s.key}>
                <dt>{s.label}</dt>
                <dd data-rpt-stat={s.key}>{s.initial}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div class="rpt-card">
          <h4>reuse decomposition</h4>
          <M
            t="\text{reuse}(\pi,\sigma)=\bigoplus_{j=1}^{m}\text{KV}(t_{1:j}) \;\Vert\; \bigoplus_{k=1}^{|\sigma|}\text{attend}(t_{1:m+k})"
            d
            cls="rpt-math"
          />
          <p>
            The walk reuses every <M t="\text{KV}(t_{1:j})" /> tile along the matched prefix, then
            attends only over freshly added tokens.
          </p>
        </div>

        <div class="rpt-card">
          <h4>DFS lower bound</h4>
          <M t="C \ge \sum_{e \in \text{edges}(T)} |e|" d cls="rpt-math" />
          <p>
            Longest-shared-prefix-first scheduling = DFS over <M t="T" />, which touches each edge
            once. FIFO ordering thrashes; LRU keeps the hot branch resident.
          </p>
        </div>
      </aside>
    </div>

    {caption ? (
      <figcaption class="rpt-caption">
        <MathText text={caption} />
      </figcaption>
    ) : null}
  </figure>
)

const RadixPrefixTreeComponent = RadixPrefixTreeImpl as QuartzMdxComponent<Props>
RadixPrefixTreeComponent.css = style
RadixPrefixTreeComponent.afterDOMLoaded = script

export const RadixPrefixTree = registerMdxComponent('RadixPrefixTree', RadixPrefixTreeComponent)

export default (() => RadixPrefixTree) satisfies (opts: undefined) => QuartzMdxComponent<Props>
