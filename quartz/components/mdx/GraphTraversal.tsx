import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/graph-traversal.inline'
import style from '../styles/graphTraversal.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Mode = 'dfs' | 'bfs'
type Props = { caption?: string; mode?: Mode; start?: number }

// The example graph of the course slides, n0 to n9; the drawn cycle n0, n7, n2, n3 is a square.
const POS: readonly (readonly [number, number])[] = [
  [330, 116],
  [230, 116],
  [430, 206],
  [330, 206],
  [330, 26],
  [130, 116],
  [30, 26],
  [430, 116],
  [230, 206],
  [230, 26],
]
const EDGES: readonly (readonly [number, number])[] = [
  [6, 9],
  [5, 6],
  [5, 9],
  [0, 9],
  [0, 4],
  [1, 8],
  [0, 7],
  [2, 7],
  [2, 3],
  [0, 3],
]
const N = POS.length
// Adj(n) lists neighbours in ascending index, as the note inspects its example alphabetically.
const ADJ: number[][] = Array.from({ length: N }, (_, u) =>
  EDGES.flatMap(([a, b]) => (a === u ? [b] : b === u ? [a] : [])).sort((a, b) => a - b),
)
const edgeIndex = (u: number, v: number) =>
  EDGES.findIndex(([a, b]) => (a === u && b === v) || (a === v && b === u))

// Node codes: u unvisited, o open (on the stack or in the queue), f finished, c current.
// Edge codes: u not inspected, s inspected without a call, t tree edge, c inspected now.
type Step = { nodes: string; edges: string; store: number[]; say: string }
type Run = { steps: Step[]; rows: [string, string[], number[]][]; reached: number }

const n = (i: number) => `n${i}`

const snapshot = (
  node: string[],
  edge: string[],
  store: number[],
  say: string,
  current: number,
  edgeNow = -1,
): Step => ({
  nodes: node.map((s, i) => (i === current ? 'c' : s)).join(''),
  edges: edge.map((s, i) => (i === edgeNow ? 'c' : s)).join(''),
  store: [...store],
  say,
})

const dfs = (s: number): Run => {
  const node = Array<string>(N).fill('u')
  const edge = Array<string>(EDGES.length).fill('u')
  const d = Array<number>(N).fill(0)
  const f = Array<number>(N).fill(0)
  const parent = Array<number>(N).fill(-1)
  const dAt = Array<number>(N).fill(Infinity)
  const fAt = Array<number>(N).fill(Infinity)
  const stack: number[] = []
  const steps: Step[] = [
    snapshot(node, edge, stack, `all vertices unmarked; call DFS-R(${n(s)})`, -1),
  ]
  let time = 0
  const visit = (v: number, from: number) => {
    node[v] = 'o'
    stack.push(v)
    d[v] = ++time
    parent[v] = from
    dAt[v] = steps.length
    const e = from < 0 ? -1 : edgeIndex(from, v)
    if (e >= 0) edge[e] = 't'
    steps.push(
      snapshot(
        node,
        edge,
        stack,
        from < 0
          ? `mark ${n(v)}; d = ${d[v]}`
          : `${n(from)} finds ${n(v)} unmarked: call DFS-R(${n(v)}), d = ${d[v]}`,
        v,
        e,
      ),
    )
    for (const m of ADJ[v]) {
      if (node[m] === 'u') {
        visit(m, v)
        continue
      }
      const k = edgeIndex(v, m)
      if (edge[k] === 'u') edge[k] = 's'
      steps.push(snapshot(node, edge, stack, `${n(v)} scans ${n(m)}: already marked`, v, k))
    }
    node[v] = 'f'
    stack.pop()
    f[v] = ++time
    fAt[v] = steps.length
    steps.push(
      snapshot(
        node,
        edge,
        stack,
        `Adj(${n(v)}) done: DFS-R(${n(v)}) returns, f = ${f[v]}`,
        stack.at(-1) ?? -1,
      ),
    )
  }
  visit(s, -1)
  return {
    steps,
    rows: [
      ['d', d.map(String), dAt],
      ['f', f.map(String), fAt],
      ['parent', parent.map(p => (p < 0 ? '' : n(p))), dAt],
    ],
    reached: node.filter(x => x !== 'u').length,
  }
}

const bfs = (s: number): Run => {
  const node = Array<string>(N).fill('u')
  const edge = Array<string>(EDGES.length).fill('u')
  const order = Array<number>(N).fill(0)
  const dist = Array<number>(N).fill(0)
  const parent = Array<number>(N).fill(-1)
  const at = Array<number>(N).fill(Infinity)
  const queue: number[] = [s]
  let marked = 1
  node[s] = 'o'
  order[s] = 1
  at[s] = 0
  const steps: Step[] = [snapshot(node, edge, queue, `mark ${n(s)}; Q holds only ${n(s)}`, -1)]
  while (queue.length > 0) {
    const v = queue.shift()!
    steps.push(snapshot(node, edge, queue, `dequeue ${n(v)}; dist = ${dist[v]}`, v))
    for (const m of ADJ[v]) {
      const k = edgeIndex(v, m)
      if (node[m] === 'u') {
        node[m] = 'o'
        order[m] = ++marked
        dist[m] = dist[v] + 1
        parent[m] = v
        at[m] = steps.length
        edge[k] = 't'
        queue.push(m)
        steps.push(
          snapshot(
            node,
            edge,
            queue,
            `${n(v)} scans ${n(m)}: mark it and enqueue, dist = ${dist[m]}`,
            v,
            k,
          ),
        )
        continue
      }
      if (edge[k] === 'u') edge[k] = 's'
      steps.push(snapshot(node, edge, queue, `${n(v)} scans ${n(m)}: already marked`, v, k))
    }
    node[v] = 'f'
  }
  steps.push(snapshot(node, edge, queue, 'Q is empty: BFS returns marked', -1))
  return {
    steps,
    rows: [
      ['order', order.map(String), at],
      ['dist', dist.map(String), at],
      ['parent', parent.map(p => (p < 0 ? '' : n(p))), at],
    ],
    reached: node.filter(x => x !== 'u').length,
  }
}

const MODES: readonly { id: Mode; label: string; store: string; rule: string }[] = [
  {
    id: 'dfs',
    label: 'DFS',
    store: 'call stack, bottom to top',
    rule: 'DFS-R marks a vertex on entry, then calls itself on each unmarked neighbour in $\\operatorname{Adj}(n)$ order. $d$ and $f$ tick one clock at each entry and return.',
  },
  {
    id: 'bfs',
    label: 'BFS',
    store: 'queue Q, front to back',
    rule: 'BFS marks a vertex when it enters $Q$, so each reached vertex is queued once. Vertices leave $Q$ in distance layers.',
  },
]

const RUNS: Record<string, Run> = {}
for (let s = 0; s < N; s++) {
  RUNS[`dfs:${s}`] = dfs(s)
  RUNS[`bfs:${s}`] = bfs(s)
}

// The inline script only applies these precomputed runs.
const CLIENT_RUNS = Object.fromEntries(
  Object.entries(RUNS).map(([key, r]) => [
    key,
    {
      steps: r.steps.map(s => [s.nodes, s.edges, s.store.join(','), s.say]),
      rows: r.rows.map(([label, vals, at]) => [
        label,
        vals,
        at.map(x => (x === Infinity ? -1 : x)),
      ]),
      reached: `${r.reached} / ${N}`,
    },
  ]),
)

const R = 13
const VIEW_W = 460
const VIEW_H = 232

let instance = 0

const GraphTraversalImpl: QuartzMdxComponent<Props> = ({ caption, mode = 'dfs', start = 0 }) => {
  const id = `gtr-${++instance}`
  const m0 = MODES.some(m => m.id === mode) ? mode : 'dfs'
  const s0 = Number.isInteger(start) && start >= 0 && start < N ? start : 0
  const run = RUNS[`${m0}:${s0}`]
  const step = run.steps[0]
  const last = run.steps.length - 1

  return (
    <figure
      class="graph-traversal"
      data-graph-traversal
      data-gtr-mode={m0}
      data-gtr-runs={JSON.stringify(CLIENT_RUNS)}
    >
      <div class="gtr-controls">
        <fieldset class="gtr-segments">
          <legend class="gtr-legend">Search</legend>
          {MODES.map(m => (
            <label class="gtr-segment">
              <input
                type="radio"
                name={`${id}-mode`}
                value={m.id}
                checked={m.id === m0}
                data-gtr-mode-input
              />
              <span>{m.label}</span>
            </label>
          ))}
        </fieldset>
        <fieldset class="gtr-segments gtr-segments--start">
          <legend class="gtr-legend">Source s</legend>
          {POS.map((_, i) => (
            <label class="gtr-segment">
              <input
                type="radio"
                name={`${id}-start`}
                value={String(i)}
                checked={i === s0}
                data-gtr-start-input
              />
              <span>{n(i)}</span>
            </label>
          ))}
        </fieldset>
      </div>

      {MODES.map(m => (
        <p class="gtr-rule" data-gtr-rule={m.id}>
          <MathText text={m.rule} mathClass="gtr-math" />
        </p>
      ))}

      <svg
        class="gtr-svg"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Undirected graph on vertices n0 to n9: a triangle n5, n6, n9; n9 joins n0; n4 hangs from n0; a four-cycle n0, n7, n2, n3; and a separate edge n1 to n8."
      >
        {EDGES.map(([a, b], k) => (
          <line
            class="gtr-edge"
            data-gtr-edge={String(k)}
            data-state={step.edges[k]}
            x1={POS[a][0]}
            y1={POS[a][1]}
            x2={POS[b][0]}
            y2={POS[b][1]}
          />
        ))}
        {POS.map(([x, y], i) => (
          <g class="gtr-node" data-gtr-node={String(i)} data-state={step.nodes[i]}>
            <rect x={x - R} y={y - R} width={2 * R} height={2 * R} />
            <text x={x} y={y + 4} text-anchor="middle">
              {n(i)}
            </text>
          </g>
        ))}
      </svg>

      <div class="gtr-stepper">
        <button type="button" class="gtr-button" data-gtr-go="first" aria-label="First step">
          first
        </button>
        <button type="button" class="gtr-button" data-gtr-go="prev" aria-label="Previous step">
          back
        </button>
        <input
          class="gtr-range"
          type="range"
          min="0"
          max={String(last)}
          step="1"
          value="0"
          aria-label="Step"
          aria-valuetext={`step 0 of ${last}`}
          data-gtr-range
        />
        <button type="button" class="gtr-button" data-gtr-go="next" aria-label="Next step">
          next
        </button>
        <button type="button" class="gtr-button" data-gtr-go="last" aria-label="Last step">
          last
        </button>
      </div>

      <p class="gtr-say" aria-live="polite">
        <span class="gtr-count" data-gtr-count>{`${0} / ${last}`}</span>
        <span data-gtr-say>{step.say}</span>
      </p>

      <div class="gtr-store">
        {MODES.map(m => (
          <span class="gtr-store-label" data-gtr-store-label={m.id}>
            {m.store}
          </span>
        ))}
        <ol class="gtr-store-cells" role="list">
          {POS.map((_, i) => (
            <li class="gtr-store-cell" data-gtr-slot={String(i)}>
              {step.store[i] === undefined ? '' : n(step.store[i])}
            </li>
          ))}
        </ol>
      </div>

      <table class="gtr-table">
        <thead>
          <tr>
            <th scope="col">vertex</th>
            {POS.map((_, i) => (
              <th scope="col">{n(i)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[0, 1, 2].map(r => (
            <tr>
              <th scope="row" data-gtr-row-label={String(r)}>
                {run.rows[r][0]}
              </th>
              {POS.map((_, i) => (
                <td data-gtr-cell={`${r}:${i}`}>
                  {run.rows[r][2][i] === 0 ? run.rows[r][1][i] || '–' : '–'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <dl class="gtr-readout">
        <div>
          <dt>Marked at the end, reachable from s</dt>
          <dd data-gtr-out="reached">{`${run.reached} / ${N}`}</dd>
        </div>
      </dl>

      {caption ? (
        <figcaption class="gtr-caption">
          <MathText text={caption} mathClass="gtr-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

GraphTraversalImpl.css = style
GraphTraversalImpl.afterDOMLoaded = script

export const GraphTraversal = registerMdxComponent('GraphTraversal', GraphTraversalImpl)

export default (() => GraphTraversal) satisfies (opts: undefined) => QuartzMdxComponent<Props>
