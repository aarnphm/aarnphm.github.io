import { MathText, renderInlineMath } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/speculative-draft-tree.inline'
import style from '../styles/speculativeDraftTree.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type MethodId = 'small' | 'medusa' | 'eagle' | 'eagle3'
type StepId = 'd1' | 'd2' | 'd3' | 'verify' | 'accept'
type Props = { caption?: string; method?: MethodId; step?: StepId }
type Dist = Record<string, number>

const REST = '…'
const PREFIX = ['How', 'can'] as const
const ROOT = 'I'
const DEPTH = 3
// Every uniform draw in the acceptance test is fixed, so the run is repeatable.
const U = 0.7
const BETA_PATH = ['help', 'you'] as const

// Synthetic target distributions p, keyed by the context after "How can".
const P: Record<string, Dist> = {
  I: { help: 0.5, make: 0.35, do: 0.05, [REST]: 0.1 },
  'I help': { you: 0.6, with: 0.25, us: 0.05, [REST]: 0.1 },
  'I make': { sure: 0.45, a: 0.3, our: 0.1, [REST]: 0.15 },
  'I help you': { with: 0.55, to: 0.15, today: 0.1, feel: 0.05, [REST]: 0.15 },
  'I help with': { the: 0.4, your: 0.35, this: 0.1, [REST]: 0.15 },
  'I make sure': { that: 0.6, it: 0.2, [REST]: 0.2 },
  'I make a': { difference: 0.5, plan: 0.2, [REST]: 0.3 },
  'I help you to': { find: 0.4, get: 0.3, [REST]: 0.3 },
  'I help you with': { this: 0.45, that: 0.25, [REST]: 0.3 },
}

// The mistakes a drafter leans toward when it errs. EAGLE's depth-3 drift toward "to" and "feel"
// follows the draft tokens in EAGLE figure 6.
const N: Record<string, Dist> = {
  I: { make: 0.45, help: 0.25, do: 0.1, [REST]: 0.2 },
  'I help': { with: 0.6, us: 0.2, [REST]: 0.2 },
  'I make': { a: 0.5, our: 0.4, [REST]: 0.1 },
  'I help you': { to: 0.55, feel: 0.35, [REST]: 0.1 },
  'I help with': { your: 0.5, the: 0.2, [REST]: 0.3 },
  'I make sure': { it: 0.5, that: 0.2, [REST]: 0.3 },
  'I make a': { plan: 0.5, lot: 0.3, [REST]: 0.2 },
}

// q = (1 - lambda) p + lambda n at each depth. A small LM errs everywhere; EAGLE's error grows as
// it regresses its own features; EAGLE-3's training-time test keeps it flat. Medusa's head k reads
// one feature, so its p is the marginal over depth-k contexts.
const LAMBDA: Record<MethodId, readonly number[]> = {
  small: [0.35, 0.45, 0.55],
  medusa: [0.2, 0.35, 0.45],
  eagle: [0.15, 0.3, 0.6],
  eagle3: [0.1, 0.12, 0.15],
}

const METHODS: readonly { id: MethodId; label: string }[] = [
  { id: 'small', label: 'small LM' },
  { id: 'medusa', label: 'Medusa' },
  { id: 'eagle', label: 'EAGLE' },
  { id: 'eagle3', label: 'EAGLE-3' },
]
const STEPS: readonly { id: StepId; label: string }[] = [
  { id: 'd1', label: 'draft 1' },
  { id: 'd2', label: 'draft 2' },
  { id: 'd3', label: 'draft 3' },
  { id: 'verify', label: 'verify' },
  { id: 'accept', label: 'accept' },
]

const keys = (...ds: Dist[]) => [...new Set(ds.flatMap(d => Object.keys(d)))]
const mix = (a: Dist, b: Dist, l: number): Dist =>
  Object.fromEntries(keys(a, b).map(k => [k, (1 - l) * (a[k] ?? 0) + l * (b[k] ?? 0)]))
const weighted = (ds: Dist[], ws: number[]): Dist => {
  const total = ws.reduce((a, b) => a + b, 0)
  return Object.fromEntries(
    keys(...ds).map(k => [k, ds.reduce((s, d, i) => s + (ws[i] / total) * (d[k] ?? 0), 0)]),
  )
}
const ranked = (d: Dist): string[] =>
  Object.entries(d)
    .filter(([k]) => k !== REST)
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .map(([k]) => k)
const ctx = (path: readonly string[]) => [ROOT, ...path].join(' ')
const target = (path: readonly string[]): Dist => {
  const p = P[ctx(path)]
  if (!p) throw new Error(`SpeculativeDraftTree: no target distribution for "${ctx(path)}"`)
  return p
}
const pathProb = (words: string[]) =>
  words.reduce((acc, t, i) => acc * (P[ctx(words.slice(0, i))]?.[t] ?? 0), 1)

const draftDist = (m: MethodId, path: readonly string[]): Dist => {
  const d = path.length
  const l = LAMBDA[m][d]
  if (m !== 'medusa') return mix(target(path), N[ctx(path)], l)
  const contexts = Object.keys(N).filter(c => c.split(' ').length === d + 1)
  const ws = contexts.map(c => pathProb(c.split(' ').slice(1)))
  return mix(
    weighted(
      contexts.map(c => P[c]),
      ws,
    ),
    weighted(
      contexts.map(c => N[c]),
      ws,
    ),
    l,
  )
}

type TreeNode = {
  id: string
  path: string[]
  token: string
  depth: number
  parent: number
  q: number
  score: number
  dist?: Dist
  children: number[]
}

// Each node drafts its two most likely tokens (one for the chain). Depth 3 expands only the two
// depth-2 nodes with the highest draft path probability.
const buildTree = (m: MethodId): TreeNode[] => {
  const width = m === 'small' ? 1 : 2
  const nodes: TreeNode[] = [
    { id: ROOT, path: [], token: ROOT, depth: 0, parent: -1, q: 1, score: 1, children: [] },
  ]
  let frontier = [0]
  for (let d = 0; d < DEPTH; d++) {
    const parents =
      d === DEPTH - 1 && width > 1
        ? [...frontier].sort((a, b) => nodes[b].score - nodes[a].score).slice(0, 2)
        : frontier
    const next: number[] = []
    for (const i of parents) {
      const parent = nodes[i]
      const dist = draftDist(m, parent.path)
      parent.dist = dist
      for (const token of ranked(dist).slice(0, width)) {
        const path = [...parent.path, token]
        nodes.push({
          id: path.join('/'),
          path,
          token,
          depth: d + 1,
          parent: i,
          q: dist[token],
          score: parent.score * dist[token],
          children: [],
        })
        parent.children.push(nodes.length - 1)
        next.push(nodes.length - 1)
      }
    }
    frontier = next
  }
  return nodes
}

const normalize = (d: Dist): Dist => {
  const s = Object.values(d).reduce((a, b) => a + b, 0)
  return Object.fromEntries(Object.entries(d).map(([k, v]) => [k, v / s]))
}
const residual = (p: Dist, q: Dist): Dist =>
  normalize(Object.fromEntries(keys(p, q).map(k => [k, Math.max(0, (p[k] ?? 0) - (q[k] ?? 0))])))
const without = (q: Dist, x: string): Dist =>
  normalize(Object.fromEntries(Object.entries(q).filter(([k]) => k !== x)))

type Test = { node: number; p: number; q: number; a: number; residual: boolean; ok: boolean }
type Run = { tests: Test[]; accepted: number[]; emit: string; emitKind: 'correction' | 'bonus' }

// Children are tested in order. A rejected child turns p into the normalized residual and is
// dropped from q before the next sibling; the one-child case is the note's chain rule.
const verify = (nodes: TreeNode[]): Run => {
  const tests: Test[] = []
  const accepted: number[] = []
  let at = 0
  for (;;) {
    const node = nodes[at]
    if (!node.children.length) {
      return { tests, accepted, emit: ranked(target(node.path))[0], emitKind: 'bonus' }
    }
    let p = target(node.path)
    let q = node.dist!
    let next = -1
    for (const c of node.children) {
      const t = nodes[c].token
      const a = Math.min(1, (p[t] ?? 0) / q[t])
      const ok = U <= a
      tests.push({ node: c, p: p[t] ?? 0, q: q[t], a, residual: q !== node.dist, ok })
      if (ok) {
        next = c
        break
      }
      p = residual(p, q)
      q = without(q, t)
    }
    if (next < 0) return { tests, accepted, emit: ranked(p)[0], emitKind: 'correction' }
    accepted.push(next)
    at = next
  }
}

const beta = (p: Dist, q: Dist) =>
  keys(p, q).reduce((s, k) => s + Math.min(p[k] ?? 0, q[k] ?? 0), 0)

const NODE_W = 54
const NODE_H = 28
const COL = 72
const PITCH = 34
const TREE_TOP = 20
const SLOTS = 6
const TREE_W = DEPTH * COL + NODE_W + 2
const TREE_H = TREE_TOP + SLOTS * PITCH

const CELL = 14
const LABEL_W = 40
const MASK_TOP = 34
const MAX_ROWS = 11
const MASK_W = LABEL_W + (PREFIX.length + MAX_ROWS) * CELL + 2
const MASK_H = MASK_TOP + MAX_ROWS * CELL + 2

type Placed = { x: number; y: number }

const layout = (nodes: TreeNode[]): Placed[] => {
  const pos: Placed[] = nodes.map(n => ({ x: 1 + n.depth * COL, y: 0 }))
  const leaves = nodes.filter(n => !n.children.length).length
  let slot = (SLOTS - leaves) / 2
  const place = (i: number): number => {
    const n = nodes[i]
    const y = n.children.length
      ? n.children.map(place).reduce((a, b) => a + b, 0) / n.children.length
      : TREE_TOP + (slot++ + 0.5) * PITCH - NODE_H / 2
    pos[i].y = y
    return y
  }
  place(0)
  return pos
}

const isAncestor = (nodes: TreeNode[], a: number, b: number) => {
  for (let i = b; i >= 0; i = nodes[i].parent) if (i === a) return true
  return false
}

const fmt = (x: number) => x.toFixed(2)

type NodeState = 'root' | 'hidden' | 'new' | 'drafted' | 'accepted' | 'rejected' | 'untested'
type MethodModel = { nodes: TreeNode[]; order: number[]; pos: Placed[]; run: Run; betas: number[] }

const MODELS = Object.fromEntries(
  METHODS.map(({ id }) => {
    const nodes = buildTree(id)
    // Mask rows follow draft order: depth first, then sibling order.
    const order = nodes.map((_, i) => i).sort((a, b) => nodes[a].depth - nodes[b].depth || a - b)
    const betas = [0, 1, 2].map(d =>
      beta(target(BETA_PATH.slice(0, d)), draftDist(id, BETA_PATH.slice(0, d))),
    )
    return [id, { nodes, order, pos: layout(nodes), run: verify(nodes), betas }]
  }),
) as Record<MethodId, MethodModel>

const stepDepth = (s: StepId) => (s === 'd1' ? 1 : s === 'd2' ? 2 : s === 'd3' ? 3 : DEPTH + 1)

const nodeState = (m: MethodModel, i: number, s: StepId): NodeState => {
  const n = m.nodes[i]
  if (i === 0) return 'root'
  const d = stepDepth(s)
  if (s !== 'accept') return n.depth > d ? 'hidden' : n.depth === d ? 'new' : 'drafted'
  if (m.run.accepted.includes(i)) return 'accepted'
  return m.run.tests.some(t => t.node === i) ? 'rejected' : 'untested'
}

// One character per mask cell, row-major over draft order: 0 masked, 1 visible, 2 lit.
const maskCode = (m: MethodModel, s: StepId): string => {
  const d = stepDepth(s)
  return m.order
    .map(r => {
      const row = m.nodes[r]
      const live = row.depth <= d
      const lit = s === 'verify' || (s === 'accept' && m.run.accepted.includes(r))
      const prefix = PREFIX.map(() => (live ? '1' : '0')).join('')
      const tree = m.order
        .map(c => (!live || !isAncestor(m.nodes, c, r) ? '0' : lit && c !== 0 ? '2' : '1'))
        .join('')
      return prefix + tree
    })
    .join('')
}

const draftPasses = (id: MethodId, d: number) => (id === 'medusa' ? 0 : d)

type Readout = Record<'drafted' | 'draft' | 'target' | 'accepted' | 'emitted' | 'plain', string>

const readout = (id: MethodId, s: StepId): Readout => {
  const m = MODELS[id]
  const d = Math.min(stepDepth(s), DEPTH)
  const drafted = m.nodes.filter(n => n.depth > 0 && n.depth <= d).length
  const done = s === 'accept'
  const emitted = [...m.run.accepted.map(i => m.nodes[i].token), m.run.emit]
  return {
    drafted: String(drafted),
    draft: String(draftPasses(id, d)),
    target: s === 'verify' || done ? '1' : '0',
    accepted: done ? `${m.run.accepted.length} of ${DEPTH}` : '–',
    emitted: done ? String(emitted.length) : '–',
    plain: done ? String(emitted.length) : '–',
  }
}

const CLIENT_STATES = Object.fromEntries(
  METHODS.flatMap(({ id }) =>
    STEPS.map(({ id: s }) => {
      const m = MODELS[id]
      return [
        `${id}:${s}`,
        {
          nodes: m.nodes.map((_, i) => nodeState(m, i, s)),
          mask: maskCode(m, s),
          out: readout(id, s),
        },
      ]
    }),
  ),
)

const RULES: Record<MethodId, Record<StepId, string>> = {
  small: {
    d1: 'A separate small LM samples from $q$ given “How can I”; the chain drafts one candidate per depth.',
    d2: 'Append each draft token and run the small LM again. Each token costs one draft pass; the LM does not use target features.',
    d3: 'The third pass drafts token three; chain depth is $\\gamma = 3$.',
    verify: 'One target pass scores I and three drafts; the chain uses a causal mask.',
    accept:
      'Test each token in order: keep it when $U \\le \\min(1, p/q)$. At the first rejection, emit the most likely token of the residual $\\max(p-q,0)$ and discard the rest of the chain.',
  },
  medusa: {
    d1: 'Head 1 drafts from $f_{\\text{can}}$, the target feature from the pass that sampled I.',
    d2: 'Head 2 sees only $f_{\\text{can}}$; make and help share the same children and $q$.',
    d3: 'Head 3 expands the two highest-probability depth-2 paths from $f_{\\text{can}}$. Heads add no draft-time forward pass.',
    verify:
      'One target pass scores I and all ten drafts. Rows attend to the cached prefix, their ancestors, and themselves; sibling branches are masked.',
    accept:
      'The simplified model uses the same test as other drafters; typical Medusa acceptance relaxes it. Head 2 averages branch proposals, so $q < p$ and the test accepts them. This lowers $\\beta$ and retains nodes such as make you.',
  },
  eagle: {
    d1: 'Pairs $(f_{\\text{how}}, e_{\\text{can}})$ and $(f_{\\text{can}}, e_{\\text{I}})$ predict $\\hat f_{\\text{I}}$; the frozen LM head drafts depth 1.',
    d2: 'Branches use $(\\hat f_{\\text{I}}, e_{\\text{help}})$ and $(\\hat f_{\\text{I}}, e_{\\text{make}})$, producing distinct features and children.',
    d3: 'Depth 3 reads EAGLE’s predicted $\\hat f_{\\text{help}}$; accumulated regression error shifts drafts toward to and feel.',
    verify:
      'One target pass scores I and all ten drafts. Rows attend to the cached prefix, their ancestors, and themselves; sibling branches are masked.',
    accept:
      'Test children in order. When a child fails, $p$ becomes the normalized residual $\\max(p-q,0)$, the child leaves $q$, and the next sibling is tested. Both depth-3 drafts fail, so the target emits the correction. A prime marks a test against the residual.',
  },
  eagle3: {
    d1: 'Fuses target layers as $g = \\mathrm{FC}([l; m; h])$; reads $(g_{\\text{how}}, e_{\\text{can}})$ and $(g_{\\text{can}}, e_{\\text{I}})$.',
    d2: 'Without target features, the draft feeds back $a$: $(a_{\\text{I}}, e_{\\text{help}})$ and $(a_{\\text{I}}, e_{\\text{make}})$.',
    d3: 'Training rehearses feedback; no loss ties $a$ to target features, keeping depth 3 close to the target.',
    verify:
      'One target pass scores I and all ten drafts. Rows attend to the cached prefix, their ancestors, and themselves; sibling branches are masked.',
    accept:
      'Test children in order with the same residual rule. Every test on the path passes, so the leaf earns a bonus token from the target distribution at that leaf.',
  },
}

type TokenKind = 'base' | 'lit' | 'target' | 'cached'
type StripCol = { token?: string; kind?: TokenKind; feature?: string; own?: boolean }
type Strip = { label: string; cols: StripCol[]; out: string }

const tex = (t: string) => `$${t}$`
// Feature subscripts follow the paper's lowercase "how".
const feat = (sym: string, word: string, hat = false) =>
  tex(`${hat ? `\\hat ${sym}` : sym}_{\\text{${word === 'How' ? 'how' : word}}}`)

const branchPath = (m: MethodModel, d: number) => {
  // The strip follows the top branch of each depth, the path the drafter ranks first.
  const path: number[] = [0]
  for (let k = 1; k < d; k++) {
    const kids = m.nodes[path[k - 1]].children
    if (!kids.length) break
    path.push(kids[0])
  }
  return path.map(i => m.nodes[i])
}

const draftedAt = (m: MethodModel, d: number) =>
  [...new Set(m.nodes.filter(n => n.depth === d).map(n => n.token))].join(', ')

const strip = (id: MethodId, s: StepId): Strip => {
  const m = MODELS[id]
  if (s === 'verify') {
    return {
      label: 'target reads',
      cols: [
        ...PREFIX.map((t): StripCol => ({ token: t, kind: 'cached' })),
        ...m.order.map((i): StripCol => ({ token: m.nodes[i].token, kind: i ? 'base' : 'target' })),
      ],
      out: 'one pass, tree mask',
    }
  }
  if (s === 'accept') {
    const r = m.run
    return {
      label: 'output',
      cols: [
        ...PREFIX.map((t): StripCol => ({ token: t, kind: 'base' })),
        { token: ROOT, kind: 'base' },
        ...r.accepted.map((i): StripCol => ({ token: m.nodes[i].token, kind: 'lit' })),
        { token: r.emit, kind: 'target' },
      ],
      out: r.emitKind === 'bonus' ? 'last token: bonus' : 'last token: correction',
    }
  }
  const d = stepDepth(s)
  const path = branchPath(m, d)
  const tokens = path.slice(1).map(n => n.token)
  const out = `depth ${d}: ${draftedAt(m, d)}`
  if (id === 'small') {
    return {
      label: 'small LM reads',
      cols: [...PREFIX, ROOT, ...tokens].map(
        (t, i, all): StripCol => ({
          token: t,
          kind: i === all.length - 1 ? (d === 1 ? 'target' : 'lit') : 'base',
        }),
      ),
      out,
    }
  }
  if (id === 'medusa') {
    return { label: `head ${d} reads`, cols: [{ feature: feat('f', 'can') }], out }
  }
  const sym = id === 'eagle' ? 'f' : 'g'
  const own = id === 'eagle' ? 'f' : 'a'
  const words = [...PREFIX, ROOT, ...tokens]
  const cols: StripCol[] = words.slice(1).map((t, i) => {
    const known = i < PREFIX.length
    const prev = words[i]
    return {
      token: t,
      feature: known ? feat(sym, prev) : feat(own, prev, id === 'eagle'),
      own: !known,
      kind: i === words.length - 2 ? (d === 1 ? 'target' : 'lit') : 'base',
    }
  })
  return { label: `${id === 'eagle' ? 'EAGLE' : 'EAGLE-3'} reads`, cols, out }
}

const OUT_ROWS: readonly { key: keyof Readout; label: string }[] = [
  { key: 'drafted', label: 'Drafted tokens' },
  { key: 'draft', label: 'Draft forward passes' },
  { key: 'target', label: 'Target forward passes' },
  { key: 'accepted', label: `Accepted drafts, $U = ${U}$` },
  { key: 'emitted', label: 'Tokens emitted' },
  { key: 'plain', label: 'Target passes plain decoding needs for them' },
]

const StripView = ({ data }: { data: Strip }) => (
  <div class="sdt-strip-row">
    <span class="sdt-strip-label">{data.label}</span>
    <span class="sdt-strip-cols">
      {data.cols.map(c => (
        <span class="sdt-col">
          {c.token ? (
            <span class="sdt-col-token" data-kind={c.kind ?? 'base'}>
              {c.token}
            </span>
          ) : null}
          {c.feature ? (
            <span
              class="sdt-col-feature"
              data-own={c.own ? 'true' : 'false'}
              dangerouslySetInnerHTML={{ __html: renderInlineMath(c.feature.slice(1, -1)) }}
            />
          ) : null}
        </span>
      ))}
    </span>
    <span class="sdt-strip-out">{`→ ${data.out}`}</span>
  </div>
)

const TreeView = ({ id, m, s0 }: { id: MethodId; m: MethodModel; s0: StepId }) => (
  <svg
    class="sdt-tree"
    viewBox={`0 0 ${TREE_W} ${TREE_H}`}
    role="img"
    aria-label={`${METHODS.find(x => x.id === id)!.label} draft tree after I: ${m.nodes
      .slice(1)
      .map(n => n.path.join(' '))
      .join('; ')}.`}
  >
    <text class="sdt-axis" x={1 + NODE_W / 2} y={10} text-anchor="middle">
      sampled
    </text>
    {[1, 2, 3].map(d => (
      <text class="sdt-axis" x={1 + d * COL + NODE_W / 2} y={10} text-anchor="middle">
        {`depth ${d}`}
      </text>
    ))}
    {m.nodes.slice(1).map((n, k) => {
      const i = k + 1
      const a = m.pos[n.parent]
      const b = m.pos[i]
      const mid = a.x + NODE_W + (COL - NODE_W) / 2
      return (
        <path
          class="sdt-edge"
          data-sdt-node={String(i)}
          data-state={nodeState(m, i, s0)}
          d={`M ${a.x + NODE_W} ${a.y + NODE_H / 2} H ${mid} V ${b.y + NODE_H / 2} H ${b.x}`}
        />
      )
    })}
    {m.nodes.map((n, i) => {
      const p = m.pos[i]
      return (
        <g
          class="sdt-node"
          data-sdt-node={String(i)}
          data-state={nodeState(m, i, s0)}
          transform={`translate(${p.x} ${p.y})`}
        >
          <rect width={NODE_W} height={NODE_H} />
          <text class="sdt-node-token" x={NODE_W / 2} y={12} text-anchor="middle">
            {n.token}
          </text>
          <text class="sdt-node-q" x={NODE_W / 2} y={23} text-anchor="middle">
            {i === 0 ? 'target' : `q ${fmt(n.q)}`}
          </text>
        </g>
      )
    })}
  </svg>
)

const MaskView = ({ m, s0 }: { m: MethodModel; s0: StepId }) => {
  const code = maskCode(m, s0)
  const cols = [...PREFIX, ...m.order.map(i => m.nodes[i].token)]
  const width = cols.length
  return (
    <svg
      class="sdt-mask"
      viewBox={`0 0 ${MASK_W} ${MASK_H}`}
      role="img"
      aria-label={`Tree attention mask: ${m.order.length} query rows by ${width} key columns, the first two columns being the cached prefix.`}
    >
      {cols.map((t, c) => (
        <text
          class="sdt-axis sdt-mask-col"
          data-sdt-node={c < PREFIX.length ? undefined : String(m.order[c - PREFIX.length])}
          data-state={c < PREFIX.length ? undefined : nodeState(m, m.order[c - PREFIX.length], s0)}
          transform={`translate(${LABEL_W + c * CELL + CELL / 2 + 3} ${MASK_TOP - 4}) rotate(-90)`}
        >
          {t}
        </text>
      ))}
      {m.order.map((i, r) => (
        <g>
          <text
            class="sdt-mask-row"
            data-sdt-node={String(i)}
            data-state={nodeState(m, i, s0)}
            x={LABEL_W - 4}
            y={MASK_TOP + r * CELL + 10}
            text-anchor="end"
          >
            {m.nodes[i].token}
          </text>
          {cols.map((_, c) => (
            <rect
              class="sdt-cell"
              data-sdt-cell={String(r * width + c)}
              data-state={code[r * width + c]}
              x={LABEL_W + c * CELL}
              y={MASK_TOP + r * CELL}
              width={CELL}
              height={CELL}
            />
          ))}
        </g>
      ))}
    </svg>
  )
}

const LogView = ({ id }: { id: MethodId }) => {
  const m = MODELS[id]
  const r = m.run
  return (
    <table class="sdt-log">
      <thead>
        <tr>
          <th scope="col">draft</th>
          <th scope="col">
            <MathText text="$p$" mathClass="sdt-math" />
          </th>
          <th scope="col">
            <MathText text="$q$" mathClass="sdt-math" />
          </th>
          <th scope="col">
            <MathText text="$\min(1, p/q)$" mathClass="sdt-math" />
          </th>
          <th scope="col">result</th>
        </tr>
      </thead>
      <tbody>
        {r.tests.map(t => (
          <tr data-ok={t.ok ? 'true' : 'false'}>
            <td>{m.nodes[t.node].token}</td>
            <td>{t.residual ? `${fmt(t.p)}′` : fmt(t.p)}</td>
            <td>{fmt(t.q)}</td>
            <td>{fmt(t.a)}</td>
            <td>{t.ok ? 'keep' : 'reject'}</td>
          </tr>
        ))}
        <tr>
          <td colSpan={4}>
            {r.emitKind === 'bonus' ? 'leaf: bonus from target' : 'all rejected: residual'}
          </td>
          <td>{r.emit}</td>
        </tr>
      </tbody>
    </table>
  )
}

let instance = 0

const SpeculativeDraftTreeImpl: QuartzMdxComponent<Props> = ({
  caption,
  method = 'eagle',
  step = 'd1',
}) => {
  const uid = `sdt-${++instance}`
  const m0 = METHODS.some(x => x.id === method) ? method : 'eagle'
  const s0 = STEPS.some(x => x.id === step) ? step : 'd1'
  const key0 = `${m0}:${s0}`
  const r0 = readout(m0, s0)

  return (
    <figure
      class="speculative-draft-tree"
      data-speculative-draft-tree
      data-sdt-method={m0}
      data-sdt-step={s0}
      data-sdt-states={JSON.stringify(CLIENT_STATES)}
    >
      <div class="sdt-controls">
        <fieldset class="sdt-segments">
          <legend class="sdt-legend">Drafter</legend>
          {METHODS.map(x => (
            <label class="sdt-segment">
              <input
                type="radio"
                name={`${uid}-method`}
                value={x.id}
                checked={x.id === m0}
                data-sdt-method-input
              />
              <span>{x.label}</span>
            </label>
          ))}
        </fieldset>
        <fieldset class="sdt-segments">
          <legend class="sdt-legend">Step</legend>
          {STEPS.map(x => (
            <label class="sdt-segment">
              <input
                type="radio"
                name={`${uid}-step`}
                value={x.id}
                checked={x.id === s0}
                data-sdt-step-input
              />
              <span>{x.label}</span>
            </label>
          ))}
        </fieldset>
      </div>

      {METHODS.flatMap(x =>
        STEPS.map(s => (
          <div
            class="sdt-step"
            data-sdt-key={`${x.id}:${s.id}`}
            hidden={`${x.id}:${s.id}` !== key0}
          >
            <p class="sdt-rule">
              <MathText text={RULES[x.id][s.id]} mathClass="sdt-math" />
            </p>
            <StripView data={strip(x.id, s.id)} />
          </div>
        )),
      )}

      <div class="sdt-stage">
        {METHODS.map(x => (
          <div class="sdt-panels" data-sdt-m={x.id} hidden={x.id !== m0}>
            <TreeView id={x.id} m={MODELS[x.id]} s0={s0} />
            <MaskView m={MODELS[x.id]} s0={s0} />
          </div>
        ))}
      </div>

      <div class="sdt-read">
        <dl class="sdt-readout">
          {OUT_ROWS.map(row => (
            <div>
              <dt>
                <span>
                  <MathText text={row.label} mathClass="sdt-math" />
                </span>
              </dt>
              <dd data-sdt-out={row.key}>{r0[row.key]}</dd>
            </div>
          ))}
          <div>
            <dt>
              <span>
                <MathText text="Acceptance $\beta$ at depth 1, 2, 3" mathClass="sdt-math" />
              </span>
            </dt>
            {METHODS.map(x => (
              <dd data-sdt-m={x.id} hidden={x.id !== m0}>
                {MODELS[x.id].betas.map(fmt).join(' · ')}
              </dd>
            ))}
          </div>
        </dl>
        {METHODS.map(x => (
          <div
            class="sdt-log-wrap"
            data-sdt-key={`${x.id}:accept`}
            hidden={key0 !== `${x.id}:accept`}
          >
            <LogView id={x.id} />
          </div>
        ))}
      </div>

      {caption ? (
        <figcaption class="sdt-caption">
          <MathText text={caption} mathClass="sdt-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const SpeculativeDraftTreeComponent = SpeculativeDraftTreeImpl as QuartzMdxComponent<Props>
SpeculativeDraftTreeComponent.css = style
SpeculativeDraftTreeComponent.afterDOMLoaded = script

export const SpeculativeDraftTree = registerMdxComponent(
  'SpeculativeDraftTree',
  SpeculativeDraftTreeComponent,
)

export default (() => SpeculativeDraftTree) satisfies (opts: undefined) => QuartzMdxComponent<Props>
