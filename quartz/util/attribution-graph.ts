// The attribution graph of Circuit Tracing, figure 5 (methods.html): an 18-layer model completing
// the acronym of "The National Digital Analytics Group (N" with "DAG". Nodes are supernodes (several
// features of one role), token embeddings and the output logit. The edge weights are illustrative:
// the paper's figure draws the graph without values. They are chosen so every node's inputs sum to
// one on the prompt, which keeps the arithmetic readable.

export type AgKind = 'token' | 'feature' | 'logit'

export interface AgNode {
  id: string
  label: string
  kind: AgKind
  /** column in token positions; a half step sets a node between two tokens */
  col: number
  /** 0 is the logit row, 5 the token row */
  row: number
  note: string
}

export interface AgEdge {
  from: string
  to: string
  /** local linear effect w_{s→t}; the attribution is A_{s→t} = a_s · w_{s→t} */
  w: number
  /** horizontal offsets of the edge's start at the source and its landing at the target */
  lane: number
  land: number
}

export const AG_NODES: readonly AgNode[] = [
  { id: 'the', label: 'The', kind: 'token', col: 0, row: 5, note: 'token embedding' },
  { id: 'national', label: 'National', kind: 'token', col: 1, row: 5, note: 'token embedding' },
  { id: 'digital', label: 'Digital', kind: 'token', col: 2, row: 5, note: 'token embedding' },
  { id: 'analytics', label: 'Analytics', kind: 'token', col: 3, row: 5, note: 'token embedding' },
  { id: 'group', label: 'Group', kind: 'token', col: 4, row: 5, note: 'token embedding' },
  { id: 'paren', label: '(', kind: 'token', col: 5, row: 5, note: 'token embedding' },
  { id: 'n', label: 'N', kind: 'token', col: 6, row: 5, note: 'token embedding' },
  { id: 'fDigital', label: 'Digital', kind: 'feature', col: 2, row: 4, note: 'fires on the word' },
  {
    id: 'fAnalytics',
    label: 'Analytics',
    kind: 'feature',
    col: 3,
    row: 4,
    note: 'fires on the word',
  },
  { id: 'fGroup', label: 'Group', kind: 'feature', col: 4, row: 4, note: 'fires on the word' },
  {
    id: 'acro',
    label: 'continue acronym',
    kind: 'feature',
    col: 6,
    row: 4,
    note: 'an acronym is open',
  },
  { id: 'sayD', label: 'say "D_"', kind: 'feature', col: 2, row: 3, note: 'first letter D' },
  { id: 'sayA', label: 'say "_A"', kind: 'feature', col: 3, row: 3, note: 'second letter A' },
  { id: 'sayG', label: 'say "_G"', kind: 'feature', col: 4, row: 3, note: 'third letter G' },
  {
    id: 'sayDA',
    label: 'say "DA_"',
    kind: 'feature',
    col: 2.5,
    row: 2,
    note: 'tokens that open DA',
  },
  { id: 'sayDAG', label: 'say "DAG"', kind: 'feature', col: 4.5, row: 1, note: 'the token DAG' },
  { id: 'logit', label: 'DAG', kind: 'logit', col: 6, row: 0, note: 'output logit' },
]

export const AG_EDGES: readonly AgEdge[] = [
  { from: 'digital', to: 'fDigital', w: 0.9, lane: 0, land: 0 },
  { from: 'analytics', to: 'fAnalytics', w: 0.9, lane: 0, land: 0 },
  { from: 'group', to: 'fGroup', w: 0.9, lane: 0, land: 0 },
  { from: 'paren', to: 'acro', w: 0.3, lane: 0, land: -10 },
  { from: 'n', to: 'acro', w: 0.6, lane: 0, land: 10 },
  { from: 'fDigital', to: 'sayD', w: 0.5, lane: -8, land: -8 },
  { from: 'fAnalytics', to: 'sayA', w: 0.5, lane: -8, land: -8 },
  { from: 'fGroup', to: 'sayG', w: 0.5, lane: -8, land: -8 },
  { from: 'acro', to: 'sayD', w: 0.4, lane: -14, land: 10 },
  { from: 'acro', to: 'sayA', w: 0.4, lane: -14, land: 10 },
  { from: 'acro', to: 'sayG', w: 0.4, lane: -14, land: 10 },
  { from: 'sayD', to: 'sayDA', w: 0.35, lane: 0, land: -12 },
  { from: 'sayA', to: 'sayDA', w: 0.35, lane: 0, land: 12 },
  { from: 'fDigital', to: 'sayDA', w: 0.1, lane: -30, land: -24 },
  { from: 'fAnalytics', to: 'sayDA', w: 0.1, lane: 30, land: 24 },
  { from: 'sayDA', to: 'sayDAG', w: 0.5, lane: 0, land: -12 },
  { from: 'sayG', to: 'sayDAG', w: 0.3, lane: 0, land: 0 },
  { from: 'fGroup', to: 'sayDAG', w: 0.1, lane: 30, land: 12 },
  { from: 'sayDAG', to: 'logit', w: 0.85, lane: 0, land: 0 },
]

/** What the graph leaves unexplained: an error node for a feature, every other path for the logit. */
export const AG_REST: Readonly<Record<string, number>> = {
  fDigital: 0.1,
  fAnalytics: 0.1,
  fGroup: 0.1,
  acro: 0.1,
  sayD: 0.1,
  sayA: 0.1,
  sayG: 0.1,
  sayDA: 0.1,
  sayDAG: 0.1,
  logit: 0.15,
}

/** node box width in viewBox pixels; the activation bar spans it less a 2px inset each side */
export const AG_NODE_W = 80
export const AG_BAR_SPAN = AG_NODE_W - 4

export const edgeKey = (e: Pick<AgEdge, 'from' | 'to'>) => `${e.from}>${e.to}`

export interface AgState {
  /** activation of every node, after the ablations */
  act: Map<string, number>
  /** attribution A_{s→t} of every edge */
  attr: Map<string, number>
}

/**
 * Evaluate the replacement model with the given nodes clamped to zero. Rows go from the tokens up,
 * so every source is known before its targets. A feature is a JumpReLU unit; the figure keeps the
 * threshold at zero, so a feature holds the positive part of its inputs.
 */
export function evaluate(ablated: ReadonlySet<string>): AgState {
  const act = new Map<string, number>()
  const attr = new Map<string, number>()
  const order = [...AG_NODES].sort((a, b) => b.row - a.row)
  for (const node of order) {
    if (ablated.has(node.id)) {
      act.set(node.id, 0)
      continue
    }
    if (node.kind === 'token') {
      act.set(node.id, 1)
      continue
    }
    let sum = AG_REST[node.id] ?? 0
    for (const e of AG_EDGES) {
      if (e.to !== node.id) continue
      const a = (act.get(e.from) ?? 0) * e.w
      attr.set(edgeKey(e), a)
      sum += a
    }
    act.set(node.id, Math.max(0, sum))
  }
  for (const e of AG_EDGES) if (!attr.has(edgeKey(e))) attr.set(edgeKey(e), 0)
  return { act, attr }
}

/** Edges at or above the display threshold, and the share of total attribution they carry. */
export function pruned(state: AgState, threshold: number) {
  let total = 0
  let kept = 0
  let count = 0
  for (const e of AG_EDGES) {
    const a = Math.abs(state.attr.get(edgeKey(e)) ?? 0)
    total += a
    if (a >= threshold && a > 0) {
      kept += a
      count++
    }
  }
  return { count, share: total > 0 ? kept / total : 0 }
}

/** Stroke width of an edge from its attribution, in viewBox pixels. */
export const strokeFor = (a: number) => 0.8 + 2.4 * Math.abs(a)

export const fixed = (v: number) => v.toFixed(2)
