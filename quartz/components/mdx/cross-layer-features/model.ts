// ---------- lifecycle: one feature through eight blocks ----------

// Synthetic stages shaped after the preview's "feature lifecycle" figure. Each block either
// leaves the feature alone, writes one unit of it, or deletes what the stream holds.
export type Stage = 'absent' | 'forms' | 'persists' | 'amplified' | 'deleted'
export const STAGES: readonly Stage[] = [
  'absent',
  'absent',
  'forms',
  'forms',
  'persists',
  'amplified',
  'persists',
  'deleted',
]

// r_{l+1} = r_l + u_l, projected onto the feature direction: s_{l+1} = s_l + w_l.
const lifecycle = () => {
  const strength = [0]
  const writes: number[] = []
  for (const stage of STAGES) {
    const before = strength[strength.length - 1]
    const w = stage === 'forms' || stage === 'amplified' ? 1 : stage === 'deleted' ? -before : 0
    writes.push(w)
    strength.push(before + w)
  }
  // Per-layer SAEs read r_1 .. r_8; a unit-norm latent there has activation s_l.
  const layers = strength.slice(1).map((s, k) => ({ station: k + 1, s }))
  const active = layers.filter(l => l.s > 0)
  const sum = active.reduce((a, l) => a + l.s, 0)
  return { writes, strength, layers, active, sum }
}

export const LC = lifecycle()
export const STATIONS = LC.strength.length
export const SAES = LC.layers.length

// ---------- circuit: the preview's three-feature example over layers L .. L+4 ----------

export const CIRCUIT_LAYERS = 5
export type Feature = { id: number; first: number; row: number }
// Features 1 and 2 are present by layer L; MLPs in layers L+2 and L+3 combine them into 3.
export const FEATURES: readonly Feature[] = [
  { id: 1, first: 0, row: 0 },
  { id: 3, first: 2, row: 1 },
  { id: 2, first: 0, row: 2 },
]
export const FORMED = 3
const PARENTS = [1, 2]
// Block b moves layer b to layer b + 1; these two blocks write feature 3.
export const FORMING_BLOCKS = [1, 2]

type Node = { f: number; l: number; label: string }
type Edge = { from: [number, number]; to: [number, number]; kind: 'persist' | 'form' }

const circuit = () => {
  const feature = (id: number) => FEATURES.find(f => f.id === id)!
  const nodes: Node[] = []
  for (const f of FEATURES)
    for (let l = f.first; l < CIRCUIT_LAYERS; l++)
      nodes.push({ f: f.id, l, label: `${f.id}${'abcde'[l - f.first]}` })
  const edges: Edge[] = []
  for (const f of FEATURES)
    for (let l = f.first; l + 1 < CIRCUIT_LAYERS; l++)
      edges.push({ from: [f.id, l], to: [f.id, l + 1], kind: 'persist' })
  for (const b of FORMING_BLOCKS)
    for (const p of PARENTS) {
      if (b < feature(p).first) continue
      edges.push({ from: [p, b], to: [FORMED, b + 1], kind: 'form' })
    }
  // A crosscoder latent spans every layer its feature occupies, so persistence edges vanish and
  // the formation edges collapse to one per parent feature.
  const latentEdges = [...new Set(edges.filter(e => e.from[0] !== e.to[0]).map(e => e.from[0]))]
  return { nodes, edges, latentEdges, feature }
}

export const CIRCUIT = circuit()
export const persistEdges = CIRCUIT.edges.filter(e => e.kind === 'persist').length
export const formEdges = CIRCUIT.edges.length - persistEdges
