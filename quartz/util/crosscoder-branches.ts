// Model behind the CrosscoderBranches figure: the counts and loadings every view draws from.

export type BranchesView = 'parallel' | 'superposition' | 'spread' | 'joint'

export const BRANCHES_VIEWS: readonly BranchesView[] = [
  'parallel',
  'superposition',
  'spread',
  'joint',
]

export const NEURONS_PER_BLOCK = 4

// One feature spread over four neurons, two in block l and two in block l+1 (after Lindsey et al.).
export const FEATURE_NEURONS = { l: [1, 3], next: [0, 2] } as const

// The crosscoder input concatenates both blocks' neurons: [a^l ; a^{l+1}].
export const concatenated = (): { layer: 'l' | 'next'; index: number; lit: boolean }[] =>
  (['l', 'next'] as const).flatMap(layer =>
    Array.from({ length: NEURONS_PER_BLOCK }, (_, index) => ({
      layer,
      index,
      lit: (FEATURE_NEURONS[layer] as readonly number[]).includes(index),
    })),
  )

// Three unit feature directions in a two-neuron space. No angle is a multiple of 90 degrees, so
// every feature loads on both neurons and each neuron reads all three features.
const ANGLES = [20, 140, 260]

export type Direction = { id: number; angle: number; n1: number; n2: number }

export const DIRECTIONS: readonly Direction[] = ANGLES.map((angle, k) => {
  const rad = (angle * Math.PI) / 180
  return { id: k + 1, angle, n1: Math.cos(rad), n2: Math.sin(rad) }
})

// The neuron the superposition view lights: its activation is a sum over all three features.
export const LIT_NEURON = 'n1' as const

export const loading = (v: number) => {
  const r = Math.round(v * 100) / 100
  return r < 0 ? `−${(-r).toFixed(2)}` : r.toFixed(2)
}
