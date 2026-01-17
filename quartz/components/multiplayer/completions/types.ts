export interface FuzzyMatch {
  score: number
  positions: number[]
}

export interface FrecencyEntry {
  count: number
  lastAccess: number
}

export interface CompletionCandidate {
  slug: string
  label: string
  detail?: string
  aliases?: string[]
}

export interface ScoredCandidate extends CompletionCandidate {
  match: FuzzyMatch
  finalScore: number
}
