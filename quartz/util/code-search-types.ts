export const CODE_SEARCH_MANIFEST = '/static/code-search/manifest.json'

export type CodeUnitKind = 'attachment' | 'block' | 'cell'

/** One public occurrence; equal source bodies may occur under multiple owners. */
export interface CodeUnit {
  id: string
  ownerSlug: string
  title: string
  path: string
  kind: CodeUnitKind
  language: string
  originalLanguage: string
  sourceId: string
  anchor: string | null
  startLine: number
  endLine: number
}

export interface CodeSource {
  id: string
  text: string
}

export interface CodeSourceShard {
  kind: 'quartz-code-sources-v1'
  generation: string
  language: string
  sources: CodeSource[]
}

export interface CodeShardReference {
  language: string
  path: string
  sha256: string
  byteLength: number
  sourceIds: string[]
}

export interface CodeSearchManifest {
  kind: 'quartz-code-search-v1'
  version: 1
  generation: string
  units: CodeUnit[]
  shards: CodeShardReference[]
  coverage: { owners: number; units: number; sources: number }
}

export interface CodeQuery {
  mode: 'literal' | 'structural'
  pattern: string
  language?: string
  path?: string
  caseSensitive: boolean
}

export interface CodePoint {
  /** Zero-based UTF-16 offset into the unmodified source body. */
  offset: number
  /** One-based line and column inside the source body. */
  line: number
  column: number
}

export interface CodeRange {
  start: CodePoint
  end: CodePoint
}

export interface CodeMatch {
  unit: CodeUnit
  ranges: CodeRange[]
  excerpt: string
  excerptStartLine: number
  score: number
}

export interface CodeSearchProgress {
  phase: 'loading' | 'searching' | 'structural'
  scannedUnits: number
  totalUnits: number
}

export interface CodeSearchResult {
  generation: string
  mode: CodeQuery['mode']
  matches: CodeMatch[]
  totalMatches: number
  scannedUnits: number
  totalUnits: number
  unsupportedLanguages: string[]
  parseErrors: number
  complete: boolean
  elapsedMs: number
}

export interface StructuralSource {
  id: string
  language: string
  text: string
}

export interface StructuralMatch {
  sourceId: string
  ranges: CodeRange[]
}

export interface StructuralResult {
  matches: StructuralMatch[]
  unsupportedLanguages: string[]
  parseErrors: number
  scannedSources: number
}

export interface StructuralSearchOptions {
  cancelled: () => boolean
  progress?: (scanned: number, total: number) => void
}
