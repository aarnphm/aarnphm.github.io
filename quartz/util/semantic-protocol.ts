import type { PdfIndex, PdfMetadata, PdfPassage } from './pdf-search'
import type { SemanticDocument, SemanticHit, SemanticQueryDType } from './semantic-search'

/** Messages between `semantic.inline.ts` and `quartz/workers/semantic.worker.ts`. */
export type SemanticWorkerConfig = {
  enable?: boolean
  model?: string
  queryDtype?: SemanticQueryDType
  /**
   * Same-origin manifest path. A root pointer (`/embeddings/pdf/manifest.json`,
   * `/embeddings/images/manifest.json`) resolves to its immutable generation in the worker.
   */
  manifestUrl?: string
}

export type SemanticInitRequest = { type: 'init'; cfg?: SemanticWorkerConfig }
export type SemanticLoadRequest = { type: 'load'; seq: number; manifestUrl: string }
export type SemanticSearchRequest = {
  type: 'search'
  seq: number
  text: string
  k: number
  manifestUrl?: string
}
export type PdfSearchRequest = {
  type: 'pdf-search'
  seq: number
  text: string
  manifestUrl: string
  semantic: boolean
}
/** Rows nearest to a stored row of an image index. Needs no encoder. */
export type SemanticSimilarRequest = {
  type: 'similar'
  seq: number
  manifestUrl: string
  seedId: string
  k: number
}
export type SemanticWorkerRequest =
  | SemanticInitRequest
  | SemanticLoadRequest
  | SemanticSearchRequest
  | PdfSearchRequest
  | SemanticSimilarRequest

export type SemanticLoadedResult = {
  type: 'loaded'
  seq: number
  rows: number
  /** Row identities, present for image indexes. */
  ids?: string[]
}

export type SemanticSearchResult = {
  type: 'search-result'
  seq: number
  /** Best row per document, best first. */
  semantic: SemanticHit[]
  /** Row identities aligned with `semantic`, present for image indexes. */
  ids?: string[]
}
export type PdfSearchResult = {
  type: 'pdf-result'
  seq: number
  generation: string
  coverage: PdfIndex['coverage']
  /** Metadata for every PDF named in `semantic` or `lexical`. */
  documents: Record<string, PdfMetadata>
  semantic: (SemanticDocument & { passage: PdfPassage })[]
  /** Passage body matches, one per PDF, in lexical rank order. */
  lexical: PdfPassage[]
}
export type ModelLoadProgress = {
  phase: 'loading' | 'initializing'
  loadedBytes: number
  totalBytes: number
}

export type SemanticWorkerResponse =
  | ({ type: 'model-progress' } & ModelLoadProgress)
  | { type: 'progress'; loadedRows: number; totalRows: number }
  | { type: 'ready' }
  | SemanticLoadedResult
  | SemanticSearchResult
  | PdfSearchResult
  | { type: 'error'; seq?: number; message: string }
