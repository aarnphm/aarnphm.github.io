import type { CodeQuery, CodeSearchProgress, CodeSearchResult } from './code-search-types'

export type CodeWorkerRequest = {
  type: 'search'
  id: number
  query: CodeQuery
  manifestUrl: string
}
export type CodeWorkerResponse =
  | { type: 'progress'; id: number; progress: CodeSearchProgress }
  | { type: 'result'; id: number; result: CodeSearchResult }
  | { type: 'error'; id: number; message: string; name: string }
