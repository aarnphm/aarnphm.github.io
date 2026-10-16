export type LexicalField = 'title' | 'fileName' | 'content' | 'tags' | 'aliases'

export type LexicalIndexPriority = 'background' | 'user-visible'

export type LexicalDocument = {
  id: number
  slug: string
  title: string
  content: string
  tags: string[]
  aliases: string[]
  target: string
  fileName: string
  protected: boolean
}

export type LexicalSearchSpec = {
  query: string
  limit: number
  index: LexicalField[]
  tag?: { tags: string }
}

export type LexicalFieldResult = { field?: LexicalField; tag?: LexicalField; result: number[] }

export type LexicalWorkerInitMessage = {
  type: 'init'
  seq: number
  fingerprint?: string
  count?: number
}

export type LexicalWorkerAddMessage = { type: 'add'; seq: number; item: LexicalDocument }

export type LexicalWorkerCommitMessage = { type: 'commit'; seq: number; count: number }

export type LexicalWorkerSearchMessage = { type: 'search'; seq: number; spec: LexicalSearchSpec }

export type LexicalWorkerRequest =
  | LexicalWorkerInitMessage
  | LexicalWorkerAddMessage
  | LexicalWorkerCommitMessage
  | LexicalWorkerSearchMessage

export type LexicalWorkerReadyMessage = { type: 'ready'; seq: number; count: number }

export type LexicalWorkerSearchResultMessage = {
  type: 'search-result'
  seq: number
  results: LexicalFieldResult[]
}

export type LexicalWorkerErrorMessage = { type: 'error'; seq?: number; message: string }

export type LexicalWorkerPrebuiltMessage = { type: 'prebuilt'; seq: number; restored: boolean }

export type LexicalWorkerResponse =
  | LexicalWorkerPrebuiltMessage
  | LexicalWorkerReadyMessage
  | LexicalWorkerSearchResultMessage
  | LexicalWorkerErrorMessage
