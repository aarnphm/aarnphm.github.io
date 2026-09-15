import type { ArenaFeedEntry } from './arena-feed'

export interface ArenaReadLink {
  articleId: string
  readAt: number | null
  updatedAt: number
  revision: number
}

export interface ArenaNoteQuote {
  exact: string
  prefix: string
  suffix: string
}

export interface ArenaNoteOccurrence {
  channelSlug: string
  blockId: string
}

export interface ArenaNote {
  id: string
  articleId: string
  sourceUrl: string
  body: string
  snapshotId: string | null
  quote: ArenaNoteQuote | null
  occurrence: ArenaNoteOccurrence | null
  createdAt: number
  updatedAt: number
  revision: number
  readyRevision: number | null
  exportedRevision: number | null
  exportReceipt: string | null
  deletedAt: number | null
}

export interface ArenaFeedResponse {
  subject: string
  revision: string
  entries: ArenaFeedEntry[]
  readLinks: ArenaReadLink[]
}

export interface ArenaReaderResource {
  id: string
  url: string
  kind: 'image' | 'pdf'
  contentType?: string
}

export interface ArenaReaderArtifactBase {
  schemaVersion: 1
  articleId: string
  snapshotId: string
  title: string
  sourceUrl: string
  finalUrl: string
  capturedAt: number
  profileVersion: string
  fingerprint: string
  resources: ArenaReaderResource[]
}

export type ArenaReaderArtifact = ArenaReaderArtifactBase &
  (
    | {
        kind: 'html'
        readerHtml: string | null
        documentHtml: string
        quality: 'complete' | 'partial'
        diagnostics: string[]
      }
    | { kind: 'pdf'; resourceId: string }
    | { kind: 'video'; embedUrl: string | null; description: string | null }
    | { kind: 'internal'; internalUrl: string }
    | { kind: 'external'; reason: string; message: string }
  )

export type ArenaReaderRenderResult =
  | { status: 'ready'; cached: boolean; artifact: ArenaReaderArtifact; warning?: string }
  | { status: 'pending'; retryAfter: number; statusUrl: string }
  | {
      status: 'unavailable'
      reason: string
      message: string
      sourceUrl: string
      retryAfter?: number
    }

export interface ArenaReaderErrorResponse {
  error: string
  message: string
  loginUrl?: string
  current?: ArenaReadLink | ArenaNote | null
}
