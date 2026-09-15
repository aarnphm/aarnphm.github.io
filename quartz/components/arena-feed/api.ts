import type { ArenaFeedEntry } from '../../util/arena-feed'
import type {
  ArenaFeedResponse,
  ArenaNote,
  ArenaNoteOccurrence,
  ArenaNoteQuote,
  ArenaReadLink,
  ArenaReaderArtifact,
  ArenaReaderRenderResult,
} from '../../util/arena-reader'

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function nullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string'
}

function nullableNumber(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isFinite(value))
}

export function isQuote(value: unknown): value is ArenaNoteQuote {
  return (
    isRecord(value) &&
    typeof value.exact === 'string' &&
    typeof value.prefix === 'string' &&
    typeof value.suffix === 'string'
  )
}

function isOccurrence(value: unknown): value is ArenaNoteOccurrence {
  return (
    isRecord(value) && typeof value.channelSlug === 'string' && typeof value.blockId === 'string'
  )
}

export function isNote(value: unknown): value is ArenaNote {
  return (
    isRecord(value) &&
    typeof value.id === 'string' &&
    typeof value.articleId === 'string' &&
    typeof value.sourceUrl === 'string' &&
    typeof value.body === 'string' &&
    nullableString(value.snapshotId) &&
    (value.quote === null || isQuote(value.quote)) &&
    (value.occurrence === null || isOccurrence(value.occurrence)) &&
    typeof value.createdAt === 'number' &&
    typeof value.updatedAt === 'number' &&
    typeof value.revision === 'number' &&
    nullableNumber(value.readyRevision) &&
    nullableNumber(value.exportedRevision) &&
    nullableString(value.exportReceipt) &&
    nullableNumber(value.deletedAt)
  )
}

export function isReadLink(value: unknown): value is ArenaReadLink {
  return (
    isRecord(value) &&
    typeof value.articleId === 'string' &&
    nullableNumber(value.readAt) &&
    typeof value.updatedAt === 'number' &&
    typeof value.revision === 'number'
  )
}

function isEntry(value: unknown): value is ArenaFeedEntry {
  return (
    isRecord(value) &&
    typeof value.articleId === 'string' &&
    typeof value.sourceUrl === 'string' &&
    typeof value.title === 'string' &&
    ['html', 'pdf', 'video', 'internal'].includes(String(value.kind)) &&
    typeof value.later === 'boolean' &&
    Array.isArray(value.tags) &&
    value.tags.every(tag => typeof tag === 'string') &&
    Array.isArray(value.occurrences) &&
    value.occurrences.every(
      occurrence =>
        isRecord(occurrence) &&
        typeof occurrence.channelSlug === 'string' &&
        typeof occurrence.channelName === 'string' &&
        typeof occurrence.blockId === 'string' &&
        nullableString(occurrence.parentBlockId) &&
        nullableString(occurrence.notesHtml),
    )
  )
}

function parseFeed(value: unknown): ArenaFeedResponse {
  if (
    isRecord(value) &&
    typeof value.subject === 'string' &&
    typeof value.revision === 'string' &&
    Array.isArray(value.entries) &&
    value.entries.every(isEntry) &&
    Array.isArray(value.readLinks) &&
    value.readLinks.every(isReadLink)
  ) {
    return {
      subject: value.subject,
      revision: value.revision,
      entries: value.entries,
      readLinks: value.readLinks,
    }
  }
  throw new Error('The reader received an invalid catalogue response.')
}

function isArtifact(value: unknown): value is ArenaReaderArtifact {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    typeof value.articleId !== 'string' ||
    typeof value.snapshotId !== 'string' ||
    typeof value.title !== 'string' ||
    typeof value.sourceUrl !== 'string' ||
    typeof value.finalUrl !== 'string' ||
    typeof value.capturedAt !== 'number' ||
    typeof value.profileVersion !== 'string' ||
    typeof value.fingerprint !== 'string' ||
    !Array.isArray(value.resources) ||
    !value.resources.every(
      resource =>
        isRecord(resource) &&
        typeof resource.id === 'string' &&
        typeof resource.url === 'string' &&
        (resource.kind === 'image' || resource.kind === 'pdf'),
    )
  )
    return false
  if (value.kind === 'html')
    return (
      nullableString(value.readerHtml) &&
      typeof value.documentHtml === 'string' &&
      (value.quality === 'complete' || value.quality === 'partial') &&
      Array.isArray(value.diagnostics) &&
      value.diagnostics.every(diagnostic => typeof diagnostic === 'string')
    )
  if (value.kind === 'pdf') return typeof value.resourceId === 'string'
  if (value.kind === 'video')
    return nullableString(value.embedUrl) && nullableString(value.description)
  if (value.kind === 'internal') return typeof value.internalUrl === 'string'
  return (
    value.kind === 'external' &&
    typeof value.reason === 'string' &&
    typeof value.message === 'string'
  )
}

function parseRender(value: unknown): ArenaReaderRenderResult {
  if (isRecord(value)) {
    if (
      value.status === 'ready' &&
      typeof value.cached === 'boolean' &&
      isArtifact(value.artifact)
    ) {
      return {
        status: 'ready',
        cached: value.cached,
        artifact: value.artifact,
        warning: typeof value.warning === 'string' ? value.warning : undefined,
      }
    }
    if (
      value.status === 'pending' &&
      typeof value.retryAfter === 'number' &&
      typeof value.statusUrl === 'string'
    ) {
      return { status: 'pending', retryAfter: value.retryAfter, statusUrl: value.statusUrl }
    }
    if (
      value.status === 'unavailable' &&
      typeof value.reason === 'string' &&
      typeof value.message === 'string' &&
      typeof value.sourceUrl === 'string'
    ) {
      return {
        status: 'unavailable',
        reason: value.reason,
        message: value.message,
        sourceUrl: value.sourceUrl,
        retryAfter: typeof value.retryAfter === 'number' ? value.retryAfter : undefined,
      }
    }
  }
  throw new Error('The reader received an invalid article response.')
}

export class ReaderApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly loginUrl?: string,
    readonly current?: ArenaNote | ArenaReadLink | null,
  ) {
    super(message)
  }
}

async function request(
  path: string,
  signal: AbortSignal,
  method = 'GET',
  body?: unknown,
  subject?: string,
  allowUnavailable = false,
): Promise<unknown> {
  const url = new URL(path, location.origin)
  if (url.origin !== location.origin || !url.pathname.startsWith('/api/arena/'))
    throw new Error('Invalid reader endpoint.')
  const headers = new Headers({ Accept: 'application/json' })
  if (body !== undefined) headers.set('Content-Type', 'application/json')
  if (subject) headers.set('X-Arena-Subject', subject)
  const options: RequestInit = {
    method,
    signal,
    credentials: 'same-origin',
    cache: 'no-store',
    headers,
  }
  if (body !== undefined) options.body = JSON.stringify(body)
  const response = await fetch(url, options)
  return readApiResponse(response, allowUnavailable)
}

export async function readApiResponse(
  response: Response,
  allowUnavailable = false,
): Promise<unknown> {
  if (!response.headers.get('content-type')?.includes('application/json')) {
    throw new ReaderApiError(
      response.status,
      'The reader API is unavailable on this host. Open the Worker-backed site to read and sync notes.',
    )
  }
  const data: unknown = await response.json()
  if (
    allowUnavailable &&
    response.status !== 401 &&
    response.status !== 403 &&
    isRecord(data) &&
    data.status === 'unavailable'
  )
    return data
  if (!response.ok && response.status !== 202) {
    const message =
      isRecord(data) && typeof data.message === 'string'
        ? data.message
        : `Reader request failed (${response.status}).`
    const loginUrl = isRecord(data) && typeof data.loginUrl === 'string' ? data.loginUrl : undefined
    const current =
      isRecord(data) && (data.current === null || isNote(data.current) || isReadLink(data.current))
        ? data.current
        : undefined
    throw new ReaderApiError(response.status, message, loginUrl, current)
  }
  return data
}

export const readerApi = {
  async feed(seed: string, signal: AbortSignal) {
    return parseFeed(await request(`/api/arena/feed?seed=${encodeURIComponent(seed)}`, signal))
  },
  async render(articleId: string, signal: AbortSignal, refresh = false) {
    return parseRender(
      await request(
        `/api/arena/articles/${encodeURIComponent(articleId)}/render`,
        signal,
        'POST',
        { refresh },
        undefined,
        true,
      ),
    )
  },
  async renderStatus(path: string, signal: AbortSignal) {
    return parseRender(await request(path, signal, 'GET', undefined, undefined, true))
  },
  async snapshot(articleId: string, snapshotId: string, signal: AbortSignal) {
    return parseRender(
      await request(
        `/api/arena/articles/${encodeURIComponent(articleId)}/snapshots/${encodeURIComponent(snapshotId)}`,
        signal,
        'GET',
        undefined,
        undefined,
        true,
      ),
    )
  },
  async read(
    articleId: string,
    read: boolean,
    revision: number,
    subject: string,
    signal: AbortSignal,
  ) {
    const data = await request(
      `/api/arena/articles/${encodeURIComponent(articleId)}/read`,
      signal,
      'PUT',
      { read, revision },
      subject,
    )
    if (isRecord(data) && isReadLink(data.readLink)) return data.readLink
    throw new Error('The read mark was not acknowledged.')
  },
  async notes(articleId: string | null, signal: AbortSignal, ready = false) {
    const path = articleId
      ? `/api/arena/articles/${encodeURIComponent(articleId)}/notes`
      : `/api/arena/notes?view=${ready ? 'ready' : 'inbox'}`
    const data = await request(path, signal)
    if (isRecord(data) && Array.isArray(data.notes) && data.notes.every(isNote)) return data.notes
    throw new Error('Notes could not be loaded.')
  },
  async save(note: ArenaNote, ready: boolean, subject: string, signal: AbortSignal) {
    const data = await request(
      `/api/arena/notes/${encodeURIComponent(note.id)}`,
      signal,
      'PUT',
      {
        articleId: note.articleId,
        body: note.body,
        snapshotId: note.snapshotId,
        quote: note.quote,
        occurrence: note.occurrence,
        revision: note.revision,
        ready,
      },
      subject,
    )
    if (isRecord(data) && isNote(data.note)) return data.note
    throw new Error('The note was not acknowledged.')
  },
  async delete(note: ArenaNote, subject: string, signal: AbortSignal) {
    await request(
      `/api/arena/notes/${encodeURIComponent(note.id)}`,
      signal,
      'DELETE',
      { revision: note.revision },
      subject,
    )
  },
}
