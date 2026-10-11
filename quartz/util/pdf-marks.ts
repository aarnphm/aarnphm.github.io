export const PDF_MANIFEST_PATH = '/static/pdf-documents.json'
export const PDF_READER_PREFIX = '/read'
export const PDF_READER_SHELL_PATH = '/static/pdf-reader'
export const PDF_MARK_ID = /^[a-z2-7]{10}$/
export const PDF_DOCUMENT_ID = /^[0-9a-f]{64}$/

const PDF_USER_SPACE_LIMIT = 14_400
const MAX_QUADS = 64
const MAX_PAGE = 100_000
const MAX_QUOTE_LENGTH = 2048
export const PDF_QUOTE_LIMIT = MAX_QUOTE_LENGTH
const MAX_HEADING_LENGTH = 256
const MAX_PASSAGE_TEXT_LENGTH = 8192
export const PDF_PASSAGE_TEXT_LIMIT = MAX_PASSAGE_TEXT_LENGTH
const MAX_CONTEXT_LENGTH = 64
const MAX_BODY_LENGTH = 16_384
export const PDF_BODY_LIMIT = MAX_BODY_LENGTH
const MAX_SRC_LENGTH = 512

/** Corners in PDF user space, ordered top-left, top-right, bottom-left, bottom-right (/QuadPoints order). */
export type PdfQuad = [number, number, number, number, number, number, number, number]
export type PdfRect = [number, number, number, number]

export interface PdfQuote {
  exact: string
  prefix: string
  suffix: string
}

/**
 * A passage of an authored lecture note (`src` = the unit slug). Page 0 keeps the shared `page`
 * column: the passage is anchored by quote and heading, never by a PDF page.
 */
export interface PdfPassageTarget {
  type: 'passage'
  page: 0
  /** Null for a note about the whole lecture. */
  quote: PdfQuote | null
  /** Id of the closest heading, which narrows quote recovery. */
  heading: string | null
  /** The quotation as Markdown, with LaTeX in place of rendered math. */
  text?: string
}

export type PdfMarkTarget =
  | { type: 'text'; page: number; quads: PdfQuad[]; quote: PdfQuote }
  | { type: 'region'; page: number; rect: PdfRect }
  | { type: 'page'; page: number }
  | PdfPassageTarget

export type PdfMarkKind = 'mark' | 'question' | 'contra'
export type PdfMarkVisibility = 'public' | 'private'

export interface PdfMarkInput {
  src: string
  doc: string
  revision: number
  kind: PdfMarkKind
  target: PdfMarkTarget
  body: string
  visibility: PdfMarkVisibility
}

export interface PdfMark {
  id: string
  doc: string
  src: string
  kind: PdfMarkKind
  target: PdfMarkTarget
  body: string
  visibility: PdfMarkVisibility
  revision: number
  createdAt: number
  updatedAt: number
}

export interface PdfCitation {
  from: string
  title: string
  page?: number
  markId?: string
  excerpt: string
}

export interface PdfDocumentRecord {
  doc: string
  bytes: number
  title: string
  titleSource?: 'frontmatter' | 'alias' | 'filename'
  citedBy: PdfCitation[]
  /** Canonical page for this PDF when a course resource page renders it (`/courses/<dir>/resources/<id>/`). */
  readPath?: string
}

export interface PdfManifest {
  version: 1
  documents: Record<string, PdfDocumentRecord>
}

export interface PdfMarksResponse {
  src: string
  doc: string
  canWrite: boolean
  signInUrl: string
  marks: PdfMark[]
  stale: PdfMark[]
}

export interface PdfReaderData {
  mode: 'document' | 'index'
  slug?: string
  readPath?: string
  /** `course`: the page carries its own strip, so the app drops the `reader /` title block. */
  chrome?: 'course'
  document?: PdfDocumentRecord
  documents?: { slug: string; readPath: string; title: string; citations: number }[]
}

export type PdfMarkCounts = Record<PdfMarkKind, number>

export interface PdfMarkCountsResponse {
  canWrite: boolean
  counts: Record<string, PdfMarkCounts>
}

/** JSON safe inside a `<script>` element: `<` and the line separators are escaped. */
export function scriptJson(value: unknown): string {
  return JSON.stringify(value)
    .replaceAll('<', '\\u003c')
    .replaceAll('\u2028', '\\u2028')
    .replaceAll('\u2029', '\\u2029')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isCoordinate(value: unknown): value is number {
  return (
    typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= PDF_USER_SPACE_LIMIT
  )
}

function isPage(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= MAX_PAGE
}

function roundCoordinate(value: number): number {
  return Math.round(value * 100) / 100
}

function readString(value: unknown, max: number): string | null {
  return typeof value === 'string' && value.length <= max ? value : null
}

export function isPdfSlug(value: string): boolean {
  return (
    value.length > 4 &&
    value.length <= MAX_SRC_LENGTH &&
    /\.pdf$/i.test(value) &&
    !value.startsWith('/') &&
    !value.split('/').some(segment => segment === '..' || segment === '.' || segment === '')
  )
}

/** Normalizes `/a/b.pdf`, `a/b.pdf` and percent-encoded paths to the manifest slug. */
export function normalizePdfSlug(value: string): string | null {
  let decoded = value.trim()
  try {
    decoded = decodeURIComponent(decoded)
  } catch {
    return null
  }
  decoded = decoded.replace(/^\/+/, '').split(/[?#]/, 1)[0]
  return isPdfSlug(decoded) ? decoded : null
}

/** Lecture notes share the marks API: `src` is the unit slug, `doc` a fixed digest of it. */
const UNIT_MARK_SRC = /^courses\/[a-z0-9][a-z0-9.-]*\/lectures\/\d{2,3}$/i

export function isUnitMarkSrc(value: string): boolean {
  return value.length <= MAX_SRC_LENGTH && UNIT_MARK_SRC.test(value) && !value.includes('..')
}

/** `sha256("unit:" + src)` in hex. Stable across note edits, so marks never go stale. */
export async function unitMarkDoc(src: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`unit:${src}`))
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}

function isReadPath(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.startsWith('/') &&
    value.length <= MAX_SRC_LENGTH &&
    !/[?#\s]/.test(value)
  )
}

/** Where a hosted PDF opens: its resource page when one claims it, else `/read/<slug>`. */
export function pdfOpenPath(slug: string, record?: Pick<PdfDocumentRecord, 'readPath'>): string {
  return record?.readPath ?? pdfReaderPath(slug)
}

export function pdfReaderKey(slug: string): string {
  return slug.replace(/\.pdf$/i, '')
}

export function pdfReaderPath(slug: string): string {
  return `${PDF_READER_PREFIX}/${pdfReaderKey(slug)}`
}

export function newPdfMarkId(): string {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz234567'
  const bytes = crypto.getRandomValues(new Uint8Array(10))
  return Array.from(bytes, byte => alphabet[byte & 31]).join('')
}

function parseQuad(value: unknown): PdfQuad | null {
  if (!Array.isArray(value) || value.length !== 8 || !value.every(isCoordinate)) return null
  return value.map(roundCoordinate) as PdfQuad
}

function parseQuote(value: unknown): PdfQuote | null {
  if (!isRecord(value)) return null
  const exact = readString(value.exact, MAX_QUOTE_LENGTH)
  const prefix = readString(value.prefix, MAX_CONTEXT_LENGTH)
  const suffix = readString(value.suffix, MAX_CONTEXT_LENGTH)
  if (exact === null || prefix === null || suffix === null || exact.trim().length === 0) return null
  return { exact, prefix, suffix }
}

function parsePassage(value: Record<string, unknown>): PdfPassageTarget | null {
  if (value.page !== 0) return null
  const quote = value.quote === null ? null : parseQuote(value.quote)
  if (value.quote !== null && quote === null) return null
  const heading = value.heading === null ? null : readString(value.heading, MAX_HEADING_LENGTH)
  if (value.heading !== null && heading === null) return null
  const text =
    value.text === undefined ? undefined : readString(value.text, MAX_PASSAGE_TEXT_LENGTH)
  if (text === null) return null
  return { type: 'passage', page: 0, quote, heading, ...(text !== undefined ? { text } : {}) }
}

export function parsePdfMarkTarget(value: unknown): PdfMarkTarget | null {
  if (isRecord(value) && value.type === 'passage') return parsePassage(value)
  if (!isRecord(value) || !isPage(value.page)) return null
  if (value.type === 'page') return { type: 'page', page: value.page }
  if (value.type === 'region') {
    const rect = value.rect
    if (!Array.isArray(rect) || rect.length !== 4 || !rect.every(isCoordinate)) return null
    const [x0, y0, x1, y1] = rect.map(roundCoordinate)
    if (x1 <= x0 || y1 <= y0) return null
    return { type: 'region', page: value.page, rect: [x0, y0, x1, y1] }
  }
  if (value.type === 'text') {
    if (!Array.isArray(value.quads) || value.quads.length < 1 || value.quads.length > MAX_QUADS) {
      return null
    }
    const quads = value.quads.map(parseQuad)
    if (quads.some(quad => quad === null)) return null
    const quote = parseQuote(value.quote)
    if (!quote) return null
    return { type: 'text', page: value.page, quads: quads as PdfQuad[], quote }
  }
  return null
}

function parseKind(value: unknown): PdfMarkKind | null {
  return value === 'mark' || value === 'question' || value === 'contra' ? value : null
}

function parseVisibility(value: unknown): PdfMarkVisibility | null {
  return value === 'public' || value === 'private' ? value : null
}

function parseRevision(value: unknown, min: number): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= min ? value : null
}

const inputKeys = new Set(['src', 'doc', 'revision', 'kind', 'target', 'body', 'visibility'])

export function parsePdfMarkInput(value: unknown): PdfMarkInput | null {
  if (!isRecord(value) || Object.keys(value).some(key => !inputKeys.has(key))) return null
  const unit = typeof value.src === 'string' && isUnitMarkSrc(value.src)
  const src = unit
    ? (value.src as string)
    : typeof value.src === 'string'
      ? normalizePdfSlug(value.src)
      : null
  const doc = typeof value.doc === 'string' && PDF_DOCUMENT_ID.test(value.doc) ? value.doc : null
  const revision = parseRevision(value.revision, 0)
  const kind = parseKind(value.kind)
  const target = parsePdfMarkTarget(value.target)
  const body = readString(value.body, MAX_BODY_LENGTH)
  const visibility = parseVisibility(value.visibility)
  // passages belong to lecture notes, pages and quads to PDFs
  if (target !== null && (target.type === 'passage') !== unit) return null
  if (
    src === null ||
    doc === null ||
    revision === null ||
    kind === null ||
    target === null ||
    body === null ||
    visibility === null
  ) {
    return null
  }
  return { src, doc, revision, kind, target, body, visibility }
}

export function parsePdfMark(value: unknown): PdfMark | null {
  if (!isRecord(value)) return null
  const id = typeof value.id === 'string' && PDF_MARK_ID.test(value.id) ? value.id : null
  const doc = typeof value.doc === 'string' && PDF_DOCUMENT_ID.test(value.doc) ? value.doc : null
  const src =
    typeof value.src === 'string' && (isPdfSlug(value.src) || isUnitMarkSrc(value.src))
      ? value.src
      : null
  const kind = parseKind(value.kind)
  const target = parsePdfMarkTarget(value.target)
  const body = readString(value.body, MAX_BODY_LENGTH)
  const visibility = parseVisibility(value.visibility)
  const revision = parseRevision(value.revision, 1)
  const createdAt = parseRevision(value.createdAt, 0)
  const updatedAt = parseRevision(value.updatedAt, 0)
  if (
    id === null ||
    doc === null ||
    src === null ||
    kind === null ||
    target === null ||
    body === null ||
    visibility === null ||
    revision === null ||
    createdAt === null ||
    updatedAt === null
  ) {
    return null
  }
  return { id, doc, src, kind, target, body, visibility, revision, createdAt, updatedAt }
}

function parseCitation(value: unknown): PdfCitation | null {
  if (!isRecord(value) || typeof value.from !== 'string' || typeof value.title !== 'string') {
    return null
  }
  if (typeof value.excerpt !== 'string') return null
  const citation: PdfCitation = { from: value.from, title: value.title, excerpt: value.excerpt }
  if (value.page !== undefined) {
    if (!isPage(value.page)) return null
    citation.page = value.page
  }
  if (value.markId !== undefined) {
    if (typeof value.markId !== 'string' || !PDF_MARK_ID.test(value.markId)) return null
    citation.markId = value.markId
  }
  return citation
}

export function parsePdfDocumentRecord(value: unknown): PdfDocumentRecord | null {
  if (!isRecord(value)) return null
  if (typeof value.doc !== 'string' || !PDF_DOCUMENT_ID.test(value.doc)) return null
  if (typeof value.bytes !== 'number' || !Number.isSafeInteger(value.bytes) || value.bytes < 0) {
    return null
  }
  if (typeof value.title !== 'string' || !Array.isArray(value.citedBy)) return null
  if (
    value.titleSource !== undefined &&
    value.titleSource !== 'frontmatter' &&
    value.titleSource !== 'alias' &&
    value.titleSource !== 'filename'
  ) {
    return null
  }
  const citedBy = value.citedBy.map(parseCitation)
  if (citedBy.some(citation => citation === null)) return null
  if (value.readPath !== undefined && !isReadPath(value.readPath)) return null
  const record: PdfDocumentRecord = {
    doc: value.doc,
    bytes: value.bytes,
    title: value.title,
    citedBy: citedBy as PdfCitation[],
  }
  if (value.titleSource !== undefined) record.titleSource = value.titleSource
  if (typeof value.readPath === 'string') record.readPath = value.readPath
  return record
}

export function parsePdfManifest(value: unknown): PdfManifest | null {
  if (!isRecord(value) || value.version !== 1 || !isRecord(value.documents)) return null
  const documents: Record<string, PdfDocumentRecord> = {}
  for (const [slug, record] of Object.entries(value.documents)) {
    const parsed = parsePdfDocumentRecord(record)
    if (!isPdfSlug(slug) || !parsed) return null
    documents[slug] = parsed
  }
  return { version: 1, documents }
}

/** Page-and-anchor fragment for links into a PDF, e.g. `#page=12` or `#^abcdefghij`. */
export function parsePdfFragment(hash: string): { page?: number; markId?: string } {
  const value = hash.replace(/^#/, '')
  const page = /^page=(\d{1,6})$/.exec(value)
  if (page) {
    const number = Number.parseInt(page[1], 10)
    return isPage(number) ? { page: number } : {}
  }
  const mark = /^\^([a-z2-7]{10})$/.exec(value)
  return mark ? { markId: mark[1] } : {}
}
