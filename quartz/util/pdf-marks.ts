export const PDF_MANIFEST_PATH = '/static/pdf-documents.json'
export const PDF_READER_PREFIX = '/read'
export const PDF_MARK_ID = /^[a-z2-7]{10}$/
export const PDF_DOCUMENT_ID = /^[0-9a-f]{64}$/

const PDF_USER_SPACE_LIMIT = 14_400
const MAX_QUADS = 64
const MAX_PAGE = 100_000
const MAX_QUOTE_LENGTH = 2048
const MAX_CONTEXT_LENGTH = 64
const MAX_BODY_LENGTH = 16_384
const MAX_SRC_LENGTH = 512

/** Corners in PDF user space, ordered top-left, top-right, bottom-left, bottom-right (/QuadPoints order). */
export type PdfQuad = [number, number, number, number, number, number, number, number]
export type PdfRect = [number, number, number, number]

export interface PdfQuote {
  exact: string
  prefix: string
  suffix: string
}

export type PdfMarkTarget =
  | { type: 'text'; page: number; quads: PdfQuad[]; quote: PdfQuote }
  | { type: 'region'; page: number; rect: PdfRect }
  | { type: 'page'; page: number }

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
  citedBy: PdfCitation[]
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
  document?: PdfDocumentRecord
  documents?: { slug: string; readPath: string; title: string; citations: number }[]
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

export function parsePdfMarkTarget(value: unknown): PdfMarkTarget | null {
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
  const src = typeof value.src === 'string' ? normalizePdfSlug(value.src) : null
  const doc = typeof value.doc === 'string' && PDF_DOCUMENT_ID.test(value.doc) ? value.doc : null
  const revision = parseRevision(value.revision, 0)
  const kind = parseKind(value.kind)
  const target = parsePdfMarkTarget(value.target)
  const body = readString(value.body, MAX_BODY_LENGTH)
  const visibility = parseVisibility(value.visibility)
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
  const src = typeof value.src === 'string' && isPdfSlug(value.src) ? value.src : null
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
  const citedBy = value.citedBy.map(parseCitation)
  if (citedBy.some(citation => citation === null)) return null
  return {
    doc: value.doc,
    bytes: value.bytes,
    title: value.title,
    citedBy: citedBy as PdfCitation[],
  }
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
