import { and, asc, eq, isNull, ne } from 'drizzle-orm'
import { drizzle } from 'drizzle-orm/d1'
import {
  parsePdfManifest,
  parsePdfMarkInput,
  PDF_MANIFEST_PATH,
  PDF_MARK_ID,
  PDF_READER_PREFIX,
  pdfReaderKey,
  pdfReaderPath,
  normalizePdfSlug,
  type PdfDocumentRecord,
  type PdfManifest,
  type PdfMark,
  type PdfMarkInput,
  type PdfMarksResponse,
  type PdfReaderData,
} from '../quartz/util/pdf-marks'
import {
  getOwnerSessionLogin,
  isArenaReaderMutationAllowed,
  type ArenaReaderAuthEnv,
} from './arena-reader-auth'
import { wantsMarkdown } from './request-utils'
import { pdfMarks } from './schema/pdf-marks'

export interface PdfReaderEnv extends ArenaReaderAuthEnv {
  ARENA_READER?: D1Database
  ASSETS: Fetcher
}

const MAX_BODY_BYTES = 64 * 1024
const MANIFEST_TTL_MS = 30_000
const TITLE_SLOT = 'PDF_READER_TITLE_SLOT'
const DATA_SLOT = '<script type="application/json" id="pdf-reader-data">null</script>'

interface ManifestCacheEntry {
  manifest: PdfManifest
  byReaderKey: Map<string, string>
  loadedAt: number
}

const manifestCache = new WeakMap<Fetcher, ManifestCacheEntry>()

class PdfRequestError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly extra: Record<string, unknown> = {},
  ) {
    super(message)
  }
}

function privateJson(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'private, no-store',
      Vary: 'Cookie',
      'X-Content-Type-Options': 'nosniff',
    },
  })
}

function signInUrl(slug: string): string {
  const returnTo = encodeURIComponent(encodeURI(pdfReaderPath(slug)))
  return `/comments/github/login?returnTo=${returnTo}`
}

async function loadManifest(env: PdfReaderEnv, origin: string): Promise<ManifestCacheEntry> {
  const cached = manifestCache.get(env.ASSETS)
  if (cached && Date.now() - cached.loadedAt < MANIFEST_TTL_MS) return cached
  const response = await env.ASSETS.fetch(new URL(PDF_MANIFEST_PATH, origin))
  if (!response.ok) {
    throw new PdfRequestError(503, 'manifest-unavailable', 'The PDF manifest is unavailable.')
  }
  const manifest = parsePdfManifest(await response.json().catch(() => null))
  if (!manifest) {
    throw new PdfRequestError(503, 'manifest-unavailable', 'The PDF manifest needs rebuilding.')
  }
  const byReaderKey = new Map<string, string>()
  for (const slug of Object.keys(manifest.documents)) byReaderKey.set(pdfReaderKey(slug), slug)
  const entry = { manifest, byReaderKey, loadedAt: Date.now() }
  manifestCache.set(env.ASSETS, entry)
  return entry
}

function rowToMark(row: typeof pdfMarks.$inferSelect): PdfMark {
  return {
    id: row.id,
    doc: row.doc,
    src: row.src,
    kind: row.kind,
    target: row.target,
    body: row.body,
    visibility: row.visibility,
    revision: row.revision,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
  }
}

function database(env: PdfReaderEnv): D1Database {
  if (!env.ARENA_READER) {
    throw new PdfRequestError(503, 'storage-unavailable', 'Mark storage is not configured.')
  }
  return env.ARENA_READER
}

async function listMarks(env: PdfReaderEnv, doc: string, includePrivate: boolean) {
  const rows = await drizzle(database(env))
    .select()
    .from(pdfMarks)
    .where(
      and(
        eq(pdfMarks.doc, doc),
        isNull(pdfMarks.deletedAt),
        includePrivate ? undefined : eq(pdfMarks.visibility, 'public'),
      ),
    )
    .orderBy(asc(pdfMarks.page), asc(pdfMarks.createdAt), asc(pdfMarks.id))
  return rows.map(rowToMark)
}

async function listStaleMarks(env: PdfReaderEnv, owner: string, slug: string, doc: string) {
  const rows = await drizzle(database(env))
    .select()
    .from(pdfMarks)
    .where(
      and(
        eq(pdfMarks.owner, owner),
        eq(pdfMarks.src, slug),
        ne(pdfMarks.doc, doc),
        isNull(pdfMarks.deletedAt),
      ),
    )
    .orderBy(asc(pdfMarks.page), asc(pdfMarks.createdAt))
  return rows.map(rowToMark)
}

async function findMark(env: PdfReaderEnv, id: string) {
  const rows = await drizzle(database(env)).select().from(pdfMarks).where(eq(pdfMarks.id, id))
  return rows[0]
}

async function saveMark(
  env: PdfReaderEnv,
  owner: string,
  id: string,
  input: PdfMarkInput,
): Promise<PdfMark> {
  const db = drizzle(database(env))
  const now = Date.now()
  const next = {
    doc: input.doc,
    src: input.src,
    kind: input.kind,
    page: input.target.page,
    target: input.target,
    body: input.body,
    visibility: input.visibility,
    revision: input.revision + 1,
    updatedAt: now,
  }
  const rows =
    input.revision === 0
      ? await db
          .insert(pdfMarks)
          .values({ id, owner, createdAt: now, ...next })
          .onConflictDoNothing()
          .returning()
      : await db
          .update(pdfMarks)
          .set(next)
          .where(
            and(
              eq(pdfMarks.id, id),
              eq(pdfMarks.owner, owner),
              eq(pdfMarks.revision, input.revision),
              isNull(pdfMarks.deletedAt),
            ),
          )
          .returning()
  if (rows[0]) return rowToMark(rows[0])
  const current = await findMark(env, id)
  throw new PdfRequestError(409, 'conflict', 'This mark changed elsewhere.', {
    current: current && current.deletedAt === null ? rowToMark(current) : null,
    deleted: current?.deletedAt != null,
  })
}

async function deleteMark(env: PdfReaderEnv, owner: string, id: string, revision: number) {
  const now = Date.now()
  const rows = await drizzle(database(env))
    .update(pdfMarks)
    .set({ deletedAt: now, updatedAt: now, revision: revision + 1 })
    .where(
      and(
        eq(pdfMarks.id, id),
        eq(pdfMarks.owner, owner),
        eq(pdfMarks.revision, revision),
        isNull(pdfMarks.deletedAt),
      ),
    )
    .returning()
  if (rows[0]) return { id, revision: rows[0].revision }
  const current = await findMark(env, id)
  if (!current || current.owner !== owner) {
    throw new PdfRequestError(404, 'not-found', 'This mark does not exist.')
  }
  if (current.deletedAt !== null && current.revision === revision + 1) {
    return { id, revision: current.revision }
  }
  throw new PdfRequestError(409, 'conflict', 'This mark changed elsewhere.', {
    current: current.deletedAt === null ? rowToMark(current) : null,
    deleted: current.deletedAt !== null,
  })
}

async function readJsonBody(request: Request): Promise<unknown> {
  if (request.headers.get('Content-Type')?.split(';')[0].trim() !== 'application/json') {
    throw new PdfRequestError(415, 'invalid-content-type', 'Send a JSON request body.')
  }
  const text = await request.text()
  if (new TextEncoder().encode(text).byteLength > MAX_BODY_BYTES) {
    throw new PdfRequestError(413, 'body-too-large', 'The mark is too large.')
  }
  try {
    return JSON.parse(text)
  } catch {
    throw new PdfRequestError(400, 'invalid-json', 'The request contains invalid JSON.')
  }
}

async function requireOwner(request: Request, env: PdfReaderEnv, slug?: string): Promise<string> {
  const owner = await getOwnerSessionLogin(request, env)
  if (!owner) {
    throw new PdfRequestError(401, 'unauthorized', 'Sign in to annotate.', {
      signInUrl: slug ? signInUrl(slug) : '/comments/github/login?returnTo=%2Fread',
    })
  }
  if (!isArenaReaderMutationAllowed(request)) {
    throw new PdfRequestError(403, 'invalid-origin', 'Reload the reader before saving.')
  }
  return owner
}

function resolveDocument(entry: ManifestCacheEntry, raw: string | null) {
  const slug = raw ? normalizePdfSlug(raw) : null
  const record = slug ? entry.manifest.documents[slug] : undefined
  if (!slug || !record) {
    throw new PdfRequestError(404, 'unknown-document', 'This PDF is not hosted here.')
  }
  return { slug, record }
}

async function handleMarksApi(request: Request, env: PdfReaderEnv, url: URL): Promise<Response> {
  const parts = url.pathname.split('/').filter(Boolean)
  const entry = await loadManifest(env, url.origin)

  if (parts.length === 3 && parts[2] === 'marks') {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      throw new PdfRequestError(405, 'method-not-allowed', 'Use GET to list marks.')
    }
    const { slug, record } = resolveDocument(entry, url.searchParams.get('src'))
    const owner = await getOwnerSessionLogin(request, env)
    const body: PdfMarksResponse = {
      src: slug,
      doc: record.doc,
      canWrite: owner !== null,
      signInUrl: signInUrl(slug),
      marks: await listMarks(env, record.doc, owner !== null),
      stale: owner ? await listStaleMarks(env, owner, slug, record.doc) : [],
    }
    return privateJson(body)
  }

  if (parts.length === 4 && parts[2] === 'marks' && parts[3] === 'export') {
    const { slug, record } = resolveDocument(entry, url.searchParams.get('src'))
    const owner = await getOwnerSessionLogin(request, env)
    if (!owner) throw new PdfRequestError(401, 'unauthorized', 'Sign in to export marks.')
    const marks = await listMarks(env, record.doc, true)
    return new Response(marksMarkdown(slug, record, marks, true), {
      headers: {
        'Content-Type': 'text/markdown; charset=utf-8',
        'Cache-Control': 'private, no-store',
        Vary: 'Cookie',
      },
    })
  }

  if (parts.length === 4 && parts[2] === 'marks') {
    const id = parts[3]
    if (!PDF_MARK_ID.test(id)) throw new PdfRequestError(404, 'not-found', 'Unknown mark.')
    if (request.method === 'PUT') {
      const owner = await requireOwner(request, env)
      const input = parsePdfMarkInput(await readJsonBody(request))
      if (!input) throw new PdfRequestError(422, 'invalid-input', 'The mark is malformed.')
      const { slug, record } = resolveDocument(entry, input.src)
      if (record.doc !== input.doc) {
        throw new PdfRequestError(409, 'stale-document', 'The PDF changed since you opened it.', {
          doc: record.doc,
        })
      }
      return privateJson(await saveMark(env, owner, id, { ...input, src: slug }))
    }
    if (request.method === 'DELETE') {
      const owner = await requireOwner(request, env)
      const body = await readJsonBody(request)
      const revision =
        typeof body === 'object' && body !== null ? Reflect.get(body, 'revision') : undefined
      if (typeof revision !== 'number' || !Number.isSafeInteger(revision) || revision < 1) {
        throw new PdfRequestError(422, 'invalid-input', 'A revision is required.')
      }
      return privateJson(await deleteMark(env, owner, id, revision))
    }
    throw new PdfRequestError(405, 'method-not-allowed', 'Use PUT or DELETE on a mark.')
  }

  throw new PdfRequestError(404, 'not-found', 'Unknown PDF route.')
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
}

function scriptJson(value: unknown): string {
  return JSON.stringify(value)
    .replaceAll('<', '\\u003c')
    .replaceAll(' ', '\\u2028')
    .replaceAll(' ', '\\u2029')
}

function quoteBlock(text: string): string {
  return text
    .replace(/\s+/g, ' ')
    .trim()
    .split('\n')
    .map(line => `> ${line}`)
    .join('\n')
}

const kindLabel = { mark: 'mark', question: 'question', contra: 'disagreement' } as const

function marksMarkdown(
  slug: string,
  record: PdfDocumentRecord,
  marks: PdfMark[],
  includeVisibility: boolean,
): string {
  const lines = [
    `# ${record.title}`,
    '',
    `PDF: [/${slug}](/${encodeURI(slug)}?raw=1)`,
    `Reader: ${encodeURI(pdfReaderPath(slug))}`,
    '',
  ]
  if (record.citedBy.length > 0) {
    lines.push('## Cited by', '')
    for (const citation of record.citedBy) {
      const where = citation.page ? ` (p. ${citation.page})` : ''
      lines.push(`- [${citation.title}](/${encodeURI(citation.from)})${where}: ${citation.excerpt}`)
    }
    lines.push('')
  }
  lines.push('## Marks', '')
  if (marks.length === 0) lines.push('No public marks yet.', '')
  for (const mark of marks) {
    const visibility = includeVisibility && mark.visibility === 'private' ? ', private' : ''
    lines.push(`### p. ${mark.target.page}, ${kindLabel[mark.kind]}${visibility} ^${mark.id}`, '')
    if (mark.target.type === 'text') lines.push(quoteBlock(mark.target.quote.exact), '')
    if (mark.body.trim()) lines.push(mark.body.trim(), '')
  }
  return `${lines.join('\n').trimEnd()}\n`
}

async function readerShell(
  env: PdfReaderEnv,
  request: Request,
  url: URL,
  title: string,
  data: PdfReaderData,
): Promise<Response> {
  const response = await env.ASSETS.fetch(new Request(new URL('/read', url.origin), request))
  if (!response.ok) return response
  // The shell is built once at `read`; client routing and link previews need the real path.
  const path = data.mode === 'document' && data.readPath ? data.readPath : PDF_READER_PREFIX
  const html = (await response.text())
    .replaceAll(TITLE_SLOT, escapeHtml(title))
    .replace(DATA_SLOT, DATA_SLOT.replace('>null<', `>${scriptJson(data)}<`))
    .replace('data-slug="read"', `data-slug="${escapeHtml(path.slice(1))}"`)
    .replace(
      /(["'])(https?:\/\/[^"'/]+)\/read\1/g,
      (_, quote, origin) => `${quote}${origin}${escapeHtml(encodeURI(path))}${quote}`,
    )
  const headers = new Headers(response.headers)
  headers.delete('ETag')
  headers.delete('Content-Length')
  headers.set('Content-Type', 'text/html; charset=utf-8')
  headers.set('Cache-Control', 'public, max-age=0, must-revalidate')
  headers.set('Vary', 'Accept, Accept-Encoding, User-Agent')
  return new Response(request.method === 'HEAD' ? null : html, { status: 200, headers })
}

async function handleReaderRoute(
  request: Request,
  env: PdfReaderEnv,
  url: URL,
): Promise<Response | null> {
  if (request.method !== 'GET' && request.method !== 'HEAD') return null
  const entry = await loadManifest(env, url.origin)
  const key = url.pathname.replace(/^\/read\/?/, '').replace(/\/+$/, '')

  if (!key) {
    const documents = Object.entries(entry.manifest.documents)
      .filter(([, record]) => record.citedBy.length > 0)
      .map(([slug, record]) => ({
        slug,
        readPath: pdfReaderPath(slug),
        title: record.title,
        citations: record.citedBy.length,
      }))
      .sort(
        (left, right) => right.citations - left.citations || left.title.localeCompare(right.title),
      )
    if (wantsMarkdown(request)) {
      const body = [
        '# Reader',
        '',
        ...documents.map(doc => `- [${doc.title}](${encodeURI(doc.readPath)}) (${doc.citations})`),
      ].join('\n')
      return new Response(`${body}\n`, {
        headers: {
          'Content-Type': 'text/markdown; charset=utf-8',
          Vary: 'Accept, Accept-Encoding, User-Agent',
        },
      })
    }
    return readerShell(env, request, url, 'reader', { mode: 'index', documents })
  }

  let decoded: string
  try {
    decoded = decodeURIComponent(key)
  } catch {
    return null
  }
  const slug = entry.byReaderKey.get(decoded)
  const record = slug ? entry.manifest.documents[slug] : undefined
  if (!slug || !record) return null

  if (wantsMarkdown(request)) {
    const marks = env.ARENA_READER ? await listMarks(env, record.doc, false) : []
    return new Response(marksMarkdown(slug, record, marks, false), {
      headers: {
        'Content-Type': 'text/markdown; charset=utf-8',
        'Cache-Control': 'public, max-age=60',
        Vary: 'Accept, Accept-Encoding, User-Agent',
      },
    })
  }
  return readerShell(env, request, url, record.title, {
    mode: 'document',
    slug,
    readPath: pdfReaderPath(slug),
    document: record,
  })
}

function isDocumentNavigation(request: Request): boolean {
  return (
    request.method === 'GET' &&
    request.headers.get('Sec-Fetch-Mode') === 'navigate' &&
    request.headers.get('Sec-Fetch-Dest') === 'document'
  )
}

async function handleHostedPdfNavigation(
  request: Request,
  env: PdfReaderEnv,
  url: URL,
): Promise<Response | null> {
  if (!isDocumentNavigation(request) || url.searchParams.has('raw')) return null
  const slug = normalizePdfSlug(url.pathname)
  if (!slug) return null
  const entry = await loadManifest(env, url.origin)
  if (!entry.manifest.documents[slug]) return null
  // no-store: a cached 301 would also answer PDF.js's same-URL byte fetches.
  return new Response(null, {
    status: 301,
    headers: {
      Location: encodeURI(pdfReaderPath(slug)),
      'Cache-Control': 'no-store',
      Vary: 'Sec-Fetch-Mode, Sec-Fetch-Dest',
    },
  })
}

export async function handlePdfReaderRequest(
  request: Request,
  env: PdfReaderEnv,
): Promise<Response | null> {
  const url = new URL(request.url)
  const isApi = url.pathname === '/api/pdf/marks' || url.pathname.startsWith('/api/pdf/marks/')
  const isReader = url.pathname === '/read' || url.pathname.startsWith('/read/')
  const isPdf = /\.pdf$/i.test(url.pathname)
  if (!isApi && !isReader && !isPdf) return null
  try {
    if (isApi) return await handleMarksApi(request, env, url)
    if (isReader) return await handleReaderRoute(request, env, url)
    return await handleHostedPdfNavigation(request, env, url)
  } catch (error) {
    if (error instanceof PdfRequestError) {
      if (!isApi) return null
      return privateJson(
        { error: error.code, message: error.message, ...error.extra },
        error.status,
      )
    }
    console.error('PDF reader request failed', error instanceof Error ? error.message : error)
    if (!isApi) return null
    return privateJson({ error: 'unavailable', message: 'Marks are unavailable right now.' }, 503)
  }
}
