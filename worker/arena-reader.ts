import { z } from 'zod'
import type { ArenaReaderErrorResponse } from '../quartz/util/arena-reader'
import {
  isArenaReadingEntry,
  orderArenaFeedEntries,
  parseArenaFeedManifest,
  type ArenaFeedManifest,
} from '../quartz/util/arena-feed'
import {
  createArenaResourceCapability,
  getArenaReaderIdentity,
  getArenaReaderLoginUrl,
  isArenaReaderMutationAllowed,
  verifyArenaResourceCapability,
} from './arena-reader-auth'
import { loadArenaReaderSnapshot } from './arena-reader-cache'
import {
  handleArenaReaderResource,
  readArenaRenderStatus,
  readArenaSnapshot,
  renderArenaArticle,
} from './arena-reader-render'
import {
  acknowledgeArenaNoteExport,
  ArenaReaderConflictError,
  ArenaReaderNoteNotFoundError,
  deleteArenaNote,
  exportReadyArenaNotes,
  listArenaNotes,
  listReadLinks,
  saveArenaNote,
  setReadLink,
} from './arena-reader-state'

const ARTICLE_ID = /^article-v1-[a-f0-9]{64}$/
const MAX_BODY_BYTES = 128 * 1024
const revision = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER)
const readInput = z.object({ read: z.boolean(), revision }).strict()
const renderInput = z.object({ refresh: z.boolean().default(false) }).strict()
const noteInput = z
  .object({
    articleId: z.string().regex(ARTICLE_ID),
    body: z
      .string()
      .max(20_000)
      .refine(value => value.trim().length > 0),
    snapshotId: z.string().uuid().nullable().default(null),
    quote: z
      .object({
        exact: z.string().min(1).max(4096),
        prefix: z.string().max(256),
        suffix: z.string().max(256),
      })
      .strict()
      .nullable()
      .default(null),
    occurrence: z
      .object({ channelSlug: z.string().max(256), blockId: z.string().max(256) })
      .strict()
      .nullable()
      .default(null),
    revision,
    ready: z.boolean().default(false),
  })
  .strict()
const deleteInput = z.object({ revision }).strict()
const receiptInput = z
  .object({ revision: revision.min(1), receipt: z.string().trim().min(1).max(4096) })
  .strict()

class ReaderRequestError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message)
  }
}

function privateResponse(response: Response): Response {
  const headers = new Headers(response.headers)
  headers.set('Cache-Control', 'private, no-store')
  headers.set('X-Content-Type-Options', 'nosniff')
  headers.set('Referrer-Policy', 'no-referrer')
  headers.delete('Access-Control-Allow-Origin')
  headers.delete('Access-Control-Allow-Credentials')
  return new Response(response.body, { status: response.status, headers })
}

function json(value: unknown, status = 200): Response {
  return privateResponse(Response.json(value, { status }))
}

function errorResponse(error: string, message: string, status: number): Response {
  return json({ error, message } satisfies ArenaReaderErrorResponse, status)
}

async function readBody<T>(request: Request, schema: z.ZodType<T>): Promise<T> {
  if (request.headers.get('Content-Type')?.split(';')[0].trim() !== 'application/json') {
    throw new ReaderRequestError(415, 'invalid-content-type', 'Send a JSON request body.')
  }
  const reader = request.body?.getReader()
  if (!reader) throw new ReaderRequestError(400, 'invalid-body', 'A JSON request body is required.')
  const chunks: Uint8Array[] = []
  let length = 0
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      length += value.byteLength
      if (length > MAX_BODY_BYTES) {
        await reader.cancel()
        throw new ReaderRequestError(413, 'body-too-large', 'The reader request is too large.')
      }
      chunks.push(value)
    }
  } finally {
    reader.releaseLock()
  }
  const bytes = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  let value: unknown
  try {
    value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes))
  } catch {
    throw new ReaderRequestError(400, 'invalid-json', 'The request contains invalid JSON.')
  }
  const result = schema.safeParse(value)
  if (!result.success) {
    throw new ReaderRequestError(
      400,
      'invalid-input',
      'The request contains invalid reader fields.',
    )
  }
  return result.data
}

async function loadCatalogue(env: Env, origin: string): Promise<ArenaFeedManifest> {
  const response = await env.ASSETS.fetch(new URL('/static/arena-feed.json', origin))
  if (!response.ok) {
    throw new ReaderRequestError(
      503,
      'catalogue-unavailable',
      'The reader catalogue is unavailable.',
    )
  }
  let value: unknown
  try {
    value = await response.json()
  } catch {
    throw new ReaderRequestError(503, 'catalogue-unavailable', 'The reader catalogue is invalid.')
  }
  const catalogue = parseArenaFeedManifest(value)
  if (!catalogue) {
    throw new ReaderRequestError(
      503,
      'catalogue-unavailable',
      'The reader catalogue needs rebuilding.',
    )
  }
  return catalogue
}

function requireMethod(request: Request, ...methods: string[]): void {
  if (!methods.includes(request.method)) {
    throw new ReaderRequestError(
      405,
      'method-not-allowed',
      'This reader route does not support that method.',
    )
  }
}

function requireMutation(request: Request, subject: string, checkSubject = true): void {
  if (
    !isArenaReaderMutationAllowed(request) ||
    (checkSubject && request.headers.get('X-Arena-Subject') !== subject)
  ) {
    throw new ReaderRequestError(
      403,
      'invalid-origin',
      'Reload the reader before saving this change.',
    )
  }
}

async function dispatch(request: Request, env: Env): Promise<Response | null> {
  const url = new URL(request.url)
  const shell = /^\/arena\/feed(?:\/|\.html|\/index(?:\.html)?)?$/.test(url.pathname)
  if (!shell && !url.pathname.startsWith('/api/arena/')) return null

  const parts = url.pathname.split('/').filter(Boolean)
  const resourceScope =
    parts.length === 8 &&
    parts[2] === 'articles' &&
    parts[4] === 'snapshots' &&
    parts[6] === 'resources'
      ? { articleId: parts[3], snapshotId: parts[5], resourceId: parts[7] }
      : null
  const identity = await getArenaReaderIdentity(request, env)
  const capability = resourceScope
    ? await verifyArenaResourceCapability(env, url.searchParams.get('token') ?? '', resourceScope)
    : false
  if (!identity && !capability) {
    const loginUrl = getArenaReaderLoginUrl(request)
    return shell
      ? privateResponse(Response.redirect(loginUrl, 302))
      : json(
          { error: 'unauthorized', message: 'Sign in to open your Arena reader.', loginUrl },
          401,
        )
  }

  if (shell) {
    requireMethod(request, 'GET', 'HEAD')
    return privateResponse(await env.ASSETS.fetch(request))
  }

  if (!env.ARENA_CONTENT || !env.ARENA_READER) {
    throw new ReaderRequestError(
      503,
      'storage-unavailable',
      'Reader storage has not been configured.',
    )
  }
  const catalogue = await loadCatalogue(env, url.origin)
  const entry =
    parts[2] === 'articles'
      ? catalogue.entries.find(item => item.articleId === parts[3])
      : undefined
  if (parts[2] === 'articles' && !entry) {
    throw new ReaderRequestError(
      404,
      'unknown-article',
      'This link is not in the current Arena catalogue.',
    )
  }
  const renderEnv = {
    ...env,
    arenaReaderResourceToken: (articleId: string, snapshotId: string, resourceId: string) =>
      createArenaResourceCapability(env, { articleId, snapshotId, resourceId }),
    arenaReaderAcquireRenderPermit: async () => {
      if (!identity || !env.ARENA_RENDER_RATE_LIMITER) return false
      const result = await env.ARENA_RENDER_RATE_LIMITER.limit({ key: identity.subject })
      return result.success
    },
  }

  if (resourceScope && entry) {
    requireMethod(request, 'GET', 'HEAD')
    const response = privateResponse(
      await handleArenaReaderResource(
        request,
        renderEnv,
        entry,
        resourceScope.snapshotId,
        resourceScope.resourceId,
      ),
    )
    if (capability) response.headers.set('Cross-Origin-Resource-Policy', 'cross-origin')
    return response
  }
  if (!identity) return errorResponse('unauthorized', 'Sign in to use the reader.', 401)
  const { subject } = identity

  if (url.pathname === '/api/arena/feed') {
    requireMethod(request, 'GET')
    const seed = url.searchParams.get('seed') ?? 'arena'
    if (seed.length > 128)
      throw new ReaderRequestError(400, 'invalid-seed', 'The shuffle seed is too long.')
    return json({
      subject,
      revision: catalogue.revision,
      entries: orderArenaFeedEntries(catalogue.entries.filter(isArenaReadingEntry), seed),
      readLinks: await listReadLinks(env.ARENA_READER, subject),
    })
  }

  if (entry && parts.length === 5) {
    switch (parts[4]) {
      case 'render': {
        requireMethod(request, 'POST')
        requireMutation(request, subject, false)
        const body = await readBody(request, renderInput)
        const renderRequest = new Request(request.url, {
          method: 'POST',
          headers: request.headers,
          body: JSON.stringify(body),
          signal: request.signal,
        })
        return privateResponse(await renderArenaArticle(renderRequest, renderEnv, entry))
      }
      case 'render-status':
        requireMethod(request, 'GET')
        return privateResponse(await readArenaRenderStatus(renderEnv, entry))
      case 'read':
        requireMethod(request, 'PUT')
        requireMutation(request, subject)
        return json({
          readLink: await setReadLink(
            env.ARENA_READER,
            subject,
            entry.articleId,
            await readBody(request, readInput),
          ),
        })
      case 'notes':
        requireMethod(request, 'GET')
        return json({ notes: await listArenaNotes(env.ARENA_READER, subject, entry.articleId) })
    }
  }
  if (entry && parts.length === 6 && parts[4] === 'snapshots') {
    requireMethod(request, 'GET')
    if (!z.string().uuid().safeParse(parts[5]).success) {
      throw new ReaderRequestError(400, 'invalid-snapshot', 'The saved copy identifier is invalid.')
    }
    return privateResponse(await readArenaSnapshot(renderEnv, entry, parts[5]))
  }

  if (url.pathname === '/api/arena/notes') {
    requireMethod(request, 'GET')
    const view = url.searchParams.get('view') ?? 'inbox'
    if (view === 'ready') return json(await exportReadyArenaNotes(env.ARENA_READER, subject))
    if (view !== 'inbox')
      throw new ReaderRequestError(400, 'invalid-view', 'Choose the Inbox or Ready notes view.')
    return json({ notes: await listArenaNotes(env.ARENA_READER, subject) })
  }
  if (
    parts[2] === 'notes' &&
    (parts.length === 4 || (parts.length === 5 && parts[4] === 'export-receipt'))
  ) {
    const noteId = parts[3]
    if (!z.string().uuid().safeParse(noteId).success) {
      throw new ReaderRequestError(400, 'invalid-note', 'The note identifier is invalid.')
    }
    requireMutation(request, subject)
    if (parts.length === 5) {
      requireMethod(request, 'POST')
      return json({
        note: await acknowledgeArenaNoteExport(
          env.ARENA_READER,
          subject,
          noteId,
          await readBody(request, receiptInput),
        ),
      })
    }
    requireMethod(request, 'PUT', 'DELETE')
    if (request.method === 'DELETE') {
      const body = await readBody(request, deleteInput)
      return json({ note: await deleteArenaNote(env.ARENA_READER, subject, noteId, body.revision) })
    }
    const input = await readBody(request, noteInput)
    const article = catalogue.entries.find(item => item.articleId === input.articleId)
    if (!article)
      throw new ReaderRequestError(
        404,
        'unknown-article',
        'This link is not in the current Arena catalogue.',
      )
    if (
      input.occurrence &&
      !article.occurrences.some(
        occurrence =>
          occurrence.channelSlug === input.occurrence?.channelSlug &&
          occurrence.blockId === input.occurrence.blockId,
      )
    ) {
      throw new ReaderRequestError(
        400,
        'invalid-occurrence',
        'The saved block no longer belongs to this link. Refresh the catalogue.',
      )
    }
    if (
      input.snapshotId &&
      !(await loadArenaReaderSnapshot(env.ARENA_CONTENT, article.articleId, input.snapshotId))
    ) {
      throw new ReaderRequestError(
        400,
        'invalid-snapshot',
        'This saved article copy is unavailable. Keep the note locally and retry.',
      )
    }
    if (input.quote && !input.snapshotId) {
      throw new ReaderRequestError(
        400,
        'missing-snapshot',
        'A quoted note must identify its saved article copy.',
      )
    }
    return json({
      note: await saveArenaNote(env.ARENA_READER, subject, noteId, {
        ...input,
        sourceUrl: article.sourceUrl,
      }),
    })
  }
  return errorResponse('not-found', 'The reader route was not found.', 404)
}

export async function handleArenaReaderRequest(
  request: Request,
  env: Env,
): Promise<Response | null> {
  try {
    return await dispatch(request, env)
  } catch (error) {
    if (error instanceof ReaderRequestError)
      return errorResponse(error.code, error.message, error.status)
    if (error instanceof ArenaReaderConflictError) {
      return json({ error: 'conflict', message: error.message, current: error.current }, 409)
    }
    if (error instanceof ArenaReaderNoteNotFoundError)
      return errorResponse('not-found', error.message, 404)
    console.error(
      'Arena reader request failed',
      error instanceof Error ? error.message : 'Unknown error',
    )
    return errorResponse(
      'reader-unavailable',
      'The reader could not complete this request. Your local notes are retained.',
      503,
    )
  }
}
