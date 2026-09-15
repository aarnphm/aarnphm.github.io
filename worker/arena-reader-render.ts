import type { BrowserWorker, HTTPRequest, Page } from '@cloudflare/puppeteer'
import puppeteer from '@cloudflare/puppeteer'
import readabilitySource from '@mozilla/readability/Readability.js'
import purifySource from 'dompurify/dist/purify.js'
import type { ArenaFeedEntry } from '../quartz/util/arena-feed'
import type {
  ArenaReaderArtifact,
  ArenaReaderArtifactBase,
  ArenaReaderRenderResult,
  ArenaReaderResource,
} from '../quartz/util/arena-reader'
import type { ArenaExtractedDocument } from './arena-reader-extraction'
import {
  ARENA_READER_COOLDOWN_MS,
  ARENA_READER_PROFILE,
  arenaReaderHash,
  arenaRenderCacheDecision,
  claimArenaRenderLease,
  keepCompleteArenaSnapshot,
  loadArenaReaderSnapshot,
  publishArenaRenderState,
  readArenaRenderCache,
  saveArenaReaderSnapshot,
} from './arena-reader-cache'
import { extractArenaReaderDocument } from './arena-reader-extraction'
import {
  fetchArenaReaderSource,
  serveArenaReaderResource,
  validateArenaReaderTarget,
} from './arena-reader-resources'

const CAPTURE_TIMEOUT_MS = 55_000
const DOCUMENT_LIMIT = 2 * 1024 * 1024
const SESSION_BYTES_LIMIT = 12 * 1024 * 1024
const SESSION_REQUEST_LIMIT = 80
const SOURCE_CSP =
  "worker-src 'none'; frame-src 'none'; object-src 'none'; media-src 'none'; connect-src http: https:; form-action 'none'"

export interface ArenaReaderRenderEnv {
  ARENA_CONTENT: R2Bucket
  BROWSER: BrowserWorker
  arenaReaderResourceToken?: (
    articleId: string,
    snapshotId: string,
    resourceId: string,
  ) => Promise<string>
  arenaReaderAcquireRenderPermit?: () => Promise<boolean>
}

class ArenaCaptureError extends Error {
  constructor(
    readonly reason: string,
    message: string,
    readonly status = 502,
    readonly retryAfter?: number,
  ) {
    super(message)
    this.name = 'ArenaCaptureError'
  }
}

function readerResponse(value: ArenaReaderRenderResult, status = 200): Response {
  const headers = new Headers({
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'private, no-store',
    'X-Content-Type-Options': 'nosniff',
  })
  if (value.status !== 'ready' && value.retryAfter)
    headers.set('Retry-After', String(value.retryAfter))
  return new Response(JSON.stringify(value), { status, headers })
}

function unavailable(
  entry: ArenaFeedEntry,
  reason: string,
  message: string,
  status = 502,
  retryAfter?: number,
): Response {
  return readerResponse(
    { status: 'unavailable', reason, message, sourceUrl: entry.sourceUrl, retryAfter },
    status,
  )
}

function pending(entry: ArenaFeedEntry): Response {
  return readerResponse(
    {
      status: 'pending',
      retryAfter: 3,
      statusUrl: `/api/arena/articles/${entry.articleId}/render-status`,
    },
    202,
  )
}

export function arenaReaderResourcePath(
  articleId: string,
  snapshotId: string,
  resourceId: string,
): string {
  return `/api/arena/articles/${articleId}/snapshots/${snapshotId}/resources/${resourceId}`
}

async function deliverArtifact(
  env: ArenaReaderRenderEnv,
  stored: ArenaReaderArtifact,
  cached: boolean,
  warning?: string,
): Promise<Response> {
  const artifact = structuredClone(stored)
  for (const resource of artifact.resources) {
    const path = arenaReaderResourcePath(artifact.articleId, artifact.snapshotId, resource.id)
    const token = await env.arenaReaderResourceToken?.(
      artifact.articleId,
      artifact.snapshotId,
      resource.id,
    )
    const deliveryUrl = token ? `${path}?token=${encodeURIComponent(token)}` : path
    resource.url = deliveryUrl
    if (artifact.kind === 'html') {
      // Only replace the exact generated attribute; publisher markup has no route authority.
      artifact.readerHtml =
        artifact.readerHtml?.replaceAll(`src="${path}"`, `src="${deliveryUrl}"`) ?? null
      artifact.documentHtml = artifact.documentHtml.replaceAll(
        `src="${path}"`,
        `src="${deliveryUrl}"`,
      )
    }
  }
  return readerResponse({ status: 'ready', cached, artifact, warning })
}

export async function readArenaSnapshot(
  env: ArenaReaderRenderEnv,
  entry: ArenaFeedEntry,
  snapshotId: string,
): Promise<Response> {
  const artifact = await loadArenaReaderSnapshot(env.ARENA_CONTENT, entry.articleId, snapshotId)
  return artifact
    ? deliverArtifact(env, artifact, true)
    : unavailable(entry, 'snapshot-missing', 'This saved article copy is unavailable.', 404)
}

export async function handleArenaReaderResource(
  request: Request,
  env: ArenaReaderRenderEnv,
  entry: ArenaFeedEntry,
  snapshotId: string,
  resourceId: string,
): Promise<Response> {
  const artifact = await loadArenaReaderSnapshot(env.ARENA_CONTENT, entry.articleId, snapshotId)
  const resource = artifact?.resources.find(item => item.id === resourceId)
  if (!artifact || !resource)
    return unavailable(
      entry,
      'resource-missing',
      'This resource is absent from the saved copy.',
      404,
    )
  return serveArenaReaderResource(request, env.ARENA_CONTENT, {
    articleId: entry.articleId,
    snapshotId,
    resourceId,
    sourceUrl: resource.url,
    purpose: resource.kind,
  })
}

export async function readArenaRenderStatus(
  env: ArenaReaderRenderEnv,
  entry: ArenaFeedEntry,
): Promise<Response> {
  const { state } = await readArenaRenderCache(env.ARENA_CONTENT, entry.articleId)
  if (state.lease && state.lease.expiresAt > Date.now()) return pending(entry)
  if (state.snapshotId) {
    const artifact = await loadArenaReaderSnapshot(
      env.ARENA_CONTENT,
      entry.articleId,
      state.snapshotId,
    )
    if (artifact) return deliverArtifact(env, artifact, true, state.failure?.message)
  }
  if (state.failure)
    return unavailable(
      entry,
      state.failure.reason,
      state.failure.message,
      200,
      Math.max(0, Math.ceil((state.failure.retryAt - Date.now()) / 1000)),
    )
  return unavailable(entry, 'not-rendered', 'Open this link to create its first saved copy.', 200)
}

async function boundedBody(response: Response, limit: number): Promise<Uint8Array<ArrayBuffer>> {
  const length = response.headers.get('Content-Length')
  if (length && Number(length) > limit) {
    await response.body?.cancel()
    throw new ArenaCaptureError('too-large', 'The source document exceeds the reader size limit.')
  }
  const reader = response.body?.getReader()
  if (!reader) return new Uint8Array()
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      total += value.byteLength
      if (total > limit)
        throw new ArenaCaptureError(
          'too-large',
          'The source document exceeds the reader size limit.',
        )
      chunks.push(value)
    }
  } finally {
    await reader.cancel().catch(() => {})
  }
  const bytes = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return bytes
}

export function arenaReaderRelayHeaders(
  headers: Headers,
  document: boolean,
): Record<string, string> {
  const output: Record<string, string> = {}
  for (const key of [
    'content-type',
    'content-language',
    'access-control-allow-origin',
    'access-control-allow-methods',
    'access-control-allow-headers',
  ]) {
    const value = headers.get(key)
    if (value) output[key] = value
  }
  if (document) output['content-security-policy'] = SOURCE_CSP
  return output
}

export function arenaReaderFailureSignals(input: {
  title: string
  text: string
  articleLength: number
  hasArticle: boolean
}): { reason: string; message: string } | null {
  const title = input.title.toLowerCase()
  const text = input.text.toLowerCase()
  const challengeTitle =
    /^(just a moment|access denied|attention required|security check|verify you are human)/.test(
      title,
    )
  const challengeText =
    /verify (?:that )?you are human|checking your browser|enable javascript and cookies to continue|complete the security check/.test(
      text,
    )
  if (challengeTitle || (challengeText && input.articleLength < 1200))
    return {
      reason: 'blocked',
      message: 'The publisher returned a browser challenge instead of the article.',
    }
  if (
    /^(sign in|log in|login|subscribe to continue)/.test(title) &&
    input.articleLength < 600 &&
    !input.hasArticle
  )
    return {
      reason: 'requires-login',
      message: 'The source requires a publisher login. Open the original to continue.',
    }
  if (input.text.trim().length < 40)
    return { reason: 'empty', message: 'The page did not produce readable article text.' }
  return null
}

function videoEmbed(url: URL): string | null {
  const host = url.hostname.toLowerCase().replace(/^www\./, '')
  let id: string | null = null
  if (host === 'youtu.be') id = url.pathname.split('/')[1] ?? null
  if (host === 'youtube.com' || host === 'm.youtube.com') {
    id = url.searchParams.get('v')
    if (!id && /^\/(shorts|embed)\//.test(url.pathname)) id = url.pathname.split('/')[2] ?? null
  }
  if (id && /^[\w-]{11}$/.test(id)) return `https://www.youtube-nocookie.com/embed/${id}`
  if (host === 'vimeo.com') {
    const vimeoId = url.pathname.split('/').find(part => /^\d+$/.test(part))
    if (vimeoId) return `https://player.vimeo.com/video/${vimeoId}?dnt=1`
  }
  return null
}

async function artifactBase(
  entry: ArenaFeedEntry,
  finalUrl = entry.sourceUrl,
): Promise<ArenaReaderArtifactBase> {
  return {
    schemaVersion: 1,
    articleId: entry.articleId,
    snapshotId: crypto.randomUUID(),
    title: entry.title,
    sourceUrl: entry.sourceUrl,
    finalUrl,
    capturedAt: Date.now(),
    profileVersion: ARENA_READER_PROFILE,
    fingerprint: await arenaReaderHash(`${entry.kind}\n${finalUrl}`),
    resources: [],
  }
}

async function pdfArtifact(
  entry: ArenaFeedEntry,
  finalUrl = entry.sourceUrl,
): Promise<ArenaReaderArtifact> {
  const base = await artifactBase(entry, finalUrl)
  const id = `resource-${(await arenaReaderHash(finalUrl)).slice(0, 32)}`
  return {
    ...base,
    kind: 'pdf',
    resourceId: id,
    resources: [{ id, url: finalUrl, kind: 'pdf', contentType: 'application/pdf' }],
  }
}

export async function buildArenaHtmlArtifact(
  base: ArenaReaderArtifactBase,
  extracted: ArenaExtractedDocument,
  diagnostics: Set<string>,
): Promise<ArenaReaderArtifact> {
  const resources: ArenaReaderResource[] = []
  const fingerprint = await arenaReaderHash(
    JSON.stringify([extracted.readerHtml, extracted.documentHtml, extracted.imageUrls]),
  )
  const prefix = `arena-${base.snapshotId}-`
  let readerHtml =
    extracted.readerHtml
      ?.replaceAll('id="arena-content-', `id="${prefix}`)
      .replaceAll('href="#arena-content-', `href="#${prefix}`) ?? null
  let documentHtml = extracted.documentHtml
    .replaceAll('id="arena-content-', `id="${prefix}`)
    .replaceAll('href="#arena-content-', `href="#${prefix}`)
  for (const [index, sourceUrl] of extracted.imageUrls.entries()) {
    const marker = `data-arena-image="${index}"`
    if (!validateArenaReaderTarget(sourceUrl)) {
      readerHtml = readerHtml?.replaceAll(marker, '') ?? null
      documentHtml = documentHtml.replaceAll(marker, '')
      continue
    }
    const id = `resource-${(await arenaReaderHash(sourceUrl)).slice(0, 32)}`
    resources.push({ id, url: sourceUrl, kind: 'image' })
    const replacement = `src="${arenaReaderResourcePath(base.articleId, base.snapshotId, id)}"`
    readerHtml = readerHtml?.replaceAll(marker, replacement) ?? null
    documentHtml = documentHtml.replaceAll(marker, replacement)
  }
  if (!readerHtml)
    diagnostics.add(
      'The article extractor could not isolate an article. The full document remains available.',
    )
  return {
    ...base,
    title: (extracted.title || base.title).slice(0, 4096),
    fingerprint,
    resources,
    kind: 'html',
    readerHtml,
    documentHtml,
    quality: readerHtml && diagnostics.size === 0 ? 'complete' : 'partial',
    diagnostics: Array.from(diagnostics).slice(0, 30),
  }
}

async function captureHtml(
  entry: ArenaFeedEntry,
  env: ArenaReaderRenderEnv,
  requestSignal: AbortSignal,
): Promise<ArenaReaderArtifact> {
  const controller = new AbortController()
  const cancel = () => controller.abort()
  requestSignal.addEventListener('abort', cancel, { once: true })
  if (requestSignal.aborted) controller.abort()
  const timer = setTimeout(() => controller.abort(), CAPTURE_TIMEOUT_MS)
  let browser: Awaited<ReturnType<typeof puppeteer.launch>> | null = null
  const diagnostics = new Set<string>()
  const pendingRequests = new Set<Promise<void>>()
  try {
    const first = await fetchArenaReaderSource(entry.sourceUrl, {
      signal: controller.signal,
      headers: { Accept: 'text/html,application/xhtml+xml,application/pdf;q=0.9' },
    })
    if (!first.response.ok) {
      await first.response.body?.cancel()
      const retry = Number(first.response.headers.get('Retry-After'))
      const status = first.response.status
      throw new ArenaCaptureError(
        status === 404
          ? 'not-found'
          : status === 401
            ? 'requires-login'
            : status === 403
              ? 'blocked'
              : 'upstream-error',
        `The publisher returned HTTP ${status}. Open the original or retry later.`,
        status === 429 ? 429 : 502,
        Number.isFinite(retry) && retry > 0 ? Math.min(retry, 86_400) : undefined,
      )
    }
    const contentType = first.response.headers
      .get('Content-Type')
      ?.split(';')[0]
      .trim()
      .toLowerCase()
    if (contentType === 'application/pdf') {
      await first.response.body?.cancel()
      return pdfArtifact(entry, first.finalUrl)
    }
    if (contentType !== 'text/html' && contentType !== 'application/xhtml+xml') {
      await first.response.body?.cancel()
      return {
        ...(await artifactBase(entry, first.finalUrl)),
        kind: 'external',
        reason: 'unsupported',
        message: 'This source is not an HTML article or PDF. Open the original to view it.',
      }
    }
    const initialBody = await boundedBody(first.response, DOCUMENT_LIMIT)
    if (env.arenaReaderAcquireRenderPermit && !(await env.arenaReaderAcquireRenderPermit()))
      throw new ArenaCaptureError(
        'rate-limited',
        'The reader has reached its rendering limit. Retry in a minute.',
        429,
        60,
      )
    controller.signal.throwIfAborted()
    // Every permitted page request is fulfilled through public-only Worker fetch. The browser
    // has no direct HTTP egress, including DNS rebinding and requests from unexpected targets.
    const launch = puppeteer
      .launch(env.BROWSER, { guardrails: { allowedDomains: [] } })
      .then(async created => {
        if (controller.signal.aborted) {
          await created.close().catch(() => {})
          controller.signal.throwIfAborted()
        }
        return created
      })
    let cancelLaunch: (() => void) | undefined
    const aborted = new Promise<never>((_, reject) => {
      cancelLaunch = () =>
        reject(new DOMException('The browser launch was cancelled.', 'AbortError'))
      controller.signal.addEventListener('abort', cancelLaunch, { once: true })
      if (controller.signal.aborted) cancelLaunch()
    })
    try {
      browser = await Promise.race([launch, aborted])
    } finally {
      if (cancelLaunch) controller.signal.removeEventListener('abort', cancelLaunch)
    }
    const activeBrowser = browser
    const closeOnAbort = () => void activeBrowser.close().catch(() => {})
    controller.signal.addEventListener('abort', closeOnAbort, { once: true })
    const sourceContext = await browser.createBrowserContext()
    const page = await sourceContext.newPage()
    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
    await page.setBypassServiceWorker(true)
    await page.setRequestInterception(true)
    let totalBytes = initialBody.byteLength
    let requestCount = 0
    let servedInitial = false
    const initialUrl = new URL(first.finalUrl)
    initialUrl.hash = ''
    const handleRequest = async (request: HTTPRequest): Promise<void> => {
      if (request.isInterceptResolutionHandled()) return
      const type = request.resourceType()
      if (
        controller.signal.aborted ||
        request.method() !== 'GET' ||
        !['document', 'script', 'stylesheet', 'xhr', 'fetch'].includes(type) ||
        (type === 'document' && request.frame() !== page.mainFrame())
      ) {
        if (request.method() !== 'GET' || ['xhr', 'fetch', 'document'].includes(type))
          diagnostics.add('Some interactive content could not be loaded anonymously.')
        await request.abort('blockedbyclient')
        return
      }
      requestCount++
      if (requestCount > SESSION_REQUEST_LIMIT || totalBytes > SESSION_BYTES_LIMIT) {
        diagnostics.add('The page reached the bounded resource-loading limit.')
        await request.abort('blockedbyclient')
        return
      }
      if (!servedInitial && request.url() === initialUrl.href) {
        servedInitial = true
        await request.respond({
          status: first.response.status,
          headers: arenaReaderRelayHeaders(first.response.headers, true),
          body: initialBody,
        })
        return
      }
      try {
        const upstream = await fetchArenaReaderSource(request.url(), {
          signal: controller.signal,
          headers: { Accept: request.headers().accept ?? '*/*' },
        })
        const final = new URL(upstream.finalUrl)
        final.hash = ''
        if (final.href !== request.url()) {
          await upstream.response.body?.cancel()
          await request.respond({ status: 302, headers: { location: final.href } })
          return
        }
        const body = await boundedBody(upstream.response, DOCUMENT_LIMIT)
        totalBytes += body.byteLength
        if (totalBytes > SESSION_BYTES_LIMIT) {
          diagnostics.add('The page reached the bounded resource-loading limit.')
          await request.abort('blockedbyclient')
          return
        }
        if (!upstream.response.ok) diagnostics.add('Some publisher resources returned an error.')
        await request.respond({
          status: upstream.response.status,
          headers: arenaReaderRelayHeaders(upstream.response.headers, type === 'document'),
          body,
        })
      } catch {
        diagnostics.add('Some publisher resources could not be loaded through the reader.')
        if (!request.isInterceptResolutionHandled()) await request.abort('failed').catch(() => {})
      }
    }
    page.on('request', request => {
      const promise = handleRequest(request).catch(() => {})
      pendingRequests.add(promise)
      void promise.finally(() => pendingRequests.delete(promise))
    })
    const navigation = await page.goto(first.finalUrl, {
      waitUntil: 'domcontentloaded',
      timeout: 25_000,
    })
    if (navigation && !navigation.ok())
      throw new ArenaCaptureError('upstream-error', 'The publisher did not produce a usable page.')
    await page.waitForNetworkIdle({ idleTime: 600, timeout: 5000 }).catch(() => {})
    for (let index = 0; index < 2; index++) {
      controller.signal.throwIfAborted()
      await page.evaluate(() => window.scrollBy(0, Math.min(window.innerHeight, 900)))
      await new Promise(resolve => setTimeout(resolve, 250))
    }
    await page.waitForNetworkIdle({ idleTime: 300, timeout: 1500 }).catch(() => {})
    const html = await page.evaluate(() => {
      for (const image of document.querySelectorAll('img')) {
        const source =
          image.currentSrc ||
          image.getAttribute('data-src') ||
          image.getAttribute('data-original') ||
          image.src
        if (source) image.setAttribute('src', source)
      }
      return document.documentElement.outerHTML.slice(0, 2 * 1024 * 1024 + 1)
    })
    if (new TextEncoder().encode(html).byteLength > DOCUMENT_LIMIT)
      throw new ArenaCaptureError(
        'too-large',
        'The rendered document exceeds the reader size limit.',
      )
    const finalUrl = page.url()
    if (!validateArenaReaderTarget(finalUrl))
      throw new ArenaCaptureError(
        'blocked',
        'The publisher navigated to an unsupported destination.',
      )
    await sourceContext.close()
    const parseContext = await browser.createBrowserContext()
    const parser: Page = await parseContext.newPage()
    await parser.setRequestInterception(true)
    parser.on('request', request => void request.abort('blockedbyclient').catch(() => {}))
    await parser.setContent('<!doctype html><html><head></head><body></body></html>')
    await parser.addScriptTag({ content: readabilitySource })
    await parser.addScriptTag({ content: purifySource })
    const base = await artifactBase(entry, finalUrl)
    const extracted = await parser.evaluate(extractArenaReaderDocument, {
      html,
      finalUrl,
      idPrefix: 'arena-content-',
    })
    const failure = arenaReaderFailureSignals(extracted)
    if (failure) throw new ArenaCaptureError(failure.reason, failure.message)
    return buildArenaHtmlArtifact(base, extracted, diagnostics)
  } catch (error) {
    if (error instanceof ArenaCaptureError) throw error
    if (requestSignal.aborted)
      throw new ArenaCaptureError('cancelled', 'The rendering request was cancelled.', 499, 1)
    if (controller.signal.aborted || (error instanceof Error && error.name === 'TimeoutError'))
      throw new ArenaCaptureError(
        'timeout',
        'The publisher did not finish rendering within the reader deadline.',
      )
    throw new ArenaCaptureError(
      'render-failed',
      'This page could not be rendered. Open the original or retry later.',
    )
  } finally {
    requestSignal.removeEventListener('abort', cancel)
    clearTimeout(timer)
    controller.abort()
    if (browser) await browser.close().catch(() => {})
    await Promise.allSettled(pendingRequests)
  }
}

async function buildArtifact(
  entry: ArenaFeedEntry,
  env: ArenaReaderRenderEnv,
  signal: AbortSignal,
): Promise<ArenaReaderArtifact> {
  const target = validateArenaReaderTarget(entry.sourceUrl)
  if (!target)
    throw new ArenaCaptureError('blocked', 'This saved URL cannot be fetched by the reader.', 400)
  if (entry.kind === 'pdf') return pdfArtifact(entry)
  if (entry.kind === 'internal')
    return {
      ...(await artifactBase(entry)),
      kind: 'internal',
      internalUrl: `${target.pathname}${target.search}${target.hash}`,
    }
  if (entry.kind === 'video')
    return {
      ...(await artifactBase(entry)),
      kind: 'video',
      embedUrl: videoEmbed(target),
      description: null,
    }
  return captureHtml(entry, env, signal)
}

export async function renderArenaArticle(
  request: Request,
  env: ArenaReaderRenderEnv,
  entry: ArenaFeedEntry,
): Promise<Response> {
  if (request.method !== 'POST')
    return unavailable(entry, 'method-not-allowed', 'Rendering requires POST.', 405)
  let refresh = false
  if (request.body) {
    const body = await boundedBody(new Response(request.body), 1024)
    if (body.byteLength) {
      let value: unknown
      try {
        value = JSON.parse(new TextDecoder().decode(body))
      } catch {
        return unavailable(
          entry,
          'invalid-body',
          'Provide a JSON object with an optional refresh flag.',
          400,
        )
      }
      if (
        !value ||
        typeof value !== 'object' ||
        Array.isArray(value) ||
        ('refresh' in value && typeof value.refresh !== 'boolean')
      )
        return unavailable(
          entry,
          'invalid-body',
          'Provide a JSON object with an optional refresh flag.',
          400,
        )
      refresh = 'refresh' in value && value.refresh === true
    }
  }
  const cached = await readArenaRenderCache(env.ARENA_CONTENT, entry.articleId)
  const previous = cached.state.snapshotId
    ? await loadArenaReaderSnapshot(env.ARENA_CONTENT, entry.articleId, cached.state.snapshotId)
    : null
  const decision = arenaRenderCacheDecision(
    previous ? cached.state : { ...cached.state, snapshotId: null },
    Date.now(),
    refresh,
  )
  if (decision === 'ready' && previous) return deliverArtifact(env, previous, true)
  if (decision === 'pending') return pending(entry)
  if (decision === 'cooldown' && cached.state.failure) {
    if (previous) return deliverArtifact(env, previous, true, cached.state.failure.message)
    return unavailable(
      entry,
      cached.state.failure.reason,
      cached.state.failure.message,
      429,
      Math.max(1, Math.ceil((cached.state.failure.retryAt - Date.now()) / 1000)),
    )
  }
  const lease = await claimArenaRenderLease(env.ARENA_CONTENT, entry.articleId, cached)
  if (!lease) return pending(entry)
  try {
    const artifact = await buildArtifact(entry, env, request.signal)
    // Retry storage from this retained extraction once before returning an unsaved copy.
    let saved = false
    for (let attempt = 0; attempt < 2 && !saved; attempt++) {
      try {
        saved = await saveArenaReaderSnapshot(env.ARENA_CONTENT, artifact)
      } catch {
        if (attempt === 1) break
      }
    }
    if (!saved) {
      await publishArenaRenderState(
        env.ARENA_CONTENT,
        entry.articleId,
        lease,
        previous?.snapshotId ?? null,
        {
          reason: 'storage-failed',
          message: 'This copy could not be saved. The previous saved copy remains available.',
          retryAt: Date.now() + 60_000,
        },
      ).catch(() => false)
      return deliverArtifact(
        env,
        artifact,
        false,
        'This copy is available for this visit, but could not be saved for later.',
      )
    }
    if (keepCompleteArenaSnapshot(previous, artifact) && previous) {
      const message =
        'The refreshed source was incomplete. Your previous complete copy has been kept.'
      await publishArenaRenderState(
        env.ARENA_CONTENT,
        entry.articleId,
        lease,
        previous.snapshotId,
        { reason: 'partial-refresh', message, retryAt: Date.now() + ARENA_READER_COOLDOWN_MS },
      )
      return deliverArtifact(env, previous, true, message)
    }
    const published = await publishArenaRenderState(
      env.ARENA_CONTENT,
      entry.articleId,
      lease,
      artifact.snapshotId,
    )
    return deliverArtifact(
      env,
      artifact,
      false,
      published
        ? undefined
        : 'This version is saved. A newer render attempt controls the current copy.',
    )
  } catch (error) {
    const failure =
      error instanceof ArenaCaptureError
        ? error
        : new ArenaCaptureError(
            'render-failed',
            'This page could not be rendered. Open the original or retry later.',
          )
    const retryAfter = failure.retryAfter ?? Math.ceil(ARENA_READER_COOLDOWN_MS / 1000)
    await publishArenaRenderState(
      env.ARENA_CONTENT,
      entry.articleId,
      lease,
      previous?.snapshotId ?? null,
      { reason: failure.reason, message: failure.message, retryAt: Date.now() + retryAfter * 1000 },
    ).catch(() => false)
    if (previous) return deliverArtifact(env, previous, true, failure.message)
    return unavailable(entry, failure.reason, failure.message, failure.status, retryAfter)
  }
}
