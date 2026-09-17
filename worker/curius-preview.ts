import type { BrowserWorker } from '@cloudflare/puppeteer'
import puppeteer from '@cloudflare/puppeteer'
import defuddleSource from 'defuddle/full'
import purifySource from 'dompurify/dist/purify.js'
import type { CuriusSavedLink } from '../quartz/util/curius-feed'
import {
  curiusPreviewImagePath,
  isCuriusPreviewImageUrl,
  parseCuriusPreview,
  type CuriusPreviewResponse,
} from '../quartz/util/curius-preview'
import { escapeHTML } from '../quartz/util/escape'
import { parseGithubRepositoryUrl } from '../quartz/util/github-embed'
import { parseTwitterPostUrl } from '../quartz/util/twitter'
import { loadCuriusFeedLinks } from './arena-reader-catalogue'
import {
  extractArenaReaderDocument,
  type ArenaExtractedDocument,
  type ArenaExtractionResponse,
} from './arena-reader-extraction'
import { arenaReaderFailureSignals } from './arena-reader-render'
import { fetchArenaReaderSource, validateArenaReaderTarget } from './arena-reader-resources'
import { readGithubRepositoryCard } from './curius-github'
import { isRecord } from './type-guards'

const CACHE_NAME = 'curius-preview-defuddle-0.19.3-v2'
const PREVIEW_MAX_AGE = 3600
const DOCUMENT_LIMIT = 2 * 1024 * 1024
const EXTRACTION_BYTES_LIMIT = 4 * 1024 * 1024
const IMAGE_LIMIT = 8 * 1024 * 1024
const IMAGE_TYPES = new Set([
  'image/avif',
  'image/bmp',
  'image/gif',
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/x-icon',
  'image/vnd.microsoft.icon',
])

export interface CuriusPreviewEnv {
  ARENA_CONTENT?: R2Bucket
  BROWSER?: BrowserWorker
  ARENA_RENDER_RATE_LIMITER?: RateLimit
}

interface CachedPreview {
  preview: CuriusPreviewResponse
  imageUrls: string[]
}

class PreviewError extends Error {
  constructor(
    readonly reason: string,
    message: string,
    readonly status = 502,
  ) {
    super(message)
  }
}

function response(preview: CuriusPreviewResponse, status = 200): Response {
  return Response.json(preview, {
    status,
    headers: {
      'Cache-Control': preview.status === 'ready' ? 'public, max-age=300' : 'no-store',
      'X-Content-Type-Options': 'nosniff',
    },
  })
}

function cacheRequest(request: Request, linkId: number): Request {
  return new Request(new URL(`/api/curius-preview-cache/v2/${linkId}`, request.url))
}

async function readCache(cache: Cache, key: Request): Promise<CachedPreview | null> {
  const stored = await cache.match(key)
  if (!stored) return null
  const value: unknown = await stored.json()
  if (!isRecord(value) || !Array.isArray(value.imageUrls)) return null
  const preview = parseCuriusPreview(value.preview)
  const imageUrls = value.imageUrls.filter((item): item is string => typeof item === 'string')
  return preview && imageUrls.length === value.imageUrls.length && imageUrls.length <= 300
    ? { preview, imageUrls }
    : null
}

async function readBody(response: Response, limit: number): Promise<Uint8Array<ArrayBuffer>> {
  const declaredLength = Number(response.headers.get('Content-Length'))
  if (declaredLength > limit) {
    await response.body?.cancel()
    throw new PreviewError('too-large', 'The source exceeds the preview size limit.', 413)
  }
  const reader = response.body?.getReader()
  if (!reader) return new Uint8Array()
  const chunks: Uint8Array[] = []
  let size = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      size += value.byteLength
      if (size > limit)
        throw new PreviewError('too-large', 'The source exceeds the preview size limit.', 413)
      chunks.push(value)
    }
  } finally {
    await reader.cancel().catch(() => {})
  }
  const bytes = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return bytes
}

export function buildCuriusPreview(
  link: CuriusSavedLink,
  finalUrl: string,
  extracted: ArenaExtractedDocument,
): CachedPreview {
  const failure = arenaReaderFailureSignals(extracted, parseTwitterPostUrl(finalUrl) ? 1 : 40)
  if (failure) throw new PreviewError(failure.reason, failure.message)
  return serializePreview(
    link,
    finalUrl,
    extracted.title,
    extracted.readerHtml ?? '',
    extracted.imageUrls,
  )
}

function serializePreview(
  link: CuriusSavedLink,
  finalUrl: string,
  title: string,
  readerHtml: string,
  sources: string[],
): CachedPreview {
  const fetchedAt = Date.now()
  const imageUrls = sources.map(source => (validateArenaReaderTarget(source) ? source : ''))
  for (const [index, source] of imageUrls.entries()) {
    readerHtml = readerHtml.replaceAll(
      `data-arena-image="${index}"`,
      source
        ? `src="${curiusPreviewImagePath(link.id, index, fetchedAt).replaceAll('&', '&amp;')}"`
        : '',
    )
  }
  if (!readerHtml || readerHtml.length > DOCUMENT_LIMIT)
    throw new PreviewError('empty', 'The page did not produce a readable article preview.')
  return {
    preview: {
      status: 'ready',
      linkId: link.id,
      title: (title || link.title || finalUrl).slice(0, 4096),
      sourceUrl: link.link,
      finalUrl,
      readerHtml,
      fetchedAt,
      cached: false,
    },
    imageUrls,
  }
}

export function buildGithubCuriusPreview(
  link: CuriusSavedLink,
  finalUrl: string,
  html: string,
): CachedPreview {
  const card = readGithubRepositoryCard(html, finalUrl)
  if (!card)
    throw new PreviewError('repository-unavailable', 'The repository preview is unavailable.')
  const dimensions =
    card.imageWidth && card.imageHeight
      ? ` width="${card.imageWidth}" height="${card.imageHeight}"`
      : ''
  const image = card.imageUrl
    ? `<a href="${escapeHTML(finalUrl)}"><img class="curius-preview-repository-card" data-arena-image="0" alt="${escapeHTML(card.title)} repository card"${dimensions}></a>`
    : ''
  const readerHtml = image + card.paragraphs.map(text => `<p>${escapeHTML(text)}</p>`).join('')
  return serializePreview(
    link,
    finalUrl,
    card.title,
    readerHtml,
    card.imageUrl ? [card.imageUrl] : [],
  )
}

async function capturePreview(
  request: Request,
  binding: BrowserWorker | undefined,
  link: CuriusSavedLink,
): Promise<CachedPreview> {
  const controller = new AbortController()
  const signal = AbortSignal.any([request.signal, controller.signal, AbortSignal.timeout(30_000)])
  let browser: Awaited<ReturnType<typeof puppeteer.launch>> | undefined
  let closeOnAbort: (() => void) | undefined
  const pending = new Set<Promise<ArenaExtractionResponse>>()
  try {
    const repository = parseGithubRepositoryUrl(link.link)
    let finalUrl = repository?.url ?? link.link
    let html = '<!doctype html><html><head></head><body></body></html>'
    if (!parseTwitterPostUrl(link.link)) {
      const upstream = await fetchArenaReaderSource(finalUrl, {
        signal,
        headers: { Accept: 'text/html, application/xhtml+xml' },
      })
      finalUrl = upstream.finalUrl
      const type = upstream.response.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase()
      if (!upstream.response.ok || !['text/html', 'application/xhtml+xml'].includes(type ?? '')) {
        await upstream.response.body?.cancel()
        throw new PreviewError(
          upstream.response.ok ? 'unsupported-type' : 'upstream-error',
          upstream.response.ok
            ? 'Open the original to view this document type.'
            : 'The publisher did not provide a readable page. Open the original to continue.',
        )
      }
      html = new TextDecoder().decode(await readBody(upstream.response, DOCUMENT_LIMIT))
    }
    signal.throwIfAborted()
    if (repository) return buildGithubCuriusPreview(link, finalUrl, html)
    if (!binding)
      throw new PreviewError('unconfigured', 'Article extraction is unavailable here.', 503)
    const launch = puppeteer
      .launch(binding, { guardrails: { allowedDomains: [] } })
      .then(async created => {
        if (signal.aborted) {
          await created.close().catch(() => {})
          signal.throwIfAborted()
        }
        return created
      })
    let cancelLaunch: (() => void) | undefined
    try {
      browser = await Promise.race([
        launch,
        new Promise<never>((_, reject) => {
          cancelLaunch = () => reject(new DOMException('Preview cancelled.', 'AbortError'))
          signal.addEventListener('abort', cancelLaunch, { once: true })
          if (signal.aborted) cancelLaunch()
        }),
      ])
    } finally {
      if (cancelLaunch) signal.removeEventListener('abort', cancelLaunch)
    }
    const activeBrowser = browser
    closeOnAbort = () => void activeBrowser.close().catch(() => {})
    signal.addEventListener('abort', closeOnAbort, { once: true })
    const parser = await browser.newPage()
    await parser.setRequestInterception(true)
    parser.on('request', upstream => void upstream.abort('blockedbyclient').catch(() => {}))
    await parser.setContent('<!doctype html><html><head></head><body></body></html>')
    await parser.addScriptTag({ content: defuddleSource })
    await parser.addScriptTag({ content: purifySource })
    let requests = 0
    let totalBytes = 0
    await parser.exposeFunction(
      'arenaReaderFetch',
      (url: string, headers: Record<string, string>) => {
        const load = async (): Promise<ArenaExtractionResponse> => {
          signal.throwIfAborted()
          if (++requests > 8) throw new Error('The preview reached its extraction request limit.')
          const upstream = await fetchArenaReaderSource(url, { signal, headers })
          const bytes = await readBody(
            upstream.response,
            Math.min(DOCUMENT_LIMIT, EXTRACTION_BYTES_LIMIT - totalBytes),
          )
          totalBytes += bytes.byteLength
          if (totalBytes > EXTRACTION_BYTES_LIMIT)
            throw new Error('The preview reached its extraction resource limit.')
          return {
            body: new TextDecoder().decode(bytes),
            status: upstream.response.status,
            contentType: upstream.response.headers.get('Content-Type') ?? 'text/plain',
            url: upstream.finalUrl,
          }
        }
        const promise = load()
        pending.add(promise)
        void promise.finally(() => pending.delete(promise)).catch(() => {})
        return promise
      },
    )
    const extracted = await parser.evaluate(extractArenaReaderDocument, {
      html,
      finalUrl,
      idPrefix: `curius-preview-${link.id}-`,
    })
    return buildCuriusPreview(link, finalUrl, extracted)
  } catch (error) {
    if (error instanceof PreviewError) throw error
    if (request.signal.aborted)
      throw new PreviewError('cancelled', 'The preview request was cancelled.', 499)
    if (signal.aborted)
      throw new PreviewError('timeout', 'The preview timed out. Open the original to continue.')
    throw new PreviewError(
      'extraction-failed',
      'This article could not be extracted. Open the original to continue.',
    )
  } finally {
    if (closeOnAbort) signal.removeEventListener('abort', closeOnAbort)
    controller.abort()
    if (browser) await browser.close().catch(() => {})
    await Promise.allSettled(pending)
  }
}

async function previewImage(
  request: Request,
  cache: Cache,
  cached: CachedPreview | null,
): Promise<Response> {
  const url = new URL(request.url)
  const index = url.searchParams.get('image') ?? ''
  const source = /^(0|[1-9]\d*)$/.test(index) ? cached?.imageUrls[Number(index)] : null
  if (
    !source ||
    cached?.preview.status !== 'ready' ||
    !isCuriusPreviewImageUrl(request.url, url.origin) ||
    url.searchParams.get('v') !== String(cached.preview.fetchedAt)
  )
    throw new PreviewError('image-unavailable', 'Open the article again to reload its images.', 404)
  const key = new Request(
    new URL(
      curiusPreviewImagePath(cached.preview.linkId, Number(index), cached.preview.fetchedAt),
      request.url,
    ),
  )
  const saved = await cache.match(key)
  if (saved) return saved
  const upstream = await fetchArenaReaderSource(source, {
    signal: request.signal,
    headers: { Accept: [...IMAGE_TYPES].join(', ') },
  })
  const type = upstream.response.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase()
  if (!upstream.response.ok || !type || !IMAGE_TYPES.has(type)) {
    await upstream.response.body?.cancel()
    throw new PreviewError('image-unavailable', 'The source image is unavailable.', 404)
  }
  const bytes = await readBody(upstream.response, IMAGE_LIMIT)
  const image = new Response(bytes, {
    headers: {
      'Content-Type': type,
      'Cache-Control': `public, max-age=${PREVIEW_MAX_AGE}`,
      'X-Content-Type-Options': 'nosniff',
      'Cross-Origin-Resource-Policy': 'same-origin',
      'Content-Security-Policy': "default-src 'none'; sandbox",
    },
  })
  await cache.put(key, image.clone())
  return image
}

export async function handleCuriusPreview(
  request: Request,
  env: CuriusPreviewEnv,
): Promise<Response> {
  let sourceUrl: string | undefined
  try {
    const url = new URL(request.url)
    const id = url.searchParams.get('id') ?? ''
    if (!/^[1-9]\d*$/.test(id) || !Number.isSafeInteger(Number(id)))
      throw new PreviewError('invalid-id', 'A saved Curius link ID is required.', 400)
    if (!env.ARENA_CONTENT)
      throw new PreviewError('unconfigured', 'Article previews are unavailable here.', 503)
    const links = await loadCuriusFeedLinks(env.ARENA_CONTENT, true)
    const link = links.find(item => item.id === Number(id))
    if (!link)
      throw new PreviewError('not-saved', 'This link is outside the saved Curius catalogue.', 404)
    if (!validateArenaReaderTarget(link.link))
      throw new PreviewError('blocked-source', 'This saved URL cannot be previewed.', 400)
    sourceUrl = link.link
    const cache = await caches.open(CACHE_NAME)
    const key = cacheRequest(request, link.id)
    const stored = await readCache(cache, key)
    const cached = stored?.preview.sourceUrl === sourceUrl ? stored : null
    if (url.searchParams.get('query') === 'preview-image')
      return await previewImage(request, cache, cached)
    if (cached)
      return response(
        cached.preview.status === 'ready' ? { ...cached.preview, cached: true } : cached.preview,
      )
    if (!env.BROWSER && !parseGithubRepositoryUrl(link.link))
      throw new PreviewError('unconfigured', 'Article extraction is unavailable here.', 503)
    if (env.ARENA_RENDER_RATE_LIMITER) {
      const permit = await env.ARENA_RENDER_RATE_LIMITER.limit({ key: 'curius-preview-public' })
      if (!permit.success)
        throw new PreviewError(
          'rate-limited',
          'Previews are busy. Retry in a minute or open the original.',
          429,
        )
    }
    let result: CachedPreview
    try {
      result = await capturePreview(request, env.BROWSER, link)
    } catch (error) {
      if (!(error instanceof PreviewError)) throw error
      if (error.reason === 'cancelled') throw error
      result = {
        preview: { status: 'unavailable', reason: error.reason, message: error.message, sourceUrl },
        imageUrls: [],
      }
    }
    const ttl = result.preview.status === 'ready' ? PREVIEW_MAX_AGE : 60
    await cache.put(
      key,
      Response.json(result, { headers: { 'Cache-Control': `public, max-age=${ttl}` } }),
    )
    return response(result.preview)
  } catch (error) {
    const failure =
      error instanceof PreviewError
        ? error
        : new PreviewError(
            'unavailable',
            'The preview could not be loaded. Open the original to continue.',
            503,
          )
    return response(
      { status: 'unavailable', reason: failure.reason, message: failure.message, sourceUrl },
      failure.status,
    )
  }
}
