import type { ElementContent, Properties, RootContent } from 'hast'
import { Semaphore } from 'async-mutex'
import { Defuddle, type DefuddleOptions, type DefuddleResponse } from 'defuddle/node'
import { fromHtml } from 'hast-util-from-html'
import { toHtml } from 'hast-util-to-html'
import { h, s } from 'hastscript'
import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { svgOptions } from './svg-options'
import { parseTwitterPostUrl } from './twitter'
import { twitterIconPath } from './twitter-icon'
import { isRecord } from './type-guards'

const EMPTY_DOCUMENT = '<!doctype html><html><head></head><body></body></html>'
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024
const DEFAULT_TWITTER_CACHE_DIR = path.join(
  process.cwd(),
  'quartz',
  '.quartz-cache',
  'twitter-posts',
)
const requests = new Semaphore(16)
const posts = new Map<string, Promise<string>>()
const ensuredCacheDirs = new Map<string, Promise<void>>()
const contentTags = new Set([
  'p',
  'br',
  'hr',
  'strong',
  'em',
  'b',
  'i',
  's',
  'del',
  'code',
  'pre',
  'a',
  'blockquote',
  'cite',
  'figure',
  'figcaption',
  'img',
  'video',
  'ul',
  'ol',
  'li',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
  'time',
  'sup',
  'sub',
])
const discardedTags = new Set([
  'script',
  'style',
  'iframe',
  'object',
  'embed',
  'svg',
  'math',
  'template',
  'form',
  'input',
  'button',
  'textarea',
  'select',
])

function publicUrl(value: unknown, base: string): string | undefined {
  if (typeof value !== 'string' || !value.trim()) return
  try {
    const url = new URL(value, base)
    if (['http:', 'https:'].includes(url.protocol) && !url.username && !url.password)
      return url.href
  } catch {
    return
  }
}

function cleanContent(nodes: RootContent[], url: string): ElementContent[] {
  return nodes.flatMap((node): ElementContent[] => {
    if (node.type === 'text') return [node]
    if (node.type !== 'element' || discardedTags.has(node.tagName)) return []
    const children = cleanContent(node.children, url)
    if (!contentTags.has(node.tagName)) return children
    const properties: Properties = {}
    if (node.tagName === 'a') {
      const href = publicUrl(node.properties.href, url)
      if (!href) return children
      Object.assign(properties, { href, target: '_blank', rel: ['noopener', 'noreferrer'] })
      if (parseTwitterPostUrl(href)) properties.dataSkipIcons = 'true'
    }
    if (node.tagName === 'img' || node.tagName === 'video') {
      const src = publicUrl(node.properties.src, url)
      if (!src) return []
      properties.src = src
      if (node.tagName === 'img') {
        properties.alt = typeof node.properties.alt === 'string' ? node.properties.alt : ''
        properties.loading = 'lazy'
        properties.decoding = 'async'
      } else {
        properties.controls = true
        properties.preload = 'none'
        properties.playsInline = true
        properties.poster = publicUrl(node.properties.poster, url)
      }
    }
    if (node.tagName === 'time' && typeof node.properties.dateTime === 'string')
      properties.dateTime = node.properties.dateTime
    return [{ type: 'element', tagName: node.tagName, properties, children }]
  })
}

export function renderTwitterPost(article: DefuddleResponse, url: string): string {
  const content = cleanContent(fromHtml(article.content, { fragment: true }).children, url)
  if (!toHtml(content).trim()) return twitterPostFallback(url)
  const date = /^\d{4}-\d{2}-\d{2}$/.test(article.published) ? article.published : null
  const author = article.author || 'X post'
  return toHtml(
    h('article.twitter-post', { 'data-twitter-source': url }, [
      h(
        'span.twitter-post-badge',
        { role: 'img', ariaLabel: 'Twitter post', title: 'Twitter post' },
        s(
          'svg',
          { ...svgOptions, viewBox: '64 64 896 896', ariaHidden: 'true', focusable: 'false' },
          s('path', { d: twitterIconPath }),
        ),
      ),
      h('header.twitter-post-header', [
        h('span.twitter-post-author', author),
        ...(date ? [h('time', { dateTime: date }, date)] : []),
        h(
          'a.twitter-post-source',
          {
            href: url,
            target: '_blank',
            rel: ['noopener', 'noreferrer'],
            ariaLabel: 'Open original post (opens in a new tab)',
            title: 'Open original post',
            dataSkipIcons: 'true',
          },
          s(
            'svg',
            {
              ...svgOptions,
              fill: 'none',
              stroke: 'currentColor',
              strokeWidth: 1.5,
              ariaHidden: 'true',
              focusable: 'false',
            },
            s('path', { d: 'M7 17 17 7M7 7h10v10' }),
          ),
        ),
      ]),
      h('div.twitter-post-body', [
        ...(article.title && !article.title.startsWith('Post by ')
          ? [h('h2.twitter-post-title', article.title)]
          : []),
        ...content,
      ]),
    ]),
  )
}

function twitterPostFallback(url: string): string {
  return toHtml(
    h('p.twitter-post-unavailable', [
      'Post unavailable. ',
      h('a', { href: url, target: '_blank', rel: ['noopener', 'noreferrer'] }, 'open original ↗'),
    ]),
  )
}

export async function parseTwitterPost(
  html: string,
  url: string,
  options: DefuddleOptions = {},
): Promise<DefuddleResponse> {
  return Defuddle(html, url, { includeReplies: false, removeSmallImages: false, ...options })
}

async function boundedResponse(response: Response): Promise<Response> {
  const reader = response.body?.getReader()
  if (!reader) return response
  const chunks: Uint8Array[] = []
  let bytes = 0
  try {
    while (true) {
      const chunk = await reader.read()
      if (chunk.done) break
      bytes += chunk.value.byteLength
      if (bytes > MAX_RESPONSE_BYTES) throw new Error('X post response exceeds the size limit.')
      chunks.push(chunk.value)
    }
  } finally {
    await reader.cancel()
  }
  const body = new Uint8Array(bytes)
  let offset = 0
  for (const chunk of chunks) {
    body.set(chunk, offset)
    offset += chunk.byteLength
  }
  return new Response([204, 205, 304].includes(response.status) ? null : body, {
    status: response.status,
    headers: { 'Content-Type': response.headers.get('Content-Type') ?? 'application/json' },
  })
}

async function extractTwitterPostWithStatus(
  url: string,
  fetchSource: typeof fetch = fetch,
): Promise<{ html: string; cacheable: boolean }> {
  const signal = AbortSignal.timeout(15_000)
  const visited = new Set<string>()
  let count = 0
  let hadTransientError = false
  let hadTerminalStatus = false

  async function fetchResponse(request: Request): Promise<Response> {
    const target = new URL(request.url)
    if (
      request.method !== 'GET' ||
      target.username ||
      target.password ||
      ![
        'https://api.fxtwitter.com',
        'https://publish.twitter.com',
        'https://publish.x.com',
      ].includes(target.origin) ||
      ++count > 8
    )
      throw new Error('Unsupported X extractor request.')
    signal.throwIfAborted()
    let response: Response
    try {
      response = await fetchSource(request, { signal, redirect: 'manual' })
    } catch (error) {
      hadTransientError = true
      throw error
    }
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      const location = response.headers.get('Location')
      await response.body?.cancel()
      const next = location ? URL.parse(location, target) : null
      if (!next) throw new Error('Invalid X extractor redirect.')
      // The oEmbed endpoint redirects to publish.x.com. Validate and count every hop.
      return fetchResponse(new Request(next, { headers: request.headers }))
    }
    if (response.status === 429 || response.status >= 500) {
      hadTransientError = true
    } else if ([400, 401, 403, 404, 410].includes(response.status)) {
      hadTerminalStatus = true
    }
    return boundedResponse(response)
  }

  async function extract(source: string, depth: number): Promise<DefuddleResponse | null> {
    const canonical = parseTwitterPostUrl(source)
    if (!canonical || visited.has(canonical)) return null
    visited.add(canonical)
    let quoteUrl: string | undefined
    const videos: ElementContent[] = []
    const article = await parseTwitterPost(EMPTY_DOCUMENT, source, {
      async fetch(resource, init) {
        const request = new Request(resource, init)
        const target = new URL(request.url)
        const response = await fetchResponse(request)
        if (target.hostname === 'api.fxtwitter.com' && response.ok) {
          const payload: unknown = await response.clone().json()
          if (isRecord(payload) && isRecord(payload.tweet)) {
            const tweet = payload.tweet
            if (isRecord(tweet.quote)) quoteUrl = publicUrl(tweet.quote.url, source)
            if (!tweet.article && isRecord(tweet.media) && Array.isArray(tweet.media.videos)) {
              for (const video of tweet.media.videos) {
                if (!isRecord(video)) continue
                const src = publicUrl(video.url, source)
                if (src)
                  videos.push(
                    h('video', {
                      src,
                      poster: publicUrl(video.thumbnail_url, source),
                      controls: true,
                    }),
                  )
              }
            }
          }
        }
        return response
      },
    })
    article.content += toHtml(videos)
    // Defuddle 0.19's network extractor omits quoted posts. Follow their URLs through
    // the same extractor; Defuddle still owns all post and quote content parsing.
    if (quoteUrl && depth < 2) {
      const quote = await extract(quoteUrl, depth + 1).catch(() => null)
      article.content += toHtml(
        h('blockquote', [
          h('cite', [h('a', { href: quoteUrl }, quote?.author || 'quoted post')]),
          ...cleanContent(
            fromHtml(quote?.content || twitterPostFallback(quoteUrl), { fragment: true }).children,
            quoteUrl,
          ),
        ]),
      )
    }
    return article
  }
  const fallback = twitterPostFallback(url)
  const article = await extract(url, 0)
  const html = article ? renderTwitterPost(article, url) : fallback
  const isFallback = html === fallback
  const cacheable = !hadTransientError && (!isFallback || hadTerminalStatus)
  return { html, cacheable }
}

export async function extractTwitterPost(
  url: string,
  fetchSource: typeof fetch = fetch,
): Promise<string> {
  const { html } = await extractTwitterPostWithStatus(url, fetchSource)
  return html
}

function ensureCacheDir(cacheDir: string): Promise<void> {
  let existing = ensuredCacheDirs.get(cacheDir)
  if (!existing) {
    existing = fs
      .mkdir(cacheDir, { recursive: true })
      .then(() => undefined)
      .catch(error => {
        ensuredCacheDirs.delete(cacheDir)
        throw error
      })
    ensuredCacheDirs.set(cacheDir, existing)
  }
  return existing
}

export interface FetchTwitterPostOptions {
  cacheDir?: string
  fetchSource?: typeof fetch
  bypassMemoryCache?: boolean
}

export function fetchTwitterPost(url: string, options?: FetchTwitterPostOptions): Promise<string> {
  const key = parseTwitterPostUrl(url)
  if (!key) return Promise.resolve('')
  const cacheDir = options?.cacheDir ?? DEFAULT_TWITTER_CACHE_DIR
  const fetchSource = options?.fetchSource ?? fetch
  const memoryKey = cacheDir === DEFAULT_TWITTER_CACHE_DIR ? key : `${cacheDir}::${key}`
  if (!options?.bypassMemoryCache) {
    const existing = posts.get(memoryKey)
    if (existing) return existing
  }

  const cacheFile = path.join(cacheDir, `${createHash('sha256').update(url).digest('hex')}.html`)

  const pending = (async () => {
    try {
      const cached = await fs.readFile(cacheFile, 'utf8')
      if (cached.trim().length > 0) return cached
    } catch {}

    return requests.runExclusive(async () => {
      try {
        const cached = await fs.readFile(cacheFile, 'utf8')
        if (cached.trim().length > 0) return cached
      } catch {}

      const { html, cacheable } = await extractTwitterPostWithStatus(url, fetchSource).catch(
        () => ({ html: twitterPostFallback(url), cacheable: false }),
      )
      if (cacheable) {
        try {
          await ensureCacheDir(cacheDir)
          const tempFile = `${cacheFile}.tmp-${process.pid}-${randomUUID()}`
          await fs.writeFile(tempFile, html, 'utf8')
          await fs.rename(tempFile, cacheFile)
        } catch {}
      }
      return html
    })
  })()

  if (!options?.bypassMemoryCache) {
    posts.set(memoryKey, pending)
  }
  return pending
}
