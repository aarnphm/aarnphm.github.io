import type { ElementContent, Properties, RootContent } from 'hast'
import { Semaphore } from 'async-mutex'
import { Defuddle, type DefuddleOptions, type DefuddleResponse } from 'defuddle/node'
import { fromHtml } from 'hast-util-from-html'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import { parseTwitterPostUrl } from './twitter'
import { isRecord } from './type-guards'

const EMPTY_DOCUMENT = '<!doctype html><html><head></head><body></body></html>'
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024
const requests = new Semaphore(4)
const posts = new Map<string, Promise<string>>()
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
      h('header.twitter-post-header', [
        h('span.twitter-post-author', author),
        ...(date ? [h('time', { dateTime: date }, date)] : []),
        h('a', { href: url, target: '_blank', rel: ['noopener', 'noreferrer'] }, 'open original ↗'),
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

export async function extractTwitterPost(
  url: string,
  fetchSource: typeof fetch = fetch,
): Promise<string> {
  const signal = AbortSignal.timeout(15_000)
  const visited = new Set<string>()
  let count = 0
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
    const response = await fetchSource(request, { signal, redirect: 'manual' })
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      const location = response.headers.get('Location')
      await response.body?.cancel()
      const next = location ? URL.parse(location, target) : null
      if (!next) throw new Error('Invalid X extractor redirect.')
      // The oEmbed endpoint redirects to publish.x.com. Validate and count every hop.
      return fetchResponse(new Request(next, { headers: request.headers }))
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
  const article = await extract(url, 0)
  return article ? renderTwitterPost(article, url) : twitterPostFallback(url)
}

export function fetchTwitterPost(url: string): Promise<string> {
  const key = parseTwitterPostUrl(url)
  if (!key) return Promise.resolve('')
  const existing = posts.get(key)
  if (existing) return existing
  const pending = requests.runExclusive(() =>
    extractTwitterPost(url).catch(() => twitterPostFallback(url)),
  )
  posts.set(key, pending)
  return pending
}
