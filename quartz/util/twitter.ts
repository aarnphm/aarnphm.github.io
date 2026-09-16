import { isRecord } from './type-guards'

const MAX_EMBED_BYTES = 256 * 1024

export function parseTwitterPostUrl(href: string): string | null {
  let url: URL
  try {
    url = new URL(href)
  } catch {
    return null
  }
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.username ||
    url.password ||
    !/^(?:(?:www|mobile|m)\.)?(?:twitter|x)\.com$/.test(url.hostname)
  )
    return null
  const post = url.pathname.match(/^\/(?:[\w]+\/status|i\/web\/status)\/(\d{1,20})(?:\/|$)/)
  if (!post) return null
  return `https://twitter.com/i/status/${post[1]}`
}

export function twitterOEmbedUrl(href: string, locale: string): URL | null {
  const post = parseTwitterPostUrl(href)
  if (!post) return null
  const url = new URL('https://publish.twitter.com/oembed')
  url.search = new URLSearchParams({
    url: post,
    dnt: 'true',
    omit_script: 'true',
    lang: locale,
  }).toString()
  return url
}

export function readTwitterEmbed(value: unknown): string | null {
  if (
    !isRecord(value) ||
    value.type !== 'rich' ||
    !['Twitter', 'X'].includes(String(value.provider_name)) ||
    typeof value.html !== 'string' ||
    !value.html.trim() ||
    value.html.length > MAX_EMBED_BYTES
  )
    return null
  return value.html
}

export async function fetchTwitterEmbed(
  href: string,
  locale: string,
  signal?: AbortSignal,
): Promise<string | null> {
  const url = twitterOEmbedUrl(href, locale)
  if (!url) return null
  const timeout = AbortSignal.timeout(10_000)
  const response = await fetch(url, {
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
  })
  if (!response.ok) {
    await response.body?.cancel()
    return null
  }
  const reader = response.body?.getReader()
  if (!reader) return null
  const decoder = new TextDecoder()
  let bytes = 0
  let body = ''
  try {
    while (true) {
      const chunk = await reader.read()
      if (chunk.done) break
      bytes += chunk.value.byteLength
      if (bytes > MAX_EMBED_BYTES) return null
      body += decoder.decode(chunk.value, { stream: true })
    }
    body += decoder.decode()
  } finally {
    await reader.cancel()
  }
  return readTwitterEmbed(JSON.parse(body))
}
