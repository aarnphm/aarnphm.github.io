import type { Env } from './environment'
import { missingSlugMarkdown } from '../../quartz/util/not-found'
import { triathlonHostLinkUrl, triathlonHostUrl } from '../../quartz/util/triathlon-host'
import { mergeVary, withHeaders } from './http-policy'

const HTML_CONTENT_TYPE = 'text/html; charset=utf-8'

export const TRIATHLON_MARKDOWN_PATH = /^\/triathlon(?:\/.*)?\.md$/

export function missingMarkdownResponse(request: Request, pathname: string): Response {
  return new Response(request.method === 'HEAD' ? null : missingSlugMarkdown(pathname), {
    status: 404,
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'Cache-Control': 'no-store',
      Vary: 'Accept, Accept-Encoding, User-Agent',
    },
  })
}

export const COOP_COEP_HEADERS: Record<string, string> = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

let notebookSlugsCache: Promise<ReadonlySet<string>> | undefined

export async function loadNotebookSlugs(
  env: { ASSETS: Fetcher },
  origin: string,
): Promise<ReadonlySet<string>> {
  if (notebookSlugsCache) return notebookSlugsCache
  notebookSlugsCache = (async () => {
    try {
      const manifestUrl = new URL('/notebook-pages.json', origin)
      const resp = await env.ASSETS.fetch(new Request(manifestUrl.toString()))
      if (!resp.ok) return new Set<string>()
      const json = (await resp.json()) as { slugs?: unknown }
      if (!Array.isArray(json.slugs)) return new Set<string>()
      return new Set(json.slugs.filter((value): value is string => typeof value === 'string'))
    } catch {
      return new Set<string>()
    }
  })()
  return notebookSlugsCache
}

function normalizeNotebookPathname(pathname: string): string {
  const trimmed = pathname.replace(/^\/+|\/+$/g, '')
  if (!trimmed) return ''
  if (trimmed.endsWith('.html') || trimmed.endsWith('.htm')) {
    return trimmed.replace(/\.html?$/, '')
  }
  if (trimmed.endsWith('/index')) return trimmed.slice(0, -'/index'.length)
  return trimmed
}

export function isNotebookPath(pathname: string, slugs: ReadonlySet<string>): boolean {
  if (slugs.size === 0) return false
  return slugs.has(normalizeNotebookPathname(pathname))
}

export function withTriathlonMarkdownHeaders(response: Response): Response {
  return withHeaders(response, {
    'Content-Type': 'text/markdown; charset=utf-8',
    'Cache-Control': 's-maxage=300, stale-while-revalidate=59',
    'Access-Control-Allow-Origin': '*',
    'X-Content-Type-Options': 'nosniff',
    Vary: mergeVary(response, 'Accept', 'Accept-Encoding', 'User-Agent'),
  })
}

export async function htmlAssetResponse(request: Request, env: Env): Promise<Response> {
  const originResp = await env.ASSETS.fetch(request)
  return withHeaders(originResp, { 'Content-Type': HTML_CONTENT_TYPE })
}

export function triathlonHostHtmlResponse(response: Response, assetUrl: URL): Response {
  if (!response.headers.get('Content-Type')?.startsWith('text/html')) return response
  // Relative links were emitted one level deeper, under /triathlon on the apex.
  const reference = new URL(assetUrl)
  reference.protocol = 'https:'
  reference.host = 'aarnphm.xyz'
  const canonical = triathlonHostUrl(assetUrl)
  const rewriteResource = (element: Element, attribute: string): void => {
    const href = element.getAttribute(attribute)
    if (!href || href.startsWith('#')) return
    const target = new URL(href, reference)
    if (target.origin === reference.origin && !URL.canParse(href) && !href.startsWith('//')) {
      target.host = assetUrl.host
      target.protocol = assetUrl.protocol
    }
    element.setAttribute(attribute, target.toString())
  }
  return new HTMLRewriter()
    .on('a[href]', {
      element(element) {
        const href = element.getAttribute('href')
        if (href) element.setAttribute('href', triathlonHostLinkUrl(href, reference))
      },
    })
    .on('[data-href]', {
      element(element) {
        const href = element.getAttribute('data-href')
        if (href) element.setAttribute('data-href', triathlonHostLinkUrl(href, reference))
      },
    })
    .on('link[href]', {
      element(element) {
        if (element.getAttribute('rel') === 'canonical') element.setAttribute('href', canonical)
        else rewriteResource(element, 'href')
      },
    })
    .on('script[src], img[src], source[src]', {
      element(element) {
        rewriteResource(element, 'src')
      },
    })
    .on('meta[property="og:url"]', {
      element(element) {
        element.setAttribute('content', canonical)
      },
    })
    .transform(response)
}
