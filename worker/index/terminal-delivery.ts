import type { Env } from './environment'
import { NOTES_HOSTNAME } from '../../quartz/util/stacked-notes'
import {
  STREAM_HOSTNAME,
  streamAssetPathname,
  streamDocumentRedirectUrl,
  streamHostPathname,
} from '../../quartz/util/stream-host'
import { TRIATHLON_HOSTNAME, triathlonHostPathname } from '../../quartz/util/triathlon-host'
import { API_CATALOG_LINK, isHomepagePathname } from '../api-catalog'
import {
  getObjectFromBucket,
  getObjectInfo,
  lfsAssetContentType,
  lfsPointerRequest,
} from '../lfs-assets'
import { publishedPdfDocument } from '../pdf-marks'
import { problemResponse } from '../problem-details'
import {
  documentDiscoveryLink,
  markdownPathname,
  resolveBaseUrl,
  shouldTreatAsDocument,
} from '../request-utils'
import { handleStackedNotesRequest } from '../stacked'
import {
  cacheHeadersForStaticAsset,
  isolationHeadersForStaticAsset,
  requestWithoutCache,
  requestWithoutStaticAssetCache,
} from '../static-assets'
import { withVisitorDaylightCell } from '../weather'
import {
  COOP_COEP_HEADERS,
  isNotebookPath,
  loadNotebookSlugs,
  missingMarkdownResponse,
  triathlonHostHtmlResponse,
} from './document-responses'
import { buildCorsHeaders, mergeVary, withHeaders, withHeadersFromSource } from './http-policy'
import { getObjectFromLFS, KEEP_HEADERS, type CfCacheStorage } from './object-delivery'

export async function handleTerminalDelivery(
  request: Request,
  url: URL,
  env: Env,
  ctx: ExecutionContext,
  localRequest: boolean,
  apiHeaders: Record<string, string>,
): Promise<Response> {
  if (url.pathname.startsWith('/fonts/')) {
    const fontFile = url.pathname.replace(/^\/fonts\//, '')

    const referer = request.headers.get('Referer')
    const allowedHosts = [
      'aarnphm.xyz',
      'notes.aarnphm.xyz',
      'stream.aarnphm.xyz',
      'localhost',
      '127.0.0.1',
    ]

    if (referer) {
      try {
        const refererUrl = new URL(referer)
        const isAllowed = allowedHosts.some(
          host => refererUrl.hostname === host || refererUrl.hostname.endsWith(`.${host}`),
        )
        if (!isAllowed) {
          return new Response('forbidden', { status: 403 })
        }
      } catch {
        return new Response('forbidden', { status: 403 })
      }
    }

    const cacheKey = new Request(url.toString(), request)
    const cache = (caches as CfCacheStorage).default

    const cached = await cache.match(cacheKey)
    if (cached) return cached

    const fontData = await env.FONTS.get(fontFile, 'arrayBuffer')
    if (!fontData) {
      return new Response('font not found', { status: 404 })
    }

    const mimeType = fontFile.endsWith('.woff2')
      ? 'font/woff2'
      : fontFile.endsWith('.woff')
        ? 'font/woff'
        : 'application/octet-stream'

    const headers = new Headers({
      'Content-Type': mimeType,
      'Cache-Control': 'public, max-age=31536000, immutable',
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    })

    const response = new Response(fontData, { headers })

    ctx.waitUntil(cache.put(cacheKey, response.clone()))

    return response
  }

  const lfsContentType = lfsAssetContentType(url.pathname)
  if (lfsContentType) {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return new Response(null, {
        status: request.method === 'OPTIONS' ? 204 : 405,
        headers: {
          Allow: 'GET, HEAD, OPTIONS',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Range',
          'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        },
      })
    }
    const assetProbeRequest = lfsPointerRequest(request)
    const originResp = await env.ASSETS.fetch(
      requestWithoutStaticAssetCache(assetProbeRequest, url.pathname),
    )
    if (originResp.ok) {
      const info = originResp.body ? await getObjectInfo(originResp.clone()) : null
      if (!info) {
        void originResp.body?.cancel().catch(() => {})
        const resp = await env.ASSETS.fetch(request)
        return withHeaders(resp, {
          ...cacheHeadersForStaticAsset(url.pathname, resp.status),
          ...isolationHeadersForStaticAsset(url.pathname, resp.status),
        })
      }

      const resp = env.LFS_BUCKET
        ? await getObjectFromBucket(env.LFS_BUCKET, info.oid, request, lfsContentType)
        : await getObjectFromLFS(info, request)
      const keep = (env.KEEP_HEADERS || KEEP_HEADERS).split(',')
      return withHeaders(withHeadersFromSource(resp, originResp, keep), {
        'Content-Type': lfsContentType,
      })
    }

    if (lfsContentType === 'application/pdf') {
      const document = await publishedPdfDocument(env, url).catch(() => undefined)
      if (document) {
        return env.LFS_BUCKET
          ? getObjectFromBucket(env.LFS_BUCKET, document.doc, request, lfsContentType)
          : getObjectFromLFS(
              { hash_algo: 'sha256', oid: document.doc, size: document.bytes },
              request,
            )
      }
    }

    const rawUrl = `https://raw.githubusercontent.com/aarnphm/aarnphm.github.io/refs/heads/main/content${url.pathname}`
    const upstream = await fetch(lfsPointerRequest(new Request(rawUrl)))
    if (upstream.ok && upstream.body) {
      const info = await getObjectInfo(upstream.clone())
      if (info) {
        const resp = env.LFS_BUCKET
          ? await getObjectFromBucket(env.LFS_BUCKET, info.oid, request, lfsContentType)
          : await getObjectFromLFS(info, request)
        const keep = (env.KEEP_HEADERS || KEEP_HEADERS).split(',')
        return withHeaders(withHeadersFromSource(resp, upstream, keep), {
          'Content-Type': lfsContentType,
        })
      }
    }
  }

  if (url.hostname === STREAM_HOSTNAME && !url.pathname.startsWith('/fonts/')) {
    const sanitizedPathname = streamHostPathname(url.pathname)

    if (sanitizedPathname !== url.pathname) {
      const redirectUrl = new URL(url)
      redirectUrl.pathname = sanitizedPathname
      return Response.redirect(redirectUrl.toString(), 308)
    }

    const pathname = sanitizedPathname
    const isDocument = shouldTreatAsDocument(pathname)
    const baseUrl = resolveBaseUrl(env, request)
    const redirectUrl = isDocument ? streamDocumentRedirectUrl(baseUrl, url) : null

    if (redirectUrl) return Response.redirect(redirectUrl, 308)

    const base = new URL(baseUrl)
    if (base.hostname === url.hostname && base.hostname.startsWith('stream.')) {
      base.hostname = base.hostname.replace(/^stream\./, '')
    }
    base.pathname = streamAssetPathname(pathname, isDocument)
    base.search = url.search
    base.hash = url.hash
    const newReq = requestWithoutStaticAssetCache(new Request(base.toString(), request), pathname)
    const resp = await env.ASSETS.fetch(newReq)
    return withHeaders(resp, {
      ...cacheHeadersForStaticAsset(pathname, resp.status),
      'X-Frame-Options': null,
      'Content-Security-Policy': "frame-ancestors 'self' *",
    })
  }

  const assetsMatch = url.pathname.match(
    /.+\.(py|go|java|c|cpp|cxx|cu|cuh|h|hpp|ts|tsx|jsx|yaml|yml|rs|m|sql|sh|zig|lua)$/i,
  )
  if (assetsMatch) {
    const originResp = await env.ASSETS.fetch(request)
    return withHeaders(originResp, {
      'Content-Type': 'text/plain; charset=utf-8',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'Cache-Control': 'no-store, no-cache, must-revalidate',
    })
  }

  if (url.pathname.startsWith('/api/')) {
    return problemResponse(request, {
      status: 404,
      title: 'API resource not found',
      detail: 'No API resource is published at this path.',
      code: 'api_resource_not_found',
      resolution: 'Read /openapi.json for the published API paths.',
      headers: buildCorsHeaders(env, request),
    })
  }

  if (request.method !== 'GET' && request.method !== 'HEAD')
    return new Response(null, {
      status: request.method === 'OPTIONS' ? 200 : 405,
      headers: { Allow: 'GET, HEAD, OPTIONS' },
    })

  const arenaJsonMatch = url.pathname.match(/^\/arena\/([^/]+)\/json$/)
  if (arenaJsonMatch) {
    const originResp = await env.ASSETS.fetch(request)
    return withHeaders(originResp, { 'Content-Type': 'application/json', ...apiHeaders })
  }

  if (
    shouldTreatAsDocument(url.pathname) &&
    (url.hostname === NOTES_HOSTNAME || url.searchParams.has('stackedNotes'))
  ) {
    const stacked = await handleStackedNotesRequest(request, env)
    if (stacked) return stacked
  }

  const shouldBypassDocumentCache = localRequest && shouldTreatAsDocument(url.pathname)
  const assetRequest = shouldBypassDocumentCache
    ? requestWithoutCache(request)
    : requestWithoutStaticAssetCache(request, url.pathname)
  const resp = await env.ASSETS.fetch(assetRequest)
  if (url.pathname.endsWith('.md') && resp.status === 404)
    return missingMarkdownResponse(request, url.pathname)
  const staticAssetHeaders = {
    ...cacheHeadersForStaticAsset(url.pathname, resp.status),
    ...isolationHeadersForStaticAsset(url.pathname, resp.status),
  }
  if (shouldTreatAsDocument(url.pathname)) {
    const slugs = await loadNotebookSlugs(env, url.origin)
    const notebookHeaders = isNotebookPath(url.pathname, slugs) ? COOP_COEP_HEADERS : {}
    const documentResponse = withHeaders(resp, {
      ...staticAssetHeaders,
      ...notebookHeaders,
      ...(shouldBypassDocumentCache
        ? { 'Cache-Control': 'no-store, no-cache, must-revalidate' }
        : {}),
      'X-Frame-Options': null,
      'Content-Security-Policy': "frame-ancestors 'self' *",
      Vary: mergeVary(resp, 'Accept', 'Accept-Encoding', 'User-Agent'),
      Link:
        url.hostname === TRIATHLON_HOSTNAME
          ? `<${triathlonHostPathname(markdownPathname(url.pathname))}>; rel="alternate"; type="text/markdown", </llms.txt>; rel="describedby"; type="text/markdown"`
          : isHomepagePathname(url.pathname)
            ? `${API_CATALOG_LINK}, ${documentDiscoveryLink(url.pathname)}`
            : documentDiscoveryLink(url.pathname),
    })
    return url.hostname === TRIATHLON_HOSTNAME
      ? triathlonHostHtmlResponse(withVisitorDaylightCell(request, documentResponse), url)
      : withVisitorDaylightCell(request, documentResponse)
  }
  return withHeaders(resp, staticAssetHeaders)
}
