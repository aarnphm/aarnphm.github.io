import type { Env } from './environment'
import {
  AI_CATALOG_PATH,
  aiCatalogResponse,
  API_CATALOG_PATH,
  API_DOCUMENTATION_PATH,
  API_HEALTH_PATH,
  apiCatalogResponse,
  apiDocumentationResponse,
  apiHealthResponse,
  OPENAPI_PATH,
  openApiResponse,
} from '../api-catalog'
import {
  handleArenaEmbedCapability,
  handleArenaEmbedCapture,
  handleArenaEmbedHtml,
} from '../arena-embed'
import handleArxiv from '../arxiv'
import handleCurius from '../curius'
import { handleFlashcardsReview, handleFlashcardsState, handleFlashcardsUndo } from '../flashcards'
import { handleLeanVerify } from '../lean'
import { MCP_SERVER_CARD_PATH, mcpServerCardResponse } from '../mcp-server-card'
import { handleMentions } from '../mentions'
import { isAgentUserAgent, resolveBaseUrl } from '../request-utils'
import { handleSaunaMap } from '../sauna-map'
import { handleStackedNoteDataRequest } from '../stacked'
import { triathlonDataHtml } from '../triathlon-data'
import { handleWeather } from '../weather'
import { htmlAssetResponse } from './document-responses'
import { buildCorsHeaders, withHeaders } from './http-policy'
import { handleLessWrong, handleSep } from './preview-routes'

export async function handleExactPath(
  request: Request,
  url: URL,
  env: Env,
  ctx: ExecutionContext,
  localRequest: boolean,
  baseUrl: string,
  apiHeaders: Record<string, string>,
): Promise<Response | null> {
  switch (url.pathname) {
    case '/are.na':
      return Response.redirect('https://aarnphm.xyz/arena', 301)
    case '/view-source':
      return Response.redirect('https://github.com/aarnphm/aarnphm.github.io', 301)
    case '/view-profile':
      return Response.redirect('https://x.com/aarnphm', 301)
    case '/github':
      return Response.redirect('https://github.com/aarnphm', 301)
    case '/substack':
      return Response.redirect('https://substack.com/@aarnphm', 301)
    case '/stream':
      if (localRequest) return null
      return Response.redirect('https://stream.aarnphm.xyz', 308)
    case '/.lfsconfig':
      return new Response(null, { status: 404 })
    case API_CATALOG_PATH:
      return apiCatalogResponse(request, baseUrl, apiHeaders)
    case AI_CATALOG_PATH:
      return aiCatalogResponse(request, baseUrl, apiHeaders)
    case OPENAPI_PATH:
      return openApiResponse(request, baseUrl, apiHeaders)
    case API_DOCUMENTATION_PATH:
      return apiDocumentationResponse(request, baseUrl, apiHeaders)
    case API_HEALTH_PATH:
      return apiHealthResponse(request, apiHeaders)
    case MCP_SERVER_CARD_PATH:
      return mcpServerCardResponse(request, baseUrl)
    case '/.well-known/openid-configuration': {
      const base = resolveBaseUrl(env, request)
      const body = JSON.stringify({
        issuer: base,
        authorization_endpoint: `${base}/authorize`,
        token_endpoint: `${base}/token`,
        registration_endpoint: `${base}/register`,
        response_types_supported: ['code'],
        subject_types_supported: ['public'],
        code_challenge_methods_supported: ['S256'],
        id_token_signing_alg_values_supported: ['none'],
      })
      return new Response(body, { headers: { 'Content-Type': 'application/json', ...apiHeaders } })
    }
    case '/site.webmanifest':
      const originResp = await env.ASSETS.fetch(request)
      return withHeaders(originResp, { ...apiHeaders, 'Access-Control-Allow-Origin': '*' })
    case '/agents.sh':
    case '/detachtools.sh': {
      const assetUrl = new URL(request.url)
      assetUrl.pathname = `/static${url.pathname}`
      const assetReq = new Request(assetUrl.toString(), request)
      const assetResp = await env.ASSETS.fetch(assetReq)
      return withHeaders(assetResp, {
        'Content-Type': 'text/plain; charset=utf-8',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'Cache-Control': 'no-store, no-cache, must-revalidate',
      })
    }
    case '/triathlon/data': {
      const assetUrl = new URL(request.url)
      assetUrl.pathname = '/static/triathlon/data.jsonl'
      const assetResp = await env.ASSETS.fetch(new Request(assetUrl.toString(), request))
      const accept = request.headers.get('Accept') ?? ''
      if (assetResp.ok && accept.includes('text/html') && !isAgentUserAgent(request)) {
        return new Response(triathlonDataHtml(await assetResp.text()), {
          headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 's-maxage=300, stale-while-revalidate=59',
            Vary: 'Accept, User-Agent',
          },
        })
      }
      return withHeaders(assetResp, {
        'Content-Type': 'application/x-ndjson; charset=utf-8',
        'Cache-Control': 's-maxage=300, stale-while-revalidate=59',
        'Access-Control-Allow-Origin': '*',
        Vary: 'Accept, User-Agent',
      })
    }
    case '/manual':
    case '/manual/':
    case '/advent':
    case '/advent/':
    case '/park/':
    case '/park': {
      return htmlAssetResponse(request, env)
    }
    case '/api/arxiv': {
      const resp = await handleArxiv(request)
      return withHeaders(resp, apiHeaders)
    }
    case '/api/lesswrong': {
      const resp = await handleLessWrong(request)
      return withHeaders(resp, apiHeaders)
    }
    case '/api/sep': {
      const resp = await handleSep(request)
      return withHeaders(resp, apiHeaders)
    }
    case '/api/curius': {
      const resp = await handleCurius(request, env)
      return withHeaders(resp, {
        ...apiHeaders,
        'Cache-Control': resp.headers.get('Cache-Control') ?? apiHeaders['Cache-Control'],
      })
    }
    case '/api/mentions': {
      const resp = await handleMentions(env)
      return withHeaders(resp, apiHeaders)
    }
    case '/api/stacked-note': {
      const resp = await handleStackedNoteDataRequest(request, env)
      return withHeaders(resp, apiHeaders)
    }
    // Same-origin and session-scoped: no shared caching, no credentialed CORS.
    case '/api/flashcards/state':
      return handleFlashcardsState(request, env)
    case '/api/flashcards/review':
      return handleFlashcardsReview(request, env)
    case '/api/flashcards/undo':
      return handleFlashcardsUndo(request, env)
    case '/api/lean/verify': {
      const resp = await handleLeanVerify(request, env)
      return withHeaders(resp, {
        ...apiHeaders,
        'Cache-Control': resp.headers.get('Cache-Control') ?? 'no-store',
      })
    }
    case '/api/arena-embed/capability': {
      const resp = await handleArenaEmbedCapability(request)
      return withHeaders(resp, buildCorsHeaders(env, request))
    }
    case '/api/arena-embed/html':
      return handleArenaEmbedHtml(request)
    case '/api/arena-embed/capture':
      return handleArenaEmbedCapture(request, env, ctx)
    case '/api/pdf-proxy': {
      const pdfUrl = url.searchParams.get('url')
      if (!pdfUrl) {
        return new Response('missing url parameter', {
          status: 400,
          headers: { ...apiHeaders, 'Content-Type': 'text/plain' },
        })
      }

      let targetUrl: URL
      try {
        targetUrl = new URL(pdfUrl)
      } catch {
        return new Response('invalid url', {
          status: 400,
          headers: { ...apiHeaders, 'Content-Type': 'text/plain' },
        })
      }

      if (targetUrl.protocol !== 'https:' && targetUrl.protocol !== 'http:') {
        return new Response('unsupported protocol', {
          status: 400,
          headers: { ...apiHeaders, 'Content-Type': 'text/plain' },
        })
      }

      try {
        const pdfResponse = await fetch(targetUrl.toString(), {
          method: 'GET',
          cf: { cacheTtl: 3600, cacheEverything: true },
        })

        if (!pdfResponse.ok) {
          return new Response(`upstream error: ${pdfResponse.status}`, {
            status: 502,
            headers: { ...apiHeaders, 'Content-Type': 'text/plain' },
          })
        }

        return new Response(pdfResponse.body, {
          status: 200,
          headers: {
            'Content-Type': 'application/pdf',
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'public, max-age=3600',
          },
        })
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : 'unknown'
        return new Response(`proxy error: ${message}`, {
          status: 502,
          headers: { ...apiHeaders, 'Content-Type': 'text/plain' },
        })
      }
    }
    case '/_plausible/event': {
      if (request.method !== 'POST') {
        return new Response('method not allowed', {
          status: 405,
          headers: { ...apiHeaders, Allow: 'POST, OPTIONS' },
        })
      }

      const upstreamHeaders = new Headers(request.headers)
      const requestOrigin = new URL(request.url).origin
      upstreamHeaders.set('Origin', requestOrigin)

      const upstream = await fetch('https://plausible.io/api/event', {
        method: 'POST',
        headers: upstreamHeaders,
        body: request.body,
      })

      const headers = new Headers(upstream.headers)
      const origin = request.headers.get('Origin') ?? requestOrigin
      headers.set('Access-Control-Allow-Origin', origin)
      headers.set('Access-Control-Allow-Methods', 'POST, OPTIONS')
      headers.set('Access-Control-Allow-Headers', 'content-type')
      headers.set('Vary', 'Origin')
      headers.set('Cache-Control', 'no-store')

      return new Response(upstream.body, {
        headers,
        status: upstream.status,
        statusText: upstream.statusText,
      })
    }
    case '/api/sauna-map':
      return handleSaunaMap(request, env, ctx)
    case '/api/weather':
      return handleWeather(request, ctx)
    case '/api/secrets': {
      if (request.method !== 'GET') {
        return new Response('method not allowed', {
          status: 405,
          headers: { ...apiHeaders, 'Cache-Control': 'no-store', Allow: 'GET' },
        })
      }

      const key = url.searchParams.get('key') ?? ''
      const allowList: Record<string, keyof Env> = { MAPBOX_API_KEY: 'MAPBOX_API_KEY' }
      const envKey = allowList[key]
      if (!envKey) {
        return new Response('not found', {
          status: 404,
          headers: { ...apiHeaders, 'Cache-Control': 'no-store' },
        })
      }

      const base = new URL(resolveBaseUrl(env, request))
      const originHeader = request.headers.get('Origin')
      if (originHeader) {
        try {
          const originUrl = new URL(originHeader)
          if (originUrl.origin !== base.origin) {
            return new Response('forbidden', {
              status: 403,
              headers: { 'Cache-Control': 'no-store' },
            })
          }
        } catch {
          return new Response('forbidden', {
            status: 403,
            headers: { 'Cache-Control': 'no-store' },
          })
        }
      }

      const refererHeader = request.headers.get('Referer')
      if (refererHeader) {
        try {
          const refererUrl = new URL(refererHeader)
          if (refererUrl.origin !== base.origin) {
            return new Response('forbidden', {
              status: 403,
              headers: { 'Cache-Control': 'no-store' },
            })
          }
        } catch {
          return new Response('forbidden', {
            status: 403,
            headers: { 'Cache-Control': 'no-store' },
          })
        }
      }

      const value = env[envKey] as string | undefined
      if (typeof value !== 'string' || value.length === 0) {
        return new Response('not found', {
          status: 404,
          headers: { ...apiHeaders, 'Cache-Control': 'no-store' },
        })
      }

      const headers = {
        'Cache-Control': 'no-store',
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': base.origin,
        'Access-Control-Allow-Credentials': 'true',
        Vary: 'Origin',
      }
      return new Response(JSON.stringify({ value }), { status: 200, headers })
    }
  }
  return null
}
