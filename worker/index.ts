import { OAuthProvider } from '@cloudflare/workers-oauth-provider'
import type { Env } from './index/environment'
import {
  isTriathlonRoutePathname,
  TRIATHLON_HOSTNAME,
  triathlonApexRedirectUrl,
  triathlonAssetPathname,
  triathlonDocumentRedirectUrl,
  triathlonHostPathname,
} from '../quartz/util/triathlon-host'
import {
  isStravaActivityIndex,
  STRAVA_ACTIVITY_INDEX_PATH,
  stravaActivityIdFromShortcutPath,
  triathlonActivityShortcutRedirectUrl,
} from '../quartz/util/triathlon-shortcut'
import { handleArenaReaderRequest } from './arena-reader'
import {
  handleAuthMarkdown,
  OAUTH_AUTHORIZATION_SERVER_PATH,
  OAUTH_SCOPES,
  oauthProtectedResourceMetadata,
  withAgentAuthMetadata,
} from './auth-md'
import {
  getGithubCommentAuthor,
  MultiplayerComments,
  normalizeAuthor,
  setGithubCommentAuthor,
} from './comments'
import {
  missingMarkdownResponse,
  TRIATHLON_MARKDOWN_PATH,
  withTriathlonMarkdownHeaders,
} from './index/document-responses'
import { handleExactPath } from './index/exact-path'
import { buildCorsHeaders, mergeVary, withHeaders } from './index/http-policy'
import { getModelObject } from './index/object-delivery'
import { handleTerminalDelivery } from './index/terminal-delivery'
import Garden from './mcp'
import { CommentsGitHubHandler, GitHubHandler } from './oauth'
import { handlePdfReaderRequest } from './pdf-marks'
import { problemResponse } from './problem-details'
import { mcpRateLimitExceededResponse, mcpRateLimitKey, withMcpRateLimitPolicy } from './rate-limit'
import {
  isLocalRequest,
  markdownPathname,
  resolveBaseUrl,
  shouldRejectDocumentResponse,
  shouldRewriteMarkdown,
  shouldTreatAsDocument,
} from './request-utils'
import { handleTrainingCalendarRequest, TrainingCalendarAccess } from './training-calendar'
import { handleWahooOAuthCallback } from './wahoo-oauth'

const commentAuthorRenameAuthorPrefix = 'comment-author-rename:author:'
const commentAuthorRenameIpPrefix = 'comment-author-rename:ip:'
const commentAuthorRenameWindowSeconds = 60 * 60 * 24 * 90

export default {
  async fetch(request, env, ctx): Promise<Response> {
    const trainingResponse = await handleTrainingCalendarRequest(request, env)
    if (trainingResponse) return trainingResponse
    const url = new URL(request.url)
    const localRequest = isLocalRequest(request)
    const apexTriathlonRedirect = triathlonApexRedirectUrl(url)
    if (apexTriathlonRedirect) return Response.redirect(apexTriathlonRedirect, 308)

    if (url.hostname === TRIATHLON_HOSTNAME) {
      const activityId = stravaActivityIdFromShortcutPath(url.pathname)
      if (activityId) {
        const indexUrl = new URL(STRAVA_ACTIVITY_INDEX_PATH, url)
        const indexResponse = await env.ASSETS.fetch(
          new Request(indexUrl, { headers: { Accept: 'application/json' } }),
        )
        const index: unknown = indexResponse.ok
          ? await indexResponse.json().catch(() => null)
          : null
        if (!isStravaActivityIndex(index)) {
          return new Response('Strava activity index unavailable', {
            status: 503,
            headers: { 'Cache-Control': 'no-store' },
          })
        }
        const redirectUrl = triathlonActivityShortcutRedirectUrl(url, index.activities)
        if (!redirectUrl) {
          return new Response(`Strava activity ${activityId} not found`, {
            status: 404,
            headers: { 'Cache-Control': 'no-store' },
          })
        }
        return Response.redirect(redirectUrl, 308)
      }

      const isDocument = shouldTreatAsDocument(url.pathname) || url.pathname.endsWith('.md')
      const documentRedirect = isDocument
        ? triathlonDocumentRedirectUrl(resolveBaseUrl(env, request), url)
        : null
      if (documentRedirect) return Response.redirect(documentRedirect, 308)
      const canonicalPathname = triathlonHostPathname(url.pathname)
      if (isTriathlonRoutePathname(url.pathname) && canonicalPathname !== url.pathname) {
        const canonicalUrl = new URL(url)
        canonicalUrl.pathname = canonicalPathname
        return Response.redirect(canonicalUrl.toString(), 308)
      }
      // Resolve the microsite before representation negotiation and /triathlon/data dispatch.
      url.pathname = triathlonAssetPathname(url.pathname)
      request = new Request(url, request)
    }

    const arenaReaderResponse = await handleArenaReaderRequest(request, env)
    if (arenaReaderResponse) return arenaReaderResponse

    const pdfReaderResponse = await handlePdfReaderRequest(request, env)
    if (pdfReaderResponse) return pdfReaderResponse

    const wahooOAuthResponse = handleWahooOAuthCallback(request)
    if (wahooOAuthResponse) return wahooOAuthResponse

    if (url.pathname.endsWith('.fc')) {
      url.pathname = url.pathname.slice(0, -3) + '/flashcards'
      request = new Request(url.toString(), request as any) as any
    }

    const baseUrl = resolveBaseUrl(env, request)
    const provider = new OAuthProvider({
      apiHandlers: {
        '/mcp': Garden.serve('/mcp', { binding: 'MCP_OBJECT' }),
        '/sse': Garden.serveSSE('/sse'),
      },
      authorizeEndpoint: '/authorize',
      clientRegistrationEndpoint: '/register',
      // @ts-ignore
      defaultHandler: GitHubHandler,
      tokenEndpoint: '/token',
      scopesSupported: [...OAUTH_SCOPES],
      resourceMetadata: oauthProtectedResourceMetadata(baseUrl),
    })

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: buildCorsHeaders(env, request) })
    }

    const authMarkdownResponse = handleAuthMarkdown(request, baseUrl)
    if (authMarkdownResponse) return authMarkdownResponse

    if (url.pathname === '/mcp') {
      const outcome = await env.MCP_RATE_LIMITER.limit({ key: mcpRateLimitKey(request) })
      if (!outcome.success) return mcpRateLimitExceededResponse(request)
    }

    const providerResp = await provider.fetch(request, env, ctx)
    if (
      request.method === 'GET' &&
      url.pathname === OAUTH_AUTHORIZATION_SERVER_PATH &&
      providerResp.ok
    ) {
      return withAgentAuthMetadata(providerResp, baseUrl)
    }
    if (providerResp.status !== 404) {
      return url.pathname === '/mcp' ? withMcpRateLimitPolicy(providerResp) : providerResp
    }

    const commentsResp = await CommentsGitHubHandler.fetch(request, env, ctx)
    if (commentsResp.status !== 404) return commentsResp

    if (request.method === 'POST' && url.pathname === '/internal/email/emit') {
      if (
        !env.EMAIL_EMITTER_SECRET ||
        request.headers.get('x-email-secret') !== env.EMAIL_EMITTER_SECRET
      ) {
        return new Response('unauthorized', { status: 401 })
      }

      let payload: unknown
      try {
        payload = await request.json()
      } catch {
        return new Response('invalid json', { status: 400 })
      }
      if (!payload || typeof payload !== 'object') {
        return new Response('invalid payload', { status: 400 })
      }

      const payloadRecord = payload as Record<string, unknown>
      const recipients = Array.isArray(payloadRecord.recipients)
        ? payloadRecord.recipients
            .filter(email => typeof email === 'string')
            .map(email => email.trim())
            .filter(email => email.length > 0)
        : typeof payloadRecord.recipients === 'string'
          ? [payloadRecord.recipients.trim()].filter(email => email.length > 0)
          : []
      if (recipients.length === 0) {
        return new Response('invalid recipients', { status: 400 })
      }

      const subject = typeof payloadRecord.subject === 'string' ? payloadRecord.subject.trim() : ''
      if (!subject) {
        return new Response('invalid subject', { status: 400 })
      }

      const text = typeof payloadRecord.text === 'string' ? payloadRecord.text : undefined
      const html = typeof payloadRecord.html === 'string' ? payloadRecord.html : undefined
      if (!text && !html) {
        return new Response('missing content', { status: 400 })
      }

      const attachments = Array.isArray(payloadRecord.attachments)
        ? payloadRecord.attachments
            .filter(entry => entry && typeof entry === 'object')
            .map(entry => {
              const record = entry as Record<string, unknown>
              return {
                contentId: typeof record.contentId === 'string' ? record.contentId : '',
                filename: typeof record.filename === 'string' ? record.filename : '',
                contentType: typeof record.contentType === 'string' ? record.contentType : '',
                content: typeof record.content === 'string' ? record.content : '',
              }
            })
            .filter(
              entry =>
                entry.contentId.length > 0 &&
                entry.filename.length > 0 &&
                entry.contentType.length > 0 &&
                entry.content.length > 0,
            )
        : []

      const apiKey = env.RESEND_API_KEY
      if (!apiKey) {
        return new Response('missing resend api key', { status: 500 })
      }
      const resendAttachments = attachments.map(a => ({
        filename: a.filename,
        content: a.content,
        content_type: a.contentType,
        content_id: a.contentId,
      }))
      const buildPayload = (recipient: string[]) => ({
        from: `Aaron Pham <${env.EMAIL_SENDER}>`,
        bcc: [...recipient],
        subject,
        ...(text ? { text } : {}),
        ...(html ? { html } : {}),
        ...(resendAttachments.length > 0 ? { attachments: resendAttachments } : {}),
      })
      if (url.searchParams.has('dry')) {
        return new Response(JSON.stringify(buildPayload(['no-reply@aarnphm.xyz']), null, 2), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }
      try {
        const res = await fetch('https://api.resend.com/emails', {
          method: 'POST',
          headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
          body: JSON.stringify(buildPayload(recipients)),
        })
        if (!res.ok) {
          const body = await res.text()
          return new Response(`resend error for ${recipients}: ${body}`, { status: 502 })
        }
      } catch (err) {
        console.error('resend request failed', err)
        return new Response('email dispatch failed', { status: 500 })
      }

      return new Response(JSON.stringify({ ok: true, sent: recipients.length }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    if (request.method === 'POST' && url.pathname === '/comments/author/rename') {
      let payload: { oldAuthor?: unknown; newAuthor?: unknown; githubLogin?: unknown } | null = null
      try {
        payload = (await request.json()) as {
          oldAuthor?: unknown
          newAuthor?: unknown
          githubLogin?: unknown
        }
      } catch {
        return new Response('invalid json', { status: 400 })
      }
      const oldAuthor =
        typeof payload?.oldAuthor === 'string' ? normalizeAuthor(payload.oldAuthor) : null
      const newAuthor =
        typeof payload?.newAuthor === 'string' ? normalizeAuthor(payload.newAuthor) : null
      const githubLogin =
        typeof payload?.githubLogin === 'string' ? normalizeAuthor(payload.githubLogin) : null

      if (!oldAuthor || !newAuthor) {
        return new Response('invalid author', { status: 400 })
      }

      if (oldAuthor === newAuthor) {
        return new Response(JSON.stringify({ updated: 0 }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }

      const ipRaw =
        request.headers.get('CF-Connecting-IP') ||
        request.headers.get('X-Forwarded-For')?.split(',')[0]?.trim() ||
        ''
      const authorKey = `${commentAuthorRenameAuthorPrefix}${oldAuthor}`
      const newAuthorKey = `${commentAuthorRenameAuthorPrefix}${newAuthor}`
      const ipKey = ipRaw ? `${commentAuthorRenameIpPrefix}${ipRaw}` : null

      let canUpdateGithub = false
      if (githubLogin) {
        const stored = await getGithubCommentAuthor(env.OAUTH_KV, githubLogin)
        canUpdateGithub = !stored || stored === oldAuthor
      }

      if (await env.OAUTH_KV.get(authorKey)) {
        return new Response('comment name can change every 3 months', { status: 429 })
      }

      if (!localRequest && ipKey && (await env.OAUTH_KV.get(ipKey))) {
        return new Response('comment name can change every 3 months', { status: 429 })
      }

      const result = await env.COMMENTS_ROOM.prepare(
        'UPDATE comments SET author = ? WHERE author = ?',
      )
        .bind(newAuthor, oldAuthor)
        .run()

      await env.OAUTH_KV.put(authorKey, `${Date.now()}`, {
        expirationTtl: commentAuthorRenameWindowSeconds,
      })
      await env.OAUTH_KV.put(newAuthorKey, `${Date.now()}`, {
        expirationTtl: commentAuthorRenameWindowSeconds,
      })

      if (!localRequest && ipKey) {
        await env.OAUTH_KV.put(ipKey, `${Date.now()}`, {
          expirationTtl: commentAuthorRenameWindowSeconds,
        })
      }

      if (githubLogin && canUpdateGithub) {
        await setGithubCommentAuthor(env.OAUTH_KV, githubLogin, newAuthor)
      }

      return new Response(JSON.stringify({ updated: result.changes ?? 0 }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    if (url.pathname.startsWith('/comments/')) {
      const id = env.MULTIPLAYER_COMMENTS.idFromName('global')
      const stub = env.MULTIPLAYER_COMMENTS.get(id)
      return stub.fetch(request)
    }

    if (url.pathname.startsWith('/models/')) {
      if (request.method !== 'GET' && request.method !== 'HEAD')
        return new Response('method not allowed', { status: 405 })
      if (localRequest) {
        const localModel = await env.ASSETS.fetch(request)
        // Speech weights exceed the 25 MiB asset limit, so in dev they live only in local R2.
        if (localModel.status !== 404)
          return withHeaders(localModel, {
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'no-store',
          })
      }
      const cacheControl = url.pathname.endsWith('/latest.json')
        ? 'public, max-age=60, must-revalidate'
        : 'public, max-age=31536000, immutable'
      return getModelObject(ctx, env.LFS_BUCKET, url.pathname.slice(1), request, cacheControl)
    }

    if (shouldRejectDocumentResponse(request, url)) {
      return problemResponse(request, {
        status: 406,
        title: 'Document representation not acceptable',
        detail: 'This document is available as text/html or text/markdown.',
        code: 'document_representation_not_acceptable',
        resolution: 'Send Accept: text/html or Accept: text/markdown.',
        headers: { Vary: 'Accept, Accept-Encoding, User-Agent' },
      })
    }

    if (shouldRewriteMarkdown(request, url)) {
      const markdownUrl = new URL(url.toString())
      markdownUrl.pathname = markdownPathname(url.pathname)
      const markdownReq = new Request(markdownUrl.toString(), request)
      const markdownResp = await env.ASSETS.fetch(markdownReq)
      if (markdownResp.status === 404) return missingMarkdownResponse(request, url.pathname)
      if (TRIATHLON_MARKDOWN_PATH.test(markdownUrl.pathname)) {
        return withTriathlonMarkdownHeaders(markdownResp)
      }
      return withHeaders(markdownResp, {
        'Content-Type': 'text/markdown; charset=utf-8',
        Vary: mergeVary(markdownResp, 'Accept', 'Accept-Encoding', 'User-Agent'),
      })
    }

    if (TRIATHLON_MARKDOWN_PATH.test(url.pathname)) {
      const markdownResp = await env.ASSETS.fetch(new Request(request.url, request))
      if (markdownResp.status === 404) return missingMarkdownResponse(request, url.pathname)
      return withTriathlonMarkdownHeaders(markdownResp)
    }

    const apiHeaders: Record<string, string> = {
      'Cache-Control': 's-maxage=300, stale-while-revalidate=59',
      ...buildCorsHeaders(env, request),
    }

    const knownPath = await handleExactPath(
      request,
      url,
      env,
      ctx,
      localRequest,
      baseUrl,
      apiHeaders,
    )
    if (knownPath) return knownPath
    return handleTerminalDelivery(request, url, env, ctx, localRequest, apiHeaders)
  },
} satisfies ExportedHandler<Env>

export { Garden, MultiplayerComments, TrainingCalendarAccess }
