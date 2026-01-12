import LFS_CONFIG from "./.lfsconfig.txt"
import handleArxiv from "./arxiv"
import handleCurius from "./curius"
import Garden from "./mcp"
import { MultiplayerComments } from "./comments"
import { CommentsGitHubHandler } from "./comments-github-handler"
import { GitHubHandler } from "./github-handler"
import { OAuthProvider } from "@cloudflare/workers-oauth-provider"
import { handleStackedNotesRequest } from "./stacked"
import { getGithubCommentAuthor, normalizeAuthor, setGithubCommentAuthor } from "./comments-auth"
import { isLocalRequest, resolveBaseUrl } from "./request-utils"

const VERSION = "version https://git-lfs.github.com/spec/v1\n"
const MIME = "application/vnd.git-lfs+json"
const KEEP_HEADERS = "Cache-Control"

type CfCacheStorage = CacheStorage & { readonly default: Cache }

function splitFirst(str: string, delim: string): [string, string?] {
  const idx = str.indexOf(delim)
  return idx === -1 ? [str] : [str.slice(0, idx), str.slice(idx + 1)]
}

function strictDecode(bytes: Uint8Array): string | null {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes)
  } catch {
    return null
  }
}

function getLfsUrl(config: string): URL | null {
  // TODO: better parser, accept remote.<remote>.lfsurl...
  let section: string | undefined
  for (const raw of config.split("\n")) {
    const line = splitFirst(raw, ";")[0].trim()
    if (line.startsWith("[") && line.endsWith("]")) {
      section = line.slice(1, -1)
    } else if (section === "lfs") {
      const [key, val] = splitFirst(line, "=")
      if (val === undefined) return null
      if (key.trimEnd() === "url") return new URL(val.trimStart())
    }
  }
  return null
}

function extendPath(url: URL | string, path: string): URL {
  const u = typeof url === "string" ? new URL(url) : new URL(url.toString())
  u.pathname = u.pathname.replace(/\/?$/, `/${path}`)
  return u
}

function withHeaders(response: Response, newHeaders: Record<string, string | null>): Response {
  if (Object.keys(newHeaders).length === 0) return response
  const headers = new Headers(response.headers)
  for (const [key, val] of Object.entries(newHeaders)) {
    if (val === null) headers.delete(key)
    else headers.set(key, val)
  }
  return new Response(response.body, {
    headers,
    status: response.status,
    statusText: response.statusText,
  })
}

function withHeadersFromSource(response: Response, source: Response, headers: string[]): Response {
  const map: Record<string, string | null> = {}
  for (const h of headers) map[h] = source.headers.get(h)
  return withHeaders(response, map)
}

function headersToObject(headers: Headers): Record<string, string> {
  const o: Record<string, string> = {}
  headers.forEach((value, key) => {
    o[key] = value
  })
  return o
}

function getExtension(pathname: string): string | null {
  const last = pathname.split("/").pop() ?? ""
  const idx = last.lastIndexOf(".")
  return idx === -1 ? null : last.slice(idx + 1).toLowerCase()
}

function shouldTreatAsDocument(pathname: string): boolean {
  const ext = getExtension(pathname)
  if (!ext) return true
  return ext === "html" || ext === "htm"
}

const commentAuthorRenameAuthorPrefix = "comment-author-rename:author:"
const commentAuthorRenameIpPrefix = "comment-author-rename:ip:"
const commentAuthorRenameWindowSeconds = 60 * 60 * 24 * 90

function getAllowedOrigin(env: Env, request: Request): string | null {
  const origin = request.headers.get("Origin")
  if (!origin) return null
  try {
    const o = new URL(origin)
    const base = env.PUBLIC_BASE_URL ? new URL(env.PUBLIC_BASE_URL) : null
    if (base && o.origin === base.origin) return origin
    if (o.hostname === "localhost" || o.hostname === "127.0.0.1") return origin
    if (o.hostname.endsWith(".workers.dev")) return origin
  } catch {}
  return null
}

function buildCorsHeaders(env: Env, request: Request): Record<string, string> {
  const origin = getAllowedOrigin(env, request)
  const headers: Record<string, string> = {
    "Access-Control-Allow-Methods": "GET,OPTIONS,PATCH,DELETE,POST,PUT",
    "Access-Control-Allow-Headers":
      "Authorization, X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version",
  }
  if (origin) {
    headers["Access-Control-Allow-Origin"] = origin
    headers["Access-Control-Allow-Credentials"] = "true"
  }
  return headers
}

async function getObjectInfo(
  response: Response,
): Promise<{ hash_algo: string; oid: string; size: number } | null> {
  // TODO: theoretically an LFS pointer could be >256 bytes.
  // however, even the LFS client spec seems to only read 100:
  // https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md
  const reader = response.body?.getReader()
  if (!reader) return null
  const { value } = await reader.read()
  if (!value) return null
  const slice = value.subarray(0, 256)
  const text = strictDecode(slice)
  if (!text || !text.startsWith(VERSION)) return null
  const rest = text.slice(VERSION.length)
  let hash_algo: string | undefined, oid: string | undefined, size: number | undefined
  for (const line of rest.split("\n")) {
    if (line === "") continue
    const [key, val] = splitFirst(line, " ")
    if (val === undefined) return null
    if (key === "oid") {
      ;[hash_algo, oid] = splitFirst(val, ":")
      if (oid === undefined) return null
    } else if (key === "size") {
      const n = parseInt(val)
      if (Number.isNaN(n)) return null
      size = n
    }
  }
  return hash_algo && oid && size ? { hash_algo, oid, size } : null
}

async function getObjectAction(
  lfsUrl: URL,
  info: { hash_algo: string; oid: string; size: number },
): Promise<{ href: string; header?: Record<string, string> } | null> {
  const url = extendPath(lfsUrl, "objects/batch")
  const headers: Record<string, string> = { Accept: MIME, "Content-Type": MIME }
  if (url.username || url.password) {
    headers["Authorization"] = `Basic ${btoa(`${url.username}:${url.password}`)}`
    url.username = url.password = ""
  }
  const body = JSON.stringify({
    operation: "download",
    transfers: ["basic"],
    objects: [{ oid: info.oid, size: info.size }],
    hash_algo: info.hash_algo,
  })
  const res = await fetch(url.toString(), { method: "POST", headers, body })
  if (res.ok && res.headers.get("Content-Type")?.startsWith(MIME)) {
    const batch: any = await res.json()
    const obj = batch.objects?.[0]
    if ((!batch.transfer || batch.transfer === "basic") && obj?.authenticated)
      return obj.actions.download
  }
  return null
}

async function getObjectFromBucket(
  ctx: ExecutionContext,
  bucket: R2Bucket,
  bucketUrl: string,
  path: string,
  request: Request,
): Promise<Response> {
  const cacheKey = new Request(extendPath(bucketUrl, path).toString(), request)
  // https://developers.cloudflare.com/workers/reference/how-the-cache-works/#cache-api
  const cache = (caches as CfCacheStorage).default
  const cached = await cache.match(cacheKey)
  if (cached) return cached
  const method = request.method.toLowerCase() as "get" | "head"
  const object = (await bucket[method](path)) as R2ObjectBody
  const headers = new Headers()
  object.writeHttpMetadata(headers)
  if (object.httpEtag) headers.set("ETag", object.httpEtag)
  const resp = new Response(object.body, { headers })
  ctx.waitUntil(
    cache.put(
      cacheKey,
      withHeaders(resp.clone(), { "Cache-Control": "immutable, max-age=31536000" }),
    ),
  )
  return resp
}

async function getObjectFromLFS(
  info: { hash_algo: string; oid: string; size: number },
  request: Request,
): Promise<Response> {
  const lfsUrl = getLfsUrl(LFS_CONFIG)
  if (!lfsUrl) return new Response(null, { status: 500 })
  const action = await getObjectAction(lfsUrl, info)
  if (!action) return new Response(null, { status: 500 })
  const headers = action.header
    ? { ...action.header, ...headersToObject(request.headers) }
    : headersToObject(request.headers)
  return fetch(action.href, { method: request.method, headers, cf: { cacheTtl: 31536000 } })
}

type Env = {
  LFS_BUCKET_URL?: string
  KEEP_HEADERS?: string
  MULTIPLAYER_COMMENTS: DurableObjectNamespace
  COMMENTS_ROOM: D1Database
  GITHUB_CLIENT_ID: string
  GITHUB_CLIENT_SECRET: string
  GITHUB_COMMENTS_CLIENT_ID?: string
  GITHUB_COMMENTS_CLIENT_SECRET?: string
  SESSION_SECRET: string
  OAUTH_KV: KVNamespace
  PUBLIC_BASE_URL?: string
  STACKED_CACHE?: KVNamespace
  MAPBOX_API_KEY?: string
} & Cloudflare.Env

export default {
  async fetch(request, env, ctx): Promise<Response> {
    const url = new URL(request.url)
    const localRequest = isLocalRequest(request)
    const method = request.method.toUpperCase()

    const provider = new OAuthProvider({
      apiHandlers: {
        // @ts-ignore
        "/mcp": Garden.serve("/mcp", { binding: "MCP_OBJECT" }),
      },
      authorizeEndpoint: "/authorize",
      clientRegistrationEndpoint: "/register",
      // @ts-ignore
      defaultHandler: GitHubHandler,
      tokenEndpoint: "/token",
    })

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: buildCorsHeaders(env, request) })
    }

    const providerResp = await provider.fetch(request, env, ctx)
    if (providerResp.status !== 404) return providerResp

    const commentsResp = await CommentsGitHubHandler.fetch(request, env, ctx)
    if (commentsResp.status !== 404) return commentsResp

    // Handle stacked notes requests with server-side rendering
    if (url.searchParams.has("stackedNotes")) {
      const stacked = await handleStackedNotesRequest(request, env, ctx)
      if (stacked) return stacked
    }

    if (request.method === "POST" && url.pathname === "/comments/author/rename") {
      let payload: { oldAuthor?: unknown; newAuthor?: unknown; githubLogin?: unknown } | null = null
      try {
        payload = (await request.json()) as {
          oldAuthor?: unknown
          newAuthor?: unknown
          githubLogin?: unknown
        }
      } catch {
        return new Response("invalid json", { status: 400 })
      }
      const oldAuthor =
        typeof payload?.oldAuthor === "string" ? normalizeAuthor(payload.oldAuthor) : null
      const newAuthor =
        typeof payload?.newAuthor === "string" ? normalizeAuthor(payload.newAuthor) : null
      const githubLogin =
        typeof payload?.githubLogin === "string" ? normalizeAuthor(payload.githubLogin) : null

      if (!oldAuthor || !newAuthor) {
        return new Response("invalid author", { status: 400 })
      }

      if (oldAuthor === newAuthor) {
        return new Response(JSON.stringify({ updated: 0 }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      }

      const ipRaw =
        request.headers.get("CF-Connecting-IP") ||
        request.headers.get("X-Forwarded-For")?.split(",")[0]?.trim() ||
        ""
      const authorKey = `${commentAuthorRenameAuthorPrefix}${oldAuthor}`
      const newAuthorKey = `${commentAuthorRenameAuthorPrefix}${newAuthor}`
      const ipKey = ipRaw ? `${commentAuthorRenameIpPrefix}${ipRaw}` : null

      let canUpdateGithub = false
      if (githubLogin) {
        const stored = await getGithubCommentAuthor(env.OAUTH_KV, githubLogin)
        canUpdateGithub = !stored || stored === oldAuthor
      }

      if (await env.OAUTH_KV.get(authorKey)) {
        return new Response("comment name can change every 3 months", { status: 429 })
      }

      if (!localRequest && ipKey && (await env.OAUTH_KV.get(ipKey))) {
        return new Response("comment name can change every 3 months", { status: 429 })
      }

      const result = await env.COMMENTS_ROOM.prepare(
        "UPDATE comments SET author = ? WHERE author = ?",
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
        headers: { "Content-Type": "application/json" },
      })
    }

    if (url.pathname.startsWith("/comments/")) {
      const id = env.MULTIPLAYER_COMMENTS.idFromName("global")
      const stub = env.MULTIPLAYER_COMMENTS.get(id)
      return stub.fetch(request)
    }

    const apiHeaders: Record<string, string> = {
      "Cache-Control": "s-maxage=300, stale-while-revalidate=59",
      ...buildCorsHeaders(env, request),
    }

    switch (url.pathname) {
      case "/are.na":
        return Response.redirect("https://aarnphm.xyz/arena", 301)
      case "/view-source":
        return Response.redirect("https://github.com/aarnphm/aarnphm.github.io", 301)
      case "/view-profile":
        return Response.redirect("https://x.com/aarnphm", 301)
      case "/github":
        return Response.redirect("https://github.com/aarnphm", 301)
      case "/substack":
        return Response.redirect("https://substack.com/@aarnphm", 301)
      case "/stream":
        if (localRequest) break
        return Response.redirect("https://stream.aarnphm.xyz", 308)
      case "/.lfsconfig":
        return new Response(null, { status: 404 })
      case "/.well-known/oauth-protected-resource": {
        const base = resolveBaseUrl(env, request)
        const body = JSON.stringify({
          resource: `${base}/mcp`,
          authorization_servers: [base],
          bearer_methods_supported: ["header"],
          resource_documentation: `${base}/`,
        })
        return new Response(body, {
          headers: { "Content-Type": "application/json", ...apiHeaders },
        })
      }
      case "/.well-known/oauth-protected-resource/mcp": {
        const base = resolveBaseUrl(env, request)
        const body = JSON.stringify({
          name: "mcp",
          sse_url: `${base}/mcp`,
          token_url: `${base}/token`,
        })
        return new Response(body, {
          headers: { "Content-Type": "application/json", ...apiHeaders },
        })
      }
      case "/.well-known/oauth-authorization-server": {
        const base = resolveBaseUrl(env, request)
        const body = JSON.stringify({
          issuer: base,
          authorization_endpoint: `${base}/authorize`,
          token_endpoint: `${base}/token`,
          registration_endpoint: `${base}/register`,
          response_types_supported: ["code"],
          grant_types_supported: ["authorization_code"],
          code_challenge_methods_supported: ["S256"],
          token_endpoint_auth_methods_supported: ["none"],
        })
        return new Response(body, {
          headers: { "Content-Type": "application/json", ...apiHeaders },
        })
      }
      case "/.well-known/openid-configuration": {
        const base = resolveBaseUrl(env, request)
        const body = JSON.stringify({
          issuer: base,
          authorization_endpoint: `${base}/authorize`,
          token_endpoint: `${base}/token`,
          registration_endpoint: `${base}/register`,
          response_types_supported: ["code"],
          subject_types_supported: ["public"],
          code_challenge_methods_supported: ["S256"],
          id_token_signing_alg_values_supported: ["none"],
        })
        return new Response(body, {
          headers: { "Content-Type": "application/json", ...apiHeaders },
        })
      }
      case "/site.webmanifest":
        const originResp = await env.ASSETS.fetch(request)
        return withHeaders(originResp, { ...apiHeaders, "Access-Control-Allow-Origin": "*" })
      case "/park": {
        const originResp = await env.ASSETS.fetch(request)
        return withHeaders(originResp, { "Content-Type": "text/html; charset=utf-8" })
      }
      case "/api/arxiv": {
        const resp = await handleArxiv(request)
        return withHeaders(resp, apiHeaders)
      }
      case "/api/curius": {
        const resp = await handleCurius(request)
        return withHeaders(resp, apiHeaders)
      }
      case "/api/pdf-proxy": {
        const pdfUrl = url.searchParams.get("url")
        if (!pdfUrl) {
          return new Response("missing url parameter", {
            status: 400,
            headers: { ...apiHeaders, "Content-Type": "text/plain" },
          })
        }

        let targetUrl: URL
        try {
          targetUrl = new URL(pdfUrl)
        } catch {
          return new Response("invalid url", {
            status: 400,
            headers: { ...apiHeaders, "Content-Type": "text/plain" },
          })
        }

        // Only allow HTTP(S)
        if (targetUrl.protocol !== "https:" && targetUrl.protocol !== "http:") {
          return new Response("unsupported protocol", {
            status: 400,
            headers: { ...apiHeaders, "Content-Type": "text/plain" },
          })
        }

        try {
          const pdfResponse = await fetch(targetUrl.toString(), {
            method: "GET",
            cf: {
              cacheTtl: 3600,
              cacheEverything: true,
            },
          })

          if (!pdfResponse.ok) {
            return new Response(`upstream error: ${pdfResponse.status}`, {
              status: 502,
              headers: { ...apiHeaders, "Content-Type": "text/plain" },
            })
          }

          return new Response(pdfResponse.body, {
            status: 200,
            headers: {
              "Content-Type": "application/pdf",
              "Access-Control-Allow-Origin": "*",
              "Cache-Control": "public, max-age=3600",
            },
          })
        } catch (err: any) {
          return new Response(`proxy error: ${err?.message ?? "unknown"}`, {
            status: 502,
            headers: { ...apiHeaders, "Content-Type": "text/plain" },
          })
        }
      }
      case "/_plausible/event": {
        if (method !== "POST") {
          return new Response("method not allowed", {
            status: 405,
            headers: { ...apiHeaders, Allow: "POST, OPTIONS" },
          })
        }

        const upstreamHeaders = new Headers(request.headers)
        const requestOrigin = new URL(request.url).origin
        upstreamHeaders.set("Origin", requestOrigin)

        const upstream = await fetch("https://plausible.io/api/event", {
          method: "POST",
          headers: upstreamHeaders,
          body: request.body,
        })

        const headers = new Headers(upstream.headers)
        const origin = request.headers.get("Origin") ?? requestOrigin
        headers.set("Access-Control-Allow-Origin", origin)
        headers.set("Access-Control-Allow-Methods", "POST, OPTIONS")
        headers.set("Access-Control-Allow-Headers", "content-type")
        headers.set("Vary", "Origin")
        headers.set("Cache-Control", "no-store")

        return new Response(upstream.body, {
          headers,
          status: upstream.status,
          statusText: upstream.statusText,
        })
      }
      case "/api/secrets": {
        if (method !== "GET") {
          return new Response("method not allowed", {
            status: 405,
            headers: { ...apiHeaders, "Cache-Control": "no-store", Allow: "GET" },
          })
        }

        const key = url.searchParams.get("key") ?? ""
        const allowList: Record<string, keyof Env> = {
          MAPBOX_API_KEY: "MAPBOX_API_KEY",
        }
        const envKey = allowList[key]
        if (!envKey) {
          return new Response("not found", {
            status: 404,
            headers: { ...apiHeaders, "Cache-Control": "no-store" },
          })
        }

        const base = new URL(resolveBaseUrl(env, request))
        const originHeader = request.headers.get("Origin")
        if (originHeader) {
          try {
            const originUrl = new URL(originHeader)
            if (originUrl.origin !== base.origin) {
              return new Response("forbidden", {
                status: 403,
                headers: { "Cache-Control": "no-store" },
              })
            }
          } catch {
            return new Response("forbidden", {
              status: 403,
              headers: { "Cache-Control": "no-store" },
            })
          }
        }

        const refererHeader = request.headers.get("Referer")
        if (refererHeader) {
          try {
            const refererUrl = new URL(refererHeader)
            if (refererUrl.origin !== base.origin) {
              return new Response("forbidden", {
                status: 403,
                headers: { "Cache-Control": "no-store" },
              })
            }
          } catch {
            return new Response("forbidden", {
              status: 403,
              headers: { "Cache-Control": "no-store" },
            })
          }
        }

        const value = env[envKey] as string | undefined
        if (typeof value !== "string" || value.length === 0) {
          return new Response("not found", {
            status: 404,
            headers: { ...apiHeaders, "Cache-Control": "no-store" },
          })
        }

        const headers = {
          "Cache-Control": "no-store",
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": base.origin,
          "Access-Control-Allow-Credentials": "true",
          Vary: "Origin",
        }
        return new Response(JSON.stringify({ value }), { status: 200, headers })
      }
    }

    // font serving from KV with edge caching
    if (url.pathname.startsWith("/fonts/")) {
      const fontFile = url.pathname.replace(/^\/fonts\//, "")

      // check referer to prevent hotlinking
      const referer = request.headers.get("Referer")
      const allowedHosts = [
        "aarnphm.xyz",
        "notes.aarnphm.xyz",
        "stream.aarnphm.xyz",
        "localhost",
        "127.0.0.1",
      ]

      if (referer) {
        try {
          const refererUrl = new URL(referer)
          const isAllowed = allowedHosts.some(
            (host) => refererUrl.hostname === host || refererUrl.hostname.endsWith(`.${host}`),
          )
          if (!isAllowed) {
            return new Response("forbidden", { status: 403 })
          }
        } catch {
          return new Response("forbidden", { status: 403 })
        }
      }

      // construct cache key for edge caching
      const cacheKey = new Request(url.toString(), request)
      const cache = (caches as CfCacheStorage).default

      // check edge cache first
      const cached = await cache.match(cacheKey)
      if (cached) return cached

      // fetch from KV
      const fontData = await env.FONTS.get(fontFile, "arrayBuffer")
      if (!fontData) {
        return new Response("font not found", { status: 404 })
      }

      // determine mime type
      const mimeType = fontFile.endsWith(".woff2")
        ? "font/woff2"
        : fontFile.endsWith(".woff")
          ? "font/woff"
          : "application/octet-stream"

      // build response with proper headers
      const headers = new Headers({
        "Content-Type": mimeType,
        "Cache-Control": "public, max-age=31536000, immutable",
        "Access-Control-Allow-Origin": "*",
        "Cross-Origin-Resource-Policy": "cross-origin",
      })

      const response = new Response(fontData, { headers })

      // cache at edge for 1 year
      ctx.waitUntil(cache.put(cacheKey, response.clone()))

      return response
    }

    // Internal rewrite for stream domain -> dedupe /stream prefix and serve canonical content
    if (url.hostname === "stream.aarnphm.xyz" && !url.pathname.startsWith("/fonts/")) {
      const streamPrefix = "/stream"
      let sanitizedPathname = url.pathname

      if (sanitizedPathname === streamPrefix || sanitizedPathname === `${streamPrefix}/`) {
        sanitizedPathname = "/"
      } else if (sanitizedPathname.startsWith(`${streamPrefix}/`)) {
        sanitizedPathname = sanitizedPathname.slice(streamPrefix.length) || "/"
      }

      if (sanitizedPathname !== url.pathname) {
        const redirectUrl = new URL(url)
        redirectUrl.pathname = sanitizedPathname
        return Response.redirect(redirectUrl.toString(), 308)
      }

      const pathname = sanitizedPathname

      const base = new URL(resolveBaseUrl(env, request))
      if (base.hostname === url.hostname && base.hostname.startsWith("stream.")) {
        base.hostname = base.hostname.replace(/^stream\./, "")
      }
      const isDocument = shouldTreatAsDocument(pathname)
      const needsStreamPrefix = isDocument && !pathname.startsWith("/stream")
      const targetPath =
        needsStreamPrefix && pathname !== "/"
          ? `/stream${pathname}`
          : needsStreamPrefix
            ? "/stream"
            : pathname
      base.pathname = targetPath
      base.search = url.search
      base.hash = url.hash
      const newReq = new Request(base.toString(), request)
      const resp = await env.ASSETS.fetch(newReq)
      return withHeaders(resp, {
        "X-Frame-Options": null,
        "Content-Security-Policy": "frame-ancestors 'self' *",
      })
    }

    // rendering supported code files as text/plain
    const assetsMatch = url.pathname.match(
      /.+\.(py|go|java|c|cpp|cxx|cu|cuh|h|hpp|ts|tsx|jsx|yaml|yml|rs|m|sql|sh|zig|lua)$/i,
    )
    if (assetsMatch) {
      const originResp = await env.ASSETS.fetch(request)
      return withHeaders(originResp, {
        "Content-Type": "text/plain; charset=utf-8",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Cache-Control": "no-store, no-cache, must-revalidate",
      })
    }

    // Deny non-GET/HEAD for other paths
    if (request.method !== "GET" && request.method !== "HEAD")
      return new Response(null, {
        status: request.method === "OPTIONS" ? 200 : 405,
        headers: { Allow: "GET, HEAD, OPTIONS" },
      })

    // arena channel json routes
    const arenaJsonMatch = url.pathname.match(/^\/arena\/([^/]+)\/json$/)
    if (arenaJsonMatch) {
      const originResp = await env.ASSETS.fetch(request)
      return withHeaders(originResp, {
        "Content-Type": "application/json",
        ...apiHeaders,
      })
    }

    // PDF redirect to R2 / LFS
    if (url.pathname.endsWith(".pdf")) {
      const rawUrl = `https://raw.githubusercontent.com/aarnphm/aarnphm.github.io/refs/heads/main/content${url.pathname}`
      const upstream = await fetch(new Request(rawUrl, { method: "GET", headers: request.headers }))
      if (upstream.body) {
        const info = await getObjectInfo(upstream.clone())
        if (info) {
          const resp =
            env.LFS_BUCKET && env.LFS_BUCKET_URL
              ? await getObjectFromBucket(
                  ctx,
                  env.LFS_BUCKET,
                  env.LFS_BUCKET_URL,
                  info.oid,
                  request,
                )
              : await getObjectFromLFS(info, request)
          const keep = (env.KEEP_HEADERS || KEEP_HEADERS).split(",")
          return withHeadersFromSource(resp, upstream, keep)
        }
      }
    }

    const resp = await env.ASSETS.fetch(request)
    if (shouldTreatAsDocument(url.pathname)) {
      return withHeaders(resp, {
        "X-Frame-Options": null,
        "Content-Security-Policy": "frame-ancestors 'self' *",
      })
    }
    return resp
  },
} satisfies ExportedHandler<Env>

export { Garden, MultiplayerComments }
