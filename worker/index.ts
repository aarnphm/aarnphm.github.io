import LFS_CONFIG from "./.lfsconfig.txt"
import handleArxiv from "./arxiv"
import handleCurius from "./curius"
import Garden from "./mcp"
import {
  handleGitHubCallback,
  handleGitHubLogin,
  handleLogout,
  handleMintToken,
  getCookieName,
} from "./github-handler"
import { getCookie } from "./auth"
import { OAuthProvider } from "@cloudflare/workers-oauth-provider"

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

function resolveBaseUrl(env: Env, request: Request): string {
  if (env.PUBLIC_BASE_URL) return env.PUBLIC_BASE_URL.replace(/\/$/, "")
  const u = new URL(request.url)
  u.pathname = ""
  u.search = ""
  u.hash = ""
  return u.toString().replace(/\/$/, "")
}

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
  GITHUB_CLIENT_ID: string
  GITHUB_CLIENT_SECRET: string
  SESSION_SECRET: string
  PUBLIC_BASE_URL?: string
  AUTH_COOKIE_NAME?: string
  AUTH_SESSION_TTL_DAYS?: string
  AUTH_TOKEN_TTL_SECONDS?: string
} & Cloudflare.Env

export default {
  async fetch(request, env, ctx): Promise<Response> {
    const url = new URL(request.url)

    const defaultAuthHandler: ExportedHandler<Env> = {
      async fetch(req, env, _ctx) {
        const u = new URL(req.url)
        switch (u.pathname) {
          case "/oauth/authorize": {
            const raw = getCookie(req, getCookieName(env))
            if (!raw) {
              const base = resolveBaseUrl(env, req)
              const loginUrl = new URL(`${base}/auth/github/login`)
              loginUrl.searchParams.set("next", u.pathname + u.search)
              return Response.redirect(loginUrl.toString(), 302)
            }
            return new Response("ok", { status: 200, headers: buildCorsHeaders(env, req) })
          }
          case "/auth/github/login":
          case "/auth/login":
            return handleGitHubLogin(req, env)
          case "/auth/github/callback":
          case "/auth/callback":
            return handleGitHubCallback(req, env)
          case "/auth/logout":
            return handleLogout(req, env)
          case "/mcp/token":
          case "/token": {
            const resp = await handleMintToken(req, env)
            return withHeaders(resp, {
              "Cache-Control": "s-maxage=300, stale-while-revalidate=59",
              ...buildCorsHeaders(env, req),
            })
          }
        }
        return new Response(null, { status: 404 })
      },
    }

    const mcpApiHandler: ExportedHandler<Env> = {
      async fetch(req, env, ctx) {
        // @ts-ignore durable object class provided by agents sdk
        return Garden.serve("/mcp", { binding: "MCP_OBJECT" }).fetch(req, env, ctx)
      },
    }

    const provider = new OAuthProvider({
      apiRoute: "/mcp",
      apiHandler: mcpApiHandler as any,
      defaultHandler: defaultAuthHandler as any,
      authorizeEndpoint: "/oauth/authorize",
      tokenEndpoint: "/oauth/token",
      clientRegistrationEndpoint: "/oauth/register",
    })

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: buildCorsHeaders(env, request) })
    }

    const providerResp = await provider.fetch(request, env, ctx)
    if (providerResp.status !== 404) return providerResp

    // Internal rewrite for notes domain root -> /notes?stackedNotes=<encoded>
    if (url.hostname === "notes.aarnphm.xyz" && url.pathname === "/") {
      const slug = "notes"
      const rewritten = new URL(`/${slug}`, url)
      rewritten.searchParams.set("stackedNotes", btoa(slug).replace(/=+$/, ""))
      const newReq = new Request(rewritten.toString(), request)
      return env.ASSETS.fetch(newReq)
    }

    // permanent redirect d.aarnphm.xyz -> aarnphm.xyz/dating
    if (url.hostname === "d.aarnphm.xyz") {
      return Response.redirect("https://aarnphm.xyz/dating/slides", 301)
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

    const apiHeaders: Record<string, string> = {
      "Cache-Control": "s-maxage=300, stale-while-revalidate=59",
      ...buildCorsHeaders(env, request),
    }

    switch (url.pathname) {
      case "/view-source":
        return Response.redirect("https://github.com/aarnphm/aarnphm.github.io", 301)
      case "/view-profile":
        return Response.redirect("https://x.com/aarnphm_", 301)
      case "/github":
        return Response.redirect("https://github.com/aarnphm", 301)
      case "/substack":
        return Response.redirect("https://substack.com/@aarnphm", 301)
      case "/.lfsconfig":
        return new Response(null, { status: 404 })
      case "/.well-known/oauth-protected-resource": {
        const base = resolveBaseUrl(env, request)
        const body = JSON.stringify({
          resource: {
            token_endpoint: `${base}/mcp/token`,
            authorization_endpoint: `${base}/auth/login`,
            resource_url: `${base}/mcp/sse`,
            aliases: [{ name: "mcp", sse_url: `${base}/mcp/sse`, token_url: `${base}/mcp/token` }],
          },
        })
        return new Response(body, {
          headers: { "Content-Type": "application/json", ...apiHeaders },
        })
      }
      case "/.well-known/oauth-protected-resource/mcp": {
        const base = resolveBaseUrl(env, request)
        const body = JSON.stringify({
          name: "mcp",
          sse_url: `${base}/mcp/sse`,
          token_url: `${base}/mcp/token`,
        })
        return new Response(body, {
          headers: { "Content-Type": "application/json", ...apiHeaders },
        })
      }
      case "/.well-known/oauth-authorization-server": {
        const base = resolveBaseUrl(env, request)
        const body = JSON.stringify({
          issuer: base,
          authorization_endpoint: `${base}/oauth/authorize`,
          token_endpoint: `${base}/oauth/token`,
          registration_endpoint: `${base}/oauth/register`,
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
          authorization_endpoint: `${base}/oauth/authorize`,
          token_endpoint: `${base}/oauth/token`,
          registration_endpoint: `${base}/oauth/register`,
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
    }

    // Deny non-GET/HEAD for other paths
    if (request.method !== "GET" && request.method !== "HEAD")
      return new Response(null, {
        status: request.method === "OPTIONS" ? 200 : 405,
        headers: { Allow: "GET, HEAD, OPTIONS" },
      })

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

    return env.ASSETS.fetch(request)
  },
} satisfies ExportedHandler<Env>

export { Garden }
