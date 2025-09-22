import {
  createSession,
  createState,
  getCookie,
  makeCookie,
  verifySession,
  verifyState,
  createAccessToken,
  verifyAccessToken,
} from "./auth"
import { Octokit } from "octokit"

type Env = {
  GITHUB_CLIENT_ID: string
  GITHUB_CLIENT_SECRET: string
  SESSION_SECRET: string
  PUBLIC_BASE_URL?: string
  AUTH_COOKIE_NAME?: string
  AUTH_SESSION_TTL_DAYS?: string
  AUTH_TOKEN_TTL_SECONDS?: string
}

function resolveBaseUrl(env: Env, request: Request): string {
  if (env.PUBLIC_BASE_URL) return env.PUBLIC_BASE_URL.replace(/\/$/, "")
  const u = new URL(request.url)
  u.pathname = ""
  u.search = ""
  u.hash = ""
  return u.toString().replace(/\/$/, "")
}

export function getCookieName(env: Env): string {
  return env.AUTH_COOKIE_NAME || "mcp_session"
}

function getSessionTtlSeconds(env: Env): number {
  const days = env.AUTH_SESSION_TTL_DAYS ? parseInt(env.AUTH_SESSION_TTL_DAYS) : 7
  return (Number.isFinite(days) ? days : 7) * 24 * 60 * 60
}

function getTokenTtlSeconds(env: Env): number {
  const secs = env.AUTH_TOKEN_TTL_SECONDS ? parseInt(env.AUTH_TOKEN_TTL_SECONDS) : 300
  return Number.isFinite(secs) ? secs : 300
}

export async function handleGitHubLogin(request: Request, env: Env): Promise<Response> {
  if (!env.SESSION_SECRET || !env.GITHUB_CLIENT_ID)
    return new Response("missing SESSION_SECRET or GITHUB_CLIENT_ID", { status: 500 })
  const url = new URL(request.url)
  const base = resolveBaseUrl(env, request)
  const next = url.searchParams.get("next") || "/mcp"
  const state = await createState(env.SESSION_SECRET, next)
  const redirect_uri = `${base}/auth/github/callback`
  const gh = new URL("https://github.com/login/oauth/authorize")
  gh.searchParams.set("client_id", env.GITHUB_CLIENT_ID)
  gh.searchParams.set("redirect_uri", redirect_uri)
  gh.searchParams.set("state", state)
  gh.searchParams.set("scope", "read:user user:email")
  return Response.redirect(gh.toString(), 302)
}

export async function handleGitHubCallback(request: Request, env: Env): Promise<Response> {
  if (!env.SESSION_SECRET || !env.GITHUB_CLIENT_ID || !env.GITHUB_CLIENT_SECRET)
    return new Response("missing SESSION_SECRET or GitHub OAuth credentials", { status: 500 })
  const url = new URL(request.url)
  const code = url.searchParams.get("code")
  const state = url.searchParams.get("state")
  if (!code || !state) return new Response(null, { status: 400 })
  const st = await verifyState(env.SESSION_SECRET, state)
  if (!st) return new Response("invalid_state", { status: 400 })
  const base = resolveBaseUrl(env, request)
  const redirect_uri = `${base}/auth/github/callback`
  const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({
      client_id: env.GITHUB_CLIENT_ID,
      client_secret: env.GITHUB_CLIENT_SECRET,
      code,
      redirect_uri,
    }),
  })
  if (!tokenRes.ok) return new Response("token_exchange_failed", { status: 502 })
  const tokenJson: any = await tokenRes.json()
  const accessToken: string | undefined = tokenJson.access_token
  if (!accessToken) return new Response("no_access_token", { status: 502 })
  const octokit = new Octokit({ auth: accessToken })
  const { data: user } = await octokit.request("GET /user", {})
  const iat = Math.floor(Date.now() / 1000)
  const exp = iat + getSessionTtlSeconds(env)
  const session = await createSession(env.SESSION_SECRET, {
    iss: base,
    sub: user.id,
    login: user.login,
    name: user.name ?? null,
    avatarUrl: user.avatar_url ?? null,
    iat,
    exp,
  })
  const cookie = makeCookie(getCookieName(env), session, {
    path: "/",
    httpOnly: true,
    secure: true,
    sameSite: "Lax",
    maxAge: getSessionTtlSeconds(env),
  })
  const dest = st.next || "/mcp"
  return new Response(null, { status: 302, headers: { Location: dest, "Set-Cookie": cookie } })
}

export async function handleLogout(_request: Request, env: Env): Promise<Response> {
  const cookie = makeCookie(getCookieName(env), "", {
    path: "/",
    httpOnly: true,
    secure: true,
    sameSite: "Lax",
    maxAge: 0,
  })
  return new Response(null, { status: 302, headers: { Location: "/", "Set-Cookie": cookie } })
}

export async function handleMintToken(request: Request, env: Env): Promise<Response> {
  const cookieName = getCookieName(env)
  const raw = getCookie(request, cookieName)
  if (!raw) return new Response("unauthorized", { status: 401 })
  const session = await verifySession(env.SESSION_SECRET, raw)
  if (!session) return new Response("unauthorized", { status: 401 })
  const ttl = getTokenTtlSeconds(env)
  const token = await createAccessToken(env.SESSION_SECRET, session, "mcp", ttl)
  const body = JSON.stringify({ access_token: token, token_type: "bearer", expires_in: ttl })
  return new Response(body, { status: 200, headers: { "Content-Type": "application/json" } })
}

export async function ensureAuthorized(request: Request, env: Env): Promise<Response | null> {
  const auth = request.headers.get("Authorization")
  if (auth?.startsWith("Bearer ")) {
    const token = auth.slice("Bearer ".length).trim()
    const verified = await verifyAccessToken(env.SESSION_SECRET, token, "mcp")
    if (verified) return null
  }
  const raw = getCookie(request, getCookieName(env))
  if (raw) {
    const session = await verifySession(env.SESSION_SECRET, raw)
    if (session) return null
  }
  const url = new URL(request.url)
  if (request.method === "GET") {
    const base = resolveBaseUrl(env, request)
    const loginUrl = new URL(`${base}/auth/github/login`)
    loginUrl.searchParams.set("next", url.pathname + url.search)
    return Response.redirect(loginUrl.toString(), 302)
  }
  return new Response("unauthorized", { status: 401 })
}
