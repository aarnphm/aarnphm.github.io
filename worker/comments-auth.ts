import { bindStateToSession, createState, OAuthError, validateState } from "./workers-oauth-utils"

type CommentAuthState = {
  returnTo: string
  author?: string | null
}

const commentAuthStatePrefix = "comment-auth:state:"
const commentAuthStateCookieName = "__Host-COMMENT_STATE"
const commentGithubAuthorPrefix = "comment-auth:github:"

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export function normalizeReturnTo(request: Request, raw: string | null): string {
  if (!raw) return "/"
  let target: URL
  try {
    target = new URL(raw, request.url)
  } catch {
    return "/"
  }
  const origin = new URL(request.url).origin
  if (target.origin !== origin) return "/"
  return `${target.pathname}${target.search}${target.hash}`
}

export function normalizeAuthor(raw: string | null): string | null {
  if (!raw) return null
  const trimmed = raw.trim()
  if (!trimmed) return null
  if (trimmed.length > 128) return null
  return trimmed
}

function parseCommentAuthState(raw: string): CommentAuthState | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }
  if (!isRecord(parsed)) return null
  const returnTo = typeof parsed.returnTo === "string" ? parsed.returnTo : null
  if (!returnTo) return null
  const author = typeof parsed.author === "string" ? parsed.author : null
  return { returnTo, author }
}

function safeJsonForHtml(value: unknown): string {
  return JSON.stringify(value).replace(/</g, "\\u003c")
}

export async function createCommentAuthState(
  kv: KVNamespace,
  returnTo: string,
  author: string | null,
): Promise<{ stateToken: string; setCookie: string }> {
  const payload: CommentAuthState = { returnTo, author }
  const { stateToken } = await createState(kv, commentAuthStatePrefix, payload, 600)
  const { setCookie } = await bindStateToSession(stateToken, commentAuthStateCookieName)
  return { stateToken, setCookie }
}

export async function validateCommentAuthState(
  request: Request,
  kv: KVNamespace,
): Promise<{ state: CommentAuthState; clearCookie: string }> {
  const { raw, clearCookie } = await validateState(request, kv, {
    statePrefix: commentAuthStatePrefix,
    cookieName: commentAuthStateCookieName,
  })
  const state = parseCommentAuthState(raw)
  if (!state) {
    throw new OAuthError("server_error", "Invalid state data", 500)
  }
  return { state, clearCookie }
}

export async function getGithubCommentAuthor(
  kv: KVNamespace,
  login: string,
): Promise<string | null> {
  const raw = await kv.get(`${commentGithubAuthorPrefix}${login}`)
  if (!raw) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }
  if (!parsed || typeof parsed !== "object") return null
  const value = parsed as { author?: unknown }
  if (typeof value.author !== "string") return null
  const normalized = normalizeAuthor(value.author)
  return normalized
}

export async function setGithubCommentAuthor(
  kv: KVNamespace,
  login: string,
  author: string,
): Promise<void> {
  await kv.put(
    `${commentGithubAuthorPrefix}${login}`,
    JSON.stringify({ author, updatedAt: Date.now() }),
  )
}

export function renderCommentAuthResponse(
  author: string,
  returnTo: string,
  login: string | null,
): Response {
  const payload = safeJsonForHtml({ author, returnTo, login })
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Comment login</title>
</head>
<body>
<script>
const payload = ${payload};
try {
  localStorage.setItem("comment-author", payload.author);
  localStorage.setItem("comment-author-source", "github");
  if (payload.login) {
    localStorage.setItem("comment-author-github-login", payload.login);
  }
} catch {}
window.location.assign(payload.returnTo);
</script>
</body>
</html>`
  return new Response(html, {
    headers: { "Content-Type": "text/html; charset=utf-8" },
  })
}
