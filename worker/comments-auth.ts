const commentGithubAuthorPrefix = "comment-auth:github:"

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

function safeJsonForHtml(value: unknown): string {
  return JSON.stringify(value).replace(/</g, "\\u003c")
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
