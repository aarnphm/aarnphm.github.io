import type { Env } from './index/environment'
import { normalizeAuthor } from './comments'

// A commenter's chosen blob (quartz/util/blob-avatar.ts) as its short trait code. Authors without
// one get the blob their name picks, so nothing is stored for them.
const avatarKeyPrefix = 'comment-avatar:'
const avatarCode = /^b1(\.[a-z][0-9]{1,2}){1,8}$/
const maxAuthors = 64

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

export async function handleCommentAvatars(request: Request, env: Env): Promise<Response | null> {
  const url = new URL(request.url)
  if (url.pathname !== '/comments/avatars') return null

  if (request.method === 'GET') {
    const authors = [
      ...new Set(
        url.searchParams
          .getAll('author')
          .map(normalizeAuthor)
          .filter((a): a is string => !!a),
      ),
    ].slice(0, maxAuthors)
    const codes = await Promise.all(authors.map(a => env.OAUTH_KV.get(`${avatarKeyPrefix}${a}`)))
    const avatars: Record<string, string> = {}
    authors.forEach((a, i) => {
      const code = codes[i]
      if (code && avatarCode.test(code)) avatars[a] = code
    })
    return json({ avatars })
  }

  if (request.method === 'POST') {
    let payload: { author?: unknown; code?: unknown }
    try {
      payload = (await request.json()) as { author?: unknown; code?: unknown }
    } catch {
      return new Response('invalid json', { status: 400 })
    }
    const author = typeof payload.author === 'string' ? normalizeAuthor(payload.author) : null
    if (!author) return new Response('invalid author', { status: 400 })
    const key = `${avatarKeyPrefix}${author}`
    // An empty code goes back to the blob the name picks.
    if (payload.code === '') {
      await env.OAUTH_KV.delete(key)
      return json({ ok: true })
    }
    if (typeof payload.code !== 'string' || !avatarCode.test(payload.code))
      return new Response('invalid avatar', { status: 400 })
    await env.OAUTH_KV.put(key, payload.code)
    return json({ ok: true })
  }

  return new Response('method not allowed', { status: 405 })
}

// A renamed commenter keeps the blob they pinned.
export async function moveCommentAvatar(env: Env, from: string, to: string) {
  const code = await env.OAUTH_KV.get(`${avatarKeyPrefix}${from}`)
  if (!code) return
  await env.OAUTH_KV.put(`${avatarKeyPrefix}${to}`, code)
  await env.OAUTH_KV.delete(`${avatarKeyPrefix}${from}`)
}
