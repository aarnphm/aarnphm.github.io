import type { OAuthHelpers } from '@cloudflare/workers-oauth-provider'
import { drizzle } from 'drizzle-orm/d1'
import { Hono } from 'hono'
import { createArenaReaderSessionCookie } from '../arena-reader-auth'
import {
  getGithubCommentAuthor,
  normalizeAuthor,
  normalizeReturnTo,
  renderCommentAuthResponse,
  setGithubCommentAuthor,
} from '../comments'
import { githubUsers } from '../schema'
import { isRecord } from '../type-guards'
import { createGithubOAuthHandler, OAuthError } from './core'

type CommentAuthState = { returnTo: string; author: string | null }

type CommentAuthResult = {
  author: string
  returnTo: string
  login: string
  readerSessionCookie: string | null
  readerAuthConfigured: boolean
}

function parseCommentAuthState(raw: string): CommentAuthState | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }
  if (!isRecord(parsed)) return null
  const returnTo = typeof parsed.returnTo === 'string' ? parsed.returnTo : null
  if (!returnTo) return null
  const author = typeof parsed.author === 'string' ? parsed.author : null
  return { returnTo, author }
}

const commentsOAuth = createGithubOAuthHandler<CommentAuthState, CommentAuthResult>(
  {
    clientId: env => env.GITHUB_COMMENTS_CLIENT_ID,
    clientSecret: env => env.GITHUB_COMMENTS_CLIENT_SECRET,
    statePrefix: 'comment-auth:state:',
    stateCookieName: '__Host-COMMENT_STATE',
    scopes: 'read:user',
    callbackPath: '/comments/github/callback',
  },
  {
    parseStatePayload: parseCommentAuthState,
    onComplete: async (req, env, state, user, _accessToken) => {
      const storedAuthor = await getGithubCommentAuthor(env.OAUTH_KV, user.login)
      const stateAuthor = normalizeAuthor(state.author)
      const resolvedAuthor =
        stateAuthor || storedAuthor || normalizeAuthor(user.login) || 'github-user'
      await setGithubCommentAuthor(env.OAUTH_KV, user.login, resolvedAuthor)

      const now = Date.now()
      const db = drizzle(env.COMMENTS_ROOM)
      await db
        .insert(githubUsers)
        .values({
          login: user.login,
          displayName: user.name || user.login,
          avatarUrl: user.avatar_url,
          lastSeenAt: now,
          firstSeenAt: now,
        })
        .onConflictDoUpdate({
          target: githubUsers.login,
          set: {
            displayName: user.name || user.login,
            avatarUrl: user.avatar_url,
            lastSeenAt: now,
          },
        })

      return {
        author: resolvedAuthor,
        returnTo: state.returnTo,
        login: user.login,
        readerSessionCookie: await createArenaReaderSessionCookie(req, env, user),
        readerAuthConfigured: Boolean(env.SESSION_SECRET?.trim()),
      }
    },
    formatResult: (result, req) => {
      const returnPath = new URL(result.returnTo, req.url).pathname
      if (
        !result.readerSessionCookie &&
        (returnPath === '/arena/feed' || returnPath === '/arena/feed/')
      ) {
        return new Response(
          result.readerAuthConfigured
            ? 'This GitHub account does not have access to the Arena reader.'
            : 'Arena reader sign-in is not configured.',
          {
            status: result.readerAuthConfigured ? 403 : 503,
            headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' },
          },
        )
      }
      const response = renderCommentAuthResponse(result.author, result.returnTo, result.login)
      if (result.readerSessionCookie) {
        response.headers.append('Set-Cookie', result.readerSessionCookie)
      }
      response.headers.set('Cache-Control', 'no-store')
      return response
    },
  },
)

const app = new Hono<{ Bindings: { OAUTH_PROVIDER: OAuthHelpers } & Env }>()

app.get('/comments/github/login', async c => {
  try {
    return await commentsOAuth.initiateAuth(c.req.raw, c.env, {
      returnTo: normalizeReturnTo(c.req.raw, c.req.query('returnTo') ?? null),
      author: normalizeAuthor(c.req.query('author') ?? null),
    })
  } catch (error: unknown) {
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text('Internal server error', 500)
  }
})

app.get('/comments/github/callback', async c => {
  try {
    return await commentsOAuth.handleCallback(c.req.raw, c.env)
  } catch (error: unknown) {
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text('Internal server error', 500)
  }
})

export { app as CommentsGitHubHandler }
