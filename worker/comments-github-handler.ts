import { Hono } from "hono"
import { Octokit } from "octokit"
import {
  createCommentAuthState,
  getCommentGithubClient,
  getGithubCommentAuthor,
  normalizeAuthor,
  normalizeReturnTo,
  renderCommentAuthResponse,
  setGithubCommentAuthor,
  validateCommentAuthState,
  type CommentAuthEnv,
} from "./comments-auth"
import { resolveBaseUrl } from "./request-utils"
import { fetchUpstreamAuthToken, getUpstreamAuthorizeUrl } from "./utils"
import { OAuthError } from "./workers-oauth-utils"

type Env = {
  OAUTH_KV: KVNamespace
} & CommentAuthEnv &
  Cloudflare.Env

const app = new Hono<{ Bindings: Env }>()

app.get("/comments/github/login", async (c) => {
  try {
    const returnTo = normalizeReturnTo(c.req.raw, c.req.query("returnTo") ?? null)
    const author = normalizeAuthor(c.req.query("author") ?? null)
    const commentClient = getCommentGithubClient(c.env)
    if (!commentClient) {
      return c.text("comment github oauth not configured", 500)
    }
    const { stateToken, setCookie } = await createCommentAuthState(
      c.env.OAUTH_KV,
      returnTo,
      author,
    )
    const redirectUri = new URL("/comments/github/callback", resolveBaseUrl(c.env, c.req.raw)).href
    const authorizeUrl = getUpstreamAuthorizeUrl({
      upstream_url: "https://github.com/login/oauth/authorize",
      client_id: commentClient.clientId,
      scope: "read:user",
      redirect_uri: redirectUri,
      state: stateToken,
    })
    return new Response(null, {
      status: 302,
      headers: {
        Location: authorizeUrl,
        "Set-Cookie": setCookie,
      },
    })
  } catch (error: unknown) {
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text("Internal server error", 500)
  }
})

app.get("/comments/github/callback", async (c) => {
  try {
    const { state, clearCookie } = await validateCommentAuthState(c.req.raw, c.env.OAUTH_KV)
    const commentClient = getCommentGithubClient(c.env)
    if (!commentClient) {
      return c.text("comment github oauth not configured", 500)
    }
    const redirectUri = new URL("/comments/github/callback", resolveBaseUrl(c.env, c.req.raw)).href
    const [accessToken, errResponse] = await fetchUpstreamAuthToken({
      client_id: commentClient.clientId,
      client_secret: commentClient.clientSecret,
      code: c.req.query("code"),
      redirect_uri: redirectUri,
      upstream_url: "https://github.com/login/oauth/access_token",
    })
    if (errResponse) return errResponse
    const { data } = await new Octokit({ auth: accessToken }).rest.users.getAuthenticated()
    const login = data.login || "github-user"
    const storedAuthor = await getGithubCommentAuthor(c.env.OAUTH_KV, login)
    const stateAuthor = normalizeAuthor(state.author ?? null)
    const resolvedAuthor = stateAuthor || storedAuthor || normalizeAuthor(login) || "github-user"
    await setGithubCommentAuthor(c.env.OAUTH_KV, login, resolvedAuthor)
    const resp = renderCommentAuthResponse(resolvedAuthor, state.returnTo, login)
    const headers = new Headers(resp.headers)
    headers.append("Set-Cookie", clearCookie)
    return new Response(resp.body, {
      status: resp.status,
      statusText: resp.statusText,
      headers,
    })
  } catch (error: unknown) {
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text("Internal server error", 500)
  }
})

export { app as CommentsGitHubHandler }
