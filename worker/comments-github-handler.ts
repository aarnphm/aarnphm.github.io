import { Hono } from "hono"
import {
  consumeCommentAuthState,
  createCommentAuthState,
  getCommentGithubClient,
  getGithubCommentAuthor,
  normalizeAuthor,
  normalizeReturnTo,
  renderCommentAuthResponse,
  setGithubCommentAuthor,
  type CommentAuthEnv,
} from "./comments-auth"
import { resolveBaseUrl } from "./request-utils"
import { fetchUpstreamAuthToken, getUpstreamAuthorizeUrl, getUserFromGitHub } from "./utils"

type Env = {
  OAUTH_KV: KVNamespace
} & CommentAuthEnv &
  Cloudflare.Env

const app = new Hono<{ Bindings: Env }>()

app.get("/comments/github/login", async (c) => {
  const returnTo = normalizeReturnTo(c.req.raw, c.req.query("returnTo") ?? null)
  const author = normalizeAuthor(c.req.query("author") ?? null)
  const commentClient = getCommentGithubClient(c.env)
  if (!commentClient) {
    return c.text("comment github oauth not configured", 500)
  }
  const stateToken = await createCommentAuthState(c.env.OAUTH_KV, returnTo, author)
  const redirectUri = new URL("/comments/github/callback", resolveBaseUrl(c.env, c.req.raw)).href
  const authorizeUrl = getUpstreamAuthorizeUrl({
    upstream_url: "https://github.com/login/oauth/authorize",
    client_id: commentClient.clientId,
    scope: "read:user",
    redirect_uri: redirectUri,
    state: stateToken,
  })
  return c.redirect(authorizeUrl, 302)
})

app.get("/comments/github/callback", async (c) => {
  const stateToken = c.req.query("state")
  if (!stateToken) return c.text("Missing state", 400)
  const state = await consumeCommentAuthState(c.env.OAUTH_KV, stateToken)
  if (!state) return c.text("Invalid state", 400)
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
  const user = await getUserFromGitHub(accessToken)
  const login = user.login || "github-user"
  const storedAuthor = await getGithubCommentAuthor(c.env.OAUTH_KV, login)
  const stateAuthor = normalizeAuthor(state.author ?? null)
  const resolvedAuthor = stateAuthor || storedAuthor || normalizeAuthor(login) || "github-user"
  await setGithubCommentAuthor(c.env.OAUTH_KV, login, resolvedAuthor)
  return renderCommentAuthResponse(resolvedAuthor, state.returnTo, login)
})

export { app as CommentsGitHubHandler }
