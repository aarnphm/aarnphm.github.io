import { Octokit } from "octokit"
import { fetchUpstreamAuthToken, getUpstreamAuthorizeUrl } from "../utils"
import { bindStateToSession, createState, OAuthError, validateState } from "../workers-oauth-utils"

export type GithubOAuthConfig = {
  clientId: (env: Env) => string
  clientSecret: (env: Env) => string
  statePrefix: string
  stateCookieName: string
  scopes: string
  callbackPath: string
}

export type GithubUser = { login: string; name: string | null; email: string | null }

export type GithubOAuthCallbacks<TState, TResult> = {
  parseStatePayload: (raw: string) => TState | null
  onComplete: (
    req: Request,
    env: Env,
    state: TState,
    user: GithubUser,
    accessToken: string,
  ) => Promise<TResult>
  formatResult: (result: TResult, req: Request) => Response
}

export type GithubOAuthHandler<TState> = {
  initiateAuth: (
    req: Request,
    env: Env,
    statePayload: TState,
    extraHeaders?: Record<string, string>,
  ) => Promise<Response>
  handleCallback: (req: Request, env: Env) => Promise<Response>
}

export function createGithubOAuthHandler<TState, TResult>(
  config: GithubOAuthConfig,
  callbacks: GithubOAuthCallbacks<TState, TResult>,
): GithubOAuthHandler<TState> {
  async function initiateAuth(
    req: Request,
    env: Env,
    statePayload: TState,
    extraHeaders: Record<string, string> = {},
  ): Promise<Response> {
    const { stateToken } = await createState(env.OAUTH_KV, config.statePrefix, statePayload, 600)
    const { setCookie } = await bindStateToSession(stateToken, config.stateCookieName)
    const redirectUri = new URL(config.callbackPath, req.url).href

    const authorizeUrl = getUpstreamAuthorizeUrl({
      upstream_url: "https://github.com/login/oauth/authorize",
      client_id: config.clientId(env),
      scope: config.scopes,
      redirect_uri: redirectUri,
      state: stateToken,
    })

    const headers = new Headers(extraHeaders)
    headers.append("Set-Cookie", setCookie)
    headers.set("Location", authorizeUrl)

    return new Response(null, { status: 302, headers })
  }

  async function handleCallback(req: Request, env: Env): Promise<Response> {
    const { raw, clearCookie } = await validateState(req, env.OAUTH_KV, {
      statePrefix: config.statePrefix,
      cookieName: config.stateCookieName,
    })

    const state = callbacks.parseStatePayload(raw)
    if (!state) {
      throw new OAuthError("server_error", "Invalid state data", 500)
    }

    const redirectUri = new URL(config.callbackPath, req.url).href
    const [accessToken, errResponse] = await fetchUpstreamAuthToken({
      client_id: config.clientId(env),
      client_secret: config.clientSecret(env),
      code: new URL(req.url).searchParams.get("code"),
      redirect_uri: redirectUri,
      upstream_url: "https://github.com/login/oauth/access_token",
    })

    if (errResponse) return errResponse

    const { data } = await new Octokit({ auth: accessToken }).rest.users.getAuthenticated()
    const user: GithubUser = {
      login: data.login,
      name: data.name ?? null,
      email: data.email ?? null,
    }

    const result = await callbacks.onComplete(req, env, state, user, accessToken)
    const response = callbacks.formatResult(result, req)

    const headers = new Headers(response.headers)
    headers.append("Set-Cookie", clearCookie)

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    })
  }

  return { initiateAuth, handleCallback }
}

export { OAuthError }
