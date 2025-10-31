import type { OAuthHelpers } from "@cloudflare/workers-oauth-provider"
import { Hono } from "hono"
import { Octokit } from "octokit"
import { fetchUpstreamAuthToken, getUpstreamAuthorizeUrl, type Props } from "./utils"
import {
  addApprovedClient,
  bindStateToSession,
  createOAuthState,
  generateCSRFProtection,
  isClientApproved,
  OAuthError,
  renderApprovalDialog,
  validateCSRFToken,
  validateOAuthState,
} from "./workers-oauth-utils"

type Env = {
  OAUTH_PROVIDER: OAuthHelpers
  GITHUB_CLIENT_ID: string
  GITHUB_CLIENT_SECRET: string
  SESSION_SECRET: string
  OAUTH_KV: KVNamespace
}

const app = new Hono<{ Bindings: Env }>()

app.get("/authorize", async (c) => {
  const oauthReqInfo = await c.env.OAUTH_PROVIDER.parseAuthRequest(c.req.raw)
  const { clientId } = oauthReqInfo
  if (!clientId) {
    return c.text("Invalid request", 400)
  }

  if (await isClientApproved(c.req.raw, clientId, c.env.SESSION_SECRET)) {
    const { stateToken } = await createOAuthState(oauthReqInfo, c.env.OAUTH_KV)
    const { setCookie: sessionBindingCookie } = await bindStateToSession(stateToken)
    return redirectToGithub(c.req.raw, stateToken, c.env.GITHUB_CLIENT_ID, {
      "Set-Cookie": sessionBindingCookie,
    })
  }

  const { token: csrfToken, setCookie } = generateCSRFProtection()

  return renderApprovalDialog(c.req.raw, {
    client: await c.env.OAUTH_PROVIDER.lookupClient(clientId),
    csrfToken,
    server: {
      description: "Aaron's notes MCP",
      logo: "https://avatars.githubusercontent.com/u/29749331?v=4",
      name: "Aaron's notes",
    },
    setCookie,
    state: { oauthReqInfo },
  })
})

app.post("/authorize", async (c) => {
  try {
    const formData = await c.req.raw.formData()

    validateCSRFToken(formData, c.req.raw)

    const encodedState = formData.get("state")
    if (!encodedState || typeof encodedState !== "string") {
      return c.text("Missing state in form data", 400)
    }

    let state: { oauthReqInfo?: any }
    try {
      state = JSON.parse(atob(encodedState))
    } catch (_e) {
      return c.text("Invalid state data", 400)
    }

    if (!state.oauthReqInfo || !state.oauthReqInfo.clientId) {
      return c.text("Invalid request", 400)
    }

    const approvedClientCookie = await addApprovedClient(
      c.req.raw,
      state.oauthReqInfo.clientId,
      c.env.SESSION_SECRET,
    )

    const { stateToken } = await createOAuthState(state.oauthReqInfo, c.env.OAUTH_KV)
    const { setCookie: sessionBindingCookie } = await bindStateToSession(stateToken)

    const headers = new Headers()
    headers.append("Set-Cookie", approvedClientCookie)
    headers.append("Set-Cookie", sessionBindingCookie)

    return redirectToGithub(
      c.req.raw,
      stateToken,
      c.env.GITHUB_CLIENT_ID,
      Object.fromEntries(headers),
    )
  } catch (error: any) {
    console.error("POST /authorize error:", error)
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text(`Internal server error: ${error.message}`, 500)
  }
})

async function redirectToGithub(
  request: Request,
  stateToken: string,
  clientId: string,
  headers: Record<string, string> = {},
) {
  return new Response(null, {
    headers: {
      ...headers,
      location: getUpstreamAuthorizeUrl({
        client_id: clientId,
        redirect_uri: new URL("/callback", request.url).href,
        scope: "read:user",
        state: stateToken,
        upstream_url: "https://github.com/login/oauth/authorize",
      }),
    },
    status: 302,
  })
}

app.get("/callback", async (c) => {
  let oauthReqInfo: any
  let clearSessionCookie: string

  try {
    const result = await validateOAuthState(c.req.raw, c.env.OAUTH_KV)
    oauthReqInfo = result.oauthReqInfo
    clearSessionCookie = result.clearCookie
  } catch (error: any) {
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text("Internal server error", 500)
  }

  if (!oauthReqInfo.clientId) {
    return c.text("Invalid OAuth request data", 400)
  }

  const [accessToken, errResponse] = await fetchUpstreamAuthToken({
    client_id: c.env.GITHUB_CLIENT_ID,
    client_secret: c.env.GITHUB_CLIENT_SECRET,
    code: c.req.query("code"),
    redirect_uri: new URL("/callback", c.req.url).href,
    upstream_url: "https://github.com/login/oauth/access_token",
  })
  if (errResponse) return errResponse

  const user = await new Octokit({ auth: accessToken }).rest.users.getAuthenticated()
  const { login, name, email } = user.data

  const { redirectTo } = await c.env.OAUTH_PROVIDER.completeAuthorization({
    metadata: {
      label: name,
    },
    props: {
      accessToken,
      email,
      login,
      name,
    } as Props,
    request: oauthReqInfo,
    scope: oauthReqInfo.scope,
    userId: login,
  })

  const headers = new Headers({ Location: redirectTo })
  if (clearSessionCookie) {
    headers.set("Set-Cookie", clearSessionCookie)
  }

  return new Response(null, {
    status: 302,
    headers,
  })
})

export { app as GitHubHandler }
