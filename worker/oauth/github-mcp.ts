import type { AuthRequest, OAuthHelpers } from "@cloudflare/workers-oauth-provider"
import { Hono } from "hono"
import type { Props } from "../utils"
import {
  addApprovedClient,
  generateCSRFProtection,
  isClientApproved,
  OAuthError,
  renderApprovalDialog,
  validateCSRFToken,
} from "../workers-oauth-utils"
import { createGithubOAuthHandler, type GithubUser } from "./core"

type McpOAuthState = {
  oauthReqInfo: AuthRequest
}

type McpOAuthResult = {
  redirectTo: string
  user: GithubUser
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function parseAuthRequestFromRecord(obj: Record<string, unknown>): AuthRequest | null {
  const responseType = typeof obj.responseType === "string" ? obj.responseType : null
  const clientId = typeof obj.clientId === "string" ? obj.clientId : null
  const redirectUri = typeof obj.redirectUri === "string" ? obj.redirectUri : null
  const state = typeof obj.state === "string" ? obj.state : null
  const scope =
    Array.isArray(obj.scope) && obj.scope.every((item) => typeof item === "string")
      ? obj.scope
      : null
  if (!responseType || !clientId || !redirectUri || !state || !scope) return null
  const codeChallenge = typeof obj.codeChallenge === "string" ? obj.codeChallenge : undefined
  const codeChallengeMethod =
    typeof obj.codeChallengeMethod === "string" ? obj.codeChallengeMethod : undefined
  let resource: string | string[] | undefined
  if (typeof obj.resource === "string") {
    resource = obj.resource
  } else if (Array.isArray(obj.resource) && obj.resource.every((item) => typeof item === "string")) {
    resource = obj.resource
  }
  return {
    responseType,
    clientId,
    redirectUri,
    scope,
    state,
    codeChallenge,
    codeChallengeMethod,
    resource,
  }
}

function parseMcpOAuthState(raw: string): McpOAuthState | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }
  if (!isRecord(parsed)) return null
  if (!isRecord(parsed.oauthReqInfo)) return null
  const oauthReqInfo = parseAuthRequestFromRecord(parsed.oauthReqInfo)
  if (!oauthReqInfo) return null
  return { oauthReqInfo }
}

const mcpOAuth = createGithubOAuthHandler<McpOAuthState, McpOAuthResult>(
  {
    clientId: (env) => env.GITHUB_CLIENT_ID,
    clientSecret: (env) => env.GITHUB_CLIENT_SECRET,
    statePrefix: "oauth:state:",
    stateCookieName: "__Host-CONSENTED_STATE",
    scopes: "read:user",
    callbackPath: "/callback",
  },
  {
    createStatePayload: async () => {
      throw new Error("createStatePayload not used for MCP flow")
    },
    parseStatePayload: parseMcpOAuthState,
    onComplete: async (req, env, state, user, accessToken) => {
      const provider = (env as any).OAUTH_PROVIDER as OAuthHelpers
      const { redirectTo } = await provider.completeAuthorization({
        metadata: { label: user.name },
        props: {
          accessToken,
          email: user.email,
          login: user.login,
          name: user.name,
        } as Props,
        request: state.oauthReqInfo,
        scope: state.oauthReqInfo.scope,
        userId: user.login,
      })
      return { redirectTo, user }
    },
    formatResult: (result, _req) => {
      const body = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>auth finished</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <p>auth finished. you can close this window.</p>
    <p><a href="${result.redirectTo}">continue</a></p>
  </body>
</html>`
      return new Response(body, {
        status: 302,
        headers: { Location: result.redirectTo },
      })
    },
  },
)

const app = new Hono<{ Bindings: { OAUTH_PROVIDER: OAuthHelpers } & Env }>()

app.get("/authorize", async (c) => {
  const oauthReqInfo = await c.env.OAUTH_PROVIDER.parseAuthRequest(c.req.raw)
  const { clientId } = oauthReqInfo
  if (!clientId) {
    return c.text("Invalid request", 400)
  }

  if (await isClientApproved(c.req.raw, clientId, c.env.SESSION_SECRET)) {
    const statePayload: McpOAuthState = { oauthReqInfo }
    return mcpOAuth.initiateAuth(c.req.raw, c.env, statePayload)
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
    } catch {
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

    const statePayload: McpOAuthState = { oauthReqInfo: state.oauthReqInfo }
    const headers = new Headers()
    headers.append("Set-Cookie", approvedClientCookie)

    const headerEntries: [string, string][] = []
    headers.forEach((value, key) => headerEntries.push([key, value]))
    return mcpOAuth.initiateAuth(c.req.raw, c.env, statePayload, Object.fromEntries(headerEntries))
  } catch (error: any) {
    console.error("POST /authorize error:", error)
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text(`Internal server error: ${error.message}`, 500)
  }
})

app.get("/callback", async (c) => {
  try {
    return await mcpOAuth.handleCallback(c.req.raw, c.env)
  } catch (error: any) {
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text("Internal server error", 500)
  }
})

export { app as GitHubHandler }
