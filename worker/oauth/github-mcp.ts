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
import { createGithubOAuthHandler } from "./core"

type McpOAuthState = {
  oauthReqInfo: AuthRequest
}

type McpOAuthResult = {
  redirectTo: string
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
  } else if (
    Array.isArray(obj.resource) &&
    obj.resource.every((item) => typeof item === "string")
  ) {
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
      return { redirectTo }
    },
    formatResult: (result, _req) => {
      return new Response("auth finished", {
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
    return mcpOAuth.initiateAuth(c.req.raw, c.env, { oauthReqInfo })
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

    let rawState: unknown
    try {
      rawState = JSON.parse(atob(encodedState))
    } catch {
      return c.text("Invalid state data", 400)
    }

    if (!isRecord(rawState) || !isRecord(rawState.oauthReqInfo)) {
      return c.text("Invalid request", 400)
    }

    const oauthReqInfo = parseAuthRequestFromRecord(rawState.oauthReqInfo)
    if (!oauthReqInfo) {
      return c.text("Invalid request", 400)
    }

    const approvedClientCookie = await addApprovedClient(
      c.req.raw,
      oauthReqInfo.clientId,
      c.env.SESSION_SECRET,
    )

    return mcpOAuth.initiateAuth(
      c.req.raw,
      c.env,
      { oauthReqInfo },
      { "Set-Cookie": approvedClientCookie },
    )
  } catch (error: unknown) {
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text("Internal server error", 500)
  }
})

app.get("/callback", async (c) => {
  try {
    return await mcpOAuth.handleCallback(c.req.raw, c.env)
  } catch (error: unknown) {
    if (error instanceof OAuthError) {
      return error.toResponse()
    }
    return c.text("Internal server error", 500)
  }
})

export { app as GitHubHandler }
