import type { OAuthHelpers } from "@cloudflare/workers-oauth-provider"
import { Hono } from "hono"
import { Octokit } from "octokit"
import { fetchUpstreamAuthToken, getUpstreamAuthorizeUrl, type Props } from "./utils"
import {
  clientIdAlreadyApproved,
  parseRedirectApproval,
  renderApprovalDialog,
} from "./workers-oauth-utils"

const app = new Hono<{
  Bindings: {
    OAUTH_PROVIDER: OAuthHelpers
    GITHUB_CLIENT_ID: string
    GITHUB_CLIENT_SECRET: string
    SESSION_SECRET: string
  }
}>()

app.get("/authorize", async (c) => {
  const oauthReqInfo = await c.env.OAUTH_PROVIDER.parseAuthRequest(c.req.raw)
  const { clientId } = oauthReqInfo
  if (!clientId) return c.text("Invalid request", 400)
  if (await clientIdAlreadyApproved(c.req.raw, clientId, c.env.SESSION_SECRET)) {
    return redirectToGithub(c.req.raw, oauthReqInfo, {}, c.env.GITHUB_CLIENT_ID)
  }
  return renderApprovalDialog(c.req.raw, {
    client: await c.env.OAUTH_PROVIDER.lookupClient(clientId),
    server: {
      description: "Aaron's notes MCP",
      logo: "https://avatars.githubusercontent.com/u/29749331?v=4",
      name: "Aaron's notes",
    },
    state: { oauthReqInfo },
  })
})

app.post("/authorize", async (c) => {
  const { state, headers } = await parseRedirectApproval(c.req.raw, c.env.SESSION_SECRET)
  if (!state.oauthReqInfo) return c.text("Invalid request", 400)
  return redirectToGithub(c.req.raw, state.oauthReqInfo, headers, c.env.GITHUB_CLIENT_ID)
})

async function redirectToGithub(
  request: Request,
  oauthReqInfo: Parameters<OAuthHelpers["completeAuthorization"]>[0]["request"],
  headers: Record<string, string> = {},
  clientId: string,
) {
  return new Response(null, {
    headers: {
      ...headers,
      location: getUpstreamAuthorizeUrl({
        client_id: clientId,
        redirect_uri: new URL("/callback", request.url).href,
        scope: "read:user",
        state: btoa(JSON.stringify(oauthReqInfo)),
        upstream_url: "https://github.com/login/oauth/authorize",
      }),
    },
    status: 302,
  })
}

app.get("/callback", async (c) => {
  const oauthReqInfo = JSON.parse(atob(c.req.query("state") as string))
  if (!oauthReqInfo?.clientId) return c.text("Invalid state", 400)
  const [accessToken, err] = await fetchUpstreamAuthToken({
    client_id: c.env.GITHUB_CLIENT_ID,
    client_secret: c.env.GITHUB_CLIENT_SECRET,
    code: c.req.query("code"),
    redirect_uri: new URL("/callback", c.req.url).href,
    upstream_url: "https://github.com/login/oauth/access_token",
  })
  if (err) return err
  const user = await new Octokit({ auth: accessToken }).rest.users.getAuthenticated()
  const { login, name, email } = user.data as any
  const { redirectTo } = await c.env.OAUTH_PROVIDER.completeAuthorization({
    metadata: { label: name },
    props: { accessToken, email, login, name } as Props,
    request: oauthReqInfo,
    scope: oauthReqInfo.scope,
    userId: login,
  })
  return Response.redirect(redirectTo)
})

export { app as GitHubHandler }
