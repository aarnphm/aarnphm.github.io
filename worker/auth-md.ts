import { isRecord } from './type-guards'

export const AUTH_MD_PATH = '/auth.md'
export const OAUTH_AUTHORIZATION_SERVER_PATH = '/.well-known/oauth-authorization-server'
export const OAUTH_SCOPES = ['mcp']

export function oauthProtectedResourceMetadata(baseUrl: string) {
  return {
    authorization_servers: [baseUrl],
    scopes_supported: [...OAUTH_SCOPES],
    bearer_methods_supported: ['header'],
    resource_name: "Aaron's notes MCP",
  }
}

export function agentAuthMetadata(baseUrl: string) {
  return {
    skill: `${baseUrl}${AUTH_MD_PATH}`,
    register_uri: `${baseUrl}/register`,
    claim_uri: `${baseUrl}/authorize`,
    revocation_uri: `${baseUrl}/token`,
    identity_types_supported: ['anonymous'],
    anonymous: { credential_types_supported: ['access_token'] },
  }
}

export function authMarkdown(baseUrl: string): string {
  return `# aarnphm.xyz auth.md

This garden exposes an MCP protected resource at \`${baseUrl}/mcp\`. Agents register as OAuth 2.0 clients, then a human approves the client and authenticates with GitHub. The service issues bearer access tokens and refresh tokens. It does not issue API keys.

## Discover

Fetch the protected resource metadata, then follow its authorization server URL:

\`\`\`text
GET ${baseUrl}/.well-known/oauth-protected-resource/mcp
GET ${baseUrl}/.well-known/oauth-authorization-server
\`\`\`

The protected resource audience is \`${baseUrl}/mcp\`. Request the \`mcp\` scope and present access tokens in the \`Authorization\` header.

## Register

Register a public OAuth client with a redirect URI controlled by the agent:

\`\`\`http
POST ${baseUrl}/register
Content-Type: application/json

{
  "client_name": "Example agent",
  "redirect_uris": ["https://agent.example.com/oauth/callback"],
  "token_endpoint_auth_method": "none",
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"]
}
\`\`\`

Store the returned \`client_id\`. This is an anonymous registration because the agent does not submit an identity assertion. A human claims access during authorization.

## Authorize

Generate a PKCE verifier and its S256 challenge, then open this URL for the human:

\`\`\`text
${baseUrl}/authorize?response_type=code&client_id=<client_id>&redirect_uri=<redirect_uri>&scope=mcp&state=<state>&code_challenge=<challenge>&code_challenge_method=S256&resource=${baseUrl}/mcp
\`\`\`

The human approves the client and authenticates with GitHub. Validate \`state\` when the authorization server redirects to the registered redirect URI.

## Exchange

Exchange the returned authorization code:

\`\`\`http
POST ${baseUrl}/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&client_id=<client_id>&code=<code>&redirect_uri=<redirect_uri>&code_verifier=<verifier>&resource=${baseUrl}/mcp
\`\`\`

Use the returned \`access_token\` as a bearer credential when calling the MCP endpoint:

\`\`\`http
POST ${baseUrl}/mcp
Authorization: Bearer <access_token>
Content-Type: application/json
\`\`\`

## Refresh and revoke

Refresh an expired access token at \`${baseUrl}/token\` with \`grant_type=refresh_token\`, the \`client_id\`, and the \`refresh_token\`.

Revoke either token with an RFC 7009 request:

\`\`\`http
POST ${baseUrl}/token
Content-Type: application/x-www-form-urlencoded

token=<access_token_or_refresh_token>&client_id=<client_id>
\`\`\`
`
}

export function handleAuthMarkdown(request: Request, baseUrl: string): Response | null {
  if (new URL(request.url).pathname !== AUTH_MD_PATH) return null
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return new Response('method not allowed', { status: 405, headers: { Allow: 'GET, HEAD' } })
  }
  return new Response(request.method === 'HEAD' ? null : authMarkdown(baseUrl), {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Cache-Control': 'public, max-age=300',
      'Content-Type': 'text/markdown; charset=utf-8',
      'X-Content-Type-Options': 'nosniff',
    },
  })
}

export async function withAgentAuthMetadata(
  response: Response,
  baseUrl: string,
): Promise<Response> {
  const metadata: unknown = await response.clone().json()
  if (!isRecord(metadata)) return response
  const headers = new Headers(response.headers)
  headers.delete('Content-Length')
  return new Response(JSON.stringify({ ...metadata, agent_auth: agentAuthMetadata(baseUrl) }), {
    headers,
    status: response.status,
    statusText: response.statusText,
  })
}
