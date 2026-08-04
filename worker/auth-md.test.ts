import assert from 'node:assert/strict'
import test from 'node:test'
import {
  agentAuthMetadata,
  authMarkdown,
  handleAuthMarkdown,
  oauthProtectedResourceMetadata,
  withAgentAuthMetadata,
} from './auth-md'

const BASE_URL = 'https://aarnphm.xyz'

test('publishes agent registration instructions at the site root', async () => {
  const response = handleAuthMarkdown(new Request(`${BASE_URL}/auth.md`), BASE_URL)
  assert.ok(response)
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('Content-Type'), 'text/markdown; charset=utf-8')
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), '*')

  const markdown = await response.text()
  assert.equal(markdown.startsWith('# aarnphm.xyz auth.md\n'), true)
  assert.equal(markdown.includes(`POST ${BASE_URL}/register`), true)
  assert.equal(markdown.includes(`GET ${BASE_URL}/.well-known/oauth-protected-resource/mcp`), true)
  assert.equal(markdown.includes('credential_types_supported'), false)
})

test('serves auth.md head requests without a body and rejects mutations', async () => {
  const head = handleAuthMarkdown(new Request(`${BASE_URL}/auth.md`, { method: 'HEAD' }), BASE_URL)
  assert.ok(head)
  assert.equal(head.status, 200)
  assert.equal(await head.text(), '')

  const post = handleAuthMarkdown(new Request(`${BASE_URL}/auth.md`, { method: 'POST' }), BASE_URL)
  assert.ok(post)
  assert.equal(post.status, 405)
  assert.equal(post.headers.get('Allow'), 'GET, HEAD')
})

test('leaves resource identifiers path-derived and configures shared OAuth metadata', () => {
  assert.deepEqual(oauthProtectedResourceMetadata(BASE_URL), {
    authorization_servers: [BASE_URL],
    scopes_supported: ['mcp'],
    bearer_methods_supported: ['header'],
    resource_name: "Aaron's notes MCP",
  })
})

test('advertises the complete supported agent registration method', () => {
  assert.deepEqual(agentAuthMetadata(BASE_URL), {
    skill: `${BASE_URL}/auth.md`,
    register_uri: `${BASE_URL}/register`,
    claim_uri: `${BASE_URL}/authorize`,
    revocation_uri: `${BASE_URL}/token`,
    identity_types_supported: ['anonymous'],
    anonymous: { credential_types_supported: ['access_token'] },
  })
})

test('extends authorization server metadata without replacing OAuth fields', async () => {
  const response = await withAgentAuthMetadata(
    new Response(
      JSON.stringify({
        issuer: BASE_URL,
        token_endpoint: `${BASE_URL}/token`,
        grant_types_supported: ['authorization_code', 'refresh_token'],
      }),
      { headers: { 'Content-Type': 'application/json' } },
    ),
    BASE_URL,
  )

  assert.deepEqual(await response.json(), {
    issuer: BASE_URL,
    token_endpoint: `${BASE_URL}/token`,
    grant_types_supported: ['authorization_code', 'refresh_token'],
    agent_auth: agentAuthMetadata(BASE_URL),
  })
})

test('generates self-contained instructions for the advertised endpoints', () => {
  const markdown = authMarkdown(BASE_URL)
  for (const path of [
    '/.well-known/oauth-protected-resource/mcp',
    '/.well-known/oauth-authorization-server',
    '/register',
    '/authorize',
    '/token',
    '/mcp',
  ]) {
    assert.equal(markdown.includes(`${BASE_URL}${path}`), true)
  }
})
