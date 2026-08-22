import assert from 'node:assert/strict'
import test from 'node:test'
import {
  MCP_RATE_LIMIT_EXHAUSTED,
  MCP_RATE_LIMIT_POLICY,
  mcpRateLimitExceededResponse,
  mcpRateLimitKey,
  withMcpRateLimitPolicy,
} from './rate-limit'

test('keys MCP rate limits by connecting IP', () => {
  assert.equal(
    mcpRateLimitKey(
      new Request('https://aarnphm.xyz/mcp', { headers: { 'CF-Connecting-IP': '203.0.113.42' } }),
    ),
    '203.0.113.42',
  )
})

test('advertises the MCP policy on ordinary responses', () => {
  const response = withMcpRateLimitPolicy(new Response('authorized', { status: 200 }))
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('RateLimit-Policy'), MCP_RATE_LIMIT_POLICY)
})

test('returns RateLimit and Retry-After when the MCP budget is exhausted', async () => {
  const response = mcpRateLimitExceededResponse(new Request('https://aarnphm.xyz/mcp'))
  assert.equal(response.status, 429)
  assert.equal(response.headers.get('RateLimit-Policy'), MCP_RATE_LIMIT_POLICY)
  assert.equal(response.headers.get('RateLimit'), MCP_RATE_LIMIT_EXHAUSTED)
  assert.equal(response.headers.get('Retry-After'), '60')
  assert.equal((await response.json()).code, 'mcp_rate_limit_exceeded')
})
