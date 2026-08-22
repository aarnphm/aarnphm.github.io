import assert from 'node:assert/strict'
import test from 'node:test'
import {
  MCP_SERVER_CARD_CONTENT_TYPE,
  MCP_SERVER_CARD_PATH,
  mcpServerCard,
  mcpServerCardResponse,
} from './mcp-server-card'

test('describes the public Streamable HTTP MCP server and its implemented capabilities', () => {
  assert.equal(MCP_SERVER_CARD_PATH, '/.well-known/mcp/server-card.json')
  assert.deepEqual(mcpServerCard('https://aarnphm.xyz'), {
    serverInfo: { name: 'aarnphm.xyz', version: '1.0.0' },
    transport: { type: 'streamable-http', endpoint: 'https://aarnphm.xyz/mcp' },
    capabilities: { tools: true },
  })
})

test('serves a cacheable, cross-origin JSON card for GET and an empty card for HEAD', async () => {
  const getResponse = mcpServerCardResponse(
    new Request(`https://aarnphm.xyz${MCP_SERVER_CARD_PATH}`),
    'https://aarnphm.xyz',
  )
  assert.equal(getResponse.status, 200)
  assert.equal(getResponse.headers.get('Content-Type'), MCP_SERVER_CARD_CONTENT_TYPE)
  assert.equal(getResponse.headers.get('Access-Control-Allow-Origin'), '*')
  assert.deepEqual(await getResponse.json(), mcpServerCard('https://aarnphm.xyz'))

  const headResponse = mcpServerCardResponse(
    new Request(`https://aarnphm.xyz${MCP_SERVER_CARD_PATH}`, { method: 'HEAD' }),
    'https://aarnphm.xyz',
  )
  assert.equal(headResponse.status, 200)
  assert.equal(await headResponse.text(), '')
})

test('rejects unsupported methods and advertises the supported methods', () => {
  const response = mcpServerCardResponse(
    new Request(`https://aarnphm.xyz${MCP_SERVER_CARD_PATH}`, { method: 'POST' }),
    'https://aarnphm.xyz',
  )
  assert.equal(response.status, 405)
  assert.equal(response.headers.get('Allow'), 'GET, HEAD')
  assert.equal(response.headers.get('Content-Type'), 'text/plain; charset=utf-8')
})
