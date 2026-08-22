export const MCP_SERVER_CARD_PATH = '/.well-known/mcp/server-card.json'
export const MCP_SERVER_CARD_CONTENT_TYPE = 'application/mcp-server-card+json; charset=utf-8'
export const MCP_SERVER_INFO = { name: 'aarnphm.xyz', version: '1.0.0' }

export function mcpServerCard(baseUrl: string) {
  return {
    serverInfo: MCP_SERVER_INFO,
    transport: { type: 'streamable-http', endpoint: `${baseUrl}/mcp` },
    capabilities: { tools: true },
  }
}

export function mcpServerCardResponse(request: Request, baseUrl: string): Response {
  const headers = new Headers({
    'Access-Control-Allow-Origin': '*',
    'Cache-Control': 'public, max-age=300, s-maxage=300, stale-while-revalidate=59',
    'X-Content-Type-Options': 'nosniff',
  })

  if (request.method !== 'GET' && request.method !== 'HEAD') {
    headers.set('Allow', 'GET, HEAD')
    headers.set('Content-Type', 'text/plain; charset=utf-8')
    return new Response('method not allowed', { status: 405, headers })
  }

  headers.set('Content-Type', MCP_SERVER_CARD_CONTENT_TYPE)
  const body = request.method === 'HEAD' ? null : JSON.stringify(mcpServerCard(baseUrl))
  return new Response(body, { headers })
}
