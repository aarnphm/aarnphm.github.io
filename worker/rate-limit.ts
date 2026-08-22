import { problemResponse } from './problem-details'

export const MCP_RATE_LIMIT_POLICY = '"mcp";q=120;w=60'
export const MCP_RATE_LIMIT_EXHAUSTED = '"mcp";r=0;t=60'

export function mcpRateLimitKey(request: Request): string {
  return request.headers.get('CF-Connecting-IP') ?? new URL(request.url).hostname
}

export function withMcpRateLimitPolicy(response: Response): Response {
  const headers = new Headers(response.headers)
  headers.set('RateLimit-Policy', MCP_RATE_LIMIT_POLICY)
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  })
}

export function mcpRateLimitExceededResponse(request: Request): Response {
  return problemResponse(request, {
    status: 429,
    title: 'MCP rate limit exceeded',
    detail: 'This client has exhausted the MCP request budget for the current 60 second window.',
    code: 'mcp_rate_limit_exceeded',
    resolution: 'Wait 60 seconds before retrying the request.',
    headers: {
      'RateLimit-Policy': MCP_RATE_LIMIT_POLICY,
      RateLimit: MCP_RATE_LIMIT_EXHAUSTED,
      'Retry-After': '60',
    },
  })
}
