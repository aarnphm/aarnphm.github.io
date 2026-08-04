export const API_CATALOG_PATH = '/.well-known/api-catalog'
export const OPENAPI_PATH = '/openapi.json'
export const API_DOCUMENTATION_PATH = '/api/docs'
export const API_HEALTH_PATH = '/api/health'
export const API_CATALOG_PROFILE = 'https://www.rfc-editor.org/info/rfc9727'
export const API_CATALOG_LINK = `<${API_CATALOG_PATH}>; rel="api-catalog"; type="application/linkset+json"; profile="${API_CATALOG_PROFILE}"`
export const API_CATALOG_CONTENT_TYPE = `application/linkset+json; profile="${API_CATALOG_PROFILE}"`

const PUBLIC_CACHE_CONTROL = 'public, max-age=300, s-maxage=300, stale-while-revalidate=59'

export function isHomepagePathname(pathname: string): boolean {
  return pathname === '/' || pathname === '/index.html'
}

export function apiCatalog(baseUrl: string) {
  return {
    linkset: [
      {
        anchor: `${baseUrl}/mcp`,
        'service-desc': [{ href: `${baseUrl}${OPENAPI_PATH}`, type: 'application/json' }],
        'service-doc': [{ href: `${baseUrl}${API_DOCUMENTATION_PATH}`, type: 'text/html' }],
        status: [{ href: `${baseUrl}${API_HEALTH_PATH}`, type: 'application/json' }],
      },
    ],
  }
}

export function openApiDocument(baseUrl: string) {
  return {
    openapi: '3.1.2',
    info: {
      title: 'aarnphm.xyz MCP API',
      version: '1.0.0',
      description:
        'OAuth-protected Model Context Protocol server for searching and retrieving notes from aarnphm.xyz.',
    },
    servers: [{ url: baseUrl }],
    paths: {
      '/mcp': {
        post: {
          summary: 'Send an MCP JSON-RPC message',
          description:
            'Use Content-Type application/json and accept both application/json and text/event-stream. Initialization creates an MCP session; subsequent requests include its Mcp-Session-Id response header.',
          security: [{ oauth: ['mcp'] }],
          parameters: [
            {
              name: 'Mcp-Session-Id',
              in: 'header',
              required: false,
              schema: { type: 'string' },
              description: 'Omit for initialization and include for subsequent session messages.',
            },
          ],
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  oneOf: [
                    { $ref: '#/components/schemas/JsonRpcMessage' },
                    {
                      type: 'array',
                      minItems: 1,
                      items: { $ref: '#/components/schemas/JsonRpcMessage' },
                    },
                  ],
                },
              },
            },
          },
          responses: {
            '200': {
              description: 'MCP messages delivered as a server-sent event stream.',
              headers: {
                'Mcp-Session-Id': {
                  description: 'Session identifier created during initialization.',
                  schema: { type: 'string' },
                },
              },
              content: { 'text/event-stream': { schema: { type: 'string' } } },
            },
            '202': { description: 'Notification or response accepted without a response body.' },
            '400': { description: 'Invalid JSON-RPC message or missing session identifier.' },
            '401': { description: 'OAuth authorization is required.' },
            '406': { description: 'The required response media types were not accepted.' },
            '413': { description: 'The request body exceeded the server limit.' },
            '415': { description: 'The request body was not application/json.' },
          },
        },
        get: {
          summary: 'Listen for MCP server messages',
          description: 'Open a server-sent event stream for an initialized MCP session.',
          security: [{ oauth: ['mcp'] }],
          parameters: [
            { name: 'Mcp-Session-Id', in: 'header', required: true, schema: { type: 'string' } },
          ],
          responses: {
            '200': {
              description: 'MCP server-sent event stream.',
              content: { 'text/event-stream': { schema: { type: 'string' } } },
            },
            '400': { description: 'The session identifier is missing.' },
            '401': { description: 'OAuth authorization is required.' },
            '404': { description: 'The MCP session does not exist.' },
            '406': { description: 'The client did not accept text/event-stream.' },
          },
        },
        delete: {
          summary: 'Terminate an MCP session',
          security: [{ oauth: ['mcp'] }],
          parameters: [
            { name: 'Mcp-Session-Id', in: 'header', required: true, schema: { type: 'string' } },
          ],
          responses: {
            '204': { description: 'The MCP session was terminated.' },
            '400': { description: 'The session identifier is missing.' },
            '401': { description: 'OAuth authorization is required.' },
            '404': { description: 'The MCP session does not exist.' },
          },
        },
      },
      [API_HEALTH_PATH]: {
        get: {
          summary: 'Check Worker liveness',
          security: [],
          responses: {
            '200': {
              description: 'The Worker is available.',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    required: ['status', 'service'],
                    properties: {
                      status: { type: 'string', const: 'ok' },
                      service: { type: 'string' },
                    },
                  },
                },
              },
            },
          },
        },
      },
    },
    components: {
      securitySchemes: {
        oauth: {
          type: 'oauth2',
          flows: {
            authorizationCode: {
              authorizationUrl: `${baseUrl}/authorize`,
              tokenUrl: `${baseUrl}/token`,
              scopes: { mcp: 'Search and retrieve garden notes through MCP.' },
            },
          },
        },
      },
      schemas: {
        JsonRpcMessage: {
          type: 'object',
          required: ['jsonrpc'],
          properties: {
            jsonrpc: { type: 'string', const: '2.0' },
            id: { oneOf: [{ type: 'string' }, { type: 'integer' }, { type: 'null' }] },
            method: { type: 'string' },
            params: {},
            result: {},
            error: {
              type: 'object',
              required: ['code', 'message'],
              properties: { code: { type: 'integer' }, message: { type: 'string' }, data: {} },
            },
          },
          additionalProperties: true,
        },
      },
    },
  }
}

export function apiDocumentation(baseUrl: string): string {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aarnphm.xyz MCP API</title>
</head>
<body>
<main>
<h1>aarnphm.xyz MCP API</h1>
<p>This OAuth-protected MCP Streamable HTTP server searches and retrieves notes from aarnphm.xyz.</p>
<dl>
<dt>Endpoint</dt><dd><a href="${baseUrl}/mcp"><code>${baseUrl}/mcp</code></a></dd>
<dt>OpenAPI</dt><dd><a href="${baseUrl}${OPENAPI_PATH}"><code>${baseUrl}${OPENAPI_PATH}</code></a></dd>
<dt>OAuth metadata</dt><dd><a href="${baseUrl}/.well-known/oauth-protected-resource/mcp"><code>${baseUrl}/.well-known/oauth-protected-resource/mcp</code></a></dd>
</dl>
<h2>Transport</h2>
<p>Request the <code>mcp</code> OAuth scope. Send JSON-RPC messages with <code>POST /mcp</code>, <code>Content-Type: application/json</code>, and <code>Accept: application/json, text/event-stream</code>. Initialization returns an <code>Mcp-Session-Id</code> header for later POST, GET, and DELETE requests.</p>
<h2>Tools</h2>
<ul>
<li><code>search</code> finds notes by semantic and lexical relevance.</li>
<li><code>retrieve</code> returns the full Markdown for a note slug.</li>
</ul>
</main>
</body>
</html>`
}

export function apiHealth() {
  return { status: 'ok', service: 'aarnphm.xyz MCP API' }
}

function resourceResponse(
  request: Request,
  body: string,
  contentType: string,
  headersInit: Record<string, string>,
  cacheControl = PUBLIC_CACHE_CONTROL,
): Response {
  const headers = new Headers(headersInit)
  headers.set('Access-Control-Allow-Origin', '*')
  headers.set('Cache-Control', cacheControl)
  headers.set('Content-Type', contentType)
  headers.set('X-Content-Type-Options', 'nosniff')

  if (request.method !== 'GET' && request.method !== 'HEAD') {
    headers.set('Allow', 'GET, HEAD')
    return new Response('method not allowed', { status: 405, headers })
  }

  return new Response(request.method === 'HEAD' ? null : body, { headers })
}

export function apiCatalogResponse(
  request: Request,
  baseUrl: string,
  headers: Record<string, string>,
): Response {
  const response = resourceResponse(
    request,
    JSON.stringify(apiCatalog(baseUrl)),
    API_CATALOG_CONTENT_TYPE,
    headers,
  )
  response.headers.set('Link', API_CATALOG_LINK)
  return response
}

export function openApiResponse(
  request: Request,
  baseUrl: string,
  headers: Record<string, string>,
): Response {
  return resourceResponse(
    request,
    JSON.stringify(openApiDocument(baseUrl)),
    'application/json; charset=utf-8',
    headers,
  )
}

export function apiDocumentationResponse(
  request: Request,
  baseUrl: string,
  headers: Record<string, string>,
): Response {
  const response = resourceResponse(
    request,
    apiDocumentation(baseUrl),
    'text/html; charset=utf-8',
    headers,
  )
  response.headers.set(
    'Content-Security-Policy',
    "default-src 'none'; base-uri 'none'; frame-ancestors 'none'",
  )
  return response
}

export function apiHealthResponse(request: Request, headers: Record<string, string>): Response {
  return resourceResponse(
    request,
    JSON.stringify(apiHealth()),
    'application/json; charset=utf-8',
    headers,
    'no-store',
  )
}
