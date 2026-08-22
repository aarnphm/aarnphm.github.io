import { MCP_SERVER_CARD_PATH } from './mcp-server-card'
import { problemResponse } from './problem-details'
import { MCP_RATE_LIMIT_EXHAUSTED, MCP_RATE_LIMIT_POLICY } from './rate-limit'

export const API_CATALOG_PATH = '/.well-known/api-catalog'
export const AI_CATALOG_PATH = '/.well-known/ai-catalog.json'
export const AGENT_SKILLS_CATALOG_PATH = '/.well-known/agent-skills/index.json'
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

export function aiCatalog(baseUrl: string) {
  return {
    specVersion: '1.0',
    host: { displayName: 'aarnphm.xyz', identifier: 'https://aarnphm.xyz' },
    entries: [
      {
        identifier: 'urn:air:aarnphm.xyz:server:garden-mcp',
        displayName: 'aarnphm.xyz Garden MCP Server',
        type: 'application/mcp-server-card+json',
        url: `${baseUrl}${MCP_SERVER_CARD_PATH}`,
        description:
          "OAuth-protected MCP server for searching and retrieving public notes from aarnphm's garden.",
        representativeQueries: [
          "search aarnphm's garden for notes about mechanistic interpretability",
          'retrieve the full text of a public garden note',
          "find aarnphm's writing about compiler construction",
        ],
      },
      {
        identifier: 'urn:air:aarnphm.xyz:schema:garden-openapi',
        displayName: 'aarnphm.xyz MCP OpenAPI Description',
        type: 'application/json',
        url: `${baseUrl}${OPENAPI_PATH}`,
        description:
          'OpenAPI description of the garden MCP Streamable HTTP, OAuth, and health endpoints.',
        representativeQueries: [
          "show me how to call aarnphm.xyz's MCP endpoint",
          'what OAuth flow and scope does the garden MCP server require',
          'show the request and response schemas for the garden MCP API',
        ],
      },
      {
        identifier: 'urn:air:aarnphm.xyz:catalog:agent-skills',
        displayName: 'aarnphm.xyz Agent Skills Catalog',
        type: 'application/json',
        url: `${baseUrl}${AGENT_SKILLS_CATALOG_PATH}`,
        description:
          'Discovery index for Garden skills covering content, flashcards, compiler construction, Quartz plugins, and interactive diagrams.',
        representativeQueries: [
          'create flashcards from a garden note',
          'add descriptions to garden content files',
          'build an interactive diagram for a technical note',
        ],
      },
    ],
  }
}

export function openApiDocument(baseUrl: string) {
  return {
    openapi: '3.1.2',
    jsonSchemaDialect: 'https://json-schema.org/draft/2020-12/schema',
    info: {
      title: 'aarnphm.xyz MCP API',
      version: '1.0.0',
      description:
        'OAuth-protected Model Context Protocol server for searching and retrieving notes from aarnphm.xyz.',
    },
    externalDocs: {
      description: 'Human-readable onboarding, authentication, tools, errors, and rate limits.',
      url: `${baseUrl}${API_DOCUMENTATION_PATH}`,
    },
    servers: [{ url: baseUrl }],
    paths: {
      '/mcp': {
        post: {
          operationId: 'sendMcpMessage',
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
            '429': {
              description: 'The client exhausted the MCP request budget.',
              headers: {
                'RateLimit-Policy': {
                  description: 'Request quota and window policy.',
                  schema: { type: 'string', example: MCP_RATE_LIMIT_POLICY },
                },
                RateLimit: {
                  description: 'Current exhausted budget and reset time.',
                  schema: { type: 'string', example: MCP_RATE_LIMIT_EXHAUSTED },
                },
                'Retry-After': {
                  description: 'Seconds until the client should retry.',
                  schema: { type: 'integer', example: 60 },
                },
              },
              content: {
                'application/problem+json': {
                  schema: { $ref: '#/components/schemas/ProblemDetails' },
                },
              },
            },
          },
        },
        get: {
          operationId: 'listenForMcpMessages',
          summary: 'Listen for MCP server messages',
          description: 'Open a server-sent event stream for an initialized MCP session.',
          security: [{ oauth: ['mcp'] }],
          parameters: [
            {
              name: 'Mcp-Session-Id',
              in: 'header',
              required: true,
              description: 'Session identifier returned by MCP initialization.',
              schema: { type: 'string' },
            },
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
            '429': {
              description: 'The client exhausted the MCP request budget.',
              content: {
                'application/problem+json': {
                  schema: { $ref: '#/components/schemas/ProblemDetails' },
                },
              },
            },
          },
        },
        delete: {
          operationId: 'terminateMcpSession',
          summary: 'Terminate an MCP session',
          description: 'Close an initialized MCP session and release its server resources.',
          security: [{ oauth: ['mcp'] }],
          parameters: [
            {
              name: 'Mcp-Session-Id',
              in: 'header',
              required: true,
              description: 'Session identifier returned by MCP initialization.',
              schema: { type: 'string' },
            },
          ],
          responses: {
            '204': { description: 'The MCP session was terminated.' },
            '400': { description: 'The session identifier is missing.' },
            '401': { description: 'OAuth authorization is required.' },
            '404': { description: 'The MCP session does not exist.' },
            '429': {
              description: 'The client exhausted the MCP request budget.',
              content: {
                'application/problem+json': {
                  schema: { $ref: '#/components/schemas/ProblemDetails' },
                },
              },
            },
          },
        },
      },
      [API_HEALTH_PATH]: {
        get: {
          operationId: 'getApiHealth',
          summary: 'Check Worker liveness',
          description: 'Return a small uncached JSON response when the public API Worker is live.',
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
        ProblemDetails: {
          type: 'object',
          required: ['type', 'title', 'status', 'detail', 'instance', 'code', 'resolution'],
          properties: {
            type: { type: 'string', format: 'uri-reference' },
            title: { type: 'string' },
            status: { type: 'integer', minimum: 400, maximum: 599 },
            detail: { type: 'string' },
            instance: { type: 'string', format: 'uri-reference' },
            code: { type: 'string' },
            resolution: { type: 'string' },
          },
          additionalProperties: true,
        },
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
<dt>Dynamic client registration</dt><dd><a href="${baseUrl}/register"><code>${baseUrl}/register</code></a></dd>
<dt>MCP server card</dt><dd><a href="${baseUrl}${MCP_SERVER_CARD_PATH}"><code>${baseUrl}${MCP_SERVER_CARD_PATH}</code></a></dd>
<dt>API catalog</dt><dd><a href="${baseUrl}${API_CATALOG_PATH}"><code>${baseUrl}${API_CATALOG_PATH}</code></a></dd>
<dt>Agent guide</dt><dd><a href="${baseUrl}/llms.txt"><code>${baseUrl}/llms.txt</code></a></dd>
</dl>
<h2>Access and onboarding</h2>
<p>The public server is free to use for read-only search and retrieval. Compatible MCP clients can register themselves and complete OAuth without an API key or sales contact. The live service is suitable for exploratory requests, so there is no separate sandbox.</p>
<h2>Transport</h2>
<p>Request the <code>mcp</code> OAuth scope. Send JSON-RPC messages with <code>POST /mcp</code>, <code>Content-Type: application/json</code>, and <code>Accept: application/json, text/event-stream</code>. Initialization returns an <code>Mcp-Session-Id</code> header for later POST, GET, and DELETE requests.</p>
<h2>Tools</h2>
<ul>
<li><code>search</code> finds notes by semantic and lexical relevance.</li>
<li><code>retrieve</code> returns the full Markdown for a note slug.</li>
</ul>
<h2>Errors</h2>
<p>MCP protocol failures use JSON-RPC error objects. HTTP discovery resources use <code>application/problem+json</code> with stable <code>code</code> and <code>resolution</code> fields. OAuth failures retain their standard OAuth error representation.</p>
<h2>Rate limits</h2>
<p>Each connecting IP receives 120 MCP requests per 60 seconds. Responses include <code>RateLimit-Policy: ${MCP_RATE_LIMIT_POLICY}</code>. A 429 response also includes <code>RateLimit: ${MCP_RATE_LIMIT_EXHAUSTED}</code> and <code>Retry-After: 60</code>.</p>
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
    return problemResponse(request, {
      status: 405,
      title: 'Method not allowed',
      detail: 'This resource supports GET and HEAD requests.',
      code: 'method_not_allowed',
      resolution: 'Retry the request with GET or HEAD.',
      headers,
    })
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

export function aiCatalogResponse(
  request: Request,
  baseUrl: string,
  headers: Record<string, string>,
): Response {
  return resourceResponse(
    request,
    JSON.stringify(aiCatalog(baseUrl)),
    'application/json; charset=utf-8',
    headers,
  )
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
