import assert from 'node:assert/strict'
import test from 'node:test'
import {
  AGENT_SKILLS_CATALOG_PATH,
  AI_CATALOG_PATH,
  API_CATALOG_CONTENT_TYPE,
  API_CATALOG_LINK,
  API_CATALOG_PATH,
  API_DOCUMENTATION_PATH,
  API_HEALTH_PATH,
  OPENAPI_PATH,
  aiCatalog,
  aiCatalogResponse,
  apiCatalog,
  apiCatalogResponse,
  apiDocumentationResponse,
  apiHealth,
  apiHealthResponse,
  isHomepagePathname,
  openApiDocument,
  openApiResponse,
} from './api-catalog'
import { MCP_SERVER_CARD_PATH } from './mcp-server-card'

const BASE_URL = 'https://aarnphm.xyz'
const API_HEADERS = { 'Access-Control-Allow-Methods': 'GET, HEAD' }

test('advertises the profiled RFC 9727 API catalog from homepage paths', () => {
  assert.equal(
    API_CATALOG_LINK,
    '</.well-known/api-catalog>; rel="api-catalog"; type="application/linkset+json"; profile="https://www.rfc-editor.org/info/rfc9727"',
  )
  assert.equal(
    API_CATALOG_CONTENT_TYPE,
    'application/linkset+json; profile="https://www.rfc-editor.org/info/rfc9727"',
  )
  assert.equal(isHomepagePathname('/'), true)
  assert.equal(isHomepagePathname('/index.html'), true)
  assert.equal(isHomepagePathname('/thoughts'), false)
})

test('builds an RFC 9727 linkset with description, documentation, and health relations', () => {
  assert.deepEqual(apiCatalog(BASE_URL), {
    linkset: [
      {
        anchor: `${BASE_URL}/mcp`,
        'service-desc': [{ href: `${BASE_URL}${OPENAPI_PATH}`, type: 'application/json' }],
        'service-doc': [{ href: `${BASE_URL}${API_DOCUMENTATION_PATH}`, type: 'text/html' }],
        status: [{ href: `${BASE_URL}${API_HEALTH_PATH}`, type: 'application/json' }],
      },
    ],
  })
})

test('builds an ARD capability manifest for deployed MCP, OpenAPI, and Agent Skills resources', () => {
  const manifest = aiCatalog(BASE_URL)
  assert.equal(manifest.specVersion, '1.0')
  assert.deepEqual(manifest.host, { displayName: 'aarnphm.xyz', identifier: 'https://aarnphm.xyz' })
  assert.equal(manifest.entries.length, 3)
  assert.deepEqual(
    manifest.entries.map(entry => ({
      identifier: entry.identifier,
      type: entry.type,
      url: entry.url,
    })),
    [
      {
        identifier: 'urn:air:aarnphm.xyz:server:garden-mcp',
        type: 'application/mcp-server-card+json',
        url: `${BASE_URL}${MCP_SERVER_CARD_PATH}`,
      },
      {
        identifier: 'urn:air:aarnphm.xyz:schema:garden-openapi',
        type: 'application/json',
        url: `${BASE_URL}${OPENAPI_PATH}`,
      },
      {
        identifier: 'urn:air:aarnphm.xyz:catalog:agent-skills',
        type: 'application/json',
        url: `${BASE_URL}${AGENT_SKILLS_CATALOG_PATH}`,
      },
    ],
  )
  for (const entry of manifest.entries) {
    assert.match(entry.identifier, /^urn:air:aarnphm\.xyz:[a-z0-9._-]+:[a-z0-9._-]+$/)
    assert.ok(entry.representativeQueries.length >= 2)
    assert.ok(entry.representativeQueries.length <= 5)
    assert.equal('data' in entry, false)
  }
})

test('serves the ARD manifest for GET and HEAD with JSON and wildcard CORS', async () => {
  const getResponse = aiCatalogResponse(
    new Request(`${BASE_URL}${AI_CATALOG_PATH}`),
    BASE_URL,
    API_HEADERS,
  )
  assert.equal(getResponse.status, 200)
  assert.equal(getResponse.headers.get('Content-Type'), 'application/json; charset=utf-8')
  assert.equal(getResponse.headers.get('Access-Control-Allow-Origin'), '*')
  assert.deepEqual(await getResponse.json(), aiCatalog(BASE_URL))

  const headResponse = aiCatalogResponse(
    new Request(`${BASE_URL}${AI_CATALOG_PATH}`, { method: 'HEAD' }),
    BASE_URL,
    API_HEADERS,
  )
  assert.equal(headResponse.status, 200)
  assert.equal(await headResponse.text(), '')
})

test('rejects unsupported ARD manifest methods with problem details', async () => {
  const response = aiCatalogResponse(
    new Request(`${BASE_URL}${AI_CATALOG_PATH}`, { method: 'POST' }),
    BASE_URL,
    API_HEADERS,
  )
  assert.equal(response.status, 405)
  assert.equal(response.headers.get('Allow'), 'GET, HEAD')
  assert.equal(response.headers.get('Content-Type'), 'application/problem+json; charset=utf-8')
  assert.equal((await response.json()).code, 'method_not_allowed')
})

test('serves the catalog for GET and HEAD with RFC discovery headers', async () => {
  const getResponse = apiCatalogResponse(
    new Request(`${BASE_URL}${API_CATALOG_PATH}`),
    BASE_URL,
    API_HEADERS,
  )
  assert.equal(getResponse.status, 200)
  assert.equal(getResponse.headers.get('Content-Type'), API_CATALOG_CONTENT_TYPE)
  assert.equal(getResponse.headers.get('Link'), API_CATALOG_LINK)
  assert.equal(getResponse.headers.get('Access-Control-Allow-Origin'), '*')
  assert.deepEqual(await getResponse.json(), apiCatalog(BASE_URL))

  const headResponse = apiCatalogResponse(
    new Request(`${BASE_URL}${API_CATALOG_PATH}`, { method: 'HEAD' }),
    BASE_URL,
    API_HEADERS,
  )
  assert.equal(headResponse.status, 200)
  assert.equal(headResponse.headers.get('Link'), API_CATALOG_LINK)
  assert.equal(await headResponse.text(), '')
})

test('rejects unsupported catalog methods with problem details', async () => {
  const response = apiCatalogResponse(
    new Request(`${BASE_URL}${API_CATALOG_PATH}`, { method: 'POST' }),
    BASE_URL,
    API_HEADERS,
  )
  assert.equal(response.status, 405)
  assert.equal(response.headers.get('Allow'), 'GET, HEAD')
  assert.equal(response.headers.get('Link'), API_CATALOG_LINK)
  assert.equal(response.headers.get('Content-Type'), 'application/problem+json; charset=utf-8')
  assert.equal((await response.json()).resolution, 'Retry the request with GET or HEAD.')
})

test('publishes a matching OpenAPI description for the MCP and health endpoints', async () => {
  const document = openApiDocument(BASE_URL)
  assert.equal(document.openapi, '3.1.2')
  assert.equal(document.servers[0]?.url, BASE_URL)
  assert.ok(document.paths['/mcp'].post)
  assert.ok(document.paths['/mcp'].get)
  assert.ok(document.paths['/mcp'].delete)
  assert.ok(document.paths[API_HEALTH_PATH].get)
  assert.equal(
    document.components.securitySchemes.oauth.flows.authorizationCode.authorizationUrl,
    `${BASE_URL}/authorize`,
  )
  assert.deepEqual(document.paths['/mcp'].post.security, [{ oauth: ['mcp'] }])
  assert.equal(
    document.components.securitySchemes.oauth.flows.authorizationCode.scopes.mcp,
    'Search and retrieve garden notes through MCP.',
  )

  const operations = [
    document.paths['/mcp'].post,
    document.paths['/mcp'].get,
    document.paths['/mcp'].delete,
    document.paths[API_HEALTH_PATH].get,
  ]
  assert.equal(new Set(operations.map(operation => operation.operationId)).size, operations.length)
  for (const operation of operations) {
    assert.ok(operation.operationId.length > 0)
    assert.ok(operation.description.length > 0)
    for (const parameter of 'parameters' in operation ? (operation.parameters ?? []) : []) {
      assert.ok(parameter.description.length > 0)
      assert.ok(parameter.schema.type.length > 0)
    }
  }
  assert.deepEqual(
    document.paths['/mcp'].post.responses['429'].content['application/problem+json'].schema,
    { $ref: '#/components/schemas/ProblemDetails' },
  )
  assert.ok(document.components.schemas.ProblemDetails.required.includes('resolution'))

  const response = openApiResponse(new Request(`${BASE_URL}${OPENAPI_PATH}`), BASE_URL, API_HEADERS)
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('Content-Type'), 'application/json; charset=utf-8')
  assert.deepEqual(await response.json(), document)
})

test('publishes human documentation and an uncached liveness response', async () => {
  const documentationResponse = apiDocumentationResponse(
    new Request(`${BASE_URL}${API_DOCUMENTATION_PATH}`),
    BASE_URL,
    API_HEADERS,
  )
  assert.equal(documentationResponse.status, 200)
  assert.equal(documentationResponse.headers.get('Content-Type'), 'text/html; charset=utf-8')
  const documentation = await documentationResponse.text()
  assert.match(documentation, /aarnphm\.xyz MCP API/)
  assert.match(documentation, /Mcp-Session-Id/)
  assert.match(documentation, /<code>search<\/code>/)
  assert.match(documentation, /<code>retrieve<\/code>/)
  assert.match(documentation, /free to use/)
  assert.match(documentation, /Dynamic client registration/)
  assert.match(documentation, /application\/problem\+json/)
  assert.match(documentation, /RateLimit-Policy/)

  const healthResponse = apiHealthResponse(
    new Request(`${BASE_URL}${API_HEALTH_PATH}`),
    API_HEADERS,
  )
  assert.equal(healthResponse.status, 200)
  assert.equal(healthResponse.headers.get('Cache-Control'), 'no-store')
  assert.deepEqual(await healthResponse.json(), apiHealth())
})
