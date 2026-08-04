import assert from 'node:assert/strict'
import test from 'node:test'
import {
  API_CATALOG_CONTENT_TYPE,
  API_CATALOG_LINK,
  API_CATALOG_PATH,
  API_DOCUMENTATION_PATH,
  API_HEALTH_PATH,
  OPENAPI_PATH,
  apiCatalog,
  apiCatalogResponse,
  apiDocumentationResponse,
  apiHealth,
  apiHealthResponse,
  isHomepagePathname,
  openApiDocument,
  openApiResponse,
} from './api-catalog'

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

test('rejects unsupported catalog methods', () => {
  const response = apiCatalogResponse(
    new Request(`${BASE_URL}${API_CATALOG_PATH}`, { method: 'POST' }),
    BASE_URL,
    API_HEADERS,
  )
  assert.equal(response.status, 405)
  assert.equal(response.headers.get('Allow'), 'GET, HEAD')
  assert.equal(response.headers.get('Link'), API_CATALOG_LINK)
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

  const healthResponse = apiHealthResponse(
    new Request(`${BASE_URL}${API_HEALTH_PATH}`),
    API_HEADERS,
  )
  assert.equal(healthResponse.status, 200)
  assert.equal(healthResponse.headers.get('Cache-Control'), 'no-store')
  assert.deepEqual(await healthResponse.json(), apiHealth())
})
