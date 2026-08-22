import assert from 'node:assert/strict'
import test from 'node:test'
import { problemResponse } from './problem-details'

test('returns RFC 9457 problem details with an agent resolution hint', async () => {
  const request = new Request('https://aarnphm.xyz/api/missing')
  const response = problemResponse(request, {
    status: 404,
    title: 'API resource not found',
    detail: 'No API resource is published at this path.',
    code: 'api_resource_not_found',
    resolution: 'Read /openapi.json for the published API paths.',
  })

  assert.equal(response.status, 404)
  assert.equal(response.headers.get('Content-Type'), 'application/problem+json; charset=utf-8')
  assert.equal(response.headers.get('Cache-Control'), 'no-store')
  assert.deepEqual(await response.json(), {
    type: 'about:blank',
    title: 'API resource not found',
    status: 404,
    detail: 'No API resource is published at this path.',
    instance: '/api/missing',
    code: 'api_resource_not_found',
    resolution: 'Read /openapi.json for the published API paths.',
  })
})

test('omits the body for HEAD while preserving problem headers', async () => {
  const response = problemResponse(
    new Request('https://aarnphm.xyz/api/missing', { method: 'HEAD' }),
    {
      status: 404,
      title: 'API resource not found',
      detail: 'No API resource is published at this path.',
      code: 'api_resource_not_found',
      resolution: 'Read /openapi.json for the published API paths.',
    },
  )

  assert.equal(response.status, 404)
  assert.equal(await response.text(), '')
})
