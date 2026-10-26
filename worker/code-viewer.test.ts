import assert from 'node:assert/strict'
import test, { describe } from 'node:test'
import { codeViewerHtml, codeViewerResponse } from './code-viewer'

const SAMPLE_PYTHON = [
  'import cutlass',
  'import cutlass.cute as cute',
  '',
  '@cute.jit',
  'def paged_attention(q: cute.Tensor, k_cache: cute.Tensor) -> cute.Tensor:',
  '    # compute <qk> & scale',
  '    return q',
  '',
].join('\n')

function mockOriginResponse(body: string, status = 200): Response {
  return new Response(body, { status, headers: { 'Content-Type': 'application/octet-stream' } })
}

describe('codeViewerHtml', () => {
  test('renders @pierre/highlights dual-theme tokens, line numbers, breadcrumb, and raw action', () => {
    const html = codeViewerHtml('/lectures/420/examples/cute_paged_attention.py', SAMPLE_PYTHON)
    assert.match(html, /^<!doctype html>/)
    assert.match(html, /<title>cute_paged_attention\.py · lectures\/420\/examples<\/title>/)
    assert.match(html, /data-language="python"/)
    assert.match(html, /data-theme="github-light github-dark"/)
    assert.match(html, /style="--line-digits:1"/)
    assert.match(html, /--syntax-light:#/)
    assert.match(html, /--syntax-dark:#/)
    assert.match(html, /data-line/)
    assert.match(html, /&lt;qk&gt; &amp; scale/)
    assert.match(html, /href="\/lectures\/420\/examples\/cute_paged_attention\.py\?raw=1"/)
    assert.match(html, /href="\/lectures\/420\/examples\/"/)
  })
})

describe('codeViewerResponse negotiation', () => {
  const path = '/lectures/420/examples/cute_paged_attention.py'

  test('serves syntax-highlighted HTML viewer to browser navigation requests', async () => {
    const url = new URL(`http://localhost:8080${path}`)
    const req = new Request(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      },
    })
    const res = await codeViewerResponse(req, url, mockOriginResponse(SAMPLE_PYTHON))
    assert.equal(res.status, 200)
    assert.equal(res.headers.get('Content-Type'), 'text/html; charset=utf-8')
    assert.match(res.headers.get('Vary') ?? '', /Accept/)
    assert.match(res.headers.get('Vary') ?? '', /User-Agent/)
    assert.equal(res.headers.get('Link'), `<${path}?raw=1>; rel="alternate"; type="text/plain"`)
    const body = await res.text()
    assert.match(body, /data-theme="github-light github-dark"/)
    assert.match(body, /paged_attention/)
  })

  test('serves raw text/plain to agent user-agents even when Accept includes text/html', async () => {
    const url = new URL(`http://localhost:8080${path}`)
    for (const ua of ['Claude-User', 'Codex/1.0', 'ChatGPT-User/1.0', 'OpenAI']) {
      const req = new Request(url, {
        headers: {
          'User-Agent': ua,
          Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        },
      })
      const res = await codeViewerResponse(req, url, mockOriginResponse(SAMPLE_PYTHON))
      assert.equal(res.status, 200, ua)
      assert.equal(res.headers.get('Content-Type'), 'text/plain; charset=utf-8', ua)
      assert.equal(await res.text(), SAMPLE_PYTHON, ua)
    }
  })

  test('serves raw text/plain for ?raw=1, Accept: */*, and Accept: text/plain', async () => {
    const browserUa = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'

    const rawUrl = new URL(`http://localhost:8080${path}?raw=1`)
    const rawRes = await codeViewerResponse(
      new Request(rawUrl, {
        headers: {
          'User-Agent': browserUa,
          Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        },
      }),
      rawUrl,
      mockOriginResponse(SAMPLE_PYTHON),
    )
    assert.equal(rawRes.headers.get('Content-Type'), 'text/plain; charset=utf-8')
    assert.equal(await rawRes.text(), SAMPLE_PYTHON)

    const url = new URL(`http://localhost:8080${path}`)
    for (const accept of ['*/*', 'text/plain, */*;q=0.8', 'text/markdown']) {
      const res = await codeViewerResponse(
        new Request(url, { headers: { 'User-Agent': browserUa, Accept: accept } }),
        url,
        mockOriginResponse(SAMPLE_PYTHON),
      )
      assert.equal(res.headers.get('Content-Type'), 'text/plain; charset=utf-8', accept)
      assert.equal(await res.text(), SAMPLE_PYTHON, accept)
    }
  })
})
