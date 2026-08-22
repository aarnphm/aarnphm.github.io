import assert from 'node:assert/strict'
import test from 'node:test'
import {
  acceptsMarkdown,
  documentDiscoveryLink,
  isAgentUserAgent,
  markdownPathname,
  resolveBaseUrl,
  shouldRejectDocumentResponse,
  shouldRewriteMarkdown,
  shouldTreatAsDocument,
} from './request-utils'

test('resolves the public base URL from the development override or request origin', () => {
  const request = new Request('http://127.0.0.1:8080/thoughts')
  assert.equal(resolveBaseUrl({}, request), 'http://127.0.0.1:8080')
  assert.equal(
    resolveBaseUrl({ PUBLIC_BASE_URL: 'https://aarnphm.xyz/' }, request),
    'https://aarnphm.xyz',
  )
})

test('recognizes agent user agents that should receive markdown', () => {
  for (const userAgent of [
    'ChatGPT-User/1.0',
    'Claude-User',
    'OAI-SearchBot/1.0',
    'Codex/1.0',
    'DeepSeekBot/1.0',
    'Google-Extended',
    'Applebot-Extended',
    'ora-agent/1.0',
  ]) {
    assert.equal(
      isAgentUserAgent(
        new Request('https://aarnphm.xyz/triathlon/analytics', {
          headers: { 'User-Agent': userAgent },
        }),
      ),
      true,
    )
  }

  assert.equal(
    isAgentUserAgent(
      new Request('https://aarnphm.xyz/triathlon/analytics', {
        headers: { 'User-Agent': 'Mozilla/5.0' },
      }),
    ),
    false,
  )

  assert.equal(
    isAgentUserAgent(
      new Request('https://aarnphm.xyz/triathlon/analytics', {
        headers: { 'User-Agent': 'PerplexityBot/1.0' },
      }),
    ),
    false,
  )
})

test('rewrites document requests for agents and markdown clients', () => {
  const url = new URL('https://aarnphm.xyz/triathlon/analytics')
  assert.equal(
    shouldRewriteMarkdown(
      new Request(url, { headers: { 'User-Agent': 'Codex/1.0', Accept: 'text/html' } }),
      url,
    ),
    false,
  )
  assert.equal(
    shouldRewriteMarkdown(
      new Request(url, { headers: { 'User-Agent': 'Codex/1.0', Accept: '*/*' } }),
      url,
    ),
    true,
  )
  assert.equal(
    shouldRewriteMarkdown(new Request(url, { headers: { Accept: 'text/markdown' } }), url),
    true,
  )
  assert.equal(
    shouldRewriteMarkdown(
      new Request(url, { headers: { 'User-Agent': 'Mozilla/5.0', Accept: 'text/html' } }),
      url,
    ),
    false,
  )
})

test('rewrites ordinary documents and excludes explicit markdown and data routes', () => {
  const markdownUrl = new URL('https://aarnphm.xyz/triathlon/analytics.md')
  const dataUrl = new URL('https://aarnphm.xyz/triathlon/data')
  const headers = { 'User-Agent': 'Codex/1.0' }

  for (const value of ['https://aarnphm.xyz/', 'https://aarnphm.xyz/thoughts']) {
    const url = new URL(value)
    assert.equal(shouldRewriteMarkdown(new Request(url, { headers }), url), true)
  }
  assert.equal(shouldRewriteMarkdown(new Request(markdownUrl, { headers }), markdownUrl), false)
  assert.equal(shouldRewriteMarkdown(new Request(dataUrl, { headers }), dataUrl), false)
})

test('honors explicit markdown accept values with positive quality', () => {
  assert.equal(
    acceptsMarkdown(
      new Request('https://aarnphm.xyz', { headers: { Accept: 'text/html, text/markdown' } }),
    ),
    true,
  )
  assert.equal(
    acceptsMarkdown(
      new Request('https://aarnphm.xyz', { headers: { Accept: 'text/markdown; q=0' } }),
    ),
    false,
  )
  assert.equal(
    acceptsMarkdown(new Request('https://aarnphm.xyz', { headers: { Accept: 'text/html' } })),
    false,
  )
})

test('honors media quality and returns 406 only when no document representation is accepted', () => {
  const url = new URL('https://aarnphm.xyz/thoughts')
  assert.equal(
    shouldRewriteMarkdown(
      new Request(url, { headers: { Accept: 'text/html;q=0.5, text/markdown;q=1' } }),
      url,
    ),
    true,
  )
  assert.equal(
    shouldRewriteMarkdown(
      new Request(url, { headers: { Accept: 'text/html;q=1, text/markdown;q=0.5' } }),
      url,
    ),
    false,
  )
  assert.equal(
    shouldRejectDocumentResponse(
      new Request(url, { headers: { Accept: 'application/json' } }),
      url,
    ),
    true,
  )
  assert.equal(
    shouldRejectDocumentResponse(
      new Request(url, { headers: { Accept: 'text/markdown;q=0, text/html;q=0' } }),
      url,
    ),
    true,
  )
  assert.equal(
    shouldRejectDocumentResponse(new Request(url, { headers: { Accept: 'text/html' } }), url),
    false,
  )
})

test('maps document paths to sibling markdown assets', () => {
  assert.equal(markdownPathname('/'), '/llms.txt')
  assert.equal(markdownPathname('/triathlon/analytics'), '/triathlon/analytics.md')
  assert.equal(markdownPathname('/triathlon/on/2026/07/'), '/triathlon/on/2026/07.md')
  assert.equal(shouldTreatAsDocument('/triathlon/analytics'), true)
  assert.equal(shouldTreatAsDocument('/triathlon/analytics.html'), true)
  assert.equal(shouldTreatAsDocument('/triathlon/analytics.md'), false)
})

test('publishes markdown alternates and the llms.txt description in document links', () => {
  assert.equal(
    documentDiscoveryLink('/'),
    '</llms.txt>; rel="alternate describedby"; type="text/markdown"',
  )
  assert.equal(
    documentDiscoveryLink('/thoughts'),
    '</thoughts.md>; rel="alternate"; type="text/markdown", </llms.txt>; rel="describedby"; type="text/markdown"',
  )
  assert.equal(
    documentDiscoveryLink('/thoughts.html'),
    '</thoughts.md>; rel="alternate"; type="text/markdown", </llms.txt>; rel="describedby"; type="text/markdown"',
  )
})
