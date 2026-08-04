import assert from 'node:assert/strict'
import test from 'node:test'
import {
  acceptsMarkdown,
  isAgentUserAgent,
  markdownPathname,
  resolveBaseUrl,
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
    'PerplexityBot/1.0',
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
})

test('rewrites document requests for agents and markdown clients', () => {
  const url = new URL('https://aarnphm.xyz/triathlon/analytics')
  assert.equal(
    shouldRewriteMarkdown(
      new Request(url, { headers: { 'User-Agent': 'Codex/1.0', Accept: 'text/html' } }),
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

test('maps document paths to sibling markdown assets', () => {
  assert.equal(markdownPathname('/'), '/llms.txt')
  assert.equal(markdownPathname('/triathlon/analytics'), '/triathlon/analytics.md')
  assert.equal(markdownPathname('/triathlon/on/2026/07/'), '/triathlon/on/2026/07.md')
  assert.equal(shouldTreatAsDocument('/triathlon/analytics'), true)
  assert.equal(shouldTreatAsDocument('/triathlon/analytics.html'), true)
  assert.equal(shouldTreatAsDocument('/triathlon/analytics.md'), false)
})
