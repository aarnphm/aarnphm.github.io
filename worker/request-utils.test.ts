import assert from 'node:assert/strict'
import test from 'node:test'
import {
  isAgentUserAgent,
  markdownPathname,
  shouldRewriteMarkdown,
  shouldTreatAsDocument,
} from './request-utils'

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

test('rewrites triathlon document requests for agents and markdown clients', () => {
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

test('keeps explicit markdown and triathlon data routes out of document rewriting', () => {
  const markdownUrl = new URL('https://aarnphm.xyz/triathlon/analytics.md')
  const dataUrl = new URL('https://aarnphm.xyz/triathlon/data')
  const headers = { 'User-Agent': 'Codex/1.0' }

  assert.equal(shouldRewriteMarkdown(new Request(markdownUrl, { headers }), markdownUrl), false)
  assert.equal(shouldRewriteMarkdown(new Request(dataUrl, { headers }), dataUrl), false)
})

test('maps document paths to sibling markdown assets', () => {
  assert.equal(markdownPathname('/'), '/index.md')
  assert.equal(markdownPathname('/triathlon/analytics'), '/triathlon/analytics.md')
  assert.equal(markdownPathname('/triathlon/on/2026/07/'), '/triathlon/on/2026/07.md')
  assert.equal(shouldTreatAsDocument('/triathlon/analytics'), true)
  assert.equal(shouldTreatAsDocument('/triathlon/analytics.html'), true)
  assert.equal(shouldTreatAsDocument('/triathlon/analytics.md'), false)
})
