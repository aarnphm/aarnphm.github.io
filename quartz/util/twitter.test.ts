import assert from 'node:assert/strict'
import test from 'node:test'
import { parseTwitterPostUrl, readTwitterEmbed, twitterOEmbedUrl } from './twitter'

test('X and Twitter post URLs share a canonical post target', () => {
  for (const href of [
    'https://x.com/karpathy/status/2021694437152157847',
    'https://twitter.com/karpathy/status/2021694437152157847?s=20',
    'https://mobile.twitter.com/karpathy/status/2021694437152157847/photo/1',
    'https://www.x.com/i/web/status/2021694437152157847#replies',
    'https://x.com/i/status/2021694437152157847',
  ]) {
    assert.equal(parseTwitterPostUrl(href), 'https://twitter.com/i/status/2021694437152157847')
  }
  assert.equal(
    parseTwitterPostUrl('https://twitter.com/Interior/status/463440424141459456'),
    'https://twitter.com/i/status/463440424141459456',
  )
})

test('profiles, X articles, malformed posts, and unrelated hosts stay out of the post parser', () => {
  for (const href of [
    'https://x.com/karpathy',
    'https://x.com/i/article/2021694437152157847',
    'https://x.com/karpathy/status/2021694437152157847suffix',
    'https://example.com/x.com/karpathy/status/2021694437152157847',
    'https://x.com.example.com/karpathy/status/2021694437152157847',
    'https://user@x.com/karpathy/status/2021694437152157847',
    'ftp://x.com/karpathy/status/2021694437152157847',
  ]) {
    assert.equal(parseTwitterPostUrl(href), null)
    assert.equal(twitterOEmbedUrl(href, 'en'), null)
  }
})

test('oEmbed requests encode the URL, omit scripts, and opt out of personalization', () => {
  const url = twitterOEmbedUrl('https://x.com/karpathy/status/2021694437152157847?s=20', 'fr')
  assert.ok(url)
  assert.equal(url.origin, 'https://publish.twitter.com')
  assert.equal(url.pathname, '/oembed')
  assert.equal(url.searchParams.get('url'), 'https://twitter.com/i/status/2021694437152157847')
  assert.equal(url.searchParams.get('omit_script'), 'true')
  assert.equal(url.searchParams.get('dnt'), 'true')
  assert.equal(url.searchParams.get('lang'), 'fr')
})

test('both provider names retain encoded post text while malformed responses are rejected', () => {
  const html = '<blockquote class="twitter-tweet"><p>x &lt; y &amp; z</p></blockquote>'
  for (const provider_name of ['Twitter', 'X']) {
    assert.equal(readTwitterEmbed({ type: 'rich', provider_name, html }), html)
  }
  for (const value of [
    null,
    { html },
    { type: 'rich', provider_name: 'Unknown', html },
    { type: 'rich', provider_name: 'X', html: ' ' },
    { type: 'rich', provider_name: 'X', html: 'a'.repeat(256 * 1024 + 1) },
  ]) {
    assert.equal(readTwitterEmbed(value), null)
  }
})
