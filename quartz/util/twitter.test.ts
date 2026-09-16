import assert from 'node:assert/strict'
import test from 'node:test'
import { parseTwitterPostUrl } from './twitter'

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
  }
})
