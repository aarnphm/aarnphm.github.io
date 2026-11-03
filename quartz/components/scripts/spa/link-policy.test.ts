import assert from 'node:assert/strict'
import test from 'node:test'
import { resolveStreamInternalHref } from './link-policy'

test('stream root survives the second host rewrite', () => {
  const once = resolveStreamInternalHref('/stream')
  assert.equal(once, 'https://stream.aarnphm.xyz/')
  assert.equal(resolveStreamInternalHref(once ?? ''), once)
  assert.equal(
    resolveStreamInternalHref('/stream/on/2026/10/28'),
    'https://stream.aarnphm.xyz/on/2026/10/28',
  )
  assert.equal(
    resolveStreamInternalHref('https://stream.aarnphm.xyz/on/2026/10/28?entry=stream-entry-2#x'),
    'https://stream.aarnphm.xyz/on/2026/10/28?entry=stream-entry-2#x',
  )
})

test('garden paths on the stream host still leave for the apex', () => {
  assert.equal(resolveStreamInternalHref('/'), 'https://aarnphm.xyz/')
  assert.equal(resolveStreamInternalHref('/colophon'), 'https://aarnphm.xyz/colophon')
})
