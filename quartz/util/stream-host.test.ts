import assert from 'node:assert/strict'
import test from 'node:test'
import {
  isStreamHostname,
  isStreamRoutePathname,
  streamAssetPathname,
  streamDocumentRedirectUrl,
  streamHostPathname,
  streamHostUrl,
} from './stream-host'

test('detects the canonical stream hostname', () => {
  assert.equal(isStreamHostname('stream.aarnphm.xyz'), true)
  assert.equal(isStreamHostname('aarnphm.xyz'), false)
})

test('canonicalizes stream host public paths', () => {
  assert.equal(streamHostPathname('/stream'), '/')
  assert.equal(streamHostPathname('/stream/'), '/')
  assert.equal(streamHostPathname('/stream/on/2026/06/27'), '/on/2026/06/27')
  assert.equal(streamHostPathname('/on/2026/06/27'), '/on/2026/06/27')
})

test('maps stream host document paths to emitted stream assets', () => {
  assert.equal(streamAssetPathname('/', true), '/stream')
  assert.equal(streamAssetPathname('/on/2026/06/27', true), '/stream/on/2026/06/27')
  assert.equal(streamAssetPathname('/stream/on/2026/06/27', true), '/stream/on/2026/06/27')
  assert.equal(streamAssetPathname('/index.xml', false), '/stream/index.xml')
  assert.equal(streamAssetPathname('/stream/index.xml', false), '/stream/index.xml')
  assert.equal(streamAssetPathname('/index-3e33b904.css', false), '/index-3e33b904.css')
  assert.equal(streamAssetPathname('/static/site.js', false), '/static/site.js')
  assert.equal(
    streamAssetPathname('/stream/static/resource-after-3693bec0.js', false),
    '/static/resource-after-3693bec0.js',
  )
})

test('detects stream route pathnames', () => {
  assert.equal(isStreamRoutePathname('/stream'), true)
  assert.equal(isStreamRoutePathname('/stream/on/2026/06/27'), true)
  assert.equal(isStreamRoutePathname('/on/2026/06/27'), true)
  assert.equal(isStreamRoutePathname('/triathlon'), false)
})

test('redirects non-stream documents to the canonical site', () => {
  assert.equal(
    streamDocumentRedirectUrl(
      'https://stream.aarnphm.xyz',
      new URL('https://stream.aarnphm.xyz/triathlon/on/2026/08/03?activity=ride#power'),
    ),
    'https://aarnphm.xyz/triathlon/on/2026/08/03?activity=ride#power',
  )
  assert.equal(
    streamDocumentRedirectUrl(
      'https://preview.aarnphm.xyz',
      'https://stream.aarnphm.xyz/thoughts?view=reader',
    ),
    'https://preview.aarnphm.xyz/thoughts?view=reader',
  )
})

test('keeps stream documents on the stream host', () => {
  assert.equal(
    streamDocumentRedirectUrl('https://stream.aarnphm.xyz', 'https://stream.aarnphm.xyz/'),
    null,
  )
  assert.equal(
    streamDocumentRedirectUrl(
      'https://stream.aarnphm.xyz',
      'https://stream.aarnphm.xyz/on/2026/08/03',
    ),
    null,
  )
})

test('builds canonical stream host URLs', () => {
  assert.equal(streamHostUrl('/stream/on'), 'https://stream.aarnphm.xyz/on')
  assert.equal(
    streamHostUrl('/stream/on/2026/06/27?entry=training#lap'),
    'https://stream.aarnphm.xyz/on/2026/06/27?entry=training#lap',
  )
  assert.equal(streamHostUrl('/stream/index.xml'), 'https://stream.aarnphm.xyz/index.xml')
})
