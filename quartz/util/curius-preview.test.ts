import assert from 'node:assert/strict'
import test from 'node:test'
import {
  curiusPreviewImagePath,
  isCuriusPreviewImageUrl,
  parseCuriusPreview,
} from './curius-preview'

test('Curius preview responses distinguish extracted articles from explicit failures', () => {
  const ready = {
    status: 'ready',
    linkId: 236997,
    title: 'An article',
    sourceUrl: 'https://example.com/article',
    finalUrl: 'https://example.com/article',
    readerHtml: '<p>A readable article.</p>',
    fetchedAt: 1_788_300_000_000,
    cached: false,
  }
  assert.deepEqual(parseCuriusPreview(ready), ready)
  const unavailable = {
    status: 'unavailable',
    reason: 'unsupported-type',
    message: 'Open the original to view this document type.',
    sourceUrl: 'https://example.com/paper.pdf',
  }
  assert.deepEqual(parseCuriusPreview(unavailable), unavailable)
  for (const change of [
    { linkId: 0 },
    { linkId: Number.MAX_SAFE_INTEGER + 1 },
    { cached: undefined },
    { readerHtml: '' },
    { readerHtml: 'a'.repeat(2 * 1024 * 1024 + 1) },
    { finalUrl: 'javascript:alert(1)' },
    { sourceUrl: 'https://reader:secret@example.com/' },
  ])
    assert.equal(parseCuriusPreview({ ...ready, ...change }), null)
  assert.equal(parseCuriusPreview({ status: 'ready', artifact: ready }), null)
  assert.equal(parseCuriusPreview({ status: 'unavailable' }), null)
})

test('preview image URLs identify a same-origin saved link, image and extraction version', () => {
  const origin = 'https://aarnphm.xyz'
  const path = curiusPreviewImagePath(236997, 0, 1_788_300_000_000)
  assert.ok(isCuriusPreviewImageUrl(path, origin))
  assert.ok(isCuriusPreviewImageUrl(`${origin}${path}`, origin))
  for (const raw of [
    `https://attacker.example${path}`,
    `https://reader:secret@aarnphm.xyz${path}`,
    path.replace('id=236997', 'id=0'),
    path.replace('id=236997', 'id=9007199254740992'),
    path.replace('image=0', 'image=-1'),
    path.replace('image=0', 'image=300'),
    path.replace('v=1788300000000', 'v=invalid'),
    path.replace('preview-image', 'preview'),
    `${path}&url=https://attacker.example/image.png`,
    `${path}&id=236997`,
    `${path}#fragment`,
    '/api/arena/articles/private/resources/image',
  ])
    assert.equal(isCuriusPreviewImageUrl(raw, origin), false, raw)
})
