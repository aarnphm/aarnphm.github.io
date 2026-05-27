import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  classifyArenaFrameHeaders,
  rebaseArenaEmbedSrcset,
  rebaseArenaEmbedUrl,
  validateArenaEmbedTarget,
} from './arena-embed'

describe('arena embed worker helpers', () => {
  test('rejects unsupported proxy targets', () => {
    assert.deepStrictEqual(validateArenaEmbedTarget(null).ok, false)
    assert.deepStrictEqual(validateArenaEmbedTarget('file:///etc/passwd').ok, false)
    assert.deepStrictEqual(validateArenaEmbedTarget('https://user:pass@example.com').ok, false)
    assert.deepStrictEqual(validateArenaEmbedTarget('https://localhost').ok, false)
    assert.deepStrictEqual(validateArenaEmbedTarget('https://127.0.0.1').ok, false)
    assert.deepStrictEqual(validateArenaEmbedTarget('https://10.0.0.1').ok, false)
    assert.deepStrictEqual(validateArenaEmbedTarget('https://192.168.1.20').ok, false)
    assert.deepStrictEqual(validateArenaEmbedTarget('https://example.com:8443').ok, false)
    assert.deepStrictEqual(validateArenaEmbedTarget('https://example.com').ok, true)
  })

  test('classifies frame-blocking headers', () => {
    const embedderOrigin = 'https://aarnphm.xyz'
    const target = new URL('https://stripe.dev/')

    assert.deepStrictEqual(
      classifyArenaFrameHeaders(
        new Headers({ 'Content-Security-Policy': "default-src 'none'; frame-ancestors 'none'" }),
        target,
        embedderOrigin,
      ),
      { mode: 'fetch', reason: 'frame-ancestors-none' },
    )

    assert.deepStrictEqual(
      classifyArenaFrameHeaders(
        new Headers({ 'Content-Security-Policy': "frame-ancestors 'self'" }),
        target,
        embedderOrigin,
      ),
      { mode: 'fetch', reason: 'frame-ancestors-mismatch' },
    )

    assert.deepStrictEqual(
      classifyArenaFrameHeaders(
        new Headers({ 'X-Frame-Options': 'SAMEORIGIN' }),
        target,
        embedderOrigin,
      ),
      { mode: 'fetch', reason: 'x-frame-options-sameorigin' },
    )

    assert.deepStrictEqual(classifyArenaFrameHeaders(new Headers({}), target, embedderOrigin), {
      mode: 'iframe',
      reason: 'ok',
    })
  })

  test('rebases fetched document assets', () => {
    const base = new URL('https://example.com/path/page')
    assert.strictEqual(rebaseArenaEmbedUrl('/asset.css', base), 'https://example.com/asset.css')
    assert.strictEqual(
      rebaseArenaEmbedUrl('./image.png', base),
      'https://example.com/path/image.png',
    )
    assert.strictEqual(rebaseArenaEmbedUrl('#section', base), '#section')
    assert.strictEqual(rebaseArenaEmbedUrl('javascript:alert(1)', base), 'about:blank')
    assert.strictEqual(
      rebaseArenaEmbedUrl('mailto:hello@example.com', base),
      'mailto:hello@example.com',
    )
    assert.strictEqual(
      rebaseArenaEmbedUrl('http://example.com/a.png', base, 'resource'),
      'about:blank',
    )
    assert.strictEqual(
      rebaseArenaEmbedUrl('data:image/png;base64,aa', base, 'resource'),
      'data:image/png;base64,aa',
    )
    assert.strictEqual(
      rebaseArenaEmbedSrcset('/a.png 1x, ./b.png 2x', base),
      'https://example.com/a.png 1x, https://example.com/path/b.png 2x',
    )
  })
})
