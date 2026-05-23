import assert from 'node:assert/strict'
import test from 'node:test'
import type { BuildCtx } from './ctx'
import { assetManifestRecord, assetPath, assetSlugForContent, resolveAsset } from './asset-manifest'

function testCtx(overrides: Partial<BuildCtx['argv']> = {}): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: 'content',
      verbose: false,
      output: 'public',
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
      ...overrides,
    },
    cfg: {} as BuildCtx['cfg'],
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

test('registers content-hashed production asset names', () => {
  const ctx = testCtx()
  const slug = assetSlugForContent(ctx, 'static/resource-after-0', '.js', 'console.log(1)')

  assert.match(slug, /^static\/resource-after-0-[0-9a-f]{8}$/)
  assert.equal(resolveAsset(ctx, 'static/resource-after-0.js'), `${slug}.js`)
  assert.deepEqual(assetManifestRecord(ctx), { 'static/resource-after-0.js': `${slug}.js` })
})

test('keeps logical names during watch and serve builds', () => {
  const watchCtx = testCtx({ watch: true })
  const serveCtx = testCtx({ serve: true, watch: true })

  assert.equal(assetSlugForContent(watchCtx, 'index', '.css', 'body{}'), 'index')
  assert.equal(assetSlugForContent(serveCtx, 'postscript', '.js', 'console.log(1)'), 'postscript')
  assert.equal(resolveAsset(watchCtx, assetPath('index', '.css')), 'index.css')
  assert.equal(resolveAsset(serveCtx, assetPath('postscript', '.js')), 'postscript.js')
})
