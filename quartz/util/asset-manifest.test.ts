import assert from 'node:assert/strict'
import test from 'node:test'
import type { BuildCtx } from './ctx'
import {
  assetReferenceForContent,
  assetManifestRecord,
  assetPath,
  assetSlugForContent,
  contentHashSlug,
  hashAssetContent,
  resolveAsset,
} from './asset-manifest'

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
  const slug = assetSlugForContent(ctx, 'postscript', '.js', 'console.log(1)')

  assert.match(slug, /^postscript-[0-9a-f]{8}$/)
  assert.equal(resolveAsset(ctx, 'postscript.js'), `${slug}.js`)
  assert.deepEqual(assetManifestRecord(ctx), { 'postscript.js': `${slug}.js` })
})

test('creates content-only extracted resource names', () => {
  assert.match(
    contentHashSlug('static/resource-after', 'console.log(1)'),
    /^static\/resource-after-[0-9a-f]{8}$/,
  )
})

test('keeps logical names during watch and serve builds', () => {
  const watchCtx = testCtx({ watch: true })
  const serveCtx = testCtx({ serve: true, watch: true })

  assert.equal(assetSlugForContent(watchCtx, 'index', '.css', 'body{}'), 'index')
  assert.equal(assetSlugForContent(serveCtx, 'postscript', '.js', 'console.log(1)'), 'postscript')
  assert.equal(resolveAsset(watchCtx, assetPath('index', '.css')), 'index.css')
  assert.equal(resolveAsset(serveCtx, assetPath('postscript', '.js')), 'postscript.js')
})

test('versions stable watch asset paths by content', () => {
  const watchCtx = testCtx({ watch: true })
  const productionCtx = testCtx()
  const watchAsset = assetReferenceForContent(watchCtx, 'static/component', '.css', 'body{}')
  const productionAsset = assetReferenceForContent(
    productionCtx,
    'static/component',
    '.css',
    'body{}',
  )

  assert.equal(watchAsset.slug, 'static/component')
  assert.equal(watchAsset.path, `static/component.css?v=${hashAssetContent('body{}')}`)
  assert.match(productionAsset.slug, /^static\/component-[0-9a-f]{8}$/)
  assert.equal(productionAsset.path, `${productionAsset.slug}.css`)
})
