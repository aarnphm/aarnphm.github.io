import assert from 'node:assert/strict'
import { mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../../util/ctx'
import { resolveAsset } from '../../../util/asset-manifest'
import { xsltPolyfillPath } from './asset-paths'
import { writeXsltPolyfillAsset } from './xslt-polyfill-assets'

function testCtx(root: string): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: path.join(root, 'content'),
      verbose: false,
      output: path.join(root, 'public'),
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'test garden',
        enableSPA: true,
        enablePopovers: true,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'modified',
        baseUrl: 'example.com',
        locale: 'en-US',
        theme: {} as BuildCtx['cfg']['configuration']['theme'],
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

test('XSLT polyfill asset is copied with a production content hash', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-xslt-polyfill-'))
  try {
    const ctx = testCtx(root)
    const file = await writeXsltPolyfillAsset(ctx)
    const emittedPath = resolveAsset(ctx, xsltPolyfillPath)

    assert.equal(file, path.join(ctx.argv.output, emittedPath))
    assert.notEqual(emittedPath, xsltPolyfillPath)
    assert.equal(path.extname(emittedPath), '.js')
    assert.ok((await stat(file)).size > 1_000_000)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
