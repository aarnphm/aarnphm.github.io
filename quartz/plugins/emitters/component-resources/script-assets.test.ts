import assert from 'node:assert/strict'
import test from 'node:test'
import type { BuildCtx } from '../../../util/ctx'
import type { ComponentResourceSet } from './resource-set'
import { registerAsset } from '../../../util/asset-manifest'
import { resolveComponentResourceAssets } from './script-assets'

function testCtx(): BuildCtx {
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
    },
    cfg: {} as BuildCtx['cfg'],
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

test('resolves component script placeholders to static-script-relative hashed paths', () => {
  const ctx = testCtx()
  registerAsset(
    ctx,
    'static/scripts/notebook-runtime.client.js',
    'static/scripts/notebook-runtime.client-11111111.js',
  )
  registerAsset(
    ctx,
    'static/scripts/notebook-pyright/typeshed/manifest.json',
    'static/scripts/notebook-pyright/typeshed/manifest-22222222.json',
  )
  registerAsset(
    ctx,
    'static/scripts/notebook-pyright/site-packages/manifest.json',
    'static/scripts/notebook-pyright/site-packages/manifest-33333333.json',
  )

  const resources: ComponentResourceSet = {
    css: [],
    componentCss: [],
    beforeDOMLoaded: [],
    afterDOMLoaded: [
      [
        'notebookRuntimeScriptUrl("notebook-runtime.client.js")',
        'notebookRuntimeScriptUrl("notebook-pyright/typeshed/manifest.json")',
        'notebookRuntimeScriptUrl("notebook-pyright/site-packages/manifest.json")',
      ].join(';'),
    ],
  }

  resolveComponentResourceAssets(ctx, resources)

  assert.equal(
    resources.afterDOMLoaded[0],
    [
      'notebookRuntimeScriptUrl("notebook-runtime.client-11111111.js")',
      'notebookRuntimeScriptUrl("notebook-pyright/typeshed/manifest-22222222.json")',
      'notebookRuntimeScriptUrl("notebook-pyright/site-packages/manifest-33333333.json")',
    ].join(';'),
  )
})
