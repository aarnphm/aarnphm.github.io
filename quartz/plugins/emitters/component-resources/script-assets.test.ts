import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { runInNewContext } from 'node:vm'
import type { BuildCtx } from '../../../util/ctx'
import { assetManifestRecord } from '../../../util/asset-manifest'
import { resetNotebookRuntimeAssetCache } from './notebook-assets'
import { writePageScripts } from './script-assets'

function testCtx(output: string): BuildCtx {
  return {
    buildId: 'test-page-script-publication',
    argv: {
      directory: 'content',
      verbose: false,
      output,
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'page script publication fixture',
        enableSPA: true,
        enablePopovers: true,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'modified',
        baseUrl: 'example.com',
        locale: 'en-US',
        theme: {
          fontOrigin: 'local',
          cdnCaching: true,
          typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
          colors: {
            lightMode: {
              light: '#fff',
              lightgray: '#eee',
              gray: '#888',
              darkgray: '#444',
              dark: '#000',
              secondary: '#456',
              tertiary: '#789',
              highlight: '#abc',
              textHighlight: '#def',
            },
            darkMode: {
              light: '#fff',
              lightgray: '#eee',
              gray: '#888',
              darkgray: '#444',
              dark: '#000',
              secondary: '#456',
              tertiary: '#789',
              highlight: '#abc',
              textHighlight: '#def',
            },
          },
        },
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

// Failures at this seam include unresolved placeholders, missing prerequisite files, omitted
// output claims, and incorrect relative URLs after content hashing.
test('page script publication resolves runtime URLs to its emitted production assets', async () => {
  const output = await mkdtemp(path.join(tmpdir(), 'quartz-page-script-publication-'))
  const ctx = testCtx(output)
  const logicalUrls = [
    'notebook-runtime.client.js',
    'notebook-runtime.worker.js',
    'notebook-runtime.javascript.worker.js',
    'notebook-runtimes/manifest.json',
    'notebook-pyright/worker/manifest.json',
    'notebook-pyright/typeshed/manifest.json',
    'notebook-pyright/site-packages/manifest.json',
    'collaborative-comments.client.js',
    'semantic.worker.js',
    'lexical.worker.js',
    'code-search.worker.js',
  ]
  try {
    const files = await writePageScripts(ctx, {
      css: [],
      componentCss: [],
      beforeDOMLoaded: [],
      afterDOMLoaded: [`window.runtimeUrls = ${JSON.stringify(logicalUrls)};`],
    })
    const manifest = assetManifestRecord(ctx)
    const script = files.find(file => file.includes('/static/scripts/script-'))
    assert.ok(script, 'publication must return its page module')
    const window: { runtimeUrls?: string[] } = {}
    runInNewContext(await readFile(script, 'utf8'), { window })
    const expectedUrls: string[] = []
    for (const logicalUrl of logicalUrls) {
      const logicalPath = `static/scripts/${logicalUrl}`
      const emittedPath = manifest[logicalPath]
      assert.ok(emittedPath, `publication must register ${logicalPath}`)
      assert.notEqual(emittedPath, logicalPath, 'production assets must carry content hashes')
      const emittedFile = path.join(output, emittedPath)
      assert.ok(
        files.some(file => file === emittedFile),
        `publication must return ${emittedPath}`,
      )
      assert.ok((await stat(emittedFile)).size > 0, `${emittedPath} must contain bytes`)
      expectedUrls.push(path.posix.relative('static/scripts', emittedPath))
    }
    assert.deepEqual(Array.from(window.runtimeUrls ?? []), expectedUrls)
    const artifacts = path.resolve(
      'docs/verification/codebase-structure/artifacts/build/publication',
    )
    await mkdir(artifacts, { recursive: true })
    await writeFile(
      path.join(artifacts, 'result.json'),
      JSON.stringify(
        { logicalUrls, resolvedUrls: expectedUrls, emittedFiles: files.length },
        null,
        2,
      ),
    )
    await writeFile(path.join(artifacts, 'asset-manifest.json'), JSON.stringify(manifest, null, 2))
  } finally {
    resetNotebookRuntimeAssetCache()
    await rm(output, { recursive: true, force: true })
  }
})
