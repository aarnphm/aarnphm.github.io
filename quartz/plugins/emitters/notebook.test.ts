import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, rm, stat, utimes, writeFile } from 'node:fs/promises'
import { registerHooks } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath } from '../../util/path'

registerHooks({
  load(url, context, nextLoad) {
    if (url.endsWith('.scss') || url.endsWith('.inline.ts') || url.endsWith('.inline')) {
      return { format: 'module', shortCircuit: true, source: 'export default ""' }
    }
    return nextLoad(url, context)
  },
})

const {
  NotebookViewer,
  affectedNotebookFiles,
  mayAffectNotebookPages,
  notebookPageCacheEntries,
  readNotebookDependencies,
  resetNotebookCaches,
} = await import('./notebook')
import { notebookAssetMinBytes } from '../../util/notebook/extract-assets'
import { getPreservedOutputFiles, resetOutputAssetClaims } from '../../util/output-assets'

function makeTestCtx(root: string, allFiles: FilePath[] = []): BuildCtx {
  const colors = {
    light: '#fff',
    lightgray: '#eee',
    gray: '#888',
    darkgray: '#444',
    dark: '#000',
    secondary: '#456',
    tertiary: '#789',
    highlight: '#abc',
    textHighlight: '#def',
  }
  return {
    buildId: 'test-notebook',
    argv: {
      directory: path.join(root, 'content'),
      output: path.join(root, 'public'),
      verbose: false,
      serve: false,
      watch: true,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'test',
        enableSPA: true,
        enablePopovers: true,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'modified',
        locale: 'en-US',
        theme: {
          fontOrigin: 'local',
          cdnCaching: true,
          typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
          colors: { lightMode: colors, darkMode: colors },
        },
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles,
    incremental: false,
  }
}

function sampleNotebookJson(codeSource: string, markdownSource = '# Notebook'): string {
  return JSON.stringify({
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
      language_info: { name: 'python' },
    },
    cells: [
      { cell_type: 'markdown', metadata: {}, source: [markdownSource] },
      { cell_type: 'code', metadata: {}, execution_count: 1, outputs: [], source: [codeSource] },
    ],
  })
}

test('readNotebookDependencies caches parsed dependency records by mtimeMs and size on globalThis', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-notebook-deps-'))
  try {
    resetNotebookCaches()
    const relPath = 'notes/demo.ipynb' as FilePath
    const fullPath = path.join(root, 'content', relPath)
    await mkdir(path.dirname(fullPath), { recursive: true })
    await writeFile(fullPath, sampleNotebookJson('import helper\n', '![diagram](./diagram.png)'))

    const ctx = makeTestCtx(root, [relPath, 'notes/helper.ipynb' as FilePath])
    const first = await readNotebookDependencies(ctx, relPath)
    assert.deepEqual([...first.imports], ['notes/helper.ipynb'])
    assert.deepEqual([...first.assets], ['notes/diagram.png'])

    const second = await readNotebookDependencies(ctx, relPath)
    assert.equal(second, first)

    const updatedContent = sampleNotebookJson(
      'import helper\nimport extra_mod\n',
      '![diagram](./diagram.png)\n<img src="chart.svg">',
    )
    await writeFile(fullPath, updatedContent)
    const future = new Date(Date.now() + 5000)
    await utimes(fullPath, future, future)

    const third = await readNotebookDependencies(ctx, relPath)
    assert.notEqual(third, first)
    assert.deepEqual([...third.imports].sort(), ['notes/extra_mod.ipynb', 'notes/helper.ipynb'])
    assert.deepEqual([...third.assets].sort(), ['notes/chart.svg', 'notes/diagram.png'])
  } finally {
    resetNotebookCaches()
    await rm(root, { recursive: true, force: true })
  }
})

test('mayAffectNotebookPages and affectedNotebookFiles use cached dependencies for partial emits', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-notebook-affected-'))
  try {
    resetNotebookCaches()
    const helperRel = 'course/helper.ipynb' as FilePath
    const mainRel = 'course/main.ipynb' as FilePath
    const helperFull = path.join(root, 'content', helperRel)
    const mainFull = path.join(root, 'content', mainRel)
    await mkdir(path.dirname(mainFull), { recursive: true })
    await writeFile(helperFull, sampleNotebookJson('x = 1'))
    await writeFile(mainFull, sampleNotebookJson('from helper import x', '![fig](./plot.png)'))

    const fps = [helperRel, mainRel]
    const ctx = makeTestCtx(root, fps)

    assert.equal(
      mayAffectNotebookPages([
        { type: 'change', path: 'thoughts/post.md' as FilePath },
        { type: 'change', path: 'university/deck.fc' as FilePath },
      ]),
      false,
    )

    const noneAffected = await affectedNotebookFiles(ctx, fps, [
      { type: 'change', path: 'course/unrelated.png' as FilePath },
    ])
    assert.deepEqual(noneAffected, [])

    const assetAffected = await affectedNotebookFiles(ctx, fps, [
      { type: 'change', path: 'course/plot.png' as FilePath },
    ])
    assert.deepEqual(assetAffected, [mainRel])

    const importAffected = await affectedNotebookFiles(ctx, fps, [
      { type: 'change', path: helperRel },
    ])
    assert.deepEqual(importAffected, [helperRel, mainRel])
  } finally {
    resetNotebookCaches()
    await rm(root, { recursive: true, force: true })
  }
})

test('NotebookViewer persists page cache to disk and hits cache across cold globalThis resets', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-notebook-emit-cache-'))
  const relPath = 'notes/__cache_test_probe__.ipynb' as FilePath
  try {
    resetNotebookCaches()
    resetOutputAssetClaims()
    const fullPath = path.join(root, 'content', relPath)
    await mkdir(path.dirname(fullPath), { recursive: true })
    const pngBytes = new Uint8Array(notebookAssetMinBytes + 128)
    for (let i = 0; i < pngBytes.length; i += 1) pngBytes[i] = (i * 7) % 256
    const dataUri = `data:image/png;base64,${Buffer.from(pngBytes).toString('base64')}`
    await writeFile(fullPath, sampleNotebookJson('y = 42', `<img src="${dataUri}">`))

    const emitter = NotebookViewer()
    const resources = { css: [], js: [], additionalHead: [] }
    const ctx1 = makeTestCtx(root, [relPath])
    const emitted1: FilePath[] = []
    for await (const fp of emitter.emit(ctx1, [], resources) as AsyncIterable<FilePath>) {
      emitted1.push(fp)
    }
    assert.equal(emitted1.length, 1)

    const entries1 = await notebookPageCacheEntries()
    const entry1 = entries1.get(relPath)
    assert.ok(entry1)
    assert.equal(entry1.extractedAssets.length, 1)
    assert.equal((await stat(entry1.blob)).isFile(), true)
    const indexJson = JSON.parse(
      await readFile(path.resolve('quartz/.quartz-cache/notebook-pages/index.json'), 'utf8'),
    )
    assert.ok(indexJson.entries[relPath])

    resetNotebookCaches()
    resetOutputAssetClaims()

    const ctx2 = makeTestCtx(root, [relPath])
    ctx2.buildId = 'test-notebook-2'
    const emitted2: FilePath[] = []
    for await (const fp of emitter.emit(ctx2, [], resources) as AsyncIterable<FilePath>) {
      emitted2.push(fp)
    }
    assert.equal(emitted2.length, 1)
    const preserved = getPreservedOutputFiles()
    const expectedAssetPath = path.resolve(
      ctx2.argv.output,
      entry1.extractedAssets[0].url.replace(/^\//, ''),
    ) as FilePath
    assert.equal(preserved.has(expectedAssetPath), true)
  } finally {
    const entries = await notebookPageCacheEntries()
    const entry = entries.get(relPath)
    if (entry) {
      await rm(entry.blob, { force: true })
      await rm(entry.blob.replace(/\.bin$/, '.meta.json'), { force: true })
      entries.delete(relPath)
    }
    resetNotebookCaches()
    resetOutputAssetClaims()
    await rm(root, { recursive: true, force: true })
  }
})
