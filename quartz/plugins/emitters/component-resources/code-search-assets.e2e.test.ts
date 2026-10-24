import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../../util/ctx'
import { resolveAsset } from '../../../util/asset-manifest'
import { writeCodeSearchAssets } from './code-search-assets'

function testCtx(output: string): BuildCtx {
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
    buildId: 'notebook-offline',
    argv: {
      directory: 'content',
      output,
      verbose: false,
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'offline notebook fixture',
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
          colors: { lightMode: colors, darkMode: colors },
        },
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

test('production worker assets execute literal first and fetch only selected AST grammar', async () => {
  const artifacts = path.resolve('docs/verification/code-search/artifacts')
  await mkdir(artifacts, { recursive: true })
  const output = await mkdtemp(path.join(artifacts, 'assets-'))
  const ctx = testCtx(output)
  const emitted = await writeCodeSearchAssets(ctx)
  const entry = resolveAsset(ctx, 'static/scripts/code-search.worker.js')
  assert.notEqual(entry, 'static/scripts/code-search.worker.js')
  assert.equal(emitted.filter(file => file.endsWith('.wasm')).length, 6)
  const text = 'print("🦆")\r\n'
  const sourceId = createHash('sha256').update(text).digest('hex')
  const generation = 'ast-acceptance'
  const shard = JSON.stringify({
    kind: 'quartz-code-sources-v1',
    generation,
    language: 'python',
    sources: [{ id: sourceId, text }],
  })
  const manifest = {
    kind: 'quartz-code-search-v1',
    version: 1,
    generation,
    units: [
      {
        id: 'fixture',
        ownerSlug: 'fixture',
        title: 'Fixture',
        path: 'fixture.py',
        kind: 'attachment',
        language: 'python',
        originalLanguage: 'python',
        sourceId,
        anchor: null,
        startLine: 1,
        endLine: 2,
      },
    ],
    shards: [
      {
        language: 'python',
        path: 'python.json',
        sha256: createHash('sha256').update(shard).digest('hex'),
        byteLength: Buffer.byteLength(shard),
        sourceIds: [sourceId],
      },
    ],
    coverage: { owners: 1, units: 1, sources: 1 },
  }
  await mkdir(path.join(output, 'static/code-search'), { recursive: true })
  await writeFile(path.join(output, 'static/code-search/python.json'), shard)
  await writeFile(path.join(output, 'static/code-search/manifest.json'), JSON.stringify(manifest))
  await writeFile(
    path.join(output, 'run.mjs'),
    String.raw`
    import assert from 'node:assert/strict'
    import { readFile, writeFile } from 'node:fs/promises'
    import { readFileSync } from 'node:fs'
    import { registerHooks } from 'node:module'
    import { performance } from 'node:perf_hooks'
    const base = new URL('./', import.meta.url)
    registerHooks({ resolve(specifier, context, next) {
      if (specifier.startsWith('https://fixture.test/') || context.parentURL?.startsWith('https://fixture.test/')) return { url: new URL(specifier, context.parentURL).href, shortCircuit: true }
      return next(specifier, context)
    }, load(url, context, next) {
      if (url.startsWith('https://fixture.test/')) return { format: 'module', source: readFileSync(new URL('.' + new URL(url).pathname, base), 'utf8'), shortCircuit: true }
      return next(url, context)
    } })
    const fetched = []
    globalThis.location = { href: 'https://fixture.test/static/scripts/code-search.worker.js' }
    globalThis.window = {}
    globalThis.process = undefined
    globalThis.fetch = async input => {
      const url = new URL(typeof input === 'string' ? input : input.href, location.href)
      fetched.push(url.pathname)
      const body = await readFile(new URL('.' + url.pathname, base))
      return new Response(body, { headers: { 'Content-Type': url.pathname.endsWith('.wasm') ? 'application/wasm' : 'application/json' } })
    }
    let handle
    let receive
    globalThis.self = { addEventListener: (_, fn) => { handle = fn }, postMessage: value => receive(value) }
    await import('https://fixture.test/` +
      entry +
      String.raw`')
    const query = async (mode, pattern) => new Promise((resolve, reject) => {
      receive = value => {
        if (value.type === 'error') reject(new Error(value.message))
        if (value.type === 'result') resolve(value.result)
      }
      handle({ data: { type: 'search', id: 1, manifestUrl: 'https://fixture.test/static/code-search/manifest.json', query: { mode, pattern, language: 'python', caseSensitive: true } } })
    })
    const literal = await query('literal', 'print')
    assert.equal(literal.totalMatches, 1)
    assert.equal(fetched.filter(url => url.endsWith('.wasm')).length, 0)
    const started = performance.now()
    const structural = await query('structural', 'print($A)')
    assert.equal(structural.totalMatches, 1)
    assert.equal(structural.matches[0].ranges[0].end.offset, 11)
    const wasm = fetched.filter(url => url.endsWith('.wasm'))
    assert.equal(wasm.length, 3)
    assert(wasm.some(url => url.includes('tree-sitter-python')))
    await writeFile(new URL('./result.json', base), JSON.stringify({ literal, structural, fetched, coldStructuralMs: performance.now() - started }, null, 2))
    console.log('Production worker acceptance passed', JSON.stringify(fetched))
  `,
  )
  const result = spawnSync(process.execPath, [path.join(output, 'run.mjs')], {
    encoding: 'utf8',
    timeout: 30_000,
  })
  await writeFile(path.join(output, 'result.txt'), result.stdout + result.stderr)
  assert.equal(result.status, 0, result.stdout + result.stderr)
})
