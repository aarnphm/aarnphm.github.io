import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import test from 'node:test'
import type { StructuralSource } from '../util/code-search-types'
import { codeSearchPlugins } from '../plugins/emitters/component-resources/code-search-assets'

test('real published AST matcher: languages, offsets, cancellation and bounded soak', async () => {
  const artifacts = path.resolve('docs/verification/code-search/artifacts')
  await mkdir(artifacts, { recursive: true })
  const dir = await mkdtemp(path.join(artifacts, 'ast-'))
  await build({
    entryPoints: ['quartz/workers/code-search-ast.ts'],
    outdir: dir,
    bundle: true,
    platform: 'browser',
    format: 'esm',
    target: 'esnext',
    plugins: codeSearchPlugins(),
    loader: { '.wasm': 'file' },
    assetNames: 'assets/[name]-[hash]',
    publicPath: 'https://code-search.test/',
  })
  const corpusPath = process.env.CODE_SEARCH_AST_CORPUS
  const corpus: unknown = corpusPath
    ? JSON.parse(await readFile(corpusPath, 'utf8'))
    : Array.from({ length: 512 }, (_, i) => ({
        id: `python-${i}`,
        language: 'python',
        text: `def report_${i}(value):\n    print("🦆", value)\n    return value + ${i}\n`,
      }))
  assert(Array.isArray(corpus))
  const subset = corpus
    .filter(
      (value: unknown): value is StructuralSource =>
        typeof value === 'object' &&
        value !== null &&
        'id' in value &&
        typeof value.id === 'string' &&
        'language' in value &&
        value.language === 'python' &&
        'text' in value &&
        typeof value.text === 'string' &&
        value.text.length < 5000,
    )
    .slice(0, 512)
  await writeFile(path.join(dir, 'corpus.json'), JSON.stringify(subset))
  const runner = String.raw`
    import assert from 'node:assert/strict'
    import { createHash } from 'node:crypto'
    import { readFile, writeFile } from 'node:fs/promises'
    const memoryUsage = process.memoryUsage.bind(process)
    const nativeFetch = globalThis.fetch
    globalThis.fetch = async (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (url.startsWith('https://code-search.test/')) return new Response(await readFile(new URL(url.replace('https://code-search.test/', './'), import.meta.url)), { headers: { 'Content-Type': 'application/wasm' } })
      if (url.startsWith('file:')) return new Response(await readFile(new URL(url)), { headers: { 'Content-Type': 'application/wasm' } })
      return nativeFetch(input, init)
    }
    globalThis.process = undefined
    globalThis.window = {}
    const { searchStructural } = await import('./code-search-ast.js')
    const query = (pattern, language) => ({ mode: 'structural', pattern, language, caseSensitive: true })
    const source = (text, language = 'javascript') => ({ id: language, language, text })
    const run = (text, pattern, language = 'javascript') => searchStructural([source(text, language)], query(pattern, language), { cancelled: () => false })
    for (const [language, text, pattern, expected] of [
      ['python', 'def f(x):\n\treturn print(x)\n', 'print($A)', 'print(x)'],
      ['javascript', 'console /*hi*/ . log (a, b)', 'console.log($$$ARGS)', 'console /*hi*/ . log (a, b)'],
      ['typescript', 'const x: number = f(1)', 'f($X)', 'f(1)'],
      ['jsx', 'const a = <Card />', '<Card />', '<Card />'],
      ['tsx', 'const a = <Card title="x" />', '<Card $$$PROPS />', '<Card title="x" />'],
      ['javascript', '// 😀\r\n\tf("🦆", x)\r\n', 'f($$$A)', 'f("🦆", x)'],
      ['javascript', 'const x = "😀"; f("🦆")', 'f("🦆")', 'f("🦆")'],
    ]) {
      const result = await run(text, pattern, language)
      assert.equal(result.matches.length, 1, language + ': ' + text)
      const range = result.matches[0].ranges[0]
      assert.equal(text.slice(range.start.offset, range.end.offset), expected)
      assert.equal(range.start.offset, text.indexOf(expected))
      if (text.startsWith('//')) assert.deepEqual(range.start, { offset: 8, line: 2, column: 2 })
    }
    assert.equal((await run('const x = "😀"; f("🐈")', 'f("🦆")')).matches.length, 0)
    assert.equal((await run('f("😀", "😀")', 'f($A, $A)')).matches.length, 1)
    assert.equal((await run('f("😀", "🦆")', 'f($A, $A)')).matches.length, 0)
    const recovered = await run('f(x); const = ;', 'f($A)')
    assert.equal(recovered.parseErrors, 1)
    assert.equal(recovered.matches.length, 1)
    const unsupported = await searchStructural([source('f(x)', 'rust')], query('f($A)'), { cancelled: () => false })
    assert.deepEqual(unsupported.unsupportedLanguages, ['rust'])
    let cancel = false
    setTimeout(() => { cancel = true }, 0)
    const cancelled = await searchStructural(Array.from({ length: 10000 }, (_, i) => ({ ...source('f(x)'), id: String(i) })), query('f($A)'), {
      cancelled: () => cancel,
    })
    assert(cancelled.scannedSources > 0 && cancelled.scannedSources < 10000)
    const broad = await searchStructural(Array.from({ length: 70 }, (_, i) => ({ ...source('f(x)'), id: String(i) })), query('$F($A)'), { cancelled: () => false })
    assert.equal(broad.matches.length, 70)
    assert.equal(broad.scannedSources, 70)
    const corpus = await readFile(new URL('./corpus.json', import.meta.url), 'utf8').then(JSON.parse)
    const progress = []
    const started = performance.now()
    const batched = await searchStructural(corpus, query('print($$$A)', 'python'), { cancelled: () => false, progress: scanned => progress.push(scanned) })
    const elapsedMs = performance.now() - started
    const report = { sources: corpus.length, progressEvents: progress.length, elapsedMs, resultHash: createHash('sha256').update(JSON.stringify(batched)).digest('hex') }
    await writeFile(new URL('./scheduling.json', import.meta.url), JSON.stringify(report, null, 2))
    assert.equal(batched.scannedSources, corpus.length)
    assert.equal(progress.at(-1), corpus.length)
    assert(progress.every((value, i) => i === 0 || value > progress[i - 1]))
    assert(progress.length < corpus.length / 2, JSON.stringify(report))
    const singles = []
    let parseErrors = 0
    for (const unit of corpus) {
      const result = await searchStructural([unit], query('print($$$A)', 'python'), { cancelled: () => false })
      singles.push(...result.matches)
      parseErrors += result.parseErrors
    }
    assert.deepEqual(batched.matches, singles)
    assert.equal(batched.parseErrors, parseErrors)
    await assert.rejects(run('f(x)', ''), /pattern|parse|syntax/i)
    for (let i = 0; i < 100; i++) await run('f(x)', 'f($A)')
    const before = memoryUsage()
    for (let i = 0; i < 1000; i++) await run('f(x); f(y)', 'f($A)')
    const after = memoryUsage()
    assert(after.arrayBuffers - before.arrayBuffers < 8 * 1024 * 1024)
    await writeFile(new URL('./soak.json', import.meta.url), JSON.stringify({ queries: 1000, before, after }, null, 2))
    console.log('AST acceptance passed: 1000-query soak', JSON.stringify({ before, after }))
  `
  await writeFile(path.join(dir, 'runner.mjs'), runner)
  const result = spawnSync(process.execPath, [path.join(dir, 'runner.mjs')], {
    encoding: 'utf8',
    timeout: 120_000,
  })
  await writeFile(path.join(dir, 'result.txt'), result.stdout + result.stderr)
  assert.equal(result.status, 0, result.stdout + result.stderr)
})
