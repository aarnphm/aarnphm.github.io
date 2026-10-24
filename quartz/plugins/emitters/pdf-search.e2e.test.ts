import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import { isFilePath, type FilePath } from '../../util/path'
import { Assets } from './assets'
import { PdfSearch } from './pdf-search'

async function collect(emitted: Promise<FilePath[]> | AsyncGenerator<FilePath> | null) {
  const result = await emitted
  assert.ok(result)
  if (Symbol.asyncIterator in result) {
    const files: FilePath[] = []
    for await (const file of result) files.push(file)
    return files
  }
  return result
}

test('PDF emitter validates all shards before switching and removing served revisions', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'pdf-publication-'))
  const content = path.join(root, 'content')
  const output = path.join(root, 'public')
  await mkdir(content)
  const colors = {
    light: '#fff',
    lightgray: '#ddd',
    gray: '#888',
    darkgray: '#444',
    dark: '#000',
    secondary: '#ddd',
    tertiary: '#eee',
    highlight: '#ddd',
    textHighlight: '#ddd',
  }
  const ctx: BuildCtx = {
    buildId: 'pdf-publication-replay',
    incremental: false,
    allFiles: [],
    allSlugs: [],
    argv: {
      directory: content,
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
        pageTitle: 'fixture',
        enableSPA: true,
        enablePopovers: false,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'modified',
        baseUrl: 'example.com',
        locale: 'en-US',
        theme: {
          typography: { header: 'sans', body: 'sans', code: 'mono' },
          cdnCaching: false,
          fontOrigin: 'local',
          colors: { lightMode: colors, darkMode: colors },
        },
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
  }
  const resources = { css: [], js: [], additionalHead: [] }
  const source = path.join(content, 'embeddings/pdf')
  const served = path.join(output, 'embeddings/pdf')
  async function generation(id: string, dimension: number) {
    const directory = path.join(source, id)
    await mkdir(directory, { recursive: true })
    const vector = new Float32Array(768)
    vector[dimension] = 1
    const bytes = Buffer.from(vector.buffer)
    const shards = [0, 1].map(i => ({
      path: `/embeddings/pdf/${id}/vectors-${i}.bin`,
      rows: 1,
      rowOffset: i,
      byteLength: bytes.length,
      byteStride: 768 * 4,
      sha256: createHash('sha256').update(bytes).digest('hex'),
    }))
    for (const shard of shards) await writeFile(path.join(content, shard.path), bytes)
    const manifest = {
      version: 2,
      kind: 'quartz-pdf-search-v1',
      modality: 'text',
      model: 'google/embeddinggemma-2',
      dims: 768,
      dtype: 'fp32',
      normalized: true,
      generation: id,
      rows: 2,
      ids: ['paper.pdf#page=1&chunk=0', 'paper.pdf#page=2&chunk=0'],
      documents: {
        'paper.pdf': { title: 'Paper', aliases: [], acronyms: [], pages: [1, 2], totalPages: 2 },
      },
      passages: [1, 2].map(page => ({
        slug: 'paper.pdf',
        page,
        pageLabel: '',
        text: `page ${page}`,
        title: 'Paper',
        chunk: 0,
      })),
      coverage: { indexedDocuments: 1, eligibleDocuments: 1, indexedPages: 2 },
      vectors: { dtype: 'fp32', rows: 2, dims: 768, shards },
    }
    const serialized = JSON.stringify(manifest)
    await writeFile(path.join(directory, 'manifest.json'), serialized)
    await writeFile(path.join(source, 'manifest.json'), serialized)
    ctx.allFiles = [
      'paper.pdf',
      ...shards.map(s => s.path.slice(1)),
      `embeddings/pdf/${id}/manifest.json`,
      'embeddings/pdf/manifest.json',
    ].filter(isFilePath)
    return { bytes, serialized, shards }
  }
  const plugin = PdfSearch()
  try {
    await writeFile(path.join(content, 'paper.pdf'), 'source fixture')
    const first = await generation('1'.repeat(64), 0)
    assert.ok(
      !(await collect(Assets().emit(ctx, [], resources))).some(file =>
        file.includes('/embeddings/pdf/'),
      ),
    )
    await collect(plugin.emit(ctx, [], resources))
    assert.equal(await readFile(path.join(served, 'manifest.json'), 'utf8'), first.serialized)
    const second = await generation('2'.repeat(64), 1)
    const damaged = path.join(content, second.shards[1].path)
    await writeFile(damaged, 'broken')
    assert.ok(plugin.partialEmit)
    ctx.incremental = true
    await collect(plugin.partialEmit(ctx, [], resources, []))
    assert.equal(await readFile(path.join(served, 'manifest.json'), 'utf8'), first.serialized)
    assert.equal((await readdir(path.join(served, '1'.repeat(64)))).length, 3)
    await writeFile(damaged, second.bytes)
    await collect(plugin.partialEmit(ctx, [], resources, []))
    assert.equal(await readFile(path.join(served, 'manifest.json'), 'utf8'), second.serialized)
    await assert.rejects(readdir(path.join(served, '1'.repeat(64))), { code: 'ENOENT' })
    for (const shard of second.shards)
      assert.deepEqual(await readFile(path.join(output, shard.path)), second.bytes)
    console.log(
      JSON.stringify({
        shards: 2,
        failedReplacementPreserved: true,
        obsoleteServedGenerationRemoved: true,
      }),
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
