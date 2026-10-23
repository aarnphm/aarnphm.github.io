import { globby } from 'globby'
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import {
  IMAGE_ENCODER_CONTRACT,
  publishImageGeneration,
  sha256,
} from '../../scripts/image-embeddings/store'
import { readImageManifest } from '../../util/image-search'
import { isFilePath, isFullSlug, type FilePath } from '../../util/path'
import { defaultProcessedContent } from '../vfile'
import { Assets } from './assets'
import { ImageSearch } from './image-search'

async function collect(result: Promise<FilePath[]> | AsyncGenerator<FilePath> | null) {
  const value = await result
  assert.ok(value)
  if (Symbol.asyncIterator in value) {
    const files = []
    for await (const file of value) files.push(file)
    return files
  }
  return value
}

function note(slug: string) {
  assert.ok(isFullSlug(slug))
  return defaultProcessedContent({
    slug,
    frontmatter: { title: slug, tags: [], pageLayout: 'default' },
  })
}

test('full and partial emit preserve unrelated rows and remove obsolete public snapshots', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'image-emitter-'))
  const content = path.join(root, 'content')
  const output = path.join(root, 'public')
  await mkdir(content)
  const files: Record<string, string> = {
    'first.md': '![[first.png]]',
    'second.md': '![[second.png]]',
    'first.png': 'first pixels',
    'second.png': 'second pixels',
    'unrelated.md': 'some words',
  }
  for (const [file, bytes] of Object.entries(files))
    await writeFile(path.join(content, file), bytes)
  const vector = new Float32Array(768)
  vector[0] = 1
  await publishImageGeneration({
    outDir: path.join(content, 'embeddings/images'),
    rows: ['first.png', 'second.png'].map(id => ({ id, vector })),
    contract: IMAGE_ENCODER_CONTRACT,
  })
  // The source is intentionally retained after the note becomes private.
  const sourcePath = path.join(content, 'embeddings/images/manifest.json')
  const source = readImageManifest(JSON.parse(await readFile(sourcePath, 'utf8')))
  source.sourceDigests = {
    'first.png': sha256(files['first.png']),
    'second.png': sha256(files['second.png']),
  }
  await writeFile(sourcePath, JSON.stringify(source))
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
    buildId: 'image-publication-replay',
    incremental: false,
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
    allSlugs: [],
    allFiles: (await globby('**/*', { cwd: content })).filter(isFilePath),
  }
  const resources = { css: [], js: [], additionalHead: [] }
  const plugin = ImageSearch()
  try {
    const assetFiles = await collect(Assets().emit(ctx, [], resources))
    assert.ok(!assetFiles.some(file => file.includes('/embeddings/images/')))
    const pages = [note('first'), note('second'), note('unrelated')]
    await collect(plugin.emit(ctx, pages, resources))
    const manifestPath = path.join(output, 'embeddings/images/manifest.json')
    const read = async () => readImageManifest(JSON.parse(await readFile(manifestPath, 'utf8')))
    const initial = await read()
    assert.deepEqual(initial.ids, ['first.png', 'second.png'])
    ctx.incremental = true
    assert.ok(plugin.partialEmit)
    const unrelatedPath = 'unrelated.md'
    assert.ok(isFilePath(unrelatedPath))
    await writeFile(path.join(content, 'unrelated.md'), 'updated unrelated prose')
    await collect(
      plugin.partialEmit(ctx, pages, resources, [{ type: 'change', path: unrelatedPath }]),
    )
    assert.equal((await read()).generation, initial.generation)
    await writeFile(path.join(content, 'first.md'), '---\nprivate: true\n---\n![[first.png]]')
    const firstPath = 'first.md'
    assert.ok(isFilePath(firstPath))
    await collect(
      plugin.partialEmit(ctx, [note('second'), note('unrelated')], resources, [
        { type: 'delete', path: firstPath },
      ]),
    )
    const after = await read()
    assert.deepEqual(after.ids, ['second.png'])
    const published = await globby('**/*', { cwd: path.join(output, 'embeddings/images') })
    assert.ok(published.every(file => !file.startsWith(initial.generation)))
    assert.equal(published.length, 3)
    assert.equal((await readFile(sourcePath, 'utf8')).includes('first.png'), true)
    console.log(
      JSON.stringify({
        initialRows: initial.rows,
        partialRows: after.rows,
        staleGenerations: 0,
        sourcePreserved: true,
      }),
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
