import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath, FullSlug, SimpleSlug } from '../../util/path'
import type { StaticResources } from '../../util/resources'
import { defaultProcessedContent } from '../vfile'
import { ContentIndex } from './contentIndex'

function testCtx(root: string): BuildCtx {
  const contentDir = path.join(root, 'content')
  return {
    buildId: 'test',
    argv: {
      directory: contentDir,
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
    allSlugs: ['bases/ideas' as FullSlug, 'thoughts/lecture/notebook' as FullSlug],
    allFiles: ['bases/ideas.base' as FilePath, 'thoughts/lecture/notebook.ipynb' as FilePath],
    incremental: false,
  }
}

const resources: StaticResources = { css: [], js: [], additionalHead: [] }

async function collectEmitted(
  emitted: Promise<FilePath[]> | AsyncGenerator<FilePath> | null,
): Promise<FilePath[]> {
  const result = await emitted
  if (result === null) {
    return []
  }
  if (Symbol.asyncIterator in result) {
    const files: FilePath[] = []
    for await (const file of result) {
      files.push(file)
    }
    return files
  }
  return result
}

function outputPaths(ctx: BuildCtx, files: FilePath[]): string[] {
  return files.map(file => path.relative(ctx.argv.output, file).split(path.sep).join('/')).sort()
}

test('content index includes notebook and base file graph pages', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-content-index-'))
  try {
    const ctx = testCtx(root)
    const baseFilePath = path.join(ctx.argv.directory, 'bases/ideas.base') as FilePath
    const baseContent = defaultProcessedContent({
      slug: 'bases/ideas' as FullSlug,
      filePath: baseFilePath,
      relativePath: 'bases/ideas.base' as FilePath,
      frontmatter: { title: 'parsed ideas base', pageLayout: 'default', tags: ['bases'] },
      text: '',
      links: ['thoughts/lecture/notebook' as SimpleSlug],
    })

    const plugin = ContentIndex({ enableAtom: false, enableSiteMap: false, enableSecurity: false })

    const emitted = await collectEmitted(plugin.emit(ctx, [baseContent], resources))
    assert.ok(emitted.some(file => file.endsWith('static/contentIndex.json')))

    const raw = await readFile(path.join(ctx.argv.output, 'static/contentIndex.json'), 'utf8')
    const index = JSON.parse(raw) as Record<string, { title: string; fileName: string }>

    assert.equal(index['bases/ideas']?.title, 'parsed ideas base')
    assert.equal(index['bases/ideas']?.fileName, 'bases/ideas.base')
    assert.equal(index['thoughts/lecture/notebook']?.title, 'notebook')
    assert.equal(index['thoughts/lecture/notebook']?.fileName, 'thoughts/lecture/notebook.ipynb')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('content index partial emit ignores source asset changes', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-content-index-partial-source-'))
  try {
    const ctx = testCtx(root)
    const plugin = ContentIndex()
    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], resources, [
        { type: 'change', path: 'quartz/static/icon.png' as FilePath },
      ]) ?? null,
    )

    assert.deepEqual(emitted, [])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('content index partial emit limits markdown changes to global indexes and affected feeds', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-content-index-partial-markdown-'))
  try {
    const ctx = testCtx(root)
    ctx.allFiles = ['thoughts/note.md' as FilePath]
    ctx.allSlugs = ['thoughts/note' as FullSlug]
    const filePath = path.join(ctx.argv.directory, 'thoughts/note.md') as FilePath
    const content = defaultProcessedContent({
      slug: 'thoughts/note' as FullSlug,
      filePath,
      relativePath: 'thoughts/note.md' as FilePath,
      frontmatter: { title: 'note', pageLayout: 'default', tags: [] },
      text: 'hello',
      links: [],
    })
    const plugin = ContentIndex({ enableSecurity: true })

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [content], resources, [
        { type: 'change', path: 'thoughts/note.md' as FilePath, file: content[1] },
      ]) ?? null,
    )

    assert.deepEqual(outputPaths(ctx, emitted), [
      'index.xml',
      'sitemap.xml',
      'static/contentIndex.json',
      'static/searchIndex.json',
      'thoughts/index.xml',
    ])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('content index partial emit updates only search index for added file assets', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-content-index-partial-asset-'))
  try {
    const ctx = testCtx(root)
    ctx.allFiles = ['thoughts/diagram.png' as FilePath]
    ctx.allSlugs = ['thoughts/diagram.png' as FullSlug]
    const plugin = ContentIndex()

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], resources, [
        { type: 'add', path: 'thoughts/diagram.png' as FilePath },
      ]) ?? null,
    )

    assert.deepEqual(outputPaths(ctx, emitted), ['static/searchIndex.json'])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('content index partial emit updates graph and search indexes for added notebooks', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-content-index-partial-notebook-'))
  try {
    const ctx = testCtx(root)
    ctx.allFiles = ['thoughts/notebook.ipynb' as FilePath]
    ctx.allSlugs = ['thoughts/notebook' as FullSlug]
    const plugin = ContentIndex()

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], resources, [
        { type: 'add', path: 'thoughts/notebook.ipynb' as FilePath },
      ]) ?? null,
    )

    assert.deepEqual(outputPaths(ctx, emitted), [
      'static/contentIndex.json',
      'static/searchIndex.json',
    ])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
