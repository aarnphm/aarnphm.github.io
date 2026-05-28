import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath, FullSlug } from '../../util/path'
import type { StaticResources } from '../../util/resources'
import { NotebookPagesManifest } from './notebookManifest'

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
        locale: 'en-US',
        theme: {} as BuildCtx['cfg']['configuration']['theme'],
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: true,
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

  if (!(Symbol.asyncIterator in result)) {
    return result
  }

  const files: FilePath[] = []
  for await (const file of result) {
    files.push(file)
  }
  return files
}

function outputPaths(ctx: BuildCtx, files: FilePath[]): string[] {
  return files.map(file => path.relative(ctx.argv.output, file).split(path.sep).join('/')).sort()
}

test('notebook pages manifest ignores markdown-only partial emits', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-notebook-manifest-markdown-'))
  try {
    const ctx = testCtx(root)
    const plugin = NotebookPagesManifest()

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], resources, [
        { type: 'change', path: 'index.md' as FilePath },
      ]) ?? null,
    )

    assert.deepEqual(emitted, [])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('notebook pages manifest rewrites when notebook membership changes', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-notebook-manifest-add-'))
  try {
    const ctx = testCtx(root)
    ctx.allFiles = ['notes/example.ipynb' as FilePath]
    ctx.allSlugs = ['notes/example' as FullSlug]
    await rm(ctx.argv.directory, { recursive: true, force: true })
    await mkdir(path.join(ctx.argv.directory, 'notes'), { recursive: true })
    await writeFile(path.join(ctx.argv.directory, 'notes/example.ipynb'), '{}')
    const plugin = NotebookPagesManifest()

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], resources, [
        { type: 'add', path: 'notes/example.ipynb' as FilePath },
      ]) ?? null,
    )

    assert.deepEqual(outputPaths(ctx, emitted), ['notebook-pages.json'])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
