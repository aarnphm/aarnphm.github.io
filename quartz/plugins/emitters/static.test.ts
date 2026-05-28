import assert from 'node:assert/strict'
import { lstat, mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath } from '../../util/path'
import type { StaticResources } from '../../util/resources'
import { Static } from './static'

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

test('static partial emit ignores content changes', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-static-partial-content-'))
  try {
    const ctx = testCtx(root)
    const plugin = Static()

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], resources, [
        { type: 'change', path: 'thoughts/note.md' as FilePath },
      ]) ?? null,
    )

    assert.deepEqual(emitted, [])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('static partial emit copies only changed quartz static files', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-static-partial-source-'))
  try {
    const ctx = testCtx(root)
    const plugin = Static()

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], resources, [
        { type: 'change', path: 'quartz/static/feed.xsl' as FilePath },
      ]) ?? null,
    )

    const expected = path.join(ctx.argv.output, 'static/feed.xsl')
    assert.deepEqual(emitted, [expected])
    await stat(expected)
    assert.equal((await lstat(expected)).isSymbolicLink(), false)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('Cloudflare Pages watch static partial emit writes regular files', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-static-partial-cf-pages-'))
  const previousCfPages = process.env.CF_PAGES
  try {
    process.env.CF_PAGES = '1'
    const ctx = testCtx(root)
    ctx.argv.watch = true
    const plugin = Static()

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], resources, [
        { type: 'change', path: 'quartz/static/feed.xsl' as FilePath },
      ]) ?? null,
    )

    const expected = path.join(ctx.argv.output, 'static/feed.xsl')
    assert.deepEqual(emitted, [expected])
    assert.equal((await lstat(expected)).isSymbolicLink(), false)
  } finally {
    if (previousCfPages === undefined) {
      delete process.env.CF_PAGES
    } else {
      process.env.CF_PAGES = previousCfPages
    }
    await rm(root, { recursive: true, force: true })
  }
})

test('static partial emit removes deleted quartz static outputs', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-static-partial-delete-'))
  try {
    const ctx = testCtx(root)
    const plugin = Static()
    const expected = path.join(ctx.argv.output, 'static/feed.xsl')
    await mkdir(path.dirname(expected), { recursive: true })
    await writeFile(expected, 'stale')

    const emitted = await collectEmitted(
      plugin.partialEmit?.(ctx, [], resources, [
        { type: 'delete', path: 'quartz/static/feed.xsl' as FilePath },
      ]) ?? null,
    )

    assert.deepEqual(emitted, [])
    await assert.rejects(stat(expected))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
