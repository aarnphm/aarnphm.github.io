import assert from 'node:assert/strict'
import { lstat, mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath } from '../../util/path'
import type { StaticResources } from '../../util/resources'
import { Assets } from './assets'

function testCtx(root: string): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: path.join(root, 'content'),
      verbose: false,
      output: path.join(root, 'public'),
      serve: false,
      watch: true,
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

async function collectEmitted(emitted: Promise<FilePath[]> | AsyncGenerator<FilePath>) {
  const result = await emitted
  if (Symbol.asyncIterator in result) {
    const files: FilePath[] = []
    for await (const file of result) {
      files.push(file)
    }
    return files
  }
  return result
}

async function touch(root: string, fp: string) {
  const fullPath = path.join(root, 'content', fp)
  await mkdir(path.dirname(fullPath), { recursive: true })
  await writeFile(fullPath, '')
}

test('watch asset emission writes regular copied files', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-assets-emit-'))
  try {
    const ctx = testCtx(root)
    ctx.argv.watch = true
    const files = [
      'media/a.png',
      'media/nested/b.png',
      'section/asset.png',
      'weird name/a file.png',
    ]
    for (const fp of files) {
      await touch(root, fp)
    }
    ctx.allFiles = files as FilePath[]
    const plugin = Assets()

    const emitted = await collectEmitted(plugin.emit(ctx, [], resources))

    assert.deepEqual(
      emitted.sort(),
      [
        path.join(ctx.argv.output, 'media/a.png'),
        path.join(ctx.argv.output, 'media/nested/b.png'),
        path.join(ctx.argv.output, 'section/asset.png'),
        path.join(ctx.argv.output, 'weird-name/a-file.png'),
      ].sort(),
    )
    const source = await stat(path.join(ctx.argv.directory, 'media/a.png'))
    const copied = await stat(path.join(ctx.argv.output, 'media/a.png'))
    assert.notEqual(source.ino, copied.ino)
    assert.equal((await lstat(path.join(ctx.argv.output, 'media/a.png'))).isSymbolicLink(), false)
    assert.equal(
      (await lstat(path.join(ctx.argv.output, 'weird-name/a-file.png'))).isSymbolicLink(),
      false,
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('watch asset emission copies referenced pdf embeds', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-assets-watch-pdf-'))
  try {
    const ctx = testCtx(root)
    ctx.argv.watch = true
    await touch(root, 'notes/index.md')
    await touch(root, 'papers/my-paper.pdf')
    await touch(root, 'papers/unused.pdf')
    await writeFile(
      path.join(ctx.argv.directory, 'notes/index.md'),
      '![[papers/my paper.pdf#{page: 2}|paper]]',
    )
    ctx.allFiles = ['notes/index.md', 'papers/my-paper.pdf', 'papers/unused.pdf'] as FilePath[]
    const plugin = Assets()

    const emitted = await collectEmitted(plugin.emit(ctx, [], resources))

    assert.deepEqual(emitted.sort(), [path.join(ctx.argv.output, 'papers/my-paper.pdf')])
    assert.equal((await lstat(path.join(ctx.argv.output, 'papers/my-paper.pdf'))).isFile(), true)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('production asset emission writes regular files', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-assets-production-'))
  try {
    const ctx = testCtx(root)
    ctx.argv.watch = false
    const files = ['media/a.png', 'weird name/a file.png']
    for (const fp of files) {
      await touch(root, fp)
    }
    const plugin = Assets()

    const emitted = await collectEmitted(plugin.emit(ctx, [], resources))

    assert.deepEqual(
      emitted.sort(),
      [
        path.join(ctx.argv.output, 'media/a.png'),
        path.join(ctx.argv.output, 'weird-name/a-file.png'),
      ].sort(),
    )
    const source = await stat(path.join(ctx.argv.directory, 'media/a.png'))
    const copied = await stat(path.join(ctx.argv.output, 'media/a.png'))
    assert.notEqual(source.ino, copied.ino)
    assert.equal((await lstat(path.join(ctx.argv.output, 'media/a.png'))).isSymbolicLink(), false)
    assert.equal(
      (await lstat(path.join(ctx.argv.output, 'weird-name/a-file.png'))).isSymbolicLink(),
      false,
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
