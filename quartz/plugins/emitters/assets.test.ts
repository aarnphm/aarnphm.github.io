import assert from 'node:assert/strict'
import { lstat, mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath } from '../../util/path'
import type { StaticResources } from '../../util/resources'
import { Assets, planDevAssetLinks } from './assets'

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

test('dev asset planner separates direct paths from slugged paths', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-assets-plan-'))
  try {
    const plan = planDevAssetLinks([
      'media/a.png',
      'media/nested/b.png',
      'section/asset.png',
      'weird name/a file.png',
    ] as FilePath[])

    assert.deepEqual(plan.stable, ['media/a.png', 'media/nested/b.png', 'section/asset.png'])
    assert.deepEqual(plan.slugged, ['weird name/a file.png'])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('dev asset emission batch-links stable paths and preserves slugged file outputs', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-assets-emit-'))
  const previousCfPages = process.env.CF_PAGES
  try {
    delete process.env.CF_PAGES
    const ctx = testCtx(root)
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
    const linked = await stat(path.join(ctx.argv.output, 'media/a.png'))
    assert.equal(source.ino, linked.ino)
    assert.equal((await lstat(path.join(ctx.argv.output, 'media'))).isSymbolicLink(), false)
    assert.equal(
      (await lstat(path.join(ctx.argv.output, 'weird-name/a-file.png'))).isSymbolicLink(),
      true,
    )
  } finally {
    if (previousCfPages === undefined) {
      delete process.env.CF_PAGES
    } else {
      process.env.CF_PAGES = previousCfPages
    }
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

test('Cloudflare Pages watch asset emission writes regular files', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-assets-cf-pages-'))
  const previousCfPages = process.env.CF_PAGES
  try {
    process.env.CF_PAGES = '1'
    const ctx = testCtx(root)
    ctx.argv.watch = true
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
    assert.equal((await lstat(path.join(ctx.argv.output, 'media/a.png'))).isSymbolicLink(), false)
    assert.equal(
      (await lstat(path.join(ctx.argv.output, 'weird-name/a-file.png'))).isSymbolicLink(),
      false,
    )
  } finally {
    if (previousCfPages === undefined) {
      delete process.env.CF_PAGES
    } else {
      process.env.CF_PAGES = previousCfPages
    }
    await rm(root, { recursive: true, force: true })
  }
})
