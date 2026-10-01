import assert from 'node:assert/strict'
import { lstat, mkdir, mkdtemp, readFile, rm, stat, truncate, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { StaticResources } from '../../util/resources'
import { isFilePath, type FilePath } from '../../util/path'
import { Assets, contentAssetClaims } from './assets'

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

async function collectEmitted(emitted: Promise<FilePath[]> | AsyncGenerator<FilePath> | null) {
  const result = await emitted
  assert.ok(result)
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

test('watch asset emission caps referenced PDF copies at 15 MB through updates', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-assets-watch-pdf-'))
  try {
    const ctx = testCtx(root)
    ctx.argv.watch = true
    await touch(root, 'notes/index.md')
    await touch(root, 'papers/my-paper.pdf')
    await touch(root, 'papers/unused.pdf')
    await touch(root, 'papers/limit.pdf')
    await touch(root, 'papers/oversized.pdf')
    await truncate(path.join(ctx.argv.directory, 'papers/my-paper.pdf'), 14_999_999)
    await truncate(path.join(ctx.argv.directory, 'papers/limit.pdf'), 15_000_000)
    await truncate(path.join(ctx.argv.directory, 'papers/oversized.pdf'), 15_000_001)
    await writeFile(
      path.join(ctx.argv.directory, 'notes/index.md'),
      '![[papers/my paper.pdf#{page: 2}|paper]]\n[[papers/limit.pdf]]\n[[papers/oversized.pdf]]',
    )
    const files = [
      'notes/index.md',
      'papers/my-paper.pdf',
      'papers/unused.pdf',
      'papers/limit.pdf',
      'papers/oversized.pdf',
    ]
    assert.ok(files.every(isFilePath))
    ctx.allFiles = files
    const plugin = Assets()
    const expected = ['papers/limit.pdf', 'papers/my-paper.pdf'].map(fp =>
      path.join(ctx.argv.output, fp),
    )

    assert.deepEqual((await contentAssetClaims(ctx)).map(claim => claim.output).sort(), expected)

    const emitted = await collectEmitted(plugin.emit(ctx, [], resources))

    assert.deepEqual(emitted.sort(), expected)
    assert.equal((await stat(expected[0])).size, 15_000_000)
    assert.equal((await stat(expected[1])).size, 14_999_999)
    await assert.rejects(stat(path.join(ctx.argv.output, 'papers/oversized.pdf')), {
      code: 'ENOENT',
    })

    const pdf = 'papers/limit.pdf'
    assert.ok(isFilePath(pdf))
    assert.ok(plugin.partialEmit)
    await truncate(path.join(ctx.argv.directory, pdf), 15_000_001)
    assert.deepEqual(
      await collectEmitted(plugin.partialEmit(ctx, [], resources, [{ type: 'change', path: pdf }])),
      [],
    )
    await assert.rejects(stat(expected[0]), { code: 'ENOENT' })

    await truncate(path.join(ctx.argv.directory, pdf), 15_000_000)
    assert.deepEqual(
      await collectEmitted(plugin.partialEmit(ctx, [], resources, [{ type: 'change', path: pdf }])),
      [expected[0]],
    )
    assert.equal((await stat(expected[0])).size, 15_000_000)
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

for (const watch of [false, true]) {
  test(`canvas assets retain their extension through emission and updates (watch=${watch})`, async t => {
    const root = await mkdtemp(path.join(tmpdir(), 'quartz-canvas-assets-'))
    t.after(() => rm(root, { recursive: true, force: true }))
    const ctx = testCtx(root)
    ctx.argv.watch = watch
    const file = 'fr/parcours.canvas'
    assert.ok(isFilePath(file))
    ctx.allFiles = [file]
    await touch(root, file)
    const source = path.join(ctx.argv.directory, file)
    const output = path.join(ctx.argv.output, file)
    const initial = JSON.stringify({ nodes: [], edges: [] })
    await writeFile(source, initial)
    const plugin = Assets()

    assert.deepEqual(await collectEmitted(plugin.emit(ctx, [], resources)), [output])
    assert.equal(await readFile(output, 'utf8'), initial)
    await assert.rejects(lstat(path.join(ctx.argv.output, 'fr/parcours')), { code: 'ENOENT' })

    const updated = JSON.stringify({ nodes: [], edges: [], version: '1.0' })
    await writeFile(source, updated)
    assert.ok(plugin.partialEmit)
    assert.deepEqual(
      await collectEmitted(
        plugin.partialEmit(ctx, [], resources, [{ type: 'change', path: file }]),
      ),
      [output],
    )
    assert.equal(await readFile(output, 'utf8'), updated)

    await collectEmitted(plugin.partialEmit(ctx, [], resources, [{ type: 'delete', path: file }]))
    await assert.rejects(lstat(output), { code: 'ENOENT' })
  })
}

test('memo audio is local while production keeps only waveform JSON', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-memo-assets-'))
  const previous = process.env.CF_PAGES
  try {
    const ctx = testCtx(root)
    for (const file of ['day.m4a', 'day.peaks.json', 'day.qta', 'day.waveform']) {
      await touch(root, `triathlon/memos/${file}`)
    }
    delete process.env.CF_PAGES
    const local = await collectEmitted(Assets().emit(ctx, [], resources))
    assert.deepEqual(local.sort(), [
      path.join(ctx.argv.output, 'triathlon/memos/day.m4a'),
      path.join(ctx.argv.output, 'triathlon/memos/day.peaks.json'),
    ])
    process.env.CF_PAGES = '1'
    const production = await collectEmitted(Assets().emit(ctx, [], resources))
    assert.deepEqual(production, [path.join(ctx.argv.output, 'triathlon/memos/day.peaks.json')])
  } finally {
    if (previous === undefined) delete process.env.CF_PAGES
    else process.env.CF_PAGES = previous
    await rm(root, { recursive: true, force: true })
  }
})
