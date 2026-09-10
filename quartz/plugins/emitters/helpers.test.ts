import assert from 'node:assert/strict'
import { mkdtemp, readFile, readdir, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'
import test from 'node:test'
import type { QuartzConfig } from '../../cfg'
import type { BuildCtx } from '../../util/ctx'
import { resetWriteCache, write, writeKnownChanged } from './helpers'

const testTheme = {
  typography: { header: 'system-ui', body: 'system-ui', code: 'monospace' },
  cdnCaching: false,
  colors: {
    lightMode: {
      light: '#ffffff',
      lightgray: '#eeeeee',
      gray: '#999999',
      darkgray: '#555555',
      dark: '#000000',
      secondary: '#000000',
      tertiary: '#000000',
      highlight: '#eeeeee',
      textHighlight: '#eeeeee',
    },
    darkMode: {
      light: '#000000',
      lightgray: '#222222',
      gray: '#999999',
      darkgray: '#dddddd',
      dark: '#ffffff',
      secondary: '#ffffff',
      tertiary: '#ffffff',
      highlight: '#222222',
      textHighlight: '#222222',
    },
  },
  fontOrigin: 'local',
} satisfies QuartzConfig['configuration']['theme']

const testConfig = {
  configuration: {
    pageTitle: 'test',
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    ignorePatterns: [],
    defaultDateType: 'created',
    theme: testTheme,
    locale: 'en-US',
  },
  plugins: { transformers: [], filters: [], emitters: [] },
} satisfies QuartzConfig

function ctx(output: string): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: 'content',
      verbose: false,
      output,
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: testConfig,
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

test('resetWriteCache invalidates cached writes after output removal', async () => {
  const output = await mkdtemp(path.join(tmpdir(), 'quartz-write-cache-'))
  try {
    const buildCtx = ctx(output)
    const file = await write({ ctx: buildCtx, slug: 'static/asset', ext: '.txt', content: 'same' })
    await rm(output, { recursive: true, force: true })

    resetWriteCache()
    await write({ ctx: buildCtx, slug: 'static/asset', ext: '.txt', content: 'same' })

    const result = await stat(file)
    assert.equal(result.isFile(), true)
  } finally {
    await rm(output, { recursive: true, force: true })
    resetWriteCache()
  }
})

test('binary write cache tracks unchanged and changed buffers', async () => {
  const output = await mkdtemp(path.join(tmpdir(), 'quartz-write-cache-'))
  try {
    const buildCtx = ctx(output)
    buildCtx.argv.watch = true
    const file = await write({
      ctx: buildCtx,
      slug: 'static/asset',
      ext: '.bin',
      content: Buffer.from('same'),
    })

    buildCtx.incremental = true
    await write({
      ctx: buildCtx,
      slug: 'static/asset',
      ext: '.bin',
      content: Buffer.from('changed'),
    })

    const changed = await stat(file)
    assert.equal(changed.size, Buffer.byteLength('changed'))

    await rm(output, { recursive: true, force: true })
    resetWriteCache()
    buildCtx.incremental = false
    await write({ ctx: buildCtx, slug: 'static/asset', ext: '.bin', content: Buffer.from('same') })

    const result = await stat(file)
    assert.equal(result.isFile(), true)
    assert.equal(result.size, Buffer.byteLength('same'))
  } finally {
    await rm(output, { recursive: true, force: true })
    resetWriteCache()
  }
})

test('incremental readers keep the complete old file until its replacement is ready', async t => {
  const directory = await mkdtemp(path.join(tmpdir(), 'quartz-write-atomic-'))
  const output = path.join(directory, 'public')
  t.after(async () => {
    await rm(directory, { recursive: true, force: true })
    resetWriteCache()
  })
  const buildCtx = ctx(output)
  buildCtx.argv.watch = true
  const file = await write({ ctx: buildCtx, slug: 'page', ext: '.html', content: 'old page' })
  buildCtx.incremental = true
  const resume = Promise.withResolvers<void>()
  const paused = Promise.withResolvers<void>()
  const content = Readable.from(
    (async function* () {
      yield 'new '
      paused.resolve()
      await resume.promise
      yield 'page'
    })(),
  )
  const pending = write({ ctx: buildCtx, slug: 'page', ext: '.html', content })
  try {
    await paused.promise
    assert.equal(await readFile(file, 'utf8'), 'old page')
    const assets = await readdir(output, { recursive: true })
    assert.deepEqual(assets, ['page.html'])
    resume.resolve()
    await pending
    // Asset scanners may stat their directory listing after the write has finished.
    for (const asset of assets) await stat(path.join(output, asset))
  } finally {
    resume.resolve()
    await pending
  }
  assert.equal(await readFile(file, 'utf8'), 'new page')
  assert.deepEqual(await readdir(output), ['page.html'])
  assert.deepEqual(await readdir(directory), ['public'])
  await write({ ctx: buildCtx, slug: 'page', ext: '.html', content: 'old page' })
  assert.equal(await readFile(file, 'utf8'), 'old page')
})

test('failed incremental writes preserve the published file and remove temporary output', async t => {
  const directory = await mkdtemp(path.join(tmpdir(), 'quartz-write-atomic-'))
  const output = path.join(directory, 'public')
  t.after(async () => {
    await rm(directory, { recursive: true, force: true })
    resetWriteCache()
  })
  const buildCtx = ctx(output)
  const file = await write({ ctx: buildCtx, slug: 'page', ext: '.html', content: 'old page' })
  buildCtx.incremental = true
  const content = Readable.from(
    (async function* () {
      yield 'incomplete '
      throw new Error('interrupted source')
    })(),
  )
  await assert.rejects(
    write({ ctx: buildCtx, slug: 'page', ext: '.html', content }),
    /interrupted source/,
  )
  assert.equal(await readFile(file, 'utf8'), 'old page')
  assert.deepEqual(await readdir(output), ['page.html'])
  assert.deepEqual(await readdir(directory), ['public'])

  await writeKnownChanged({ ctx: buildCtx, slug: 'page', ext: '.html', content: 'recovered page' })
  assert.equal(await readFile(file, 'utf8'), 'recovered page')
  assert.deepEqual(await readdir(output), ['page.html'])
  assert.deepEqual(await readdir(directory), ['public'])
})
