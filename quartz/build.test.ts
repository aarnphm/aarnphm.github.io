import { Mutex } from 'async-mutex'
import { toHtml } from 'hast-util-to-html'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { registerHooks } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'
import type { QuartzConfig } from './cfg'
import type { Argv } from './util/ctx'
import { write } from './plugins/emitters/helpers'
import { resetProcessedContentCache } from './processors/parse'

test('a content edit during the initial build is rebuilt before becoming idle', async () => {
  const assets = registerHooks({
    resolve(specifier, context, next) {
      return next(specifier.endsWith('.inline') ? `${specifier}.ts` : specifier, context)
    },
    load(url, context, next) {
      if (/\.(?:scss|css|inline\.ts)$/.test(url)) {
        return {
          format: 'module',
          source: `export default ${JSON.stringify(readFileSync(new URL(url), 'utf8'))}`,
          shortCircuit: true,
        }
      }
      return next(url, context)
    },
  })
  const { buildQuartz } = await import('./build')
  const originalDirectory = process.cwd()
  const directory = await mkdtemp(path.join(tmpdir(), 'quartz-initial-rebuild-'))
  const input = path.join(directory, 'content')
  const output = path.join(directory, 'public')
  await mkdir(input)
  await mkdir(path.join(directory, 'quartz/static'), { recursive: true })
  await mkdir(path.join(directory, '.agents/skills'), { recursive: true })
  const source = path.join(input, 'note.md')
  await writeFile(source, 'before initial build finishes')
  const entered = Promise.withResolvers<void>()
  const resume = Promise.withResolvers<void>()
  const refreshed = Promise.withResolvers<void>()
  let initial = true
  const colors = {
    light: '#fff',
    lightgray: '#eee',
    gray: '#999',
    darkgray: '#555',
    dark: '#111',
    secondary: '#123',
    tertiary: '#456',
    highlight: '#def',
    textHighlight: '#fed',
  }
  const config: QuartzConfig = {
    configuration: {
      pageTitle: 'rebuild fixture',
      enableSPA: false,
      enablePopovers: false,
      analytics: null,
      ignorePatterns: [],
      defaultDateType: 'created',
      locale: 'en-US',
      theme: {
        typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
        cdnCaching: false,
        colors: { lightMode: colors, darkMode: colors },
        fontOrigin: 'local',
      },
    },
    plugins: {
      transformers: [],
      filters: [],
      emitters: [
        {
          name: 'FixturePage',
          async *emit(ctx, content) {
            for (const [tree, file] of content) {
              if (!file.data.slug) throw new Error('missing slug')
              yield write({ ctx, slug: file.data.slug, ext: '.html', content: toHtml(tree) })
            }
            if (initial) {
              initial = false
              entered.resolve()
              await resume.promise
            }
          },
        },
      ],
    },
  }
  const argv: Argv = {
    directory: input,
    output,
    watch: true,
    serve: false,
    verbose: false,
    force: false,
    port: 0,
    wsPort: 0,
    concurrency: 1,
  }
  process.chdir(directory)
  resetProcessedContentCache()
  const build = buildQuartz(config, argv, new Mutex(), () => refreshed.resolve())
  try {
    await entered.promise
    assert.match(await readFile(path.join(output, 'note.html'), 'utf8'), /before initial/)
    await writeFile(source, 'edited during initial build')
    resume.resolve()
    await build
    await Promise.race([
      refreshed.promise,
      delay(2_000, undefined, { ref: false }).then(() => {
        throw new Error('the edit made during initial emission was never rebuilt')
      }),
    ])
    assert.match(await readFile(path.join(output, 'note.html'), 'utf8'), /edited during initial/)
  } finally {
    resume.resolve()
    await (await build)?.dispose()
    resetProcessedContentCache()
    process.chdir(originalDirectory)
    await rm(directory, { recursive: true, force: true })
    assets.deregister()
  }
})
