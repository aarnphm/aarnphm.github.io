import assert from 'node:assert/strict'
import { mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { QuartzConfig } from '../../cfg'
import type { BuildCtx } from '../../util/ctx'
import { resetWriteCache, write } from './helpers'

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
