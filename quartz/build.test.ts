import assert from 'node:assert/strict'
import test from 'node:test'
import { type QuartzConfig } from './cfg'
import { type ChangeEvent, type QuartzEmitterPluginInstance } from './types/plugin'
import { type BuildCtx } from './util/ctx'
import { emitChangedContent } from './util/emit-scheduler'
import { isFilePath, type FilePath } from './util/path'
import { type StaticResources } from './util/resources'

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

function filePath(value: string): FilePath {
  if (!isFilePath(value)) {
    throw new Error(`${value} is not a file path`)
  }
  return value
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

test('incremental emit runs component resources before concurrent page emitters', async () => {
  const resources: StaticResources = { css: [], js: [], additionalHead: [] }
  const ctx = {
    buildId: 'test',
    argv: {
      directory: 'content',
      verbose: false,
      output: 'public',
      serve: false,
      watch: false,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: testConfig,
    allSlugs: [],
    allFiles: [],
    incremental: true,
  } satisfies BuildCtx

  const events: string[] = []
  let activePageEmitters = 0
  let maxActivePageEmitters = 0
  const changeEvents: ChangeEvent[] = [{ type: 'change', path: filePath('index.md') }]

  function emitter(name: string, ms: number): QuartzEmitterPluginInstance {
    return {
      name,
      async emit() {
        return []
      },
      async partialEmit() {
        const isComponentResources = name === 'ComponentResources'
        events.push(`${name}:start`)
        if (!isComponentResources) {
          assert.equal(events.includes('ComponentResources:end'), true)
          activePageEmitters += 1
          maxActivePageEmitters = Math.max(maxActivePageEmitters, activePageEmitters)
        }
        await delay(ms)
        if (!isComponentResources) {
          activePageEmitters -= 1
        }
        events.push(`${name}:end`)
        return [filePath(`${name}.txt`)]
      },
    }
  }

  const emittedFiles = await emitChangedContent(
    ctx,
    [emitter('PageA', 20), emitter('ComponentResources', 5), emitter('PageB', 20)],
    [],
    resources,
    changeEvents,
  )

  assert.equal(emittedFiles, 3)
  assert.equal(events.at(0), 'ComponentResources:start')
  assert.equal(events.at(1), 'ComponentResources:end')
  assert.equal(maxActivePageEmitters, 2)
})
