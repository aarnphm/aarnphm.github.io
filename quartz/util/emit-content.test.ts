import assert from 'node:assert/strict'
import test from 'node:test'
import type { QuartzConfig } from '../cfg'
import type { QuartzEmitterPluginInstance } from '../types/plugin'
import type { BuildCtx } from './ctx'
import { emitContent } from '../processors/emit'
import { FilePath, isFilePath } from './path'

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

function filePath(value: string): FilePath {
  if (isFilePath(value)) return value
  throw new Error(`invalid file path ${value}`)
}

function testCtx(emitters: QuartzEmitterPluginInstance[]): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: 'content',
      verbose: false,
      output: 'public',
      serve: false,
      watch: true,
      port: 8080,
      wsPort: 3001,
      force: false,
      slowBuildThreshold: 0,
    },
    cfg: {
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
      plugins: { transformers: [], filters: [], emitters },
    },
    allSlugs: [],
    allFiles: [],
    incremental: true,
  }
}

function emitter(
  name: string,
  calls: string[],
  partialResult: FilePath[] | null = [filePath(`${name}-partial.txt`)],
): QuartzEmitterPluginInstance {
  const instance: QuartzEmitterPluginInstance = {
    name,
    async emit() {
      calls.push(`${name}:emit`)
      return [filePath(`${name}-full.txt`)]
    },
  }

  instance.partialEmit = (_ctx, _content, _resources, changeEvents) => {
    calls.push(`${name}:partial:${changeEvents.map(event => event.path).join(',')}`)
    if (partialResult === null) return null
    return Promise.resolve(partialResult)
  }

  return instance
}

function emitterWithoutPartial(name: string, calls: string[]): QuartzEmitterPluginInstance {
  return {
    name,
    async emit() {
      calls.push(`${name}:emit`)
      return [filePath(`${name}-full.txt`)]
    },
  }
}

test('incremental emit dispatches existing partial emitters', async () => {
  const calls: string[] = []
  const ctx = testCtx([emitter('ContentPage', calls), emitter('ComponentResources', calls)])

  await emitContent(ctx, [], [{ type: 'change', path: filePath('note.md') }])

  assert.deepEqual(calls, ['ComponentResources:partial:note.md', 'ContentPage:partial:note.md'])
})

test('incremental emit falls back to full emit when a plugin has no partial path', async () => {
  const calls: string[] = []
  const ctx = testCtx([
    emitter('ComponentResources', calls, null),
    emitterWithoutPartial('LegacyEmitter', calls),
  ])

  await emitContent(ctx, [], [{ type: 'change', path: filePath('note.md') }])

  assert.deepEqual(calls, ['ComponentResources:partial:note.md', 'LegacyEmitter:emit'])
})
