import type { Root as HtmlRoot } from 'hast'
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { VFile } from 'vfile'
import type { QuartzTransformerPluginInstance } from '../types/plugin'
import type { BuildCtx } from '../util/ctx'
import type { FilePath } from '../util/path'
import { ProcessedContent } from '../plugins/vfile'
import {
  canReuseProcessedHtml,
  HTML_PARSE_CHUNK_SIZE,
  parseMarkdown,
  parseWorkerConcurrency,
  resetProcessedContentCache,
  TEXT_PARSE_CHUNK_SIZE,
  titleOnlyFrontmatterChange,
} from './parse'

function processedContent(source: string, frontmatter: Record<string, unknown>): ProcessedContent {
  const tree: HtmlRoot = { type: 'root', children: [] }
  const file = new VFile(source)
  file.data.frontmatter = { title: 'note', pageLayout: 'default', ...frontmatter }
  return [tree, file]
}

function parsedFile(source: string, frontmatter: Record<string, unknown>): VFile {
  const file = new VFile(source)
  file.data.frontmatter = { title: 'note', pageLayout: 'default', ...frontmatter }
  return file
}

function markerTransformer(id: string): QuartzTransformerPluginInstance {
  return {
    name: `Marker:${id}`,
    htmlPlugins: () => [
      () => (tree: HtmlRoot) => {
        tree.children.push({ type: 'element', tagName: 'span', properties: { id }, children: [] })
      },
    ],
  }
}

function testCtx(directory: string, marker: string): BuildCtx {
  const colorScheme = {
    light: '#ffffff',
    lightgray: '#eeeeee',
    gray: '#999999',
    darkgray: '#555555',
    dark: '#111111',
    secondary: '#0055aa',
    tertiary: '#aa5500',
    highlight: '#ffffcc',
    textHighlight: '#ffff00',
  }
  return {
    buildId: marker,
    argv: {
      directory,
      verbose: false,
      output: join(directory, 'public'),
      serve: false,
      watch: true,
      port: 8080,
      wsPort: 3001,
      force: false,
      concurrency: 1,
    },
    cfg: {
      configuration: {
        pageTitle: 'test',
        enableSPA: false,
        enablePopovers: false,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'created',
        theme: {
          typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
          cdnCaching: false,
          colors: { lightMode: colorScheme, darkMode: colorScheme },
          fontOrigin: 'local',
        },
        locale: 'en-US',
      },
      plugins: { transformers: [markerTransformer(marker)], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

function hasMarker(content: ProcessedContent, id: string): boolean {
  return content[0].children.some(child => child.type === 'element' && child.properties?.id === id)
}

test('parse worker concurrency avoids worker overhead for tiny rebuilds', () => {
  assert.equal(parseWorkerConcurrency(0, 10), 1)
  assert.equal(parseWorkerConcurrency(1, 10), 1)
})

test('parse worker concurrency uses html jobs to balance AST to HAST work', () => {
  assert.equal(parseWorkerConcurrency(TEXT_PARSE_CHUNK_SIZE - 1, 10), 8)
  assert.equal(parseWorkerConcurrency(TEXT_PARSE_CHUNK_SIZE, 10), 8)
  assert.equal(parseWorkerConcurrency(HTML_PARSE_CHUNK_SIZE * 14, 10), 10)
})

test('processed html can be reused for title-only frontmatter changes', () => {
  const previous = processedContent('---\ntitle: old\n---\n# Body\n', { title: 'old' })
  const current = parsedFile('---\ntitle: new\n---\n# Body\n', { title: 'new' })

  assert.equal(canReuseProcessedHtml(current, previous), true)
})

test('title-only frontmatter change returns the current title', () => {
  assert.equal(
    titleOnlyFrontmatterChange('---\ntitle: new\n---\n# Body\n', '---\ntitle: old\n---\n# Body\n'),
    'new',
  )
})

test('title-only frontmatter change is blocked by body changes', () => {
  assert.equal(
    titleOnlyFrontmatterChange(
      '---\ntitle: new\n---\n# Changed\n',
      '---\ntitle: old\n---\n# Body\n',
    ),
    undefined,
  )
})

test('processed html reuse is blocked by body changes', () => {
  const previous = processedContent('---\ntitle: old\n---\n# Body\n', { title: 'old' })
  const current = parsedFile('---\ntitle: new\n---\n# Other\n', { title: 'new' })

  assert.equal(canReuseProcessedHtml(current, previous), false)
})

test('processed html reuse is blocked by html-derived frontmatter changes', () => {
  const previous = processedContent('---\ndescription: old\n---\n# Body\n', { description: 'old' })
  const current = parsedFile('---\ndescription: new\n---\n# Body\n', { description: 'new' })

  assert.equal(canReuseProcessedHtml(current, previous), false)
})

test('processed content cache can be reset after source changes', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'quartz-parse-cache-'))
  t.after(async () => {
    resetProcessedContentCache()
    await rm(directory, { recursive: true, force: true })
  })

  const fp = join(directory, 'note.md') as FilePath
  await writeFile(fp, '# note\n')
  resetProcessedContentCache()

  const first = await parseMarkdown(testCtx(directory, 'first'), [fp])
  assert.equal(hasMarker(first[0], 'first'), true)

  const cached = await parseMarkdown(testCtx(directory, 'second'), [fp])
  assert.equal(hasMarker(cached[0], 'first'), true)
  assert.equal(hasMarker(cached[0], 'second'), false)

  resetProcessedContentCache()
  const reparsed = await parseMarkdown(testCtx(directory, 'second'), [fp])
  assert.equal(hasMarker(reparsed[0], 'second'), true)
})
