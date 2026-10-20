import type { Root as HtmlRoot } from 'hast'
import type { Processor } from 'unified'
import { toHtml } from 'hast-util-to-html'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { VFile } from 'vfile'
import type { QuartzTransformerPluginInstance } from '../types/plugin'
import type { BuildCtx } from '../util/ctx'
import { wikilink, wikilinkFromMarkdown } from '../extensions/micromark-extension-ofm-wikilinks'
import { CodeViewer } from '../plugins/transformers/codeViewer'
import { ProcessedContent } from '../plugins/vfile'
import { isFilePath, type FilePath } from '../util/path'
import {
  canReuseProcessedHtml,
  parseMarkdown,
  resetProcessedContentCache,
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

test('editing a transcluded code file invalidates the unchanged Markdown parse', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'quartz-code-dependency-'))
  t.after(async () => {
    resetProcessedContentCache()
    await rm(directory, { recursive: true, force: true })
  })
  const fp = join(directory, 'note.md')
  const dependency = 'example.py'
  if (!isFilePath(fp) || !isFilePath(dependency)) throw new Error('invalid fixture path')
  await writeFile(fp, '![[example.py]]\n')
  await writeFile(join(directory, 'example.py'), 'print("before")\n')
  const ctx = testCtx(directory, 'code')
  ctx.allFiles = [dependency]
  ctx.cfg.plugins.transformers = [
    {
      name: 'Wikilinks',
      markdownPlugins: () => [
        function (this: Processor) {
          const data = this.data()
          ;(data.micromarkExtensions ??= []).push(wikilink())
          ;(data.fromMarkdownExtensions ??= []).push(wikilinkFromMarkdown())
        },
      ],
    },
    CodeViewer(),
  ]
  resetProcessedContentCache()

  const [first] = await parseMarkdown(ctx, [fp])
  assert.deepEqual(first[1].data.codeDependencies, ['example.py'])
  assert.match(toHtml(first[0]), /before/)
  const [cached] = await parseMarkdown(ctx, [fp])
  assert.equal(cached, first)

  await writeFile(join(directory, 'example.py'), 'print("after dependency edit")\n')
  const [rebuilt] = await parseMarkdown(ctx, [fp])
  assert.notEqual(rebuilt, first)
  assert.match(toHtml(rebuilt[0]), /after dependency edit/)
  assert.doesNotMatch(toHtml(rebuilt[0]), /before/)
})

test('notebook files render Markdown, code, and stored outputs through the file parser', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'quartz-notebook-parse-'))
  t.after(async () => {
    resetProcessedContentCache()
    await rm(directory, { recursive: true, force: true })
  })
  const fp = join(directory, 'fixture.ipynb')
  if (!isFilePath(fp)) throw new Error('invalid fixture path')
  const source = JSON.stringify({
    metadata: { language_info: { name: 'python' } },
    cells: [
      { cell_type: 'markdown', source: '# Heading\n\nbody text' },
      {
        cell_type: 'code',
        source: 'print("hi")',
        execution_count: 1,
        outputs: [{ output_type: 'stream', name: 'stdout', text: ['hi\n'] }],
      },
      {
        cell_type: 'code',
        source: '1 + 1',
        execution_count: 2,
        outputs: [
          { output_type: 'execute_result', data: { 'text/plain': '2' }, execution_count: 2 },
        ],
      },
    ],
  })
  await writeFile(fp, source)
  const ctx = testCtx(directory, 'notebook')
  ctx.cfg.plugins.transformers = []
  resetProcessedContentCache()
  const parsed = await parseMarkdown(ctx, [fp])
  assert.equal(parsed.length, 1)
  const html = toHtml(parsed[0][0], { allowDangerousHtml: true })

  const artifacts = process.env.QUARTZ_PARSE_TEST_ARTIFACTS
  if (artifacts) {
    await mkdir(artifacts, { recursive: true })
    await writeFile(join(artifacts, 'fixture.ipynb'), source)
    await writeFile(join(artifacts, 'fixture.html'), html)
  }

  assert.match(html, /<h1>Heading<\/h1>/)
  assert.match(html, /<p>body text<\/p>/)
  assert.match(html, /<code class="language-python">print\("hi"\)/)
  assert.match(html, /data-output-name="stdout"><samp>hi<\/samp>/)
  assert.match(html, /data-output-name="result"><samp>2<\/samp>/)
  assert.match(html, /id="cell-1"/)
  assert.match(html, /id="cell-2"/)
  assert.match(html, /id="notebook-md-1"/)
})

test('editing stylesheet or inline client script files in parseSourceInputs does not wipe cached Markdown parses', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'quartz-parse-style-script-'))
  const previousInputs = globalThis.__quartzParseSourceInputs
  t.after(async () => {
    globalThis.__quartzParseSourceInputs = previousInputs
    resetProcessedContentCache()
    await rm(directory, { recursive: true, force: true })
  })

  const fp = join(directory, 'note.md') as FilePath
  const styleFile = join(directory, 'widget.scss')
  const inlineScriptFile = join(directory, 'widget.inline.ts')
  const parserSourceFile = join(directory, 'parser-helper.ts')
  await writeFile(fp, '# cached note\n')
  await writeFile(styleFile, '.widget { color: red; }\n')
  await writeFile(inlineScriptFile, 'console.log("v1")\n')
  await writeFile(parserSourceFile, 'export const v = 1\n')

  globalThis.__quartzParseSourceInputs = [styleFile, inlineScriptFile, parserSourceFile]
  resetProcessedContentCache()

  const [first] = await parseMarkdown(testCtx(directory, 'first'), [fp])
  assert.equal(hasMarker(first, 'first'), true)

  // Editing an SCSS file or inline script imported by an MDX component must not clear the AST cache
  await writeFile(styleFile, '.widget { color: blue; padding: 4px; }\n')
  await writeFile(inlineScriptFile, 'console.log("v2-updated")\n')

  const [afterClientAssetEdit] = await parseMarkdown(testCtx(directory, 'second'), [fp])
  assert.equal(afterClientAssetEdit, first)
  assert.equal(hasMarker(afterClientAssetEdit, 'first'), true)

  // Editing an actual parser source file must invalidate the AST cache when unpinned
  await writeFile(parserSourceFile, 'export const v = 222\n')
  const [afterParserEdit] = await parseMarkdown(testCtx(directory, 'second'), [fp])
  assert.notEqual(afterParserEdit, first)
  assert.equal(hasMarker(afterParserEdit, 'second'), true)
})

test('pinned bundle parseSourceSalt prevents mid-build disk edits from wiping cached Markdown parses before bundle reload', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'quartz-parse-pinned-salt-'))
  const previousInputs = globalThis.__quartzParseSourceInputs
  const previousSalt = (globalThis as { __quartzParseSourceSalt?: string }).__quartzParseSourceSalt
  t.after(async () => {
    globalThis.__quartzParseSourceInputs = previousInputs
    ;(globalThis as { __quartzParseSourceSalt?: string }).__quartzParseSourceSalt = previousSalt
    resetProcessedContentCache()
    await rm(directory, { recursive: true, force: true })
  })

  const fp = join(directory, 'note.md') as FilePath
  const parserSourceFile = join(directory, 'transformer.ts')
  await writeFile(fp, '# note\n')
  await writeFile(parserSourceFile, 'export const version = 1\n')

  globalThis.__quartzParseSourceInputs = [parserSourceFile]
  ;(globalThis as { __quartzParseSourceSalt?: string }).__quartzParseSourceSalt = 'bundle-epoch-1'
  resetProcessedContentCache()

  const [first] = await parseMarkdown(testCtx(directory, 'epoch-1'), [fp])
  assert.equal(hasMarker(first, 'epoch-1'), true)

  // Simulate a disk edit to transformer.ts while a content rebuild is running on bundle-epoch-1
  await writeFile(parserSourceFile, 'export const version = 200\n')
  const [duringContentRebuild] = await parseMarkdown(testCtx(directory, 'epoch-1-stale'), [fp])
  assert.equal(duringContentRebuild, first)

  // Once rebuildQuartzBundle finishes and updates __quartzParseSourceSalt, the cache invalidates cleanly
  ;(globalThis as { __quartzParseSourceSalt?: string }).__quartzParseSourceSalt = 'bundle-epoch-2'
  const [afterBundleReload] = await parseMarkdown(testCtx(directory, 'epoch-2'), [fp])
  assert.notEqual(afterBundleReload, first)
  assert.equal(hasMarker(afterBundleReload, 'epoch-2'), true)
})
