import type { Element } from 'hast'
import { h } from 'hastscript'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { registerHooks } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { StreamEntry } from '../transformers/stream'
import type { ProcessedContent } from '../vfile'
import { emitContent } from '../../processors/emit'
import { compileBaseConfig } from '../../util/base/compile'
import { isFilePath, isFullSlug } from '../../util/path'
import { staticCssBundleKey, staticJsBundleKey } from '../../util/resource-bundles'
import { getStaticResourcesFromPlugins } from '../../util/static-resources'
import { defaultProcessedContent } from '../vfile'
import { renderedStreamEntries } from './streamRenderedText'

function note(name: string, children: Element[]): ProcessedContent {
  const path = `content/${name}.md`
  const relativePath = `${name}.md`
  if (!isFullSlug(name) || !isFilePath(path) || !isFilePath(relativePath)) throw new Error(name)
  const result = defaultProcessedContent({
    slug: name,
    filePath: path,
    relativePath,
    frontmatter: { title: name, pageLayout: 'default', tags: [] },
  })
  result[0].children = children
  result[1].data.htmlAst = result[0]
  return result
}

function embed(target: string): Element {
  return h('blockquote', { className: ['transclude'] }, [
    h('a', { dataSlug: target, href: `./${target}` }, target),
  ])
}

test('stream rebuilds every daily fragment and search text after a nested target changes', async t => {
  // Quartz loads browser scripts and styles as text during server rendering.
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
  t.after(() => assets.deregister())
  const { StreamPage } = await import('./streamPage')
  const { StreamIndex } = await import('./streamIndex')
  const { default: collapseHeaderStyle } =
    await import('../../components/styles/collapseHeader.inline.scss')
  const output = await mkdtemp(join(tmpdir(), 'quartz-stream-rebuild-'))
  t.after(() => rm(output, { recursive: true, force: true }))
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
  const ctx: BuildCtx = {
    buildId: 'initial',
    argv: {
      directory: 'content',
      output,
      verbose: false,
      serve: false,
      watch: true,
      port: 0,
      wsPort: 0,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'stream fixture',
        enableSPA: false,
        enablePopovers: false,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'created',
        locale: 'en-US',
        baseUrl: 'example.com',
        theme: {
          typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
          cdnCaching: false,
          colors: { lightMode: colors, darkMode: colors },
          fontOrigin: 'local',
        },
      },
      plugins: { transformers: [], filters: [], emitters: [StreamPage(), StreamIndex()] },
    },
    allFiles: [],
    allSlugs: [],
    incremental: true,
    extractedStaticResources: new Map([
      [staticCssBundleKey(collapseHeaderStyle), 'static/collapse.css'],
    ]),
  }
  ctx.argv.watch = false
  assert.deepEqual(getStaticResourcesFromPlugins(ctx).js, [])
  ctx.argv.watch = true
  const scripts = getStaticResourcesFromPlugins(ctx).js
  assert.equal(scripts.length, 1)
  for (const resource of scripts) {
    if (resource.contentType !== 'inline') throw new Error('expected inline reload script')
    ctx.extractedStaticResources?.set(
      staticJsBundleKey(resource.loadTime, [resource.script]),
      'static/reload.js',
    )
  }
  const base = note('library', [])
  const compiled = compileBaseConfig(
    'filters: file.inFolder("library")\nviews:\n  - type: table\n    name: books\n    order: [title]',
    'library.base',
  )
  Object.assign(base[1].data, {
    bases: true,
    basesConfig: compiled.config,
    basesExpressions: compiled.expressions,
    basesDiagnostics: compiled.diagnostics,
  })
  let book = note('library/book', [])
  book[1].data.frontmatter = { title: 'original book title', pageLayout: 'default' }
  const entries: StreamEntry[] = ['2026-09-09', '2026-09-08'].map(date => {
    const node = embed('nested')
    node.data = { position: {} }
    Object.assign(node.data, { streamEntryId: date, streamEntryContentIndex: 0 })
    const view = embed('library/books')
    view.data = { position: {} }
    Object.assign(view.data, { streamEntryId: date, streamEntryContentIndex: 1 })
    return { id: date, date, metadata: {}, content: [node, view] }
  })
  const originalEntries = structuredClone(entries)
  const stream = note(
    'stream',
    entries.flatMap(entry => entry.content).filter(node => node.type === 'element'),
  )
  stream[1].data.streamData = { entries }
  const nested = note('nested', [embed('source')])
  let source = note('source', [h('p', 'before nested edit')])
  await emitContent(ctx, [source, nested, base, book, stream])

  const assertRendered = async (expected: string, obsolete: string) => {
    const root = await readFile(join(output, 'stream.html'), 'utf8')
    assert.ok(root.includes('./static/reload.js'))
    assert.equal(renderedStreamEntries(root).size, 1)
    assert.match(root, /data-stream-lazy="true"/)
    assert.ok(root.includes(expected))
    assert.ok(!root.includes(obsolete))
    for (const day of ['09', '08']) {
      const html = await readFile(join(output, `stream/on/2026/09/${day}.html`), 'utf8')
      const rendered = renderedStreamEntries(html)
      assert.equal(rendered.size, 1)
      assert.ok(rendered.get(`2026-09-${day}`)?.content.includes(expected))
      assert.ok(!html.includes(obsolete))
      assert.ok(!html.includes('transclude-inner'))
    }
    const manifest = await readFile(join(output, 'streams.jsonl'), 'utf8')
    assert.equal(manifest.split('\n').filter(line => line.includes(expected)).length, 2)
    assert.ok(!manifest.includes(obsolete))
  }
  await assertRendered('before nested edit', 'after nested edit')
  assert.ok((await readFile(join(output, 'streams.jsonl'), 'utf8')).includes('original book title'))

  for (const [index, message] of ['after nested edit', 'after second edit'].entries()) {
    const previous = source[1]
    source = note('source', [h('p', message)])
    ctx.buildId = `rebuild-${index}`
    const path = source[1].data.relativePath
    if (!path) throw new Error('missing source path')
    await emitContent(
      ctx,
      [source, nested, base, book, stream],
      [{ type: 'change', path, file: source[1], previousFile: previous }],
    )
    await assertRendered(message, index === 0 ? 'before nested edit' : 'after nested edit')
  }
  assert.deepEqual(entries, originalEntries)

  const previousBook = book[1]
  book = note('library/book', [])
  book[1].data.frontmatter = { title: 'updated book title', pageLayout: 'default' }
  const bookPath = book[1].data.relativePath
  if (!bookPath) throw new Error('missing book path')
  ctx.buildId = 'updated-base-member'
  await emitContent(
    ctx,
    [source, nested, base, book, stream],
    [{ type: 'change', path: bookPath, file: book[1], previousFile: previousBook }],
  )
  await assertRendered('updated book title', 'original book title')

  let previousStream = stream
  for (const remaining of [entries.slice(0, 1), []]) {
    const revised = note(
      'stream',
      remaining.flatMap(entry => entry.content).filter(node => node.type === 'element'),
    )
    revised[1].data.streamData = { entries: remaining }
    const path = revised[1].data.relativePath
    if (!path) throw new Error('missing stream path')
    ctx.buildId = `remove-${remaining.length}`
    await emitContent(
      ctx,
      [source, nested, base, book, revised],
      [{ type: 'change', path, file: revised[1], previousFile: previousStream[1] }],
    )
    const manifest = await readFile(join(output, 'streams.jsonl'), 'utf8')
    assert.equal(manifest.split('\n').filter(Boolean).length, remaining.length)
    assert.equal(
      renderedStreamEntries(await readFile(join(output, 'stream.html'), 'utf8')).size,
      remaining.length,
    )
    for (const day of remaining.length === 0 ? ['09', '08'] : ['08']) {
      await assert.rejects(readFile(join(output, `stream/on/2026/09/${day}.html`)), {
        code: 'ENOENT',
      })
    }
    previousStream = revised
  }
})
