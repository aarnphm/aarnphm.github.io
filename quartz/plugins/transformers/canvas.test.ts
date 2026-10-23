import type { Root } from 'hast'
import { h } from 'hastscript'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { registerHooks } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test, { type TestContext } from 'node:test'
import { unified } from 'unified'
import { VFile } from 'vfile'
import type { BuildCtx } from '../../util/ctx'
import { isFilePath, isFullSlug, simplifySlug } from '../../util/path'
import { staticCssBundleKey } from '../../util/resource-bundles'
import { JsonCanvas } from './canvas'
import { Description } from './description'

async function loadCrawlLinks(t: TestContext) {
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
  return (await import('./links')).CrawlLinks
}

function slug(value: string) {
  assert.ok(isFullSlug(value))
  return value
}

function testContext(): BuildCtx {
  const colors = {
    light: '#ffffff',
    lightgray: '#eeeeee',
    gray: '#888888',
    darkgray: '#444444',
    dark: '#000000',
    secondary: '#0055aa',
    tertiary: '#228855',
    highlight: '#dddddd',
    textHighlight: '#ffffaa',
  }
  return {
    buildId: 'canvas-links',
    argv: {
      directory: 'content',
      output: 'public',
      verbose: false,
      serve: false,
      watch: false,
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
        locale: 'fr-FR',
        theme: {
          typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
          fontOrigin: 'local',
          cdnCaching: false,
          colors: { lightMode: colors, darkMode: colors },
        },
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: ['fr/index', 'fr/episode-1', 'fr/phrases', 'fr/parcours'].map(slug),
    allFiles: [],
    incremental: false,
  }
}

test('canvas file nodes and wikilinks remain backlinks after link processing', async t => {
  const CrawlLinks = await loadCrawlLinks(t)
  const ctx = testContext()
  const file = new VFile({
    path: 'content/fr/parcours.canvas',
    value: JSON.stringify({
      nodes: [
        { id: 'index', type: 'file', file: 'fr/index.md', x: 0, y: 0, width: 200, height: 120 },
        {
          id: 'episode',
          type: 'file',
          file: 'fr/episode-1.md',
          x: 300,
          y: 0,
          width: 200,
          height: 120,
        },
        {
          id: 'links',
          type: 'text',
          text: '[[fr/phrases]] et [[fr/episode-1]]',
          x: 600,
          y: 0,
          width: 200,
          height: 120,
        },
      ],
      edges: [],
    }),
  })
  file.data.slug = slug('fr/parcours')
  file.data.jsonCanvas = true
  const tree: Root = { type: 'root', children: [] }
  await unified()
    .use(JsonCanvas().htmlPlugins?.(ctx) ?? [])
    .use(CrawlLinks().htmlPlugins?.(ctx) ?? [])
    .run(tree, file)

  assert.deepEqual(file.data.links, ['fr/', 'fr/episode-1', 'fr/phrases'])
  assert.equal(file.data.canvas?.data.nodeMap.size, 3)
})

test('ordinary notes discard stale outgoing links when reprocessed', async t => {
  const CrawlLinks = await loadCrawlLinks(t)
  const ctx = testContext()
  const file = new VFile({ path: 'content/fr/index.md', value: '' })
  file.data.slug = slug('fr/index')
  file.data.links = [simplifySlug(slug('fr/episode-1'))]
  const tree: Root = { type: 'root', children: [] }
  await unified()
    .use(CrawlLinks().htmlPlugins?.(ctx) ?? [])
    .run(tree, file)

  assert.deepEqual(file.data.links, [])
})

test('canvas links open the interactive page and use its canonical backlink slug', async t => {
  const CrawlLinks = await loadCrawlLinks(t)
  const ctx = testContext()
  const file = new VFile({ path: 'content/fr/index.md', value: '' })
  file.data.slug = slug('fr/index')
  const link = h('a', { href: 'fr/parcours.canvas' }, 'carte des notes')
  const tree: Root = { type: 'root', children: [link] }
  await unified()
    .use(CrawlLinks().htmlPlugins?.(ctx) ?? [])
    .run(tree, file)

  assert.equal(link.properties.href, '../fr/parcours')
  assert.deepEqual(file.data.links, ['fr/parcours'])
})

test('incremental canvas publishing creates and refreshes the page and metadata', async t => {
  const CrawlLinks = await loadCrawlLinks(t)
  const { emitPartialEmitter } = await import('../../processors/emit')
  const { CanvasPage } = await import('../emitters/canvas')
  const { default: collapseHeaderStyle } =
    await import('../../components/styles/collapseHeader.inline.scss')
  const output = await mkdtemp(path.join(tmpdir(), 'quartz-canvas-'))
  t.after(() => rm(output, { recursive: true, force: true }))
  const ctx = testContext()
  ctx.argv.output = output
  ctx.incremental = true
  ctx.cfg.plugins.emitters = [CanvasPage()]
  ctx.extractedStaticResources = new Map([
    [staticCssBundleKey(collapseHeaderStyle), 'static/collapse.css'],
  ])
  const filePath = 'content/fr/parcours.canvas'
  const relativePath = 'fr/parcours.canvas'
  assert.ok(isFilePath(filePath) && isFilePath(relativePath))
  const file = new VFile({ path: filePath })
  file.data.slug = slug('fr/parcours')
  file.data.filePath = filePath
  file.data.relativePath = relativePath
  file.data.jsonCanvas = true
  const tree: Root = { type: 'root', children: [] }

  for (const [position, target] of ['fr/index.md', 'fr/phrases.md'].entries()) {
    ctx.buildId = `canvas-${position}`
    file.value = JSON.stringify({
      nodes: [{ id: 'note', type: 'file', file: target, x: 0, y: 0, width: 200, height: 120 }],
      edges: [],
    })
    await unified()
      .use(JsonCanvas().htmlPlugins?.(ctx) ?? [])
      .use(CrawlLinks().htmlPlugins?.(ctx) ?? [])
      .run(tree, file)
    await emitPartialEmitter(
      ctx,
      [[tree, file]],
      [{ type: position === 0 ? 'add' : 'change', path: filePath, file }],
      'CanvasPage',
    )

    const html = await readFile(path.join(output, 'fr/parcours.html'), 'utf8')
    assert.match(html, /data-canvas="\/fr\/parcours\.canvas"/)
    assert.match(html, /data-meta="\/fr\/parcours\.meta\.json"/)
    const metadata: unknown = JSON.parse(
      await readFile(path.join(output, 'fr/parcours.meta.json'), 'utf8'),
    )
    const targetSlug = target.replace(/\.md$/, '')
    assert.deepEqual(metadata, {
      note: { slug: targetSlug, href: targetSlug, displayName: path.basename(target, '.md') },
    })
  }
})

test('Description resolves canvas descriptions from content/description.json, preserves canvas searchable text, and omits default fallback text when absent', async t => {
  const { writeFile } = await import('node:fs/promises')
  const { CanvasPage } = await import('../emitters/canvas')
  const { emitPartialEmitter } = await import('../../processors/emit')
  const { default: collapseHeaderStyle } =
    await import('../../components/styles/collapseHeader.inline.scss')
  const dir = await mkdtemp(path.join(tmpdir(), 'quartz-canvas-desc-'))
  const output = await mkdtemp(path.join(tmpdir(), 'quartz-canvas-desc-out-'))
  t.after(async () => {
    await rm(dir, { recursive: true, force: true })
    await rm(output, { recursive: true, force: true })
  })

  await writeFile(
    path.join(dir, 'description.json'),
    JSON.stringify([
      {
        '[[singularity.canvas]]':
          'systems map connecting [[fr/index|autoregressive models]] to inference engines',
      },
    ]),
    'utf8',
  )

  const ctx = testContext()
  ctx.argv.directory = dir
  ctx.argv.output = output
  ctx.incremental = true
  ctx.cfg.plugins.emitters = [CanvasPage()]
  ctx.extractedStaticResources = new Map([
    [staticCssBundleKey(collapseHeaderStyle), 'static/collapse.css'],
  ])

  const canvasPath = 'singularity.canvas'
  assert.ok(isFilePath(canvasPath))
  const canvasFile = new VFile({
    path: path.join(dir, canvasPath),
    value: JSON.stringify({
      nodes: [
        {
          id: 'n1',
          type: 'text',
          text: 'speculative decoding tokens',
          x: 0,
          y: 0,
          width: 200,
          height: 100,
        },
      ],
      edges: [],
    }),
  })
  canvasFile.data.slug = slug('singularity')
  canvasFile.data.filePath = canvasPath
  canvasFile.data.relativePath = canvasPath
  canvasFile.data.jsonCanvas = true
  canvasFile.data.frontmatter = { title: 'singularity', pageLayout: 'default' }
  const tree: Root = { type: 'root', children: [] }

  await unified()
    .use(JsonCanvas().htmlPlugins?.(ctx) ?? [])
    .use(Description().htmlPlugins?.(ctx) ?? [])
    .run(tree, canvasFile)

  assert.equal(canvasFile.data.text, 'speculative decoding tokens')
  assert.equal(
    canvasFile.data.rawDescription,
    'systems map connecting [[fr/index|autoregressive models]] to inference engines',
  )
  assert.match(
    canvasFile.data.description ?? '',
    /<a href="\/fr\/index" class="internal">autoregressive models<\/a>/,
  )
  assert.ok(canvasFile.data.links?.includes(simplifySlug(slug('fr/index'))))

  await emitPartialEmitter(
    ctx,
    [[tree, canvasFile]],
    [{ type: 'add', path: canvasPath, file: canvasFile }],
    'CanvasPage',
  )
  const preview = JSON.parse(
    await readFile(path.join(output, 'static/preview/singularity.json'), 'utf8'),
  ) as { description: string | null }
  assert.equal(
    preview.description,
    'systems map connecting autoregressive models to inference engines',
  )

  // A canvas or empty note not listed in description.json gets an empty description, never 'aucune description fournie'
  const unlistedCanvas = new VFile({
    path: path.join(dir, 'unlisted.canvas'),
    value: JSON.stringify({
      nodes: [
        { id: 'n1', type: 'text', text: 'raw node label', x: 0, y: 0, width: 100, height: 80 },
      ],
      edges: [],
    }),
  })
  unlistedCanvas.data.slug = slug('unlisted')
  unlistedCanvas.data.jsonCanvas = true
  await unified()
    .use(JsonCanvas().htmlPlugins?.(ctx) ?? [])
    .use(Description().htmlPlugins?.(ctx) ?? [])
    .run({ type: 'root', children: [] } as Root, unlistedCanvas)

  assert.equal(unlistedCanvas.data.text, 'raw node label')
  assert.equal(unlistedCanvas.data.description, '')
  assert.equal(unlistedCanvas.data.rawDescription, '')
})
