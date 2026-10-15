import matter from 'gray-matter'
import { toHtml } from 'hast-util-to-html'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { toHast } from 'mdast-util-to-hast'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import fs from 'node:fs/promises'
import { registerHooks } from 'node:module'
import path from 'node:path'
import { unified } from 'unified'
import { visit } from 'unist-util-visit'
import type { QuartzPluginData } from '../plugins/vfile'
import type { BuildCtx } from '../util/ctx'
import { defaultProcessedContent } from '../plugins/vfile'
import { compileBaseConfig } from '../util/base/compile'
import { renderBaseViewsForFile } from '../util/base/render'
import { isFilePath, slugifyFilePath } from '../util/path'
import { isRecord } from '../util/type-guards'

const notePath = process.argv[2]
if (!notePath || !path.resolve(notePath).startsWith(path.resolve('content/places') + path.sep)) {
  throw new Error('Pass a note path inside content/places.')
}
const raw = await fs.readFile(notePath, 'utf8')
const data = matter(raw).data
assert.equal(typeof data.title, 'string')
assert.equal(typeof data.url, 'string')
assert.ok(Array.isArray(data.category) && data.category.includes('[[places to go]]'))
assert.ok(Array.isArray(data.type) && data.type.length > 0)
const url = String(data.url)
const arenaSource = await fs.readFile('content/are.na.md', 'utf8')
const section = arenaSource.split(/^## places\s*$/m)[1]?.split(/^## /m)[0]
assert.ok(section, 'Arena places section is missing')
const lines = section.split('\n')
const starts = lines.flatMap((line, index) => (line.startsWith(`- ${url} -- `) ? [index] : []))
assert.equal(starts.length, 1, 'Expected one Arena places item with the note URL')
const start = starts[0]
let end = start + 1
while (end < lines.length && !lines[end].startsWith('- ')) end++
const fixture = `## places\n\n${lines.slice(start, end).join('\n')}\n`
const out = path.resolve(
  'docs/verification/garden-add-place/artifacts',
  path.basename(notePath, '.md'),
)
await fs.mkdir(out, { recursive: true })
await fs.writeFile(path.join(out, 'arena-input.md'), fixture)
await fs.writeFile(path.join(out, 'place-input.md'), raw)

const hooks = registerHooks({
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
  buildId: 'place-ingestion-verification',
  argv: {
    directory: 'content',
    output: out,
    verbose: false,
    serve: false,
    watch: false,
    port: 0,
    wsPort: 0,
    force: false,
  },
  cfg: {
    configuration: {
      pageTitle: 'Place verification',
      enableSPA: false,
      enablePopovers: false,
      analytics: null,
      ignorePatterns: [],
      defaultDateType: 'created',
      locale: 'en-US',
      baseUrl: 'aarnphm.xyz',
      theme: {
        typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
        cdnCaching: false,
        colors: { lightMode: colors, darkMode: colors },
        fontOrigin: 'local',
      },
    },
    plugins: { transformers: [], filters: [], emitters: [] },
  },
  allFiles: [],
  allSlugs: [],
  incremental: false,
}
try {
  const { Arena } = await import('../plugins/transformers/arena')
  const arenaPath = 'are.na.md'
  assert.ok(isFilePath(arenaPath))
  const [, arenaFile] = defaultProcessedContent({
    slug: slugifyFilePath(arenaPath),
    frontmatter: { title: 'Arena', pageLayout: 'default' },
  })
  const tree = toHast(fromMarkdown(fixture))
  assert.ok(tree?.type === 'root')
  const plugins = Arena().htmlPlugins?.(ctx)
  assert.ok(plugins)
  await unified().use(plugins).run(tree, arenaFile)
  const block = arenaFile.data.arenaData?.channels
    .find(channel => channel.slug === 'places')
    ?.blocks.find(block => block.url === url)
  assert.ok(block, 'Arena transformer did not expose the place')
  assert.equal(block.title, data.title)
  assert.deepEqual(block.tags, data.type)
  let coordinates: number[] | undefined
  if (data.coordinates !== undefined) {
    assert.ok(Array.isArray(data.coordinates) && data.coordinates.length === 2)
    coordinates = data.coordinates.map(Number)
    assert.ok(coordinates.every(Number.isFinite))
    assert.ok(Math.abs(coordinates[0]) <= 90 && Math.abs(coordinates[1]) <= 180)
    assert.deepEqual(block.coordinates, { lat: coordinates[0], lon: coordinates[1] })
  } else {
    assert.equal(block.coordinates, undefined)
  }
  assert.equal(block.metadata?.address, data.address)
  const relativePath = path.relative('content', notePath)
  assert.ok(isFilePath(relativePath))
  const place: QuartzPluginData = {
    slug: slugifyFilePath(relativePath),
    filePath: relativePath,
    relativePath,
    frontmatter: { ...data, title: String(data.title), pageLayout: 'default' },
  }
  const source = await fs.readFile('content/places to go.base', 'utf8')
  const compiled = compileBaseConfig(source, 'content/places to go.base')
  assert.equal(compiled.diagnostics.length, 0)
  const basePath = 'places to go.base'
  assert.ok(isFilePath(basePath))
  const base: QuartzPluginData = {
    slug: slugifyFilePath(basePath),
    basesConfig: compiled.config,
    basesExpressions: compiled.expressions,
    frontmatter: { title: 'Places', pageLayout: 'default' },
  }
  const rendered = renderBaseViewsForFile(base, [place])
  for (const view of rendered.views) {
    assert.equal(view.resultCount, 1, `Place missing from Base view ${view.view.name}`)
    await fs.writeFile(path.join(out, `${view.view.name}.html`), toHtml(view.tree))
  }
  const map = rendered.views.find(view => view.view.type === 'map')
  assert.ok(map)
  const markers: Record<string, unknown>[] = []
  visit(map.tree, 'element', node => {
    const payload = node.properties.dataMarkers
    if (typeof payload !== 'string') return
    const parsed: unknown = JSON.parse(payload)
    assert.ok(Array.isArray(parsed))
    for (const marker of parsed) {
      assert.ok(isRecord(marker))
      markers.push(marker)
    }
  })
  assert.equal(markers.length, coordinates ? 1 : 0)
  if (coordinates) {
    assert.equal(markers[0].lat, coordinates[0])
    assert.equal(markers[0].lon, coordinates[1])
  }
  await fs.writeFile(
    path.join(out, 'results.json'),
    JSON.stringify(
      {
        command: ['pnpm', 'exec', 'tsx', 'quartz/scripts/verify-place.ts', notePath],
        input: notePath,
        arena: block,
        markers,
        passed: true,
      },
      null,
      2,
    ),
  )
  console.log(
    `Verified ${data.title}: Arena item, Base table inclusion, and ${markers.length} map marker(s).`,
  )
} finally {
  hooks.deregister()
}
