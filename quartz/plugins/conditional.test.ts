import type { Root as HastRoot } from 'hast'
import type { Root as MdRoot } from 'mdast'
import assert from 'node:assert/strict'
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import test from 'node:test'
import type { QuartzComponent } from '../types/component'
import type { QuartzEmitterPluginInstance, QuartzTransformerPluginInstance } from '../types/plugin'
import type { Argv, BuildCtx } from '../util/ctx'
import { emitContent, emitPartialEmitter } from '../processors/emit'
import { filterContent } from '../processors/filter'
import {
  createFileParser,
  createHtmlProcessor,
  createMarkdownParser,
  createMdProcessor,
} from '../processors/parse'
import { isFilePath } from '../util/path'
import { getStaticResourcesFromPlugins } from '../util/static-resources'
import { Conditional, ConditionalTransform } from './conditional'
import { write } from './emitters/helpers'
import { RemoveDrafts } from './filters/draft'
import { FrontMatter } from './transformers/frontmatter'

const artifacts = path.resolve('docs/verification/conditional-plugins/artifacts')
const source = '---\ntitle: Fixture\ndraft: true\n---\n\nseed\n'

function transform(label: string): QuartzTransformerPluginInstance {
  return {
    name: label,
    textTransform(_ctx, source) {
      return source.replace('seed', `seed:${label}`)
    },
    markdownPlugins() {
      return [
        () => (tree: MdRoot) => {
          tree.children.push({ type: 'paragraph', children: [{ type: 'text', value: label }] })
        },
      ]
    },
    htmlPlugins() {
      return [
        () => (tree: HastRoot) => {
          tree.children.push({
            type: 'element',
            tagName: 'span',
            properties: {},
            children: [{ type: 'text', value: label }],
          })
        },
      ]
    },
    externalResources() {
      return { css: [{ content: `.${label} { color: red; }`, inline: true }] }
    },
  }
}

const component: QuartzComponent = () => null
component.css = '.conditional-component { color: blue; }'

const conditionalEmitter: QuartzEmitterPluginInstance = {
  name: 'ConditionalOutput',
  async *emit(ctx) {
    yield write({ ctx, slug: 'conditional', ext: '.html', content: 'initial' })
  },
  partialEmit(ctx, _content, _resources, changes) {
    if (changes.length === 0) return null
    return (async function* () {
      yield write({ ctx, slug: 'conditional', ext: '.html', content: 'partial' })
    })()
  },
  getQuartzComponents() {
    return [component]
  },
  externalResources() {
    return { css: [{ content: '.emitter { color: green; }', inline: true }] }
  },
}

// A real file emitter without partialEmit exercises Quartz's fallback path.
const snapshot: QuartzEmitterPluginInstance = {
  name: 'Snapshot',
  async emit(ctx, content, resources) {
    const result = {
      incremental: ctx.incremental,
      text: content.map(([, file]) => String(file.value)),
      trees: content.map(([tree]) => tree),
      css: resources.css,
    }
    return [
      await write({
        ctx,
        slug: 'snapshot',
        ext: '.json',
        content: JSON.stringify(result, null, 2),
      }),
    ]
  },
}

const before = transform('before')
const gated = transform('gated')
const after = transform('after')
const transformers = Conditional({
  plugins: [FrontMatter(), before, { plugin: gated, dev: false }, { plugin: after, dev: true }],
})

test('fallbacks own every hook and retain partial-emission fallback in both directions', async () => {
  const filePath = path.join(artifacts, 'fallback.md')
  assert.ok(isFilePath(filePath))
  await mkdir(artifacts, { recursive: true })
  await writeFile(filePath, source)
  const fallbackComponent: QuartzComponent = () => null
  fallbackComponent.css = '.fallback-component { color: green; }'
  const fallbackEmitter: QuartzEmitterPluginInstance = {
    ...snapshot,
    getQuartzComponents: () => [fallbackComponent],
    externalResources: () => ({ css: [{ content: '.fallback-emitter {}', inline: true }] }),
  }
  const fallbackTransformer: QuartzTransformerPluginInstance = {
    name: 'Fallback',
    htmlPlugins: transform('fallback').htmlPlugins,
  }
  const configuredTransformers = Conditional({
    plugins: [
      FrontMatter(),
      { plugin: gated, dev: false, fallback: fallbackTransformer },
      { plugin: { name: 'MissingHooks' }, dev: false, fallback: transform('extra') },
    ],
  })
  const configuredEmitters = Conditional({
    plugins: [{ plugin: conditionalEmitter, dev: false, fallback: fallbackEmitter }],
  })
  const configuredFilters = Conditional({
    plugins: [
      {
        plugin: RemoveDrafts(),
        dev: false,
        fallback: {
          name: 'OnlyDrafts',
          shouldPublish: (_ctx, [, file]) => file.data.frontmatter?.draft === true,
        },
      },
    ],
  })
  for (const force of [false, true]) {
    const output = path.join(artifacts, force ? 'fallback-forced' : 'fallback-watch')
    await rm(output, { recursive: true, force: true })
    await mkdir(output, { recursive: true })
    const ctx = context(output, { watch: true, force, serve: false })
    ctx.cfg.plugins = {
      transformers: configuredTransformers,
      filters: configuredFilters,
      emitters: configuredEmitters,
    }
    const markdown = await createFileParser(ctx, [filePath])(createMdProcessor(ctx))
    const content = await createMarkdownParser(ctx, markdown)(createHtmlProcessor(ctx))
    assert.equal(content.length, 1)
    const file = content[0][1]
    assert.ok(String(file.value).includes(force ? 'seed:gated' : 'seed:extra'))
    const serialized = JSON.stringify(content[0][0])
    assert.equal(serialized.includes('"value":"gated"'), force)
    assert.equal(serialized.includes('"value":"fallback"'), !force)
    assert.equal(serialized.includes('"value":"extra"'), !force)
    assert.equal(filterContent(ctx, content).length, force ? 0 : 1)
    const resources = getStaticResourcesFromPlugins(ctx)
    assert.deepEqual(
      resources.css.map(resource => resource.content),
      force
        ? ['.gated { color: red; }', '.emitter { color: green; }']
        : ['.extra { color: red; }', '.fallback-emitter {}'],
    )
    assert.deepEqual(configuredEmitters[0].getQuartzComponents?.(ctx), [
      force ? component : fallbackComponent,
    ])
    await emitContent(ctx, content)
    assert.deepEqual(await readdir(output), force ? ['conditional.html'] : ['snapshot.json'])
    ctx.incremental = true
    await emitContent(ctx, content, [{ type: 'change', path: filePath, file }])
    if (force)
      assert.equal(await readFile(path.join(output, 'conditional.html'), 'utf8'), 'partial')
    else
      assert.ok(
        (await readFile(path.join(output, 'snapshot.json'), 'utf8')).includes(
          '"incremental": true',
        ),
      )

    ctx.argv.output = path.join(output, 'reverse')
    await mkdir(ctx.argv.output, { recursive: true })
    ctx.cfg.plugins.emitters = Conditional({
      plugins: [{ plugin: snapshot, dev: false, fallback: conditionalEmitter }],
    })
    await emitContent(ctx, content, [{ type: 'change', path: filePath, file }])
    assert.deepEqual(
      await readdir(ctx.argv.output),
      force ? ['snapshot.json'] : ['conditional.html'],
    )
    if (!force) {
      assert.equal(
        await readFile(path.join(ctx.argv.output, 'conditional.html'), 'utf8'),
        'partial',
      )
      await emitContent(ctx, content, [])
      assert.equal(
        await readFile(path.join(ctx.argv.output, 'conditional.html'), 'utf8'),
        'partial',
      )
    }
  }
  assert.deepEqual(
    configuredTransformers.map(plugin => plugin.name),
    ['FrontMatter', 'gated', 'MissingHooks'],
  )
  assert.equal(configuredEmitters[0].name, 'ConditionalOutput')
  assert.equal(configuredFilters[0].name, 'RemoveDrafts')
})
const filters = Conditional({ plugins: [{ plugin: RemoveDrafts(), dev: false }] })
const emitters = Conditional({ plugins: [snapshot, { plugin: conditionalEmitter, dev: false }] })

function context(output: string, flags: Pick<Argv, 'watch' | 'force' | 'serve'>): BuildCtx {
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
  return {
    buildId: output,
    argv: { directory: artifacts, output, verbose: false, port: 0, wsPort: 0, ...flags },
    cfg: {
      configuration: {
        pageTitle: 'Conditional fixture',
        enableSPA: false,
        enablePopovers: false,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'created',
        locale: 'en-US',
        theme: {
          typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
          cdnCaching: false,
          fontOrigin: 'local',
          colors: { lightMode: colors, darkMode: colors },
        },
      },
      plugins: { transformers, filters, emitters },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

test('conditional plugins preserve the parse/filter/emit contract across build modes', async () => {
  await mkdir(artifacts, { recursive: true })
  const filePath = path.join(artifacts, 'fixture.md')
  assert.ok(isFilePath(filePath))
  await writeFile(filePath, source)
  const modes = [
    { name: 'watch', watch: true, force: false, serve: false, enabled: false },
    { name: 'forced-watch', watch: true, force: true, serve: false, enabled: true },
    { name: 'build', watch: false, force: false, serve: false, enabled: true },
    { name: 'forced-build', watch: false, force: true, serve: false, enabled: true },
    { name: 'serve', watch: false, force: false, serve: true, enabled: true },
    { name: 'serve-watch', watch: true, force: false, serve: true, enabled: false },
  ]
  const results: { mode: string; enabled: boolean; files: string[] }[] = []
  for (const mode of modes) {
    const output = path.join(artifacts, mode.name)
    await rm(output, { recursive: true, force: true })
    await mkdir(output, { recursive: true })
    const ctx = context(output, mode)
    const markdown = await createFileParser(ctx, [filePath])(createMdProcessor(ctx))
    const content = await createMarkdownParser(ctx, markdown)(createHtmlProcessor(ctx))
    assert.equal(content.length, 1)
    const [, file] = content[0]
    assert.equal(file.data.frontmatter?.title, 'Fixture')
    assert.ok(
      String(file.value).includes(mode.enabled ? 'seed:after:gated:before' : 'seed:after:before'),
    )
    const spans = content[0][0].children
      .filter(node => node.type === 'element')
      .filter(node => node.tagName === 'span')
    assert.deepEqual(
      spans.map(node => node.children),
      (mode.enabled ? ['before', 'gated', 'after'] : ['before', 'after']).map(value => [
        { type: 'text', value },
      ]),
    )
    assert.equal(filterContent(ctx, content).length, mode.enabled ? 0 : 1)
    assert.ok(file.data.frontmatter)
    file.data.frontmatter.draft = false
    const published = filterContent(ctx, content)
    assert.equal(published.length, 1)
    const resources = getStaticResourcesFromPlugins(ctx)
    assert.deepEqual(
      resources.css.map(resource => resource.content),
      (mode.enabled ? ['before', 'gated', 'after', 'emitter'] : ['before', 'after']).map(
        label => `.${label} { color: ${label === 'emitter' ? 'green' : 'red'}; }`,
      ),
    )
    assert.deepEqual(emitters[1].getQuartzComponents?.(ctx), mode.enabled ? [component] : [])

    await emitContent(ctx, published)
    const initial = await readFile(path.join(output, 'snapshot.json'), 'utf8')
    assert.ok(initial.includes('"incremental": false'))
    assert.equal(initial.includes('"value": "gated"'), mode.enabled)
    assert.equal((await readdir(output)).includes('conditional.html'), mode.enabled)
    if (mode.enabled)
      assert.equal(await readFile(path.join(output, 'conditional.html'), 'utf8'), 'initial')

    ctx.incremental = true
    await emitContent(ctx, published, [{ type: 'change', path: filePath, file }])
    assert.ok(
      (await readFile(path.join(output, 'snapshot.json'), 'utf8')).includes('"incremental": true'),
    )
    if (mode.enabled)
      assert.equal(await readFile(path.join(output, 'conditional.html'), 'utf8'), 'partial')
    await emitPartialEmitter(ctx, published, [], 'ConditionalOutput')
    if (mode.enabled)
      assert.equal(await readFile(path.join(output, 'conditional.html'), 'utf8'), 'partial')
    else assert.deepEqual(await readdir(output), ['snapshot.json'])
    results.push({ mode: mode.name, enabled: mode.enabled, files: (await readdir(output)).sort() })
    ctx.argv.output = path.join(output, 'fallback')
    await mkdir(ctx.argv.output, { recursive: true })
    ctx.cfg.plugins.emitters = Conditional({ plugins: [{ plugin: snapshot, dev: false }] })
    assert.equal(ctx.cfg.plugins.emitters[0].partialEmit, undefined)
    await emitContent(ctx, published, [{ type: 'change', path: filePath, file }])
    assert.deepEqual(await readdir(ctx.argv.output), mode.enabled ? ['snapshot.json'] : [])
  }
  assert.deepEqual(
    transformers.map(plugin => plugin.name),
    ['FrontMatter', 'before', 'gated', 'after'],
  )
  assert.deepEqual(
    emitters.map(plugin => plugin.name),
    ['Snapshot', 'ConditionalOutput'],
  )
  assert.equal(transformers[1], before)
  assert.equal(transformers[3], after)
  assert.equal(emitters[0].partialEmit, undefined)
  assert.deepEqual(Conditional({ plugins: [] }), [])
  assert.equal(ConditionalTransform, Conditional)
  await writeFile(
    path.join(artifacts, 'results.json'),
    JSON.stringify(
      { command: 'pnpm test quartz/plugins/conditional.test.ts', status: 'passed', results },
      null,
      2,
    ),
  )
})
