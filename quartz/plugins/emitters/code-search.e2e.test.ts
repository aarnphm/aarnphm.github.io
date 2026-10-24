import type { Processor } from 'unified'
import assert from 'node:assert/strict'
import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { registerHooks } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { visit } from 'unist-util-visit'
import { VFile } from 'vfile'
import type { CodeSearchManifest, CodeSourceShard } from '../../util/code-search-types'
import type { BuildCtx } from '../../util/ctx'
import { createMdProcessor, mdastToHastRoot } from '../../processors/parse'
import { isFilePath, slugifyFilePath } from '../../util/path'
import { CodeSearch } from './code-search'
import { codeSearchTestContext } from './code-search-test-context'

registerHooks({
  load(url, context, nextLoad) {
    if (/\.(scss|inline(?:\.ts)?)$/.test(url))
      return { format: 'module', shortCircuit: true, source: 'export default ""' }
    return nextLoad(url, context)
  },
})

// Failure gates: hidden owners leaking, lost duplicate occurrences, source normalization,
// notebook output indexing, dead anchors, and stale generations after update/deletion.

async function note(ctx: BuildCtx, name: string, raw: string, flags: Record<string, boolean> = {}) {
  assert.ok(isFilePath(name))
  const file = new VFile({ path: path.join(ctx.argv.directory, name), value: raw })
  file.data.relativePath = name
  file.data.slug = slugifyFilePath(name)
  file.data.frontmatter = { title: name, pageLayout: 'default', tags: [], ...flags }
  const processor = createMdProcessor(ctx)
  const tree = await processor.run(processor.parse(raw), file)
  return [mdastToHastRoot(tree), file] satisfies import('../vfile').ProcessedContent
}
const resources = { css: [], js: [], additionalHead: [] }
async function emit(result: ReturnType<ReturnType<typeof CodeSearch>['emit']> | null) {
  const files = await result
  assert.ok(files)
  if (Symbol.asyncIterator in files)
    for await (const _file of files) {
      /* exhaust publication */
    }
}
async function manifest(ctx: BuildCtx): Promise<CodeSearchManifest> {
  return JSON.parse(
    await readFile(path.join(ctx.argv.output, 'static/code-search/manifest.json'), 'utf8'),
  )
}
async function texts(ctx: BuildCtx, index: CodeSearchManifest) {
  const result = new Map<string, string>()
  for (const ref of index.shards) {
    const shard: CodeSourceShard = JSON.parse(
      await readFile(path.join(ctx.argv.output, ref.path), 'utf8'),
    )
    assert.equal(shard.generation, index.generation)
    for (const source of shard.sources) result.set(source.id, source.text)
  }
  return result
}

async function saveReceipt(ctx: BuildCtx, name: string) {
  const destination = process.env.CODE_SEARCH_TEST_ARTIFACTS
  if (!destination) return
  const root = path.join(destination, name)
  await mkdir(root, { recursive: true })
  await cp(ctx.argv.output, path.join(root, 'public'), { recursive: true })
  await cp(ctx.argv.directory, path.join(root, 'content'), { recursive: true })
}

test('publishes public occurrences, raw ranges and notebooks, then replaces and prunes generations', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'code-search-'))
  try {
    const ctx = codeSearchTestContext(root)
    await mkdir(ctx.argv.directory)
    const source = 'print("😀")\r\n\tpass  '
    await writeFile(path.join(ctx.argv.directory, 'sample.py'), source)
    await writeFile(path.join(ctx.argv.directory, 'data.json'), '{"secret":true}')
    await mkdir(path.join(ctx.argv.directory, 'ignored'))
    await writeFile(path.join(ctx.argv.directory, 'ignored/leak.py'), 'IGNORED')
    const notebook = {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: { language_info: { name: 'python' } },
      cells: [
        {
          cell_type: 'code',
          metadata: {},
          source,
          execution_count: 1,
          outputs: [
            {
              output_type: 'display_data',
              metadata: {},
              data: { 'text/markdown': '```python\nOUTPUT_ONLY\n```' },
            },
          ],
        },
        { cell_type: 'markdown', metadata: {}, source: '# Example\n\n```py\nprint(2)\n```' },
      ],
    }
    await writeFile(path.join(ctx.argv.directory, 'book.ipynb'), JSON.stringify(notebook))
    ctx.allFiles = ['sample.py', 'data.json', 'book.ipynb', 'ignored/leak.py'].filter(isFilePath)
    const raw =
      '> ```py\r\n> print("😀")\r\n> \tpass  \r\n> ```\n\n```py\nprint(2)\n```\n\n```py\nprint(2)\n```'
    const page = await note(ctx, 'note.md', raw)
    const hidden = await Promise.all(
      ['protected', 'private', 'draft', 'noindex'].map(flag =>
        note(ctx, `${flag}.md`, '```py\nHIDDEN\n```', { [flag]: true }),
      ),
    )
    const plugin = CodeSearch()
    await emit(plugin.emit(ctx, [page, ...hidden], resources))
    const first = await manifest(ctx)
    assert.equal(first.units.length, 6)
    assert.equal(first.units.filter(unit => unit.ownerSlug === 'note').length, 3)
    assert.ok(first.units.every(unit => !unit.path.includes('ignored')))
    const bodies = await texts(ctx, first)
    assert.ok([...bodies.values()].includes(source))
    assert.ok(![...bodies.values()].some(body => /HIDDEN|OUTPUT_ONLY|IGNORED/.test(body)))
    const anchors = new Set<string>()
    visit(page[0], 'element', node => {
      if (typeof node.properties.id === 'string') anchors.add(node.properties.id)
    })
    for (const unit of first.units.filter(unit => unit.ownerSlug === 'note'))
      assert.ok(anchors.has(unit.anchor ?? ''))
    const nested = first.units.find(unit => unit.ownerSlug === 'note' && unit.startLine === 2)
    assert.ok(nested)
    assert.equal(bodies.get(nested.sourceId), source)
    assert.equal(nested.endLine, 3)
    assert.equal(first.units.find(unit => unit.kind === 'cell')?.anchor, 'cell-1')
    assert.ok(first.coverage.sources < first.coverage.units)
    await saveReceipt(ctx, 'initial')
    await writeFile(path.join(ctx.argv.directory, 'sample.py'), 'updated()')
    ctx.allFiles = ['sample.py'].filter(isFilePath)
    assert.ok(plugin.partialEmit)
    await emit(plugin.partialEmit(ctx, [], resources, []))
    const second = await manifest(ctx)
    assert.notEqual(second.generation, first.generation)
    assert.equal(second.units.length, 1)
    await saveReceipt(ctx, 'updated')
    assert.deepEqual(
      [...(await texts(ctx, second))].map(([, text]) => text),
      ['updated()'],
    )
    assert.deepEqual(
      (await readdir(path.join(ctx.argv.output, 'static/code-search'))).sort(),
      [second.generation, 'manifest.json'].sort(),
    )
    await writeFile(path.join(ctx.argv.directory, 'sample.py'), source)
    ctx.allFiles = ['sample.py', 'data.json', 'book.ipynb', 'ignored/leak.py'].filter(isFilePath)
    await emit(plugin.partialEmit(ctx, [page], resources, []))
    assert.equal((await manifest(ctx)).generation, first.generation)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('indexes resolved local code transclusions and retains ordinary path-labelled fences', async () => {
  const { CodeViewer } = await import('../transformers/codeViewer')
  const { CodeSearchBlocks } = await import('../transformers/code-search')
  const { wikilink, wikilinkFromMarkdown } =
    await import('../../extensions/micromark-extension-ofm-wikilinks')
  const root = await mkdtemp(path.join(tmpdir(), 'code-transclude-'))
  try {
    const ctx = codeSearchTestContext(root)
    ctx.cfg.plugins.transformers = [
      {
        name: 'Wikilinks',
        markdownPlugins: () => [
          function (this: Processor) {
            const data = this.data()
            data.micromarkExtensions = [wikilink()]
            data.fromMarkdownExtensions = [wikilinkFromMarkdown()]
          },
        ],
      },
      CodeViewer(),
      CodeSearchBlocks(),
    ]
    await mkdir(ctx.argv.directory)
    await writeFile(path.join(ctx.argv.directory, 'shared.py'), 'print("shared")\n')
    ctx.allFiles = ['shared.py'].filter(isFilePath)
    const page = await note(
      ctx,
      'owner.md',
      '![[shared.py]]\n\n```py path="display.py"\nprint("authored")\n```',
    )
    await emit(CodeSearch().emit(ctx, [page], resources))
    const index = await manifest(ctx)
    const bodies = await texts(ctx, index)
    assert.equal(index.units.length, 3)
    const owner = index.units.filter(unit => unit.ownerSlug === 'owner')
    assert.equal(owner.length, 2)
    assert.equal(
      owner.find(unit => bodies.get(unit.sourceId) === 'print("shared")\n')?.path,
      'shared.py',
    )
    assert.ok(owner.some(unit => bodies.get(unit.sourceId) === 'print("shared")\n'))
    assert.ok(owner.some(unit => bodies.get(unit.sourceId) === 'print("authored")'))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('code anchors survive Mermaid node-data replacement in the production transform order', async () => {
  const { CodeSearchBlocks } = await import('../transformers/code-search')
  const { SyntaxHighlighting } = await import('../transformers/syntax')
  const { ObsidianFlavoredMarkdown } = await import('../transformers/ofm')
  const { createHtmlProcessor } = await import('../../processors/parse')
  const root = await mkdtemp(path.join(tmpdir(), 'code-anchors-'))
  try {
    const ctx = codeSearchTestContext(root)
    ctx.cfg.plugins.transformers = [
      CodeSearchBlocks(),
      SyntaxHighlighting(),
      ObsidianFlavoredMarkdown(),
    ]
    await mkdir(ctx.argv.directory)
    const page = await note(ctx, 'diagram.md', '```mermaid\ngraph TD\n A --> B\n```')
    const rendered = await createHtmlProcessor(ctx).run(page[0], page[1])
    const anchors = new Set<string>()
    visit(rendered, 'element', node => {
      if (typeof node.properties.id === 'string') anchors.add(node.properties.id)
    })
    await emit(CodeSearch().emit(ctx, [[rendered, page[1]]], resources))
    const index = await manifest(ctx)
    assert.equal(index.units.length, 1)
    assert.ok(anchors.has(index.units[0].anchor ?? ''))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('publishes a shared body in each selected language shard', async () => {
  const { readCodeManifest, readCodeShard } = await import('../../util/code-search-format')
  const root = await mkdtemp(path.join(tmpdir(), 'code-languages-'))
  try {
    const ctx = codeSearchTestContext(root)
    await mkdir(ctx.argv.directory)
    ctx.allFiles = ['same.py', 'same.js'].filter(isFilePath)
    for (const fp of ctx.allFiles) await writeFile(path.join(ctx.argv.directory, fp), 'shared()')
    await emit(CodeSearch().emit(ctx, [], resources))
    const index = readCodeManifest(await manifest(ctx))
    assert.equal(new Set(index.units.map(unit => unit.sourceId)).size, 1)
    await saveReceipt(ctx, 'cross-language')
    assert.deepEqual(index.shards.map(shard => shard.language).sort(), ['javascript', 'python'])
    for (const reference of index.shards) {
      const bytes = new Uint8Array(await readFile(path.join(ctx.argv.output, reference.path)))
      const shard = await readCodeShard(bytes, reference, index.generation)
      assert.equal(shard.sources[0].text, 'shared()')
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
