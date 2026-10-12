import assert from 'node:assert/strict'
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { StaticResources } from '../../util/resources'
import { createHtmlProcessor, createMdProcessor, mdastToHastRoot } from '../../processors/parse'
import { isFilePath, isFullSlug, type FilePath } from '../../util/path'
import { LLM } from '../transformers/llm'
import { Stream } from '../transformers/stream'
import { defaultProcessedContent } from '../vfile'
import { LLMText, llmsIndex } from './llm'

function testCtx(root: string): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: path.join(root, 'content'),
      verbose: false,
      output: path.join(root, 'public'),
      serve: false,
      watch: true,
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
        locale: 'en-US',
        theme: {} as BuildCtx['cfg']['configuration']['theme'],
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
}

const resources: StaticResources = { css: [], js: [], additionalHead: [] }

const note = (slug: string, text: string) => {
  const filePath = `${slug}.md`
  assert.ok(isFullSlug(slug))
  assert.ok(isFilePath(filePath))
  const content = defaultProcessedContent({
    slug,
    filePath,
    relativePath: filePath,
    frontmatter: { title: slug, pageLayout: 'default', tags: [] },
    text,
    links: [],
  })
  content[1].data.llmsText = text
  return content
}

async function collectEmitted(
  emitted: Promise<FilePath[]> | AsyncGenerator<FilePath> | null,
): Promise<FilePath[]> {
  const result = await emitted
  if (result === null) return []
  if (!(Symbol.asyncIterator in result)) return result
  const files: FilePath[] = []
  for await (const file of result) files.push(file)
  return files
}

const relativeOutputs = (ctx: BuildCtx, outputs: FilePath[]): string[] =>
  outputs.map(output => path.relative(ctx.argv.output, output)).sort()

test('publishes a proposal-compliant llms.txt with use cases and discovery links', () => {
  const content = llmsIndex('aarnphm.xyz')
  const lines = content.split('\n')

  assert.equal(lines[0], '# aarnphm.xyz')
  assert.ok(lines[2]?.startsWith('> '))
  assert.match(content, /## When to use this site/)
  assert.match(content, /Use the read-only MCP tools/)
  assert.match(content, /https:\/\/aarnphm\.xyz\/api\/docs/)
  assert.match(content, /https:\/\/aarnphm\.xyz\/openapi\.json/)
  assert.match(content, /https:\/\/aarnphm\.xyz\/\.well-known\/api-catalog/)
  assert.match(content, /https:\/\/aarnphm\.xyz\/about\.md/)

  for (const line of lines.filter(line => line.startsWith('- '))) {
    assert.match(line, /^- \[[^\]]+\]\(https:\/\/aarnphm\.xyz\/[^)]+\): .+/)
  }
})

test('watch emit publishes triathlon and Arena Markdown without rebuilding the garden corpus', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-llm-watch-'))
  try {
    const ctx = testCtx(root)
    const plugin = LLMText()
    const outputs = await collectEmitted(
      plugin.emit(
        ctx,
        [
          note('thoughts/example', '# example'),
          note('triathlon', '# triathlon'),
          note('are.na', '## engineering\n\n- https://example.com -- Example'),
        ],
        resources,
      ),
    )

    assert.deepEqual(relativeOutputs(ctx, outputs), ['are.na.md', 'llms.txt', 'triathlon.md'])
    const arenaMarkdown = await readFile(path.join(ctx.argv.output, 'are.na.md'), 'utf8')
    assert.match(arenaMarkdown, /## engineering\n\n- https:\/\/example.com -- Example/)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('watch partial emit refreshes only the changed triathlon or Arena source', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-llm-watch-partial-'))
  try {
    const ctx = testCtx(root)
    const plugin = LLMText()
    const triathlon = note('triathlon', '# triathlon')
    const arena = note('are.na', '## engineering\n\n- https://example.com -- Updated entry')
    const example = note('thoughts/example', '# example')
    const content = [triathlon, arena, example]
    const partialEmit = plugin.partialEmit
    assert.ok(partialEmit)

    const triathlonOutputs = await collectEmitted(
      partialEmit(ctx, content, resources, [
        { type: 'change', path: triathlon[1].data.filePath!, file: triathlon[1] },
      ]),
    )
    const arenaOutputs = await collectEmitted(
      partialEmit(ctx, content, resources, [
        { type: 'change', path: arena[1].data.filePath!, file: arena[1] },
      ]),
    )
    const exampleOutputs = await collectEmitted(
      partialEmit(ctx, content, resources, [
        { type: 'change', path: example[1].data.filePath!, file: example[1] },
      ]),
    )

    assert.deepEqual(relativeOutputs(ctx, triathlonOutputs), ['triathlon.md'])
    assert.deepEqual(relativeOutputs(ctx, arenaOutputs), ['are.na.md'])
    const arenaMarkdown = await readFile(path.join(ctx.argv.output, 'are.na.md'), 'utf8')
    assert.match(arenaMarkdown, /Updated entry/)
    assert.deepEqual(exampleOutputs, [])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

const streamSource = `## public fixture

- [meta]
  - date: 2026-10-12

PUBLIC_STREAM_FIXTURE

---

## protected fixture

- [meta]
  - date: 2026-10-11
  - protected: true

PROTECTED_STREAM_FIXTURE

---

## private fixture

- [meta]
  - date: 2026-10-10
  - private: true

PRIVATE_STREAM_FIXTURE
`

for (const mode of ['full', 'watch', 'forced-watch']) {
  test(`${mode} LLM pipeline excludes stream and removes obsolete Markdown`, async () => {
    const root = await mkdtemp(path.join(tmpdir(), 'quartz-llm-stream-'))
    const artifacts = path.resolve('docs/verification/stream-llm-exclusion/artifacts', mode)
    try {
      const ctx = testCtx(root)
      ctx.argv.watch = mode !== 'full'
      ctx.argv.force = mode === 'forced-watch'
      ctx.cfg.plugins.transformers = [LLM(), Stream()]
      const stream = note('stream', streamSource)
      const file = stream[1]
      file.value = streamSource
      delete file.data.llmsText
      const md = createMdProcessor(ctx)
      const tree = await md.run(md.parse(file), file)
      const html = await createHtmlProcessor(ctx).run(mdastToHastRoot(tree), file)
      stream[0] = html
      assert.equal(file.data.streamData?.entries.length, 3)
      const serializedStream = file.data.llmsText

      // A cached serialization must not bypass the emitter's publication policy.
      file.data.llmsText = streamSource
      const index = note('index', 'PUBLIC_HOMEPAGE_FIXTURE')
      const triathlon = note('triathlon', 'PUBLIC_TRIATHLON_FIXTURE')
      const plugin = LLMText()
      await mkdir(ctx.argv.output, { recursive: true })
      await writeFile(path.join(root, 'stream-source.md'), streamSource)
      await writeFile(path.join(ctx.argv.output, 'stream.md'), streamSource)
      const outputs = await collectEmitted(plugin.emit(ctx, [index, stream, triathlon], resources))
      const expected =
        mode === 'watch'
          ? ['llms.txt', 'triathlon.md']
          : ['llms-full.txt', 'llms.txt', 'triathlon.md']
      await writeFile(
        path.join(root, 'results.json'),
        JSON.stringify(
          {
            mode,
            serializedStream: serializedStream !== undefined,
            emitted: relativeOutputs(ctx, outputs),
          },
          null,
          2,
        ),
      )

      assert.equal(serializedStream, undefined)
      assert.deepEqual(relativeOutputs(ctx, outputs), expected)
      await assert.rejects(readFile(path.join(ctx.argv.output, 'stream.md')), { code: 'ENOENT' })
      for (const output of outputs) {
        const text = await readFile(output, 'utf8')
        assert.ok(!text.includes('STREAM_FIXTURE'))
        assert.ok(!text.includes('<document slug=stream>'))
      }
      const markdown = await readFile(path.join(ctx.argv.output, 'triathlon.md'), 'utf8')
      assert.ok(markdown.includes('PUBLIC_TRIATHLON_FIXTURE'))
      if (mode !== 'watch') {
        const corpus = await readFile(path.join(ctx.argv.output, 'llms-full.txt'), 'utf8')
        assert.ok(corpus.includes('PUBLIC_HOMEPAGE_FIXTURE'))
        assert.ok(corpus.includes('PUBLIC_TRIATHLON_FIXTURE'))
      }

      const partial = plugin.partialEmit
      assert.ok(partial)
      ctx.incremental = true
      await writeFile(path.join(ctx.argv.output, 'stream.md'), streamSource)
      const updated = await collectEmitted(
        partial(ctx, [index, stream, triathlon], resources, [
          { type: 'change', path: file.data.filePath!, file },
        ]),
      )
      assert.deepEqual(updated, [])
      await assert.rejects(readFile(path.join(ctx.argv.output, 'stream.md')), { code: 'ENOENT' })

      // Deletion can leave no stream VFile in the current content inventory.
      await writeFile(path.join(ctx.argv.output, 'stream.md'), streamSource)
      await collectEmitted(
        partial(ctx, [index, triathlon], resources, [
          { type: 'delete', path: file.data.filePath!, previousFile: file },
        ]),
      )
      await assert.rejects(readFile(path.join(ctx.argv.output, 'stream.md')), { code: 'ENOENT' })
      await writeFile(
        path.join(root, 'results.json'),
        JSON.stringify(
          {
            mode,
            status: 'passed',
            serializedStream: false,
            emitted: relativeOutputs(ctx, outputs),
            obsoleteMarkdownRemoved: true,
            partialChangeRemovedMarkdown: true,
            partialDeleteRemovedMarkdown: true,
          },
          null,
          2,
        ),
      )
    } finally {
      await rm(artifacts, { recursive: true, force: true })
      await mkdir(artifacts, { recursive: true })
      await cp(root, artifacts, { recursive: true })
      await rm(root, { recursive: true, force: true })
    }
  })
}
