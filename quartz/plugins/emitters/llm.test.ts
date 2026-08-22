import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath, FullSlug } from '../../util/path'
import type { StaticResources } from '../../util/resources'
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

const note = (slug: FullSlug, text: string) => {
  const content = defaultProcessedContent({
    slug,
    filePath: `${slug}.md` as FilePath,
    relativePath: `${slug}.md` as FilePath,
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

test('watch emit publishes the triathlon route index without rebuilding the garden corpus', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-llm-watch-'))
  try {
    const ctx = testCtx(root)
    const plugin = LLMText()
    const outputs = await collectEmitted(
      plugin.emit(
        ctx,
        [
          note('thoughts/example' as FullSlug, '# example'),
          note('triathlon' as FullSlug, '# triathlon'),
        ],
        resources,
      ),
    )

    assert.deepEqual(relativeOutputs(ctx, outputs), ['llms.txt', 'triathlon.md'])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('watch partial emit refreshes the triathlon route index only when its source changes', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-llm-watch-partial-'))
  try {
    const ctx = testCtx(root)
    const plugin = LLMText()
    const triathlon = note('triathlon' as FullSlug, '# triathlon')
    const example = note('thoughts/example' as FullSlug, '# example')
    const partialEmit = plugin.partialEmit
    assert.ok(partialEmit)

    const triathlonOutputs = await collectEmitted(
      partialEmit(ctx, [triathlon, example], resources, [
        { type: 'change', path: 'triathlon.md' as FilePath, file: triathlon[1] },
      ]),
    )
    const exampleOutputs = await collectEmitted(
      partialEmit(ctx, [triathlon, example], resources, [
        { type: 'change', path: 'thoughts/example.md' as FilePath, file: example[1] },
      ]),
    )

    assert.deepEqual(relativeOutputs(ctx, triathlonOutputs), ['triathlon.md'])
    assert.deepEqual(exampleOutputs, [])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
