import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import type { BuildCtx } from '../../util/ctx'
import type { StaticResources } from '../../util/resources'
import { isFilePath, isFullSlug, type FilePath } from '../../util/path'
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
