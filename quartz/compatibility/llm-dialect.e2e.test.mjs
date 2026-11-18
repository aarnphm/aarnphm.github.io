import { toHtml } from 'hast-util-to-html'
import assert from 'node:assert/strict'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import test from 'node:test'
import './pipeline.mjs'

const [
  { parseMarkdown, resetProcessedContentCache },
  { codeSearchTestContext },
  { FrontMatter },
  { ObsidianFlavoredMarkdown },
  { GitHubFlavoredMarkdown },
  { Speech },
  { LLM },
] = await Promise.all([
  import('../processors/parse.ts'),
  import('../plugins/emitters/code-search-test-context.ts'),
  import('../plugins/transformers/frontmatter.ts'),
  import('../plugins/transformers/ofm.ts'),
  import('../plugins/transformers/gfm.ts'),
  import('../plugins/transformers/speech.ts'),
  import('../plugins/transformers/llm.ts'),
])

const directory = path.resolve(`docs/verification/llm-highlight/artifacts/run-${Date.now()}`)

async function render(name, source, watch = false) {
  const root = path.join(directory, name)
  const ctx = codeSearchTestContext(root)
  ctx.argv.watch = watch
  ctx.argv.concurrency = 1
  ctx.cfg.plugins.transformers = [
    FrontMatter(),
    ObsidianFlavoredMarkdown(),
    GitHubFlavoredMarkdown(),
    Speech(),
    LLM(),
  ]
  const filename = path.join(ctx.argv.directory, 'fr/lessons/260810.md')
  await mkdir(path.dirname(filename), { recursive: true })
  await writeFile(filename, source)
  ctx.allFiles.push(filename)
  resetProcessedContentCache()
  const content = await parseMarkdown(ctx, [filename])
  assert.equal(content.length, 1)
  const [tree, file] = content[0]
  const html = toHtml(tree)
  const markdown = file.data.llmsText
  assert.equal(typeof markdown, 'string')
  await writeFile(path.join(root, 'output.html'), html)
  await writeFile(path.join(root, 'output.md'), markdown)
  return { html, markdown }
}

test('the reported French lesson serializes its highlight through the real Markdown pipeline', async () => {
  const source = await readFile('content/fr/lessons/260810.md', 'utf8')
  const result = await render('reported-lesson', source)
  assert.ok(result.html.includes('<mark>engelures</mark>'))
  assert.ok(result.markdown.includes('<mark>engelures</mark>'))
})

for (const watch of [false, true]) {
  test(`LLM highlights retain nested phrasing and literal boundaries in ${watch ? 'watch' : 'full'} mode`, async () => {
    const source = [
      '---',
      'flashcards:',
      '  default: true',
      '---',
      '==plain== and ==**bold** _italic_ `inline` [site](https://example.com)==.',
      '',
      '> ==🔴colored **content**== with ^[==footnote==].',
      '',
      '| label | value |',
      '| --- | --- |',
      '| marked | ==table== |',
      '',
      '`==literal==` and \\==escaped\\==.',
      '',
      '```text',
      '==fenced==',
      '```',
    ].join('\n')
    const result = await render(watch ? 'watch' : 'full', source, watch)
    assert.ok(result.markdown.includes('<mark>plain</mark>'))
    assert.ok(
      result.markdown.includes(
        '<mark>**bold** _italic_ `inline` [site](https://example.com)</mark>',
      ),
    )
    assert.ok(result.markdown.includes('<mark>colored **content**</mark>'))
    assert.ok(result.markdown.includes('<mark>footnote</mark>'))
    assert.ok(result.markdown.includes('<mark>table</mark>'))
    assert.ok(result.markdown.includes('`==literal==`'))
    assert.ok(result.markdown.includes('==fenced=='))
    assert.equal(result.markdown.includes('<mark>escaped</mark>'), false)
    assert.ok(result.html.includes('data-highlight-color="red"'))
    assert.ok(result.html.includes('<mark><strong>bold</strong>'))
  })
}
