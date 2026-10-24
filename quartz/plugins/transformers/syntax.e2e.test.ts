import type { Element, Root } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { toString as toText } from 'hast-util-to-string'
import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import test from 'node:test'
import rehypeStringify from 'rehype-stringify'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'
import { visit } from 'unist-util-visit'
import type { BuildCtx } from '../../util/ctx'
import { SyntaxHighlighting } from './syntax'

function elements(tree: Root | Element, matches: (node: Element) => boolean): Element[] {
  const result: Element[] = []
  visit(tree, 'element', node => {
    if (matches(node)) result.push(node)
  })
  return result
}
function tags(tree: Root | Element, tag: string) {
  return elements(tree, node => node.tagName === tag)
}
function marked(tree: Root | Element, key: string, value?: string) {
  return elements(
    tree,
    node =>
      node.properties[key] !== undefined && (value === undefined || node.properties[key] === value),
  )
}

async function render(source: string, keepBackground = true) {
  const syntax = SyntaxHighlighting({ keepBackground })
  const processor = unified().use(remarkParse)
  const context: BuildCtx = {
    buildId: 'syntax-test',
    argv: {
      directory: 'content',
      output: 'public',
      verbose: false,
      serve: false,
      watch: false,
      port: 0,
      wsPort: 0,
      force: false,
    },
    get cfg(): BuildCtx['cfg'] {
      throw new Error('Syntax plugins must not require configuration')
    },
    allSlugs: [],
    allFiles: [],
    incremental: false,
  }
  processor.use(syntax.markdownPlugins?.(context) ?? []).use(remarkRehype)
  processor.use(syntax.htmlPlugins?.(context) ?? []).use(rehypeStringify)
  return String(await processor.process(source))
}

const fixture = [
  '```typescript title="example.ts" caption="Example" {2}#focus /answer/#value showLineNumbers{9} disableLineNumber=true',
  'const answer = "<script>&"',
  'console.log(answer)',
  '',
  '```',
  '',
  'Inline `const n = 1{:js}` and escaped `literal\\{:js}`.',
  '',
  '```unknown-language',
  '<script>alert("unsafe")</script>',
  '```',
].join('\n')

test('Markdown highlighting preserves figure metadata, source, lines, inline code and copy controls', async () => {
  const html = await render(fixture)
  const tree = fromHtml(html, { fragment: true })
  const figure = tags(tree, 'figure')[0]
  assert.ok(figure)
  assert.equal(marked(figure, 'dataRehypePrettyCodeTitle')[0]?.tagName, 'div')
  assert.equal(toText(marked(figure, 'dataRehypePrettyCodeTitle')[0]!), 'example.ts')
  assert.equal(toText(marked(figure, 'dataRehypePrettyCodeCaption')[0]!), 'Example')
  const code = tags(figure, 'code')[0]
  assert.ok(code)
  assert.match(toText(code), /const answer = "<script>&"\nconsole.log\(answer\)/)
  assert.equal(marked(code, 'dataLine').length, 3)
  assert.equal(marked(code, 'dataHighlightedLineId', 'focus')[0]?.tagName, 'span')
  assert.equal(marked(code, 'dataCharsId', 'value').length, 2)
  assert.equal(
    tags(figure, 'pre')[0]?.properties['data-disable-line-number'] ??
      tags(figure, 'pre')[0]?.properties.dataDisableLineNumber,
    'true',
  )
  assert.equal(
    elements(
      tree,
      node =>
        node.tagName === 'button' &&
        Array.isArray(node.properties.className) &&
        node.properties.className.includes('clipboard-button'),
    ).length,
    2,
  )
  assert.equal(tags(tree, 'script').length, 0)
  assert.equal(
    marked(tree, 'dataRehypePrettyCodeFigure').filter(node => node.tagName === 'span').length,
    1,
  )
  assert.match(html, /literal\{:js\}/)
  assert.match(html, /--syntax-light:/)
  assert.match(html, /--syntax-dark:/)
  if (process.env.HIGHLIGHTS_ARTIFACTS) {
    await mkdir(process.env.HIGHLIGHTS_ARTIFACTS, { recursive: true })
    await writeFile(`${process.env.HIGHLIGHTS_ARTIFACTS}/fixture.md`, fixture)
    await writeFile(`${process.env.HIGHLIGHTS_ARTIFACTS}/output.html`, html)
  }
})

test('background opt-out and unlabelled code preserve plain source', async () => {
  const html = await render('```python\nx = 1\n```\n\n```\n<raw>&\n```', false)
  const tree = fromHtml(html, { fragment: true })
  assert.equal(tags(tree, 'pre')[0]?.properties.style, undefined)
  assert.equal(
    elements(
      tree,
      node =>
        node.tagName === 'button' &&
        Array.isArray(node.properties.className) &&
        node.properties.className.includes('clipboard-button'),
    ).length,
    2,
  )
  assert.match(toText(tree), /<raw>&/)
})

test('character ranges cross token boundaries and count selected occurrences across lines', async () => {
  const html = await render(
    '```js /const answer/2#selected {1,3-4} showLineNumbers{99}\nconst answer = 1\nconst answer = 2\n\nconst other = "é😀"\n```',
  )
  const tree = fromHtml(html, { fragment: true })
  const marks = marked(tree, 'dataCharsId', 'selected')
  assert.equal(marks.length, 1)
  assert.equal(toText(marks[0]), 'const answer')
  assert.equal(marked(tree, 'dataHighlightedLine').length, 3)
  const code = tags(tree, 'code')[0]
  assert.ok(code)
  assert.equal(code.properties.dataLineNumbersMaxDigits, '3')
  assert.match(String(code.properties.style), /counter-set: line 98/)
  assert.match(toText(code), /é😀/)
})

test('Prolog keeps syntax colors without Shiki and unsupported languages keep literal source', async () => {
  const html = await render(
    '```prolog\n% comment\nlikes(alice, "tea").\n```\n\n```lean\n#check Nat → Nat\n```\n\n```mojo\ndef f():\n    return 42\n```',
  )
  const tree = fromHtml(html, { fragment: true })
  const figures = tags(tree, 'figure')
  assert.equal(figures[0].properties.dataHighlighter, 'highlight.js')
  assert.ok(marked(figures[0], 'dataSyntaxRole', 'comment')[0])
  assert.equal(figures[1].properties.dataHighlightLanguage, 'plain')
  assert.equal(tags(figures[1], 'code')[0]?.properties.dataLanguage, 'lean')
  assert.equal(toText(tags(figures[1], 'code')[0]!), '#check Nat → Nat')
  assert.equal(figures[2].properties.dataHighlightLanguage, 'python')
  assert.ok(marked(figures[2], 'dataSyntaxRole', 'keyword')[0])
})
