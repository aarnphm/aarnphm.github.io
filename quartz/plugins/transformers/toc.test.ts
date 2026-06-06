import type { Element, Root } from 'hast'
import type { Heading, Root as MarkdownRoot } from 'mdast'
import { h } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import { visit } from 'unist-util-visit'
import { collectHtmlToc, collectHtmlTocData, collectMarkdownTocData } from './toc'

function markdownHeading(depth: Heading['depth'], text: string): Heading {
  return { type: 'heading', depth, children: [{ type: 'text', value: text }] }
}

function katexInline(tex: string): Element {
  return h('span.katex', [
    h('span.katex-mathml', [
      h('math', [
        h('semantics', [
          h('mrow', [h('mi', 'A'), h('mo', '∩'), h('mi', 'B')]),
          h('annotation', { encoding: 'application/x-tex' }, tex),
        ]),
      ]),
    ]),
    h('span.katex-html', { ariaHidden: 'true' }, 'A∩B'),
  ])
}

test('collects toc entries from final html headings', () => {
  const tree: Root = {
    type: 'root',
    children: [
      h('h2#local', 'Local'),
      h('blockquote.transclude', [
        h('div.transclude-content', [h('h3#remote', 'Remote'), h('h4', 'Generated remote')]),
      ]),
      h('h6', 'Too deep'),
    ],
  }

  const toc = collectHtmlToc(tree, { maxDepth: 4, minEntries: 1 })

  assert.deepEqual(toc, [
    { depth: 0, text: 'Local', slug: 'local' },
    { depth: 1, text: 'Remote', slug: 'remote' },
    { depth: 2, text: 'Generated remote', slug: 'generated-remote' },
  ])

  let generatedId: unknown
  visit(tree, 'element', node => {
    if (node.tagName === 'h4') {
      generatedId = node.properties?.id
    }
  })
  assert.equal(generatedId, 'generated-remote')
})

test('uses semantic unicode math text for html toc headings', () => {
  const tree: Root = {
    type: 'root',
    children: [h('h2#intersection', ['(', katexInline('A \\cap B'), ')']), h('h2#next', 'Next')],
  }

  const toc = collectHtmlToc(tree, { maxDepth: 4, minEntries: 1 })

  assert.deepEqual(toc, [
    { depth: 0, text: '(A∩B)', slug: 'intersection' },
    { depth: 0, text: 'Next', slug: 'next' },
  ])

  const heading = tree.children[0]
  if (heading.type !== 'element') throw new Error('expected heading')
  assert.equal(heading.properties?.dataHeadingAlias, '(A∩B)')
})

test('skips headings rendered inside base embeds', () => {
  const tree: Root = {
    type: 'root',
    children: [
      h('h2#before', 'Before'),
      h('div.base-embed', [
        h('div.base-embed-view', [
          h('h2#book', 'Book title'),
          h('section', [h('h3', 'Generated base heading')]),
        ]),
      ]),
      h('h2#after', 'After'),
    ],
  }

  const toc = collectHtmlToc(tree, { maxDepth: 4, minEntries: 1 })

  assert.deepEqual(toc, [
    { depth: 0, text: 'Before', slug: 'before' },
    { depth: 0, text: 'After', slug: 'after' },
  ])
})

test('limits long html toc to the shallowest heading depth', () => {
  const children: Root['children'] = []
  for (let index = 0; index < 26; index++) {
    children.push(h(`h2#section-${index}`, `Section ${index}`))
    children.push(h(`h3#section-${index}-detail`, `Detail ${index}`))
  }

  const toc = collectHtmlTocData({ type: 'root', children }, { maxDepth: 3, minEntries: 1 })

  assert.equal(toc.sourceEntries, 52)
  assert.equal(toc.entries.length, 26)
  assert.deepEqual(toc.entries.slice(0, 2), [
    { depth: 0, text: 'Section 0', slug: 'section-0' },
    { depth: 0, text: 'Section 1', slug: 'section-1' },
  ])
  assert.equal(
    toc.entries.some(entry => entry.text.startsWith('Detail')),
    false,
  )
})

test('keeps detailed html toc at fifty entries', () => {
  const children: Root['children'] = []
  for (let index = 0; index < 25; index++) {
    children.push(h(`h2#section-${index}`, `Section ${index}`))
    children.push(h(`h3#section-${index}-detail`, `Detail ${index}`))
  }

  const toc = collectHtmlToc({ type: 'root', children }, { maxDepth: 3, minEntries: 1 })

  assert.equal(toc.length, 50)
  assert.equal(
    toc.some(entry => entry.text.startsWith('Detail')),
    true,
  )
})

test('limits long markdown toc to the shallowest heading depth', () => {
  const children: MarkdownRoot['children'] = []
  for (let index = 0; index < 26; index++) {
    children.push(markdownHeading(2, `Section ${index}`))
    children.push(markdownHeading(3, `Detail ${index}`))
  }

  const toc = collectMarkdownTocData({ type: 'root', children }, { maxDepth: 3, minEntries: 1 })

  assert.equal(toc.sourceEntries, 52)
  assert.equal(toc.entries.length, 26)
  assert.deepEqual(toc.entries.slice(0, 2), [
    { depth: 0, text: 'Section 0', slug: 'section-0' },
    { depth: 0, text: 'Section 1', slug: 'section-1' },
  ])
  assert.equal(
    toc.entries.some(entry => entry.text.startsWith('Detail')),
    false,
  )
})
