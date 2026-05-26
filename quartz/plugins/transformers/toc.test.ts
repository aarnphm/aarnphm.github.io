import type { Root } from 'hast'
import { h } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import { visit } from 'unist-util-visit'
import { collectHtmlToc } from './toc'

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
