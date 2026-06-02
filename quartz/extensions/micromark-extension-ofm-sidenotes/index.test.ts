import type { Paragraph, PhrasingContent, Root } from 'mdast'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { mathFromMarkdown, type InlineMath } from 'mdast-util-math'
import { math } from 'micromark-extension-math'
import assert from 'node:assert/strict'
import test, { describe } from 'node:test'
import { sidenote, sidenoteFromMarkdown, type Sidenote } from './index'

function parse(markdown: string): Root {
  const innerMathExtension = math()
  const innerMathFromMarkdown = mathFromMarkdown()

  return fromMarkdown(markdown, {
    extensions: [sidenote(), math()],
    mdastExtensions: [
      sidenoteFromMarkdown({
        micromarkExtensions: [innerMathExtension],
        mdastExtensions: [innerMathFromMarkdown],
      }),
      mathFromMarkdown(),
    ],
  })
}

function firstParagraph(tree: Root): Paragraph {
  const node = tree.children[0]
  if (node.type !== 'paragraph') {
    assert.fail(`expected paragraph, received ${node.type}`)
  }
  return node
}

function firstSidenote(tree: Root): Sidenote {
  const note = firstParagraph(tree).children.find(isSidenote)
  assert(note)
  return note
}

function isSidenote(node: PhrasingContent): node is Sidenote {
  return node.type === 'sidenote'
}

function isInlineMath(node: PhrasingContent): node is InlineMath {
  return node.type === 'inlineMath'
}

describe('micromark sidenote extension', () => {
  test('parses inline sidenote content with LaTeX braces', () => {
    const content =
      '$e^{x_i}$ exceeds the FP16 ceiling ($\\approx 65504$) once $x_i > 11$, and FP32 near $x_i \\approx 88$.'
    const tree = parse(
      `With numerically-stable softmax, exponentiating directly will result in {{sidenotes[overflows]: ${content}}}`,
    )
    const note = firstSidenote(tree)

    assert.equal(note.data?.sidenoteParsed?.label, 'overflows')
    assert.equal(note.data?.sidenoteParsed?.content, ` ${content}`)
    assert.deepEqual(
      note.children.filter(isInlineMath).map(node => node.value),
      ['e^{x_i}', '\\approx 65504', 'x_i > 11', 'x_i \\approx 88'],
    )
  })

  test('keeps single closing braces inside sidenote content', () => {
    const tree = parse('Text {{sidenotes[label]: value } keeps going}} after')
    const note = firstSidenote(tree)

    assert.equal(note.data?.sidenoteParsed?.content, ' value } keeps going')
  })
})
