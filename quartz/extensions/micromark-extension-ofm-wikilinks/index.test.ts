/**
 * tests for micromark wikilink extension.
 * verifies tokenization and mdast node creation.
 */

import type { Root } from 'mdast'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { toHast } from 'mdast-util-to-hast'
import { toMarkdown } from 'mdast-util-to-markdown'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import { referenceHastHandler } from '../../util/references/render'
import { wikilinkFromMarkdown, Wikilink, isWikilink } from './fromMarkdown'
import { wikilink } from './syntax'
import { wikilinkToMarkdown } from './toMarkdown'

/**
 * helper to parse markdown with wikilink extension.
 */
function parse(markdown: string): Root {
  return fromMarkdown(markdown, {
    extensions: [wikilink()],
    mdastExtensions: [wikilinkFromMarkdown()],
  })
}

/**
 * helper to extract first wikilink node from AST.
 */
function extractWikilink(tree: Root): Wikilink | null {
  for (const child of tree.children) {
    if (child.type === 'paragraph') {
      for (const node of child.children) {
        if (isWikilink(node)) {
          return node
        }
      }
    }
  }
  return null
}

/**
 * helper to serialize markdown with wikilink extension.
 */
function serialize(tree: Root): string {
  return toMarkdown(tree, { extensions: [wikilinkToMarkdown()] })
}

function render(node: Wikilink) {
  const element = toHast(node, { handlers: { wikilink: referenceHastHandler } })
  assert(element?.type === 'element')
  return element
}

describe('micromark wikilink extension', () => {
  describe('edge cases', () => {
    test('parses empty alias', () => {
      const tree = parse('[[target|]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.referenceFields?.target, 'target')
      // Empty authored labels retain their source delimiter.
      assert.strictEqual(wikilink.data?.referenceFields?.label, '')
    })

    test('parses empty anchor', () => {
      const tree = parse('[[target#]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.referenceFields?.target, 'target')
      // empty anchor results in undefined (no content after #)
      assert.deepStrictEqual(wikilink.data?.referenceFields?.fragments, [])
    })

    test('handles multiple anchors (subheadings)', () => {
      const tree = parse('[[file#heading#subheading]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
      // Parsing retains every fragment before target interpretation.
      assert.deepStrictEqual(wikilink.data?.referenceFields?.fragments, ['heading', 'subheading'])
    })
  })

  describe('escaping', () => {
    test('handles escaped pipe in target', () => {
      const tree = parse('[[file\\|name]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      // backslash is consumed, and the pipe acts as an alias delimiter
      assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
      assert.strictEqual(wikilink.data?.referenceFields?.label, 'name')
    })

    test('handles escaped hash in target', () => {
      const tree = parse('[[file\\#name]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      // backslash is consumed during parsing
      assert.strictEqual(wikilink.data?.referenceFields?.target, 'file#name')
      assert.deepStrictEqual(wikilink.data?.referenceFields?.fragments, [])
    })

    test('handles escaped pipe with alias', () => {
      const tree = parse('[[file\\|name|alias]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      // the first escaped pipe acts as an alias delimiter, the second is just text in the alias
      assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
      assert.strictEqual(wikilink.data?.referenceFields?.label, 'name|alias')
    })

    test('handles escaped bracket in alias', () => {
      const tree = parse('[[file|alias\\]text]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
      // backslash is consumed, bracket is part of alias
      assert.strictEqual(wikilink.data?.referenceFields?.label, 'alias]text')
    })
  })
  describe('context awareness', () => {
    test('does not parse in code block', () => {
      const tree = parse('```\n[[not a link]]\n```')
      const codeBlock = tree.children[0]

      assert(codeBlock.type === 'code')
      assert(codeBlock.value.includes('[[not a link]]'))
    })

    test('does not parse in inline code', () => {
      const tree = parse('`[[not a link]]`')
      const paragraph = tree.children[0]
      assert(paragraph.type === 'paragraph')
      const inlineCode = paragraph.children[0]

      assert(inlineCode.type === 'inlineCode')
      assert(inlineCode.value.includes('[[not a link]]'))
    })
  })

  describe('malformed input', () => {
    test('does not parse single bracket', () => {
      const tree = parse('[not a link]')
      const wikilink = extractWikilink(tree)

      assert.strictEqual(wikilink, null)
    })

    test('does not parse mismatched brackets', () => {
      const tree = parse('[[incomplete')
      const wikilink = extractWikilink(tree)

      assert.strictEqual(wikilink, null, 'incomplete wikilink should not parse')
    })

    test('does not parse reversed brackets', () => {
      const tree = parse(']]reversed[[')
      const wikilink = extractWikilink(tree)

      assert.strictEqual(wikilink, null)
    })
  })
  describe('toMarkdown serialization', () => {
    describe('embed serialization', () => {
      test('serializes embed with anchor', () => {
        const tree = parse('![[file#section]]')
        const result = serialize(tree)
        assert(result.includes('![[file#section]]'))
      })
    })

    describe('edge case serialization', () => {
      test('serializes multiple anchors (subheadings)', () => {
        const tree = parse('[[file#heading#subheading]]')
        const result = serialize(tree)
        assert(result.includes('[[file#heading#subheading]]'))
      })
    })

    describe('escaping in serialization', () => {
      test('escapes pipe with alias', () => {
        const tree = parse('[[file\\|name|alias]]')
        const result = serialize(tree)
        // pipe in target should be escaped, alias delimiter should not
        assert(result.includes('[[file\\|name|alias]]'))
      })
    })

    describe('round-trip consistency', () => {
      const testCases = [
        '[[test]]',
        '[[page|alias]]',
        '[[page#section]]',
        '[[page#section|alias]]',
        '[[file#^block]]',
        '![[embed]]',
        '![[image.png|100x200]]',
        '[[]]',
        '[[#heading]]',
        '[[|display]]',
        '[[file\\|name]]',
        '[[file\\#name]]',
        '[[file|alias\\]text]]',
        '[[path/to/file]]',
        '[[file (with) parens]]',
      ]

      for (const input of testCases) {
        test(`round-trip: ${input}`, () => {
          const tree = parse(input)
          const result = serialize(tree)
          // normalize whitespace for comparison
          const normalized = result.trim()
          assert(normalized.includes(input), `expected "${normalized}" to include "${input}"`)
        })
      }
    })

    describe('multiple wikilinks serialization', () => {
      test('serializes multiple wikilinks in paragraph', () => {
        const input = 'Here is [[link1]] and [[link2|alias]].'
        const tree = parse(input)
        const result = serialize(tree)
        assert(result.includes('[[link1]]'))
        assert(result.includes('[[link2|alias]]'))
      })
    })
  })

  describe('syntax and standalone target interpretation', () => {
    describe('nested heading anchors', () => {
      test('uses the final heading for standalone fallback rendering', () => {
        const tree = parse('[[file#Parent#Child]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
        // Standalone rendering has no destination heading index.
        assert.strictEqual(render(wikilink).properties.href, 'file#child')
      })
    })

    describe('anchor slugification', () => {
      test('preserves math without adjacent spaces', () => {
        const tree = parse('[[file#$mu$ value]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
        // no spaces to remove around $
        assert.strictEqual(render(wikilink).properties.href, 'file#mu-value')
      })

      test('handles mixed spacing patterns', () => {
        const tree = parse('[[file#test $ a$ and $b $ plus $c$]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
        // "$ a$" → "$a$", "$b $" → "$b$", "$c$" unchanged
        assert.strictEqual(render(wikilink).properties.href, 'file#test-a-and-b-plus-c')
      })

      test('preserves authored heading text in syntax', () => {
        const tree = parse('[[file#Section Title]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
        // Decoded fields retain the authored heading text.
        assert.deepStrictEqual(wikilink.data?.referenceFields?.fragments, ['Section Title'])
      })
    })
    describe('block reference fragments', () => {
      test('preserves each authored block-reference fragment', () => {
        const tree = parse('[[file#^block#with#hashes]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
        // The block marker stays with its authored fragment.
        assert.deepStrictEqual(wikilink.data?.referenceFields?.fragments, [
          '^block',
          'with',
          'hashes',
        ])
      })
    })

    describe('fragment edge cases', () => {
      test('handles empty segments in nested path', () => {
        const tree = parse('[[file#Parent##Child]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
        // last segment is "Child", empty segment is ignored
        assert.strictEqual(render(wikilink).properties.href, 'file#child')
      })

      test('works with embeds', () => {
        const tree = parse('![[file#Parent#Child]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.referenceFields?.target, 'file')
        const element = render(wikilink)
        const inner = element.children[0]
        assert(inner.type === 'element')
        assert.strictEqual(inner.properties.href, 'file#child')
        assert.strictEqual(wikilink.data?.reference?.embed, true)
      })

      test('keeps authored path-context text in the default label', () => {
        const tree = parse('[[#thoughts/Epistemology epistemic humility]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.referenceFields?.target, '')
        // should extract "epistemic humility" from after the last /
        assert.strictEqual(render(wikilink).properties.href, '#epistemic-humility')
        assert.deepStrictEqual(wikilink.data?.referenceFields?.fragments, [
          'thoughts/Epistemology epistemic humility',
        ])
        assert.deepStrictEqual(render(wikilink).children, [
          { type: 'text', value: 'thoughts/Epistemology epistemic humility' },
        ])
      })
    })
  })

  describe('explicit HAST conversion', () => {
    describe('image embeds', () => {
      test('annotates image with multi-part caption', () => {
        const tree = parse('![[photo.jpg|Caption with|multiple|pipes]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(render(wikilink).tagName, 'img')

        const figcaption = render(wikilink).properties?.alt
        assert.strictEqual(figcaption, 'Caption with|multiple|pipes')
      })

      test('annotates image with multi-part caption and dimensions', () => {
        const tree = parse('![[photo.jpg|Caption with|multiple parts|800x600]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(render(wikilink).tagName, 'img')

        assert.strictEqual(String(render(wikilink).properties?.width), '800')
        assert.strictEqual(String(render(wikilink).properties?.height), '600')
        assert.strictEqual(render(wikilink).properties?.alt, 'Caption with|multiple parts')
      })
    })
    describe('parser data', () => {
      test('keeps parsed nodes independent of HAST rendering', () => {
        const tree = parse('[[target]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.hName, undefined)
        assert.strictEqual(wikilink.data?.hProperties, undefined)
        assert.strictEqual(wikilink.data?.hChildren, undefined)
        assert.strictEqual(wikilink.data?.referenceFields?.target, 'target')
      })
    })
  })

  describe('anchor display text', () => {
    test('keeps original anchor text in data when no alias', () => {
      const tree = parse('[[page#Section Title]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(render(wikilink).properties.href, 'page#section-title')
      assert.deepStrictEqual(wikilink.data?.referenceFields?.fragments, ['Section Title'])
      assert.deepStrictEqual(render(wikilink).children, [
        { type: 'text', value: 'page > Section Title' },
      ])
    })

    test('keeps anchor text for same-file reference', () => {
      const tree = parse('[[#knowledge]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.referenceFields?.target, '')
      assert.strictEqual(render(wikilink).properties.href, '#knowledge')
      assert.deepStrictEqual(wikilink.data?.referenceFields?.fragments, ['knowledge'])
      assert.deepStrictEqual(render(wikilink).children, [{ type: 'text', value: 'knowledge' }])
    })

    test('keeps anchor text for block reference', () => {
      const tree = parse('[[file#^block-id]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.deepStrictEqual(wikilink.data?.referenceFields?.fragments, ['^block-id'])
      assert.deepStrictEqual(render(wikilink).children, [
        { type: 'text', value: 'file > block-id' },
      ])
    })
  })
})
