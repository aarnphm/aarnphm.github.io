/**
 * tests for micromark wikilink extension.
 * verifies tokenization and mdast node creation.
 */

import type { Root } from 'mdast'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { toMarkdown } from 'mdast-util-to-markdown'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import { wikilinkFromMarkdown, Wikilink, isWikilink } from './fromMarkdown'
import { wikilink } from './syntax'
import { wikilinkToMarkdown } from './toMarkdown'

/**
 * helper to parse markdown with wikilink extension.
 */
function parse(
  markdown: string,
  options?: { obsidian?: boolean; stripExtensions?: string[]; hasSlug?: (slug: string) => boolean },
): Root {
  return fromMarkdown(markdown, {
    extensions: [wikilink()],
    mdastExtensions: [wikilinkFromMarkdown(options)],
  })
}

/**
 * helper to extract first wikilink node from AST.
 */
function extractWikilink(tree: Root): Wikilink | null {
  for (const child of tree.children) {
    if (child.type === 'paragraph') {
      for (const node of (child as any).children) {
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

describe('micromark wikilink extension', () => {
  describe('edge cases', () => {
    test('parses empty alias', () => {
      const tree = parse('[[target|]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'target')
      // empty alias results in undefined (no content after |)
      assert.strictEqual(wikilink.data?.wikilink.alias, undefined)
    })

    test('parses empty anchor', () => {
      const tree = parse('[[target#]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'target')
      // empty anchor results in undefined (no content after #)
      assert.strictEqual(wikilink.data?.wikilink.anchor, undefined)
    })

    test('handles multiple anchors (subheadings)', () => {
      const tree = parse('[[file#heading#subheading]]', { obsidian: false })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'file')
      // non-obsidian mode preserves anchor as-is
      assert.strictEqual(wikilink.data?.wikilink.anchor, '#heading#subheading')
    })
  })

  describe('escaping', () => {
    test('handles escaped pipe in target', () => {
      const tree = parse('[[file\\|name]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      // backslash is consumed, and the pipe acts as an alias delimiter
      assert.strictEqual(wikilink.data?.wikilink.target, 'file')
      assert.strictEqual(wikilink.data?.wikilink.alias, 'name')
    })

    test('handles escaped hash in target', () => {
      const tree = parse('[[file\\#name]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      // backslash is consumed during parsing
      assert.strictEqual(wikilink.data?.wikilink.target, 'file#name')
      assert.strictEqual(wikilink.data?.wikilink.anchor, undefined)
    })

    test('handles escaped pipe with alias', () => {
      const tree = parse('[[file\\|name|alias]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      // the first escaped pipe acts as an alias delimiter, the second is just text in the alias
      assert.strictEqual(wikilink.data?.wikilink.target, 'file')
      assert.strictEqual(wikilink.data?.wikilink.alias, 'name|alias')
    })

    test('handles escaped bracket in alias', () => {
      const tree = parse('[[file|alias\\]text]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'file')
      // backslash is consumed, bracket is part of alias
      assert.strictEqual(wikilink.data?.wikilink.alias, 'alias]text')
    })
  })

  describe('complex paths', () => {
    test('handles paths with slashes', () => {
      const tree = parse('[[path/to/file]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'path/to/file')
    })

    test('handles paths with special chars', () => {
      const tree = parse('[[file (with) parens]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'file (with) parens')
    })
  })

  describe('context awareness', () => {
    test('does not parse in code block', () => {
      const tree = parse('```\n[[not a link]]\n```')
      const codeBlock = tree.children[0]

      assert.strictEqual(codeBlock.type, 'code')
      assert((codeBlock as any).value.includes('[[not a link]]'))
    })

    test('does not parse in inline code', () => {
      const tree = parse('`[[not a link]]`')
      const paragraph = tree.children[0] as any
      const inlineCode = paragraph.children[0]

      assert.strictEqual(inlineCode.type, 'inlineCode')
      assert((inlineCode as any).value.includes('[[not a link]]'))
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

  describe('position tracking', () => {
    test('preserves position information', () => {
      const tree = parse('[[test]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert(wikilink.position, 'position should be defined')
      assert.strictEqual(wikilink.position!.start.column, 1)
      assert.strictEqual(wikilink.position!.start.line, 1)
      assert(wikilink.position!.end.column > wikilink.position!.start.column)
    })

    test('tracks position with embed prefix', () => {
      const tree = parse('![[embed]]')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert(wikilink.position, 'position should be defined')
      assert.strictEqual(wikilink.position!.start.column, 1)
    })

    test('tracks position in middle of line', () => {
      const tree = parse('some text [[link]] more text')
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert(wikilink.position, 'position should be defined')
      assert(wikilink.position!.start.column > 1, "should start after 'some text '")
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

  describe('obsidian mode', () => {
    describe('nested heading anchors', () => {
      test('uses last segment with obsidian: true', () => {
        const tree = parse('[[file#Parent#Child#Grandchild]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // should use only "grandchild" and slugify it
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#grandchild')
      })

      test('preserves full anchor with obsidian: false (default)', () => {
        const tree = parse('[[file#Parent#Child#Grandchild]]', { obsidian: false })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // non-obsidian mode: preserves full anchor as-is
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#Parent#Child#Grandchild')
      })

      test('uses last segment when option not specified (defaults to obsidian: true)', () => {
        const tree = parse('[[file#Parent#Child]]')
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // default mode (obsidian: true): uses last segment
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#child')
      })
    })

    describe('anchor slugification', () => {
      test('slugifies LaTeX in anchor', () => {
        const tree = parse('[[file#architectural skeleton of $ mu$]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // spaces around $ should be removed before slugification
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#architectural-skeleton-of-mu')
      })

      test('normalizes inline math in context', () => {
        const tree = parse('[[file#Section with $ x + y$ equation]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // spaces around $ removed, then slugified
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#section-with-x--y-equation')
      })

      test('normalizes multiple math expressions', () => {
        const tree = parse('[[file#$ alpha$ and $ beta$]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // "$ alpha$ and $ beta$" → "$alpha$ and $beta$" → slugged: "alpha-and-beta"
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#alpha-and-beta')
      })

      test('handles math with multiple spaces', () => {
        const tree = parse('[[file#$  x  $]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // "$  x  $" → "$x$" → slugged: "x"
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#x')
      })

      test('preserves math without adjacent spaces', () => {
        const tree = parse('[[file#$mu$ value]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // no spaces to remove around $
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#mu-value')
      })

      test('handles mixed spacing patterns', () => {
        const tree = parse('[[file#test $ a$ and $b $ plus $c$]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // "$ a$" → "$a$", "$b $" → "$b$", "$c$" unchanged
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#test-a-and-b-plus-c')
      })

      test('preserves anchors without obsidian mode', () => {
        const tree = parse('[[file#Section Title]]', { obsidian: false })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // non-obsidian mode: preserves anchor as-is
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#Section Title')
      })
    })

    describe('nested headings with slugification', () => {
      test('combines nested heading extraction and slugification', () => {
        const tree = parse('[[file#Parent Heading#Child Section]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // should extract "Child Section" and slugify to "child-section"
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#child-section')
      })

      test('handles mixed case nested headings', () => {
        const tree = parse('[[file#First#SECOND#ThIrD]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#third')
      })
    })

    describe('block references with obsidian mode', () => {
      test('preserves block reference marker', () => {
        const tree = parse('[[file#^block-id]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // block reference should keep ^ marker
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#^block-id')
      })

      test('does not process # in block reference content', () => {
        const tree = parse('[[file#^block#with#hashes]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // should extract last segment "hashes" but keep block marker
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#^hashes')
      })
    })

    describe('edge cases with obsidian mode', () => {
      test('handles empty segments in nested path', () => {
        const tree = parse('[[file#Parent##Child]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        // last segment is "Child", empty segment is ignored
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#child')
      })

      test('works with embeds', () => {
        const tree = parse('![[file#Parent#Child]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, 'file')
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#child')
        assert.strictEqual(wikilink.data?.wikilink.embed, true)
      })

      test('handles anchors with path context (Obsidian format)', () => {
        const tree = parse('[[#thoughts/Epistemology epistemic humility]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.wikilink.target, '')
        // should extract "epistemic humility" from after the last /
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#epistemic-humility')
        assert.strictEqual(wikilink.data?.wikilink.anchorText, 'epistemic humility')
        assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: 'text', value: '' }])
      })
    })
  })

  describe('hName annotation (automatic hast conversion)', () => {
    describe('image embeds', () => {
      test('annotates image with caption text (figure)', () => {
        const tree = parse('![[photo.jpg|A beautiful sunset]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.hName, 'img')
        assert.strictEqual(wikilink.data?.hProperties?.src, 'photo.jpg')
        assert.strictEqual(wikilink.data?.hProperties?.width, 'auto')
        assert.strictEqual(wikilink.data?.hProperties?.height, 'auto')
        assert.strictEqual(wikilink.data?.hProperties?.alt, 'A beautiful sunset')
      })

      test('annotates image with dimensions (width only)', () => {
        const tree = parse('![[photo.jpg|400]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.hName, 'img')
        assert.strictEqual(wikilink.data?.hProperties?.src, 'photo.jpg')
        assert.strictEqual(wikilink.data?.hProperties?.width, '400')
        assert.strictEqual(wikilink.data?.hProperties?.height, 'auto')
      })

      test('annotates image with multi-part caption', () => {
        const tree = parse('![[photo.jpg|Caption with|multiple|pipes]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.hName, 'img')

        const figcaption = wikilink.data?.hProperties?.alt as any
        assert.strictEqual(figcaption, 'Caption with|multiple|pipes')
      })

      test('annotates image with multi-part caption and dimensions', () => {
        const tree = parse('![[photo.jpg|Caption with|multiple parts|800x600]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.hName, 'img')

        assert.strictEqual(wikilink?.data.hProperties?.width, '800')
        assert.strictEqual(wikilink?.data.hProperties?.height, '600')
        assert.strictEqual(wikilink?.data.hProperties?.alt, 'Caption with|multiple parts')
      })
    })

    describe('PDF embeds', () => {
      test('annotates PDF embed options', () => {
        const tree = parse(
          "![[paper.pdf#{page: 3, fit: 'page', height: 720, overscan: 4}|slides]]",
          { obsidian: true },
        )
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.hName, 'div')
        assert.strictEqual(wikilink.data?.hProperties?.['data-pdf-src'], 'paper.pdf')
        assert.strictEqual(wikilink.data?.hProperties?.['data-pdf-title'], 'slides')
        assert.strictEqual(wikilink.data?.hProperties?.['data-pdf-page'], '3')
        assert.strictEqual(wikilink.data?.hProperties?.['data-pdf-fit'], 'page')
        assert.strictEqual(wikilink.data?.hProperties?.['data-pdf-height'], '720')
        assert.strictEqual(wikilink.data?.hProperties?.['data-pdf-overscan'], '4')
      })
    })

    describe('block transcludes', () => {
      test('annotates transclude with anchor', () => {
        const tree = parse('![[notes#summary]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.hName, 'blockquote')
        assert.strictEqual(wikilink.data?.hProperties?.class, 'transclude')
        assert.strictEqual(wikilink.data?.hProperties?.['data-url'], 'notes')
        assert.strictEqual(wikilink.data?.hProperties?.['data-block'], '#summary')
        assert(Array.isArray(wikilink.data?.hChildren), 'should have hChildren array')

        const innerLink = wikilink.data?.hChildren?.[0] as any
        assert.strictEqual(innerLink?.type, 'element')
        assert.strictEqual(innerLink?.tagName, 'a')
        assert.strictEqual(innerLink?.properties?.href, 'notes#summary')
        assert.strictEqual(innerLink?.properties?.class, 'transclude-inner')
      })

      test('annotates transclude with metadata', () => {
        const tree = parse('![[thoughts/Attention#{collapsed: true}]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.hName, 'blockquote')
        assert.strictEqual(wikilink.data?.hProperties?.['data-url'], 'thoughts/Attention')
        assert.strictEqual(wikilink.data?.hProperties?.['data-block'], '')
        assert.strictEqual(wikilink.data?.hProperties?.['data-metadata'], '{"collapsed":true}')
      })
    })

    describe('without explicit options', () => {
      test('does not annotate when explicitly obsidian: false', () => {
        const tree = parse('[[target]]', { obsidian: false })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        assert.strictEqual(wikilink.data?.hName, undefined)
        assert.strictEqual(wikilink.data?.hProperties, undefined)
        assert.strictEqual(wikilink.data?.hChildren, undefined)
        assert.strictEqual(wikilink.data?.wikilink.target, 'target')
      })
    })

    describe('integration with obsidian mode', () => {
      test('combines obsidian anchor handling with hName annotation', () => {
        const tree = parse('[[file#Parent#Child|display]]', { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, 'wikilink node should exist')
        // obsidian mode: uses last anchor segment
        assert.strictEqual(wikilink.data?.wikilink.anchor, '#child')
        // hName annotation
        assert.strictEqual(wikilink.data?.hName, 'a')
        assert.strictEqual(wikilink.data?.hProperties?.href, 'file#child')
        assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: 'text', value: 'display' }])
      })
    })
  })

  describe('stripExtensions option', () => {
    test('only strips first matching extension', () => {
      const tree = parse('[[file.md.backup]]', { obsidian: true, stripExtensions: ['.backup'] })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.hName, 'a')
      // slugAnchor removes dots, so file.md becomes filemd
      assert.strictEqual(wikilink.data?.hProperties?.href, 'file.md')
    })
  })

  describe('implicit alias from space in target', () => {
    const implicitAliasSlugs = new Set(['Target', 'thoughts/Epistemology', 'folder/file', 'file'])
    const hasSlug = (slug: string) => implicitAliasSlugs.has(slug)

    test('extracts display text after space in simple target', () => {
      const tree = parse('[[Target display text]]', { obsidian: true, hasSlug })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'Target')
      assert.strictEqual(wikilink.data?.wikilink.alias, 'display text')
      assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: 'text', value: 'display text' }])
    })

    test('extracts display text after space in path with slashes', () => {
      const tree = parse('[[thoughts/Epistemology epistemic humility]]', {
        obsidian: true,
        hasSlug,
      })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'thoughts/Epistemology')
      assert.strictEqual(wikilink.data?.wikilink.alias, 'epistemic humility')
      assert.deepStrictEqual(wikilink.data?.hChildren, [
        { type: 'text', value: 'epistemic humility' },
      ])
    })

    test('handles multiple words in display text', () => {
      const tree = parse('[[folder/file this is the display]]', { obsidian: true, hasSlug })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'folder/file')
      assert.strictEqual(wikilink.data?.wikilink.alias, 'this is the display')
    })

    test('does not extract when explicit alias is present', () => {
      const tree = parse('[[target with space|explicit alias]]', { obsidian: true, hasSlug })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'target with space')
      assert.strictEqual(wikilink.data?.wikilink.alias, 'explicit alias')
    })

    test('works with anchors', () => {
      const tree = parse('[[file name display#section]]', { obsidian: true, hasSlug })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, 'file')
      assert.strictEqual(wikilink.data?.wikilink.alias, 'name display')
      assert.strictEqual(wikilink.data?.wikilink.anchor, '#section')
    })
  })

  describe('anchor display text', () => {
    test('keeps original anchor text in data when no alias', () => {
      const tree = parse('[[page#Section Title]]', { obsidian: true })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.anchor, '#section-title')
      assert.strictEqual(wikilink.data?.wikilink.anchorText, 'Section Title')
      assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: 'text', value: 'page' }])
    })

    test('keeps anchor text for same-file reference', () => {
      const tree = parse('[[#knowledge]]', { obsidian: true })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.target, '')
      assert.strictEqual(wikilink.data?.wikilink.anchor, '#knowledge')
      assert.strictEqual(wikilink.data?.wikilink.anchorText, 'knowledge')
      assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: 'text', value: '' }])
    })

    test('keeps anchor text for block reference', () => {
      const tree = parse('[[file#^block-id]]', { obsidian: true })
      const wikilink = extractWikilink(tree)

      assert(wikilink, 'wikilink node should exist')
      assert.strictEqual(wikilink.data?.wikilink.anchorText, 'block-id')
      assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: 'text', value: 'file' }])
    })
  })
})
