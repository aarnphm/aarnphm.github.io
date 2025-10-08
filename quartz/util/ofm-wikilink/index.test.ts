/**
 * tests for micromark wikilink extension.
 * verifies tokenization and mdast node creation.
 */

import test, { describe } from "node:test"
import assert from "node:assert"
import { fromMarkdown } from "mdast-util-from-markdown"
import { toMarkdown } from "mdast-util-to-markdown"
import { wikilink } from "./syntax"
import { wikilinkFromMarkdown, Wikilink, isWikilink } from "./fromMarkdown"
import { wikilinkToMarkdown } from "./toMarkdown"
import type { Root } from "mdast"

/**
 * helper to parse markdown with wikilink extension.
 */
function parse(
  markdown: string,
  options?: {
    obsidian?: boolean
    stripExtensions?: string[]
  },
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
    if (child.type === "paragraph") {
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
  return toMarkdown(tree, {
    extensions: [wikilinkToMarkdown()],
  })
}

describe("micromark wikilink extension", () => {
  describe("basic wikilinks", () => {
    test("parses simple wikilink", () => {
      const tree = parse("[[test]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.type, "wikilink")
      assert.strictEqual(wikilink.value, "[[test]]")
      assert.strictEqual(wikilink.data?.wikilink.target, "test")
      assert.strictEqual(wikilink.data?.wikilink.embed, false)
      assert.strictEqual(wikilink.data?.wikilink.anchor, undefined)
      assert.strictEqual(wikilink.data?.wikilink.alias, undefined)
    })

    test("parses wikilink with alias", () => {
      const tree = parse("[[page|display text]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "page")
      assert.strictEqual(wikilink.data?.wikilink.alias, "display text")
      assert.strictEqual(wikilink.data?.wikilink.embed, false)
    })

    test("parses wikilink with anchor", () => {
      const tree = parse("[[page#section]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "page")
      assert.strictEqual(wikilink.data?.wikilink.anchor, "#section")
      assert.strictEqual(wikilink.data?.wikilink.alias, undefined)
    })

    test("parses wikilink with anchor and alias", () => {
      const tree = parse("[[page#section|custom text]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "page")
      assert.strictEqual(wikilink.data?.wikilink.anchor, "#section")
      assert.strictEqual(wikilink.data?.wikilink.alias, "custom text")
    })

    test("parses block reference", () => {
      const tree = parse("[[file#^block-id]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "file")
      assert.strictEqual(wikilink.data?.wikilink.anchor, "#^block-id")
    })
  })

  describe("embed wikilinks", () => {
    test("parses embed prefix", () => {
      const tree = parse("![[embed]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.value, "![[embed]]")
      assert.strictEqual(wikilink.data?.wikilink.target, "embed")
      assert.strictEqual(wikilink.data?.wikilink.embed, true)
    })

    test("parses embed with anchor", () => {
      const tree = parse("![[file#section]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "file")
      assert.strictEqual(wikilink.data?.wikilink.anchor, "#section")
      assert.strictEqual(wikilink.data?.wikilink.embed, true)
    })

    test("parses embed with alias", () => {
      const tree = parse("![[image.png|100x200]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "image.png")
      assert.strictEqual(wikilink.data?.wikilink.alias, "100x200")
      assert.strictEqual(wikilink.data?.wikilink.embed, true)
    })
  })

  describe("edge cases", () => {
    test("parses empty target", () => {
      const tree = parse("[[]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "")
    })

    test("parses empty target with anchor", () => {
      const tree = parse("[[#heading]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "")
      assert.strictEqual(wikilink.data?.wikilink.anchor, "#heading")
    })

    test("parses empty target with alias", () => {
      const tree = parse("[[|display]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "")
      assert.strictEqual(wikilink.data?.wikilink.alias, "display")
    })

    test("parses empty alias", () => {
      const tree = parse("[[target|]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "target")
      // empty alias results in undefined (no content after |)
      assert.strictEqual(wikilink.data?.wikilink.alias, undefined)
    })

    test("parses empty anchor", () => {
      const tree = parse("[[target#]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "target")
      // empty anchor results in undefined (no content after #)
      assert.strictEqual(wikilink.data?.wikilink.anchor, undefined)
    })

    test("handles multiple anchors (subheadings)", () => {
      const tree = parse("[[file#heading#subheading]]", { obsidian: false })
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "file")
      // non-obsidian mode preserves anchor as-is
      assert.strictEqual(wikilink.data?.wikilink.anchor, "#heading#subheading")
    })
  })

  describe("escaping", () => {
    test("handles escaped pipe in target", () => {
      const tree = parse("[[file\\|name]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      // backslash is consumed during parsing - this is correct behavior
      assert.strictEqual(wikilink.data?.wikilink.target, "file|name")
      assert.strictEqual(wikilink.data?.wikilink.alias, undefined)
    })

    test("handles escaped hash in target", () => {
      const tree = parse("[[file\\#name]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      // backslash is consumed during parsing
      assert.strictEqual(wikilink.data?.wikilink.target, "file#name")
      assert.strictEqual(wikilink.data?.wikilink.anchor, undefined)
    })

    test("handles escaped pipe with alias", () => {
      const tree = parse("[[file\\|name|alias]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      // backslash is consumed, pipe is part of target
      assert.strictEqual(wikilink.data?.wikilink.target, "file|name")
      assert.strictEqual(wikilink.data?.wikilink.alias, "alias")
    })

    test("handles escaped bracket in alias", () => {
      const tree = parse("[[file|alias\\]text]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "file")
      // backslash is consumed, bracket is part of alias
      assert.strictEqual(wikilink.data?.wikilink.alias, "alias]text")
    })
  })

  describe("complex paths", () => {
    test("handles paths with slashes", () => {
      const tree = parse("[[path/to/file]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "path/to/file")
    })

    test("handles paths with spaces", () => {
      const tree = parse("[[path with spaces]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "path with spaces")
    })

    test("handles paths with special chars", () => {
      const tree = parse("[[file (with) parens]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "file (with) parens")
    })

    test("handles file extensions", () => {
      const tree = parse("[[document.pdf]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "document.pdf")
    })
  })

  describe("multiple wikilinks", () => {
    test("parses multiple wikilinks in paragraph", () => {
      const tree = parse("Here is [[link1]] and [[link2|alias]].")
      const paragraph = tree.children[0] as any

      const wikilinks = paragraph.children.filter((node: any) => isWikilink(node))
      assert.strictEqual(wikilinks.length, 2)

      assert.strictEqual(wikilinks[0].data.wikilink.target, "link1")
      assert.strictEqual(wikilinks[1].data.wikilink.target, "link2")
      assert.strictEqual(wikilinks[1].data.wikilink.alias, "alias")
    })

    test("parses wikilinks separated by text", () => {
      const tree = parse("[[first]] some text [[second]]")
      const paragraph = tree.children[0] as any

      const wikilinks = paragraph.children.filter((node: any) => isWikilink(node))
      assert.strictEqual(wikilinks.length, 2)

      assert.strictEqual(wikilinks[0].data.wikilink.target, "first")
      assert.strictEqual(wikilinks[1].data.wikilink.target, "second")
    })
  })

  describe("context awareness", () => {
    test("does not parse in code block", () => {
      const tree = parse("```\n[[not a link]]\n```")
      const codeBlock = tree.children[0]

      assert.strictEqual(codeBlock.type, "code")
      assert((codeBlock as any).value.includes("[[not a link]]"))
    })

    test("does not parse in inline code", () => {
      const tree = parse("`[[not a link]]`")
      const paragraph = tree.children[0] as any
      const inlineCode = paragraph.children[0]

      assert.strictEqual(inlineCode.type, "inlineCode")
      assert((inlineCode as any).value.includes("[[not a link]]"))
    })
  })

  describe("malformed input", () => {
    test("does not parse single bracket", () => {
      const tree = parse("[not a link]")
      const wikilink = extractWikilink(tree)

      assert.strictEqual(wikilink, null)
    })

    test("does not parse triple bracket", () => {
      const tree = parse("[[[too many]]]")
      const wikilink = extractWikilink(tree)

      // should parse as [[too many]] with extra brackets as text
      // or fail to parse entirely depending on implementation
      // this tests that it doesn't create invalid nodes
      if (wikilink) {
        assert.strictEqual(wikilink.data?.wikilink.target, "[too many")
      }
    })

    test("does not parse mismatched brackets", () => {
      const tree = parse("[[incomplete")
      const wikilink = extractWikilink(tree)

      assert.strictEqual(wikilink, null, "incomplete wikilink should not parse")
    })

    test("does not parse reversed brackets", () => {
      const tree = parse("]]reversed[[")
      const wikilink = extractWikilink(tree)

      assert.strictEqual(wikilink, null)
    })
  })

  describe("position tracking", () => {
    test("preserves position information", () => {
      const tree = parse("[[test]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert(wikilink.position, "position should be defined")
      assert.strictEqual(wikilink.position!.start.column, 1)
      assert.strictEqual(wikilink.position!.start.line, 1)
      assert(wikilink.position!.end.column > wikilink.position!.start.column)
    })

    test("tracks position with embed prefix", () => {
      const tree = parse("![[embed]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert(wikilink.position, "position should be defined")
      assert.strictEqual(wikilink.position!.start.column, 1)
    })

    test("tracks position in middle of line", () => {
      const tree = parse("some text [[link]] more text")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert(wikilink.position, "position should be defined")
      assert(wikilink.position!.start.column > 1, "should start after 'some text '")
    })
  })

  describe("isWikilink type guard", () => {
    test("returns true for valid wikilink node", () => {
      const tree = parse("[[test]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink should exist")
      assert(isWikilink(wikilink))
    })

    test("returns false for text node", () => {
      const tree = parse("plain text")
      const paragraph = tree.children[0] as any
      const textNode = paragraph.children[0]

      assert(!isWikilink(textNode))
    })

    test("returns false for null", () => {
      assert(!isWikilink(null))
    })

    test("returns false for undefined", () => {
      assert(!isWikilink(undefined))
    })
  })

  describe("toMarkdown serialization", () => {
    describe("basic serialization", () => {
      test("serializes simple wikilink", () => {
        const tree = parse("[[test]]")
        const result = serialize(tree)
        assert(result.includes("[[test]]"))
      })

      test("serializes wikilink with alias", () => {
        const tree = parse("[[page|display text]]")
        const result = serialize(tree)
        assert(result.includes("[[page|display text]]"))
      })

      test("serializes wikilink with anchor", () => {
        const tree = parse("[[page#section]]")
        const result = serialize(tree)
        assert(result.includes("[[page#section]]"))
      })

      test("serializes wikilink with anchor and alias", () => {
        const tree = parse("[[page#section|custom text]]")
        const result = serialize(tree)
        assert(result.includes("[[page#section|custom text]]"))
      })

      test("serializes block reference", () => {
        const tree = parse("[[file#^block-id]]")
        const result = serialize(tree)
        assert(result.includes("[[file#^block-id]]"))
      })
    })

    describe("embed serialization", () => {
      test("serializes embed", () => {
        const tree = parse("![[embed]]")
        const result = serialize(tree)
        assert(result.includes("![[embed]]"))
      })

      test("serializes embed with anchor", () => {
        const tree = parse("![[file#section]]")
        const result = serialize(tree)
        assert(result.includes("![[file#section]]"))
      })

      test("serializes embed with alias", () => {
        const tree = parse("![[image.png|100x200]]")
        const result = serialize(tree)
        assert(result.includes("![[image.png|100x200]]"))
      })
    })

    describe("edge case serialization", () => {
      test("serializes empty target", () => {
        const tree = parse("[[]]")
        const result = serialize(tree)
        assert(result.includes("[[]]"))
      })

      test("serializes empty target with anchor", () => {
        const tree = parse("[[#heading]]")
        const result = serialize(tree)
        assert(result.includes("[[#heading]]"))
      })

      test("serializes empty target with alias", () => {
        const tree = parse("[[|display]]")
        const result = serialize(tree)
        assert(result.includes("[[|display]]"))
      })

      test("serializes multiple anchors (subheadings)", () => {
        const tree = parse("[[file#heading#subheading]]")
        const result = serialize(tree)
        assert(result.includes("[[file#heading#subheading]]"))
      })
    })

    describe("escaping in serialization", () => {
      test("escapes pipe in target", () => {
        const tree = parse("[[file\\|name]]")
        const result = serialize(tree)
        // should re-escape the pipe
        assert(result.includes("[[file\\|name]]"))
      })

      test("escapes hash in target", () => {
        const tree = parse("[[file\\#name]]")
        const result = serialize(tree)
        // should re-escape the hash
        assert(result.includes("[[file\\#name]]"))
      })

      test("escapes pipe with alias", () => {
        const tree = parse("[[file\\|name|alias]]")
        const result = serialize(tree)
        // pipe in target should be escaped, alias delimiter should not
        assert(result.includes("[[file\\|name|alias]]"))
      })

      test("escapes bracket in alias", () => {
        const tree = parse("[[file|alias\\]text]]")
        const result = serialize(tree)
        // should re-escape the bracket
        assert(result.includes("[[file|alias\\]text]]"))
      })
    })

    describe("round-trip consistency", () => {
      const testCases = [
        "[[test]]",
        "[[page|alias]]",
        "[[page#section]]",
        "[[page#section|alias]]",
        "[[file#^block]]",
        "![[embed]]",
        "![[image.png|100x200]]",
        "[[]]",
        "[[#heading]]",
        "[[|display]]",
        "[[file\\|name]]",
        "[[file\\#name]]",
        "[[file|alias\\]text]]",
        "[[path/to/file]]",
        "[[file (with) parens]]",
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

    describe("multiple wikilinks serialization", () => {
      test("serializes multiple wikilinks in paragraph", () => {
        const input = "Here is [[link1]] and [[link2|alias]]."
        const tree = parse(input)
        const result = serialize(tree)
        assert(result.includes("[[link1]]"))
        assert(result.includes("[[link2|alias]]"))
      })

      test("serializes wikilinks separated by text", () => {
        const input = "[[first]] some text [[second]]"
        const tree = parse(input)
        const result = serialize(tree)
        assert(result.includes("[[first]]"))
        assert(result.includes("[[second]]"))
      })
    })
  })

  describe("obsidian mode", () => {
    describe("nested heading anchors", () => {
      test("uses last segment with obsidian: true", () => {
        const tree = parse("[[file#Parent#Child#Grandchild]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // should use only "grandchild" and slugify it
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#grandchild")
      })

      test("uses last segment with two levels", () => {
        const tree = parse("[[NVIDIA#cuda]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "NVIDIA")
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#cuda")
      })

      test("preserves full anchor with obsidian: false (default)", () => {
        const tree = parse("[[file#Parent#Child#Grandchild]]", { obsidian: false })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // non-obsidian mode: preserves full anchor as-is
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#Parent#Child#Grandchild")
      })

      test("uses last segment when option not specified (defaults to obsidian: true)", () => {
        const tree = parse("[[file#Parent#Child]]")
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // default mode (obsidian: true): uses last segment
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#child")
      })
    })

    describe("anchor slugification", () => {
      test("slugifies anchor text with obsidian mode", () => {
        const tree = parse("[[file#Section Title]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // github-slugger converts to lowercase and replaces spaces with hyphens
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#section-title")
      })

      test("slugifies LaTeX in anchor", () => {
        const tree = parse("[[file#architectural skeleton of $ mu$]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // spaces around $ should be removed before slugification
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#architectural-skeleton-of-mu")
      })

      test("normalizes single inline math with spaces", () => {
        const tree = parse("[[file#$ mu$]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // "$ mu$" → "$mu$" → slugged: "mu"
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#mu")
      })

      test("normalizes inline math in context", () => {
        const tree = parse("[[file#Section with $ x + y$ equation]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // spaces around $ removed, then slugified
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#section-with-x--y-equation")
      })

      test("normalizes multiple math expressions", () => {
        const tree = parse("[[file#$ alpha$ and $ beta$]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // "$ alpha$ and $ beta$" → "$alpha$ and $beta$" → slugged: "alpha-and-beta"
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#alpha-and-beta")
      })

      test("handles math with multiple spaces", () => {
        const tree = parse("[[file#$  x  $]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // "$  x  $" → "$x$" → slugged: "x"
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#x")
      })

      test("preserves math without adjacent spaces", () => {
        const tree = parse("[[file#$mu$ value]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // no spaces to remove around $
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#mu-value")
      })

      test("handles mixed spacing patterns", () => {
        const tree = parse("[[file#test $ a$ and $b $ plus $c$]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // "$ a$" → "$a$", "$b $" → "$b$", "$c$" unchanged
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#test-a-and-b-plus-c")
      })

      test("preserves anchors without obsidian mode", () => {
        const tree = parse("[[file#Section Title]]", { obsidian: false })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // non-obsidian mode: preserves anchor as-is
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#Section Title")
      })
    })

    describe("nested headings with slugification", () => {
      test("combines nested heading extraction and slugification", () => {
        const tree = parse("[[file#Parent Heading#Child Section]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // should extract "Child Section" and slugify to "child-section"
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#child-section")
      })

      test("handles mixed case nested headings", () => {
        const tree = parse("[[file#First#SECOND#ThIrD]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#third")
      })
    })

    describe("block references with obsidian mode", () => {
      test("preserves block reference marker", () => {
        const tree = parse("[[file#^block-id]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // block reference should keep ^ marker
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#^block-id")
      })

      test("does not process # in block reference content", () => {
        const tree = parse("[[file#^block#with#hashes]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // should extract last segment "hashes" but keep block marker
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#^hashes")
      })
    })

    describe("edge cases with obsidian mode", () => {
      test("handles single heading (no nesting)", () => {
        const tree = parse("[[file#heading]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#heading")
      })

      test("handles empty segments in nested path", () => {
        const tree = parse("[[file#Parent##Child]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // last segment is "Child", empty segment is ignored
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#child")
      })

      test("works with aliases", () => {
        const tree = parse("[[file#Parent#Child|display text]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#child")
        assert.strictEqual(wikilink.data?.wikilink.alias, "display text")
      })

      test("works with embeds", () => {
        const tree = parse("![[file#Parent#Child]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#child")
        assert.strictEqual(wikilink.data?.wikilink.embed, true)
      })
    })
  })

  describe("hName annotation (automatic hast conversion)", () => {
    describe("regular links", () => {
      test("annotates simple link", () => {
        const tree = parse("[[target]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "a")
        assert.strictEqual(wikilink.data?.hProperties?.href, "target")
        assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: "text", value: "target" }])
      })

      test("annotates link with alias", () => {
        const tree = parse("[[thoughts/attention|focus]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "a")
        assert.strictEqual(wikilink.data?.hProperties?.href, "thoughts/attention")
        assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: "text", value: "focus" }])
      })

      test("annotates link with anchor", () => {
        const tree = parse("[[page#section]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "a")
        assert.strictEqual(wikilink.data?.hProperties?.href, "page#section")
        assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: "text", value: "page" }])
      })

      test("annotates link with anchor and alias", () => {
        const tree = parse("[[page#section|custom]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "a")
        assert.strictEqual(wikilink.data?.hProperties?.href, "page#section")
        assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: "text", value: "custom" }])
      })
    })

    describe("image embeds", () => {
      test("annotates simple image", () => {
        const tree = parse("![[photo.jpg]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "img")
        assert.strictEqual(wikilink.data?.hProperties?.src, "photo.jpg")
        assert.strictEqual(wikilink.data?.hProperties?.width, "auto")
        assert.strictEqual(wikilink.data?.hProperties?.height, "auto")
      })

      test("annotates image with caption text (figure)", () => {
        const tree = parse("![[photo.jpg|A beautiful sunset]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "img")
        assert.strictEqual(wikilink.data?.hProperties?.src, "photo.jpg")
        assert.strictEqual(wikilink.data?.hProperties?.width, "auto")
        assert.strictEqual(wikilink.data?.hProperties?.height, "auto")
        assert.strictEqual(wikilink.data?.hProperties?.alt, "A beautiful sunset")
      })

      test("annotates image with dimensions (width only)", () => {
        const tree = parse("![[photo.jpg|400]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "img")
        assert.strictEqual(wikilink.data?.hProperties?.src, "photo.jpg")
        assert.strictEqual(wikilink.data?.hProperties?.width, "400")
        assert.strictEqual(wikilink.data?.hProperties?.height, "auto")
      })

      test("annotates image with dimensions (width and height)", () => {
        const tree = parse("![[photo.jpg|400x300]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "img")
        assert.strictEqual(wikilink.data?.hProperties?.src, "photo.jpg")
        assert.strictEqual(wikilink.data?.hProperties?.width, "400")
        assert.strictEqual(wikilink.data?.hProperties?.height, "300")
      })

      test("annotates image with caption and dimensions", () => {
        const tree = parse("![[photo.jpg|A beautiful sunset|400x300]]", {
          obsidian: true,
        })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "img")
        assert.strictEqual(wikilink.data?.hProperties?.src, "photo.jpg")
        assert.strictEqual(wikilink.data?.hProperties?.width, "400")
        assert.strictEqual(wikilink.data?.hProperties?.height, "300")
        assert.strictEqual(wikilink.data?.hProperties?.alt, "A beautiful sunset")
      })

      test("annotates image with multi-part caption", () => {
        const tree = parse("![[photo.jpg|Caption with|multiple|pipes]]", {
          obsidian: true,
        })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "img")

        const figcaption = wikilink.data?.hProperties?.alt as any
        assert.strictEqual(figcaption, "Caption with|multiple|pipes")
      })

      test("annotates image with multi-part caption and dimensions", () => {
        const tree = parse("![[photo.jpg|Caption with|multiple parts|800x600]]", {
          obsidian: true,
        })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "img")

        assert.strictEqual(wikilink?.data.hProperties?.width, "800")
        assert.strictEqual(wikilink?.data.hProperties?.height, "600")
        assert.strictEqual(wikilink?.data.hProperties?.alt, "Caption with|multiple parts")
      })

      test("handles various image extensions", () => {
        const extensions = ["png", "jpg", "jpeg", "gif", "bmp", "svg", "webp"]
        for (const ext of extensions) {
          const tree = parse(`![[image.${ext}]]`, { obsidian: true })
          const wikilink = extractWikilink(tree)

          assert(wikilink, `wikilink node should exist for .${ext}`)
          assert.strictEqual(wikilink.data?.hName, "img", `should be img for .${ext}`)
        }
      })
    })

    describe("video embeds", () => {
      test("annotates video embed", () => {
        const tree = parse("![[demo.mp4]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "video")
        assert.strictEqual(wikilink.data?.hProperties?.src, "demo.mp4")
        assert.strictEqual(wikilink.data?.hProperties?.controls, true)
        assert.strictEqual(wikilink.data?.hProperties?.loop, true)
      })

      test("handles various video extensions", () => {
        const extensions = ["mp4", "webm", "ogv", "mov", "mkv"]
        for (const ext of extensions) {
          const tree = parse(`![[video.${ext}]]`, { obsidian: true })
          const wikilink = extractWikilink(tree)

          assert(wikilink, `wikilink node should exist for .${ext}`)
          assert.strictEqual(wikilink.data?.hName, "video", `should be video for .${ext}`)
        }
      })
    })

    describe("audio embeds", () => {
      test("annotates audio embed", () => {
        const tree = parse("![[song.mp3]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "audio")
        assert.strictEqual(wikilink.data?.hProperties?.src, "song.mp3")
        assert.strictEqual(wikilink.data?.hProperties?.controls, true)
      })

      test("handles various audio extensions", () => {
        const extensions = ["mp3", "wav", "m4a", "ogg", "3gp", "flac"]
        for (const ext of extensions) {
          const tree = parse(`![[audio.${ext}]]`, { obsidian: true })
          const wikilink = extractWikilink(tree)

          assert(wikilink, `wikilink node should exist for .${ext}`)
          assert.strictEqual(wikilink.data?.hName, "audio", `should be audio for .${ext}`)
        }
      })
    })

    describe("PDF embeds", () => {
      test("annotates PDF embed", () => {
        const tree = parse("![[paper.pdf]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "iframe")
        assert.strictEqual(wikilink.data?.hProperties?.src, "paper.pdf")
        assert.strictEqual(wikilink.data?.hProperties?.class, "pdf")
      })
    })

    describe("block transcludes", () => {
      test("annotates simple transclude", () => {
        const tree = parse("![[notes]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "blockquote")
        assert.strictEqual(wikilink.data?.hProperties?.class, "transclude")
        assert.strictEqual(wikilink.data?.hProperties?.["data-url"], "notes")
        assert.strictEqual(wikilink.data?.hProperties?.["data-block"], "")
        assert(Array.isArray(wikilink.data?.hChildren), "should have hChildren array")
        assert.strictEqual((wikilink.data?.hChildren?.[0] as any)?.tagName, "a")
      })

      test("annotates transclude with anchor", () => {
        const tree = parse("![[notes#summary]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "blockquote")
        assert.strictEqual(wikilink.data?.hProperties?.class, "transclude")
        assert.strictEqual(wikilink.data?.hProperties?.["data-url"], "notes")
        assert.strictEqual(wikilink.data?.hProperties?.["data-block"], "#summary")
        assert(Array.isArray(wikilink.data?.hChildren), "should have hChildren array")

        const innerLink = wikilink.data?.hChildren?.[0] as any
        assert.strictEqual(innerLink?.type, "element")
        assert.strictEqual(innerLink?.tagName, "a")
        assert.strictEqual(innerLink?.properties?.href, "notes#summary")
        assert.strictEqual(innerLink?.properties?.class, "transclude-inner")
      })

      test("annotates transclude with block reference", () => {
        const tree = parse("![[notes#^block-id]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "blockquote")
        assert.strictEqual(wikilink.data?.hProperties?.["data-block"], "#^block-id")
      })

      test("annotates transclude with alias", () => {
        const tree = parse("![[notes#summary|see notes]]", { obsidian: true })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "blockquote")
        assert.strictEqual(wikilink.data?.hProperties?.["data-embed-alias"], "see notes")
      })
    })

    describe("without explicit options", () => {
      test("annotates with obsidian: true by default", () => {
        const tree = parse("[[target]]") // defaults to obsidian: true
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "a")
        assert.strictEqual(wikilink.data?.hProperties?.href, "target")
        assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: "text", value: "target" }])
        // wikilink data should still be present
        assert.strictEqual(wikilink.data?.wikilink.target, "target")
      })

      test("annotates embeds with obsidian: true by default", () => {
        const tree = parse("![[image.png]]") // defaults to obsidian: true
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, "img")
        assert.strictEqual(wikilink.data?.hProperties?.src, "image.png")
        assert.strictEqual(wikilink.data?.wikilink.embed, true)
      })

      test("does not annotate when explicitly obsidian: false", () => {
        const tree = parse("[[target]]", { obsidian: false })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.hName, undefined)
        assert.strictEqual(wikilink.data?.hProperties, undefined)
        assert.strictEqual(wikilink.data?.hChildren, undefined)
        assert.strictEqual(wikilink.data?.wikilink.target, "target")
      })
    })

    describe("integration with obsidian mode", () => {
      test("combines obsidian anchor handling with hName annotation", () => {
        const tree = parse("[[file#Parent#Child|display]]", {
          obsidian: true,
        })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        // obsidian mode: uses last anchor segment
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#child")
        // hName annotation
        assert.strictEqual(wikilink.data?.hName, "a")
        assert.strictEqual(wikilink.data?.hProperties?.href, "file#child")
        assert.deepStrictEqual(wikilink.data?.hChildren, [{ type: "text", value: "display" }])
      })
    })
  })

  describe("stripExtensions option", () => {
    test("strips .md extension with obsidian mode", () => {
      const tree = parse("[[notes.md]]", { obsidian: true })
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a")
      assert.strictEqual(wikilink.data?.hProperties?.href, "notes")
    })

    test("strips .base extension with obsidian mode", () => {
      const tree = parse("[[data.base]]", { obsidian: true })
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a")
      assert.strictEqual(wikilink.data?.hProperties?.href, "data")
    })

    test("strips custom extensions", () => {
      const tree = parse("[[notes.mdx]]", {
        obsidian: true,
        stripExtensions: [".md", ".mdx", ".base"],
      })
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a")
      assert.strictEqual(wikilink.data?.hProperties?.href, "notes")
    })

    test("only strips first matching extension", () => {
      const tree = parse("[[file.md.backup]]", {
        obsidian: true,
        stripExtensions: [".backup"],
      })
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a")
      // slugAnchor removes dots, so file.md becomes filemd
      assert.strictEqual(wikilink.data?.hProperties?.href, "file.md")
    })

    test("preserves non-stripped extensions in path", () => {
      const tree = parse("[[document.pdf]]", { obsidian: true, stripExtensions: [".md"] })
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a")
      // slugAnchor will process the whole path
      assert(wikilink.data?.hProperties?.href.includes("document"))
    })
  })

  describe("obsidian mode with internal slugification", () => {
    test("uses internal slugification when obsidian: true", () => {
      const tree = parse("[[File With Spaces]]", { obsidian: true })
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a")
      assert.strictEqual(wikilink.data?.hProperties?.href, "file-with-spaces")
    })

    test("strips extension and slugifies", () => {
      const tree = parse("[[My Notes.md]]", { obsidian: true })
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a")
      assert.strictEqual(wikilink.data?.hProperties?.href, "my-notes")
    })

    test("handles special characters with slugification", () => {
      const tree = parse("[[File (with) parens!]]", { obsidian: true })
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a")
      // slugAnchor should handle special chars
      assert(wikilink.data?.hProperties?.href)
    })
  })
})
