/**
 * tests for micromark wikilink extension.
 * verifies tokenization and mdast node creation.
 */

import test, { describe } from "node:test"
import assert from "node:assert"
import { fromMarkdown } from "mdast-util-from-markdown"
import { toMarkdown } from "mdast-util-to-markdown"
import { wikilink } from "./syntax"
import { wikilinkFromMarkdown, Wikilink, isWikilinkNode } from "./fromMarkdown"
import { wikilinkToMarkdown } from "./toMarkdown"
import type { Root } from "mdast"

/**
 * helper to parse markdown with wikilink extension.
 */
function parse(markdown: string, options?: { obsidian?: boolean }): Root {
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
        if (isWikilinkNode(node)) {
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
      const tree = parse("[[file#heading#subheading]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.wikilink.target, "file")
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

      const wikilinks = paragraph.children.filter((node: any) => isWikilinkNode(node))
      assert.strictEqual(wikilinks.length, 2)

      assert.strictEqual(wikilinks[0].data.wikilink.target, "link1")
      assert.strictEqual(wikilinks[1].data.wikilink.target, "link2")
      assert.strictEqual(wikilinks[1].data.wikilink.alias, "alias")
    })

    test("parses wikilinks separated by text", () => {
      const tree = parse("[[first]] some text [[second]]")
      const paragraph = tree.children[0] as any

      const wikilinks = paragraph.children.filter((node: any) => isWikilinkNode(node))
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

  describe("isWikilinkNode type guard", () => {
    test("returns true for valid wikilink node", () => {
      const tree = parse("[[test]]")
      const wikilink = extractWikilink(tree)

      assert(wikilink, "wikilink should exist")
      assert(isWikilinkNode(wikilink))
    })

    test("returns false for text node", () => {
      const tree = parse("plain text")
      const paragraph = tree.children[0] as any
      const textNode = paragraph.children[0]

      assert(!isWikilinkNode(textNode))
    })

    test("returns false for null", () => {
      assert(!isWikilinkNode(null))
    })

    test("returns false for undefined", () => {
      assert(!isWikilinkNode(undefined))
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
        // should keep full anchor path without obsidian mode
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#Parent#Child#Grandchild")
      })

      test("preserves full anchor when option not specified", () => {
        const tree = parse("[[file#Parent#Child]]")
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#Parent#Child")
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
        // LaTeX symbols should be normalized
        assert.strictEqual(wikilink.data?.wikilink.anchor, "#architectural-skeleton-of--mu")
      })

      test("does not slugify without obsidian mode", () => {
        const tree = parse("[[file#Section Title]]", { obsidian: false })
        const wikilink = extractWikilink(tree)

        assert(wikilink, "wikilink node should exist")
        assert.strictEqual(wikilink.data?.wikilink.target, "file")
        // should preserve original anchor text
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
})
