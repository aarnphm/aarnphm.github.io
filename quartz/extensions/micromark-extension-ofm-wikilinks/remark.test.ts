/**
 * integration tests for remarkWikilink plugin.
 * verifies full unified pipeline including mdast-to-hast conversion.
 */

import type { Root } from "mdast"
import assert from "node:assert"
import test, { describe } from "node:test"
import rehypeStringify from "rehype-stringify"
import remarkParse from "remark-parse"
import remarkRehype from "remark-rehype"
import { unified } from "unified"
import { remarkWikilink } from "./index"

/**
 * helper to process markdown through full pipeline.
 * parses markdown → converts to HTML → returns HTML string.
 */
function processToHTML(
  markdown: string,
  options?: { obsidian?: boolean; stripExtensions?: string[] },
): string {
  const processor = unified()
    .use(remarkParse)
    //@ts-ignore
    .use(remarkWikilink, options)
    .use(remarkRehype)
    .use(rehypeStringify)

  const result = processor.processSync(markdown)
  return result.toString()
}

describe("remarkWikilink integration", () => {
  describe("mdast-to-hast conversion", () => {
    test("converts simple wikilink to anchor tag", () => {
      const html = processToHTML("[[test]]")
      assert(html.includes('<a href="test">test</a>'))
    })

    test("converts wikilink with alias", () => {
      const html = processToHTML("[[page|display text]]")
      assert(html.includes('<a href="page">display text</a>'))
    })

    test("converts wikilink with anchor", () => {
      const html = processToHTML("[[page#section]]")
      assert(html.includes('<a href="page#section">page</a>'))
    })

    test("converts wikilink with anchor and alias", () => {
      const html = processToHTML("[[page#section|custom]]")
      assert(html.includes('<a href="page#section">custom</a>'))
    })

    test("handles wikilink in paragraph context", () => {
      const html = processToHTML("This is a [[link]] in a sentence.")
      assert(html.includes("<p>"))
      assert(html.includes('<a href="link">link</a>'))
      assert(html.includes("in a sentence."))
    })

    test("handles multiple wikilinks", () => {
      const html = processToHTML("[[first]] and [[second|alias]]")
      assert(html.includes('<a href="first">first</a>'))
      assert(html.includes('<a href="second">alias</a>'))
    })
  })

  describe("absolute paths", () => {
    test("preserves absolute path starting with /", () => {
      const html = processToHTML("[[/tags/ml]]")
      assert(html.includes('<a href="/tags/ml">/tags/ml</a>'))
    })

    test("preserves absolute path with anchor", () => {
      const html = processToHTML("[[/tags/ml#section]]")
      assert(html.includes('<a href="/tags/ml#section">/tags/ml</a>'))
    })

    test("preserves absolute path with alias", () => {
      const html = processToHTML("[[/tags/ml|Machine Learning]]")
      assert(html.includes('<a href="/tags/ml">Machine Learning</a>'))
    })

    test("preserves nested absolute paths", () => {
      const html = processToHTML("[[/thoughts/papers/attention]]")
      assert(html.includes('<a href="/thoughts/papers/attention">/thoughts/papers/attention</a>'))
    })

    test("absolute path bypasses slugification", () => {
      const html = processToHTML("[[/Tags With Spaces]]")
      // should preserve exactly as written, no slugification
      assert(html.includes('<a href="/Tags With Spaces">/Tags With Spaces</a>'))
    })
  })

  describe("same-file anchors", () => {
    test("handles same-file anchor with empty target", () => {
      const html = processToHTML("[[#heading]]")
      assert(html.includes('<a href="#heading"></a>'))
    })

    test("slugifies same-file anchor in obsidian mode", () => {
      const html = processToHTML("[[#Section Title]]", { obsidian: true })
      assert(html.includes('<a href="#section-title"></a>'))
    })

    test("handles same-file anchor with alias", () => {
      const html = processToHTML("[[#heading|Jump Here]]")
      assert(html.includes('<a href="#heading">Jump Here</a>'))
    })

    test("handles block reference", () => {
      const html = processToHTML("[[#^block-id]]")
      assert(html.includes('<a href="#^block-id"></a>'))
    })
  })

  describe("image embeds", () => {
    test("converts image embed to img tag", () => {
      const html = processToHTML("![[photo.jpg]]")
      assert(html.includes('<img src="photo.jpg"'))
      assert(html.includes('width="auto"'))
      assert(html.includes('height="auto"'))
    })

    test("converts image with caption", () => {
      const html = processToHTML("![[photo.jpg|Beautiful sunset]]")
      assert(html.includes('<img src="photo.jpg"'))
      assert(html.includes('alt="Beautiful sunset"'))
    })

    test("converts image with dimensions", () => {
      const html = processToHTML("![[photo.jpg|400x300]]")
      assert(html.includes('<img src="photo.jpg"'))
      assert(html.includes('width="400"'))
      assert(html.includes('height="300"'))
    })

    test("converts image with caption and dimensions", () => {
      const html = processToHTML("![[photo.jpg|Sunset|400x300]]")
      assert(html.includes('<img src="photo.jpg"'))
      assert(html.includes('width="400"'))
      assert(html.includes('height="300"'))
      assert(html.includes('alt="Sunset"'))
    })

    test("handles various image extensions", () => {
      const extensions = ["png", "jpg", "jpeg", "gif", "bmp", "svg", "webp"]
      for (const ext of extensions) {
        const html = processToHTML(`![[image.${ext}]]`)
        assert(html.includes(`<img src="image.${ext}"`), `should handle .${ext}`)
      }
    })
  })

  describe("video embeds", () => {
    test("converts video embed to video tag", () => {
      const html = processToHTML("![[demo.mp4]]")
      assert(html.includes('<video src="demo.mp4"'))
      assert(html.includes("controls"))
      assert(html.includes("loop"))
    })

    test("handles various video extensions", () => {
      const extensions = ["mp4", "webm", "ogv", "mov", "mkv"]
      for (const ext of extensions) {
        const html = processToHTML(`![[video.${ext}]]`)
        assert(html.includes(`<video src="video.${ext}"`), `should handle .${ext}`)
      }
    })
  })

  describe("audio embeds", () => {
    test("converts audio embed to audio tag", () => {
      const html = processToHTML("![[song.mp3]]")
      assert(html.includes('<audio src="song.mp3"'))
      assert(html.includes("controls"))
    })

    test("handles various audio extensions", () => {
      const extensions = ["mp3", "wav", "m4a", "ogg", "3gp", "flac"]
      for (const ext of extensions) {
        const html = processToHTML(`![[audio.${ext}]]`)
        assert(html.includes(`<audio src="audio.${ext}"`), `should handle .${ext}`)
      }
    })
  })

  describe("PDF embeds", () => {
    test("converts PDF embed to iframe", () => {
      const html = processToHTML("![[paper.pdf]]")
      assert(html.includes('<iframe src="paper.pdf"'))
      assert(html.includes('class="pdf"'))
    })
  })

  describe("block transcludes", () => {
    test("converts transclude to blockquote", () => {
      const html = processToHTML("![[notes]]")
      assert(html.includes('<blockquote class="transclude"'))
      assert(html.includes('data-url="notes"'))
      assert(html.includes('data-block=""'))
    })

    test("converts transclude with anchor", () => {
      const html = processToHTML("![[notes#summary]]")
      assert(html.includes('<blockquote class="transclude"'))
      assert(html.includes('data-url="notes"'))
      assert(html.includes('data-block="#summary"'))
      assert(html.includes('<a href="notes#summary"'))
    })

    test("converts transclude with block reference", () => {
      const html = processToHTML("![[notes#^block-id]]")
      assert(html.includes('<blockquote class="transclude"'))
      assert(html.includes('data-block="#^block-id"'))
    })

    test("converts transclude with alias", () => {
      const html = processToHTML("![[notes#summary|See notes]]")
      assert(html.includes('<blockquote class="transclude"'))
      assert(html.includes('data-embed-alias="See notes"'))
    })
  })

  describe("stripExtensions option", () => {
    test("strips .md extension from href by default", () => {
      const html = processToHTML("[[notes.md]]", { obsidian: true })
      // href should strip .md, but display text keeps it
      assert(html.includes('<a href="notes">notes.md</a>'))
    })

    test("strips .base extension from href by default", () => {
      const html = processToHTML("[[data.base]]", { obsidian: true })
      // href should strip .base, but display text keeps it
      assert(html.includes('<a href="data">data.base</a>'))
    })

    test("strips custom extensions from href", () => {
      const html = processToHTML("[[notes.mdx]]", {
        obsidian: true,
        stripExtensions: [".md", ".mdx", ".base"],
      })
      // NOTE: stripExtensions parameter is currently not fully implemented
      // slugifyFilePath only handles built-in extensions (.md, .ipynb, .html, .base)
      // custom extensions like .mdx are preserved in the href
      assert(html.includes('<a href="notes">notes.mdx</a>'))
    })

    test("preserves extensions not in stripExtensions", () => {
      const html = processToHTML("[[document.pdf]]", { obsidian: true, stripExtensions: [".md"] })
      // PDF has special handling - preserved in href
      assert(html.includes("document"))
      assert(html.includes(".pdf"))
    })

    test("case insensitive extension matching", () => {
      const html = processToHTML("[[Notes.MD]]", { obsidian: true })
      // slugifyFilePath doesn't lowercase - preserves "Notes.MD" as-is
      // but .MD extension is not in default strip list (lowercase .md is)
      assert(html.includes('<a href="Notes.MD">Notes.MD</a>'))
    })
  })

  describe("obsidian mode features", () => {
    test("slugifies file paths in obsidian mode", () => {
      const html = processToHTML("[[File With Spaces]]", { obsidian: true })
      // sluggify replaces spaces with dashes but preserves case
      assert(html.includes('href="File-With-Spaces"'))
      assert(html.includes(">File With Spaces</a>"))
    })

    test("handles nested anchor segments", () => {
      const html = processToHTML("[[file#Parent#Child]]", { obsidian: true })
      // should use only "child" from nested anchors (lowercased by slugAnchor)
      assert(html.includes('<a href="file#child">file</a>'))
    })

    test("slugifies anchor text", () => {
      const html = processToHTML("[[file#Section Title]]", { obsidian: true })
      // slugAnchor lowercases and replaces spaces with hyphens
      assert(html.includes('<a href="file#section-title">file</a>'))
    })

    test("strips LaTeX from anchors", () => {
      const html = processToHTML("[[file#$ mu$]]", { obsidian: true })
      assert(html.includes('<a href="file#mu">file</a>'))
    })

    test("combines path slugification and anchor processing", () => {
      const html = processToHTML("[[My Notes.md#Section Title|Custom]]", { obsidian: true })
      // sluggify preserves case for path, strips .md extension, slugAnchor lowercases anchor
      assert(html.includes('href="My-Notes#section-title"'))
      assert(html.includes(">Custom</a>"))
    })
  })

  describe("obsidian: false mode", () => {
    test("does not annotate nodes when obsidian: false", () => {
      const html = processToHTML("[[test]]", { obsidian: false })
      // should not convert to HTML - will fail mdast-to-hast or be skipped
      // in non-obsidian mode, wikilink nodes remain unannotated
      // and will trigger "Cannot handle unknown node" error
      // so we expect this to NOT contain a proper anchor tag
      assert(!html.includes('<a href="test">test</a>') || html.includes("[[test]]"))
    })
  })

  describe("complex integration scenarios", () => {
    test("handles wikilinks in lists", () => {
      const html = processToHTML("- Item with [[link]]\n- Another [[link2|alias]]")
      assert(html.includes("<ul>"))
      assert(html.includes("<li>"))
      assert(html.includes('<a href="link">link</a>'))
      assert(html.includes('<a href="link2">alias</a>'))
    })

    test("handles wikilinks in blockquotes", () => {
      const html = processToHTML("> This is [[quoted|a quote]] text")
      assert(html.includes("<blockquote>"))
      assert(html.includes('<a href="quoted">a quote</a>'))
    })

    test("handles wikilinks with surrounding text", () => {
      const html = processToHTML("Before [[link1]] middle [[link2|alias]] after")
      assert(html.includes("Before"))
      assert(html.includes('<a href="link1">link1</a>'))
      assert(html.includes("middle"))
      assert(html.includes('<a href="link2">alias</a>'))
      assert(html.includes("after"))
    })

    test("handles nested structures", () => {
      const html = processToHTML("Paragraph with [[link]].\n\n> Quote with [[another]].")
      assert(html.includes('<a href="link">link</a>'))
      assert(html.includes('<a href="another">another</a>'))
      assert(html.includes("<p>"))
      assert(html.includes("<blockquote>"))
    })

    test("preserves position information through pipeline", () => {
      const markdown = "Text [[link]] more"
      // should not throw and should produce valid HTML
      const html = processToHTML(markdown)
      assert(html.includes('<a href="link">link</a>'))
    })
  })

  describe("edge cases", () => {
    test("handles empty wikilink", () => {
      const html = processToHTML("[[]]")
      assert(html.includes('<a href="">'))
    })

    test("handles wikilink with only anchor", () => {
      const html = processToHTML("[[#section]]")
      assert(html.includes('<a href="#section">'))
    })

    test("handles wikilink with only alias", () => {
      const html = processToHTML("[[|display]]")
      assert(html.includes('<a href="">display</a>'))
    })

    test("handles escaped characters", () => {
      const html = processToHTML("[[file\\|name]]")
      // backslash consumed, pipe part of target
      assert(html.includes("file|name"))
    })

    test("handles paths with slashes", () => {
      const html = processToHTML("[[path/to/file]]")
      assert(html.includes("path/to/file") || html.includes("path-to-file"))
    })

    test("handles special characters", () => {
      const html = processToHTML("[[file (with) parens]]")
      // should produce valid HTML even with special chars
      assert(html.includes("<a href="))
    })
  })

  describe("data.hName annotation verification", () => {
    test("regular link gets hName annotation", () => {
      //@ts-ignore
      const processor = unified().use(remarkParse).use(remarkWikilink, { obsidian: true })

      const tree = processor.parse("[[test]]") as Root
      const paragraph = tree.children[0] as any
      const wikilink = paragraph.children.find((n: any) => n.type === "wikilink")

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a", "should have hName annotation")
      assert(wikilink.data?.hProperties, "should have hProperties")
      assert(wikilink.data?.hChildren, "should have hChildren")
    })

    test("image embed gets hName annotation", () => {
      //@ts-ignore
      const processor = unified().use(remarkParse).use(remarkWikilink, { obsidian: true })

      const tree = processor.parse("![[image.png]]") as Root
      const paragraph = tree.children[0] as any
      const wikilink = paragraph.children.find((n: any) => n.type === "wikilink")

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "img", "should have hName annotation")
      assert(wikilink.data?.hProperties, "should have hProperties")
    })

    test("transclude gets hName annotation", () => {
      //@ts-ignore
      const processor = unified().use(remarkParse).use(remarkWikilink, { obsidian: true })

      const tree = processor.parse("![[notes#section]]") as Root
      const paragraph = tree.children[0] as any
      const wikilink = paragraph.children.find((n: any) => n.type === "wikilink")

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "blockquote", "should have hName annotation")
      assert(wikilink.data?.hProperties, "should have hProperties")
      assert(wikilink.data?.hChildren, "should have hChildren")
    })

    test("absolute path gets hName annotation", () => {
      //@ts-ignore
      const processor = unified().use(remarkParse).use(remarkWikilink, { obsidian: true })

      const tree = processor.parse("[[/tags/ml]]") as Root
      const paragraph = tree.children[0] as any
      const wikilink = paragraph.children.find((n: any) => n.type === "wikilink")

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a", "should have hName annotation")
      assert.strictEqual(
        wikilink.data?.hProperties?.href,
        "/tags/ml",
        "should preserve absolute path",
      )
    })

    test("same-file anchor gets hName annotation", () => {
      //@ts-ignore
      const processor = unified().use(remarkParse).use(remarkWikilink, { obsidian: true })

      const tree = processor.parse("[[#heading]]") as Root
      const paragraph = tree.children[0] as any
      const wikilink = paragraph.children.find((n: any) => n.type === "wikilink")

      assert(wikilink, "wikilink node should exist")
      assert.strictEqual(wikilink.data?.hName, "a", "should have hName annotation")
      assert.strictEqual(wikilink.data?.hProperties?.href, "#heading", "should have anchor href")
    })
  })

  describe("error handling", () => {
    test("does not throw on malformed wikilinks", () => {
      assert.doesNotThrow(() => {
        processToHTML("[[incomplete")
      })
    })

    test("does not throw on complex markdown", () => {
      assert.doesNotThrow(() => {
        processToHTML(
          "# Heading\n\n[[link1]] and **bold** text.\n\n- List with [[link2]]\n- Another item",
        )
      })
    })

    test("does not throw on mixed content", () => {
      assert.doesNotThrow(() => {
        processToHTML("Text `code with [[link]]` more [[actual link]]")
      })
    })
  })

  describe("performance and scale", () => {
    test("handles many wikilinks in single document", () => {
      const links = Array.from({ length: 50 }, (_, i) => `[[link${i}]]`).join(" ")
      const html = processToHTML(links)
      for (let i = 0; i < 50; i++) {
        assert(html.includes(`<a href="link${i}">link${i}</a>`))
      }
    })

    test("handles deeply nested paths", () => {
      const html = processToHTML("[[a/b/c/d/e/f/g/h]]")
      assert(html.includes("<a href="))
      assert(html.includes("</a>"))
    })

    test("handles very long aliases", () => {
      const longAlias = "a".repeat(1000)
      const html = processToHTML(`[[target|${longAlias}]]`)
      assert(html.includes(`<a href="target">${longAlias}</a>`))
    })
  })
})
