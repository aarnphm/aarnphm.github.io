import test, { describe } from "node:test"
import assert from "node:assert"
import {
  wikilinkToMdast,
  containsWikilinks,
  escapeWikilinkForTable,
  WikilinkToMdastOptions,
} from "./mdast-util-wikilinks"
import { ParsedWikilink, parseWikilink } from "./wikilinks"
import { Image, Html, Link } from "mdast"

describe("containsWikilinks", () => {
  test("detects wikilinks in text", () => {
    assert(containsWikilinks("[[test]]"))
    assert(containsWikilinks("some text [[link]] more text"))
    assert(containsWikilinks("![[embed]]"))
    assert(containsWikilinks("[[page#header|alias]]"))

    assert(!containsWikilinks("no links here"))
    assert(!containsWikilinks("[regular markdown](url)"))
    assert(!containsWikilinks("[[incomplete"))
    assert(!containsWikilinks("incomplete]]"))
  })
})

describe("escapeWikilinkForTable", () => {
  test("escapes pipes in wikilinks", () => {
    assert.strictEqual(escapeWikilinkForTable("[[page|alias]]"), "[[page\\|alias]]")
    assert.strictEqual(escapeWikilinkForTable("[[page]]"), "[[page]]")
    assert.strictEqual(escapeWikilinkForTable("[[a|b|c]]"), "[[a\\|b\\|c]]")
  })

  test("escapes hashes in wikilinks", () => {
    assert.strictEqual(escapeWikilinkForTable("[[page#header]]"), "[[page\\#header]]")
    assert.strictEqual(escapeWikilinkForTable("[[page#header|alias]]"), "[[page\\#header\\|alias]]")
  })

  test("preserves already-escaped characters", () => {
    assert.strictEqual(escapeWikilinkForTable("[[page\\|already]]"), "[[page\\|already]]")
    // double backslash should allow escaping the pipe
    assert.strictEqual(escapeWikilinkForTable("[[page\\\\|alias]]"), "[[page\\\\\\|alias]]")
  })
})

describe("wikilinkToMdast", () => {
  describe("basic links", () => {
    test("converts simple wikilink to link node", () => {
      const parsed: ParsedWikilink = {
        raw: "[[test]]",
        target: "test",
        embed: false,
      }

      const result = wikilinkToMdast(parsed) as Link
      assert.strictEqual(result.type, "link")
      assert.strictEqual(result.url, "test")
      assert.strictEqual(result.children[0].type, "text")
      assert.strictEqual((result.children[0] as any).value, "test")
    })

    test("converts wikilink with alias", () => {
      const parsed = parseWikilink("[[page|display text]]")!
      const result = wikilinkToMdast(parsed) as Link

      assert.strictEqual(result.type, "link")
      assert.strictEqual(result.url, "page")
      assert.strictEqual((result.children[0] as any).value, "display text")
    })

    test("converts wikilink with header anchor", () => {
      const parsed = parseWikilink("[[page#section]]")!
      const result = wikilinkToMdast(parsed) as Link

      assert.strictEqual(result.type, "link")
      assert.strictEqual(result.url, "page#section")
      assert.strictEqual((result.children[0] as any).value, "page")
    })

    test("converts wikilink with block reference", () => {
      const parsed = parseWikilink("[[page#^block-id]]")!
      const result = wikilinkToMdast(parsed) as Link

      assert.strictEqual(result.type, "link")
      assert.strictEqual(result.url, "page#^block-id")
      assert.strictEqual((result.children[0] as any).value, "page")
    })

    test("converts wikilink with header and alias", () => {
      const parsed = parseWikilink("[[page#section|custom text]]")!
      const result = wikilinkToMdast(parsed) as Link

      assert.strictEqual(result.type, "link")
      assert.strictEqual(result.url, "page#section")
      assert.strictEqual((result.children[0] as any).value, "custom text")
    })
  })

  describe("broken link detection", () => {
    const options: WikilinkToMdastOptions = {
      enableBrokenWikilinks: true,
      allSlugs: ["existing-page", "another/page"],
    }

    test("marks non-existent links as broken", () => {
      const parsed = parseWikilink("[[nonexistent]]")!
      const result = wikilinkToMdast(parsed, options) as Link

      assert.strictEqual(result.type, "link")
      assert.deepStrictEqual(result.data?.hProperties?.className, ["broken"])
    })

    test("does not mark existing links as broken", () => {
      const parsed = parseWikilink("[[existing-page]]")!
      const result = wikilinkToMdast(parsed, options) as Link

      assert.strictEqual(result.type, "link")
      assert(!result.data?.hProperties?.className)
    })

    test("skips broken detection when disabled", () => {
      const parsed = parseWikilink("[[nonexistent]]")!
      const result = wikilinkToMdast(parsed, {
        enableBrokenWikilinks: false,
        allSlugs: ["existing"],
      }) as Link

      assert.strictEqual(result.type, "link")
      assert(!result.data?.hProperties?.className)
    })
  })

  describe("image embeds", () => {
    test("converts image embed to image node", () => {
      const parsed = parseWikilink("![[image.png]]")!
      const result = wikilinkToMdast(parsed) as Image

      assert.strictEqual(result.type, "image")
      assert(result.url.includes("image.png"))
      assert.strictEqual(result.data?.hProperties?.width, "auto")
      assert.strictEqual(result.data?.hProperties?.height, "auto")
    })

    test("handles image with width", () => {
      const parsed = parseWikilink("![[image.png|200]]")!
      const result = wikilinkToMdast(parsed) as Image

      assert.strictEqual(result.type, "image")
      assert.strictEqual(result.data?.hProperties?.width, "200")
      assert.strictEqual(result.data?.hProperties?.height, "auto")
    })

    test("handles image with width and height", () => {
      const parsed = parseWikilink("![[image.png|200x300]]")!
      const result = wikilinkToMdast(parsed) as Image

      assert.strictEqual(result.type, "image")
      assert.strictEqual(result.data?.hProperties?.width, "200")
      assert.strictEqual(result.data?.hProperties?.height, "300")
    })

    test("handles image with alt text and dimensions", () => {
      const parsed = parseWikilink("![[image.png|alt text|100x200]]")!
      const result = wikilinkToMdast(parsed) as Image

      assert.strictEqual(result.type, "image")
      assert.strictEqual(result.data?.hProperties?.alt, "alt text")
      assert.strictEqual(result.data?.hProperties?.width, "100")
      assert.strictEqual(result.data?.hProperties?.height, "200")
    })

    test("supports multiple image formats", () => {
      const formats = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"]

      for (const ext of formats) {
        const parsed = parseWikilink(`![[test${ext}]]`)!
        const result = wikilinkToMdast(parsed)
        assert.strictEqual(result?.type, "image", `failed for ${ext}`)
      }
    })
  })

  describe("video embeds", () => {
    test("converts video embed to html node", () => {
      const parsed = parseWikilink("![[video.mp4]]")!
      const result = wikilinkToMdast(parsed) as Html

      assert.strictEqual(result.type, "html")
      assert(result.value?.includes("<video"))
      assert(result.value?.includes("video.mp4"))
      assert(result.value?.includes("controls"))
      assert(result.value?.includes("loop"))
    })

    test("supports multiple video formats", () => {
      const formats = [".mp4", ".webm", ".ogv", ".mov", ".mkv"]

      for (const ext of formats) {
        const parsed = parseWikilink(`![[test${ext}]]`)!
        const result = wikilinkToMdast(parsed) as Html
        assert.strictEqual(result.type, "html", `failed for ${ext}`)
        assert(result.value?.includes("<video"), `failed for ${ext}`)
      }
    })
  })

  describe("audio embeds", () => {
    test("converts audio embed to html node", () => {
      const parsed = parseWikilink("![[audio.mp3]]")!
      const result = wikilinkToMdast(parsed) as Html

      assert.strictEqual(result.type, "html")
      assert(result.value?.includes("<audio"))
      assert(result.value?.includes("audio.mp3"))
      assert(result.value?.includes("controls"))
    })

    test("supports multiple audio formats", () => {
      const formats = [".mp3", ".wav", ".m4a", ".ogg", ".3gp", ".flac"]

      for (const ext of formats) {
        const parsed = parseWikilink(`![[test${ext}]]`)!
        const result = wikilinkToMdast(parsed) as Html
        assert.strictEqual(result.type, "html", `failed for ${ext}`)
        assert(result.value?.includes("<audio"), `failed for ${ext}`)
      }
    })
  })

  describe("pdf embeds", () => {
    test("converts pdf embed to iframe", () => {
      const parsed = parseWikilink("![[document.pdf]]")!
      const result = wikilinkToMdast(parsed) as Html

      assert.strictEqual(result.type, "html")
      assert(result.value?.includes("<iframe"))
      assert(result.value?.includes("document.pdf"))
      assert(result.value?.includes('class="pdf"'))
    })
  })

  describe("block transclusion", () => {
    test("converts markdown embed to transclude blockquote", () => {
      const parsed = parseWikilink("![[note]]")!
      const result = wikilinkToMdast(parsed) as Html

      assert.strictEqual(result.type, "html")
      assert(result.value?.includes("<blockquote"))
      assert(result.value?.includes('class="transclude"'))
      assert(result.value?.includes("note"))
      assert.strictEqual(result.data?.hProperties?.transclude, true)
    })

    test("includes block reference in transclude", () => {
      const parsed = parseWikilink("![[note#^block]]")!
      const result = wikilinkToMdast(parsed) as Html

      assert.strictEqual(result.type, "html")
      assert(result.value?.includes('data-block="#^block"'))
      assert(result.value?.includes("note"))
    })

    test("includes alias in transclude data", () => {
      const parsed = parseWikilink("![[note|custom]]")!
      const result = wikilinkToMdast(parsed) as Html

      assert.strictEqual(result.type, "html")
      assert(result.value?.includes('data-embed-alias="custom"'))
    })
  })

  describe("edge cases", () => {
    test("handles empty target", () => {
      const parsed: ParsedWikilink = {
        raw: "[[]]",
        target: "",
        embed: false,
      }
      const result = wikilinkToMdast(parsed) as Link

      assert.strictEqual(result.type, "link")
      assert.strictEqual(result.url, "")
    })

    test("handles whitespace in targets", () => {
      const parsed = parseWikilink("[[ spaced ]]")!
      const result = wikilinkToMdast(parsed) as Link

      assert.strictEqual(result.type, "link")
      // target is trimmed
      assert.strictEqual(result.url, "spaced")
    })

    test("preserves wikilink metadata in data field", () => {
      const parsed = parseWikilink("[[page#section|alias]]")!
      const result = wikilinkToMdast(parsed) as any

      assert(result.data?.wikilink)
      assert.strictEqual(result.data.wikilink.target, "page")
      assert.strictEqual(result.data.wikilink.anchor, "#section")
      assert.strictEqual(result.data.wikilink.alias, "alias")
    })
  })
})
