import { test } from "node:test"
import * as assert from "node:assert/strict"
import { micromark } from "micromark"
import { sidenote, sidenoteDefinition, sidenoteFromMarkdown, sidenoteToMarkdown } from "./index.js"
import { fromMarkdown } from "mdast-util-from-markdown"
import { toMarkdown } from "mdast-util-to-markdown"
import { math } from "micromark-extension-math"
import { mathFromMarkdown } from "mdast-util-math"

test("sidenotes", async (t) => {
  const htmlOptions = {
    extensions: [sidenote(), sidenoteDefinition()],
    htmlExtensions: [
      {
        enter: {
          sidenote(token) {
            this.tag("<sidenote>")
          },
          sidenoteProperties(token) {
            this.tag("<properties>")
          },
          sidenoteLabel(token) {
            this.tag("<label>")
          },
          sidenoteContent(token) {
            this.tag("<content>")
          },
          sidenoteReference(token) {
            this.tag("<sidenote-ref>")
          },
          sidenoteDefinition(token) {
            this.tag("<sidenote-def>")
          },
        },
        exit: {
          sidenote(token) {
            this.tag("</sidenote>")
          },
          sidenoteProperties(token) {
            this.tag("</properties>")
          },
          sidenoteLabel(token) {
            this.tag("</label>")
          },
          sidenoteContent(token) {
            this.tag("</content>")
          },
          sidenoteLabelChunk(token) {
            this.raw(this.sliceSerialize(token))
          },
          sidenoteContentChunk(token) {
            this.raw(this.sliceSerialize(token))
          },
          sidenotePropertiesChunk(token) {
            this.raw(this.sliceSerialize(token))
          },
          sidenoteReference(token) {
            this.tag("</sidenote-ref>")
          },
          sidenoteReferenceLabelChunk(token) {
            this.raw(this.sliceSerialize(token))
          },
          sidenoteDefinition(token) {
            this.tag("</sidenote-def>")
          },
          sidenoteDefinitionLabelChunk(token) {
            this.raw(this.sliceSerialize(token))
          }
        },
      },
    ],
  }

  await t.test("basic sidenote", () => {
    const input = "{{sidenotes[some text.]: another text}}"
    const output = micromark(input, htmlOptions)
    assert.match(output, /<sidenote>/)
    assert.match(output, /<label>some text.<\/label>/)
    assert.match(output, /<content>\s*another text<\/content>/)
  })

  await t.test("mixed markdown in label", () => {
    const input = "{{sidenotes[_some markdown mixed_]: content}}"
    const output = micromark(input, htmlOptions)
    assert.match(output, /<label>_some markdown mixed_<\/label>/)
  })

  await t.test("properties after label", () => {
    const input = "{{sidenotes[text]<dropdown: true>: items}}"
    const output = micromark(input, htmlOptions)
    assert.match(output, /<label>text<\/label>/)
    assert.match(output, /<properties>dropdown: true<\/properties>/)
    assert.match(output, /<content>\s*items<\/content>/)
  })

  await t.test("balanced brackets in label", () => {
    const input = "{{sidenotes[[[wikilinks]]]: supported}}"
    const output = micromark(input, htmlOptions)
    assert.match(output, /<label>\[\[wikilinks\]\]<\/label>/)
    assert.match(output, /<content>\s*supported<\/content>/)
  })

  await t.test("math in label", () => {
    const input = "{{sidenotes[$E=mc^2$]: content}}"

    const tree = fromMarkdown(input, {
      extensions: [sidenote(), math()],
      mdastExtensions: [
        sidenoteFromMarkdown({
          micromarkExtensions: [math()],
          mdastExtensions: [mathFromMarkdown()],
        }),
      ],
    })

    // @ts-ignore
    const node = tree.children[0].children[0]
    assert.equal(node.type, "sidenote")
    assert.equal(node.data.sidenoteParsed.labelNodes[0].type, "inlineMath")
    assert.equal(node.data.sidenoteParsed.labelNodes[0].value, "E=mc^2")
  })

  await t.test("sidenote reference", () => {
    const input = "{{sidenotes[^ref]}}"
    const output = micromark(input, htmlOptions)
    assert.match(output, /<sidenote-ref>\^ref<\/sidenote-ref>/)
  })

  await t.test("long label with spaces", () => {
    const input = "{{sidenotes[^there is many text in here.]}}"
    const output = micromark(input, htmlOptions)
    assert.match(output, /<sidenote-ref>\^there is many text in here.<\/sidenote-ref>/)
  })

  await t.test("sidenote definition", () => {
    const input = "{{sidenotes[ref]}}:\n  content"
    const output = micromark(input, htmlOptions)
    assert.match(output, /<sidenote-def>ref/)
  })

  await t.test("stream.md crash reproduction", () => {
    const input = `{{sidenotes[pick up the rock]}}:
    The closest pre-modern candidate...`
    const output = micromark(input, htmlOptions)
    assert.match(output, /<sidenote-def>pick up the rock<\/sidenote-def>/)
    // Check if content follows (it might be a code block due to indentation, but verifying no crash is key)
  })

  await t.test("ast transformation", () => {
    const input = "{{sidenotes[label]<key: val>: **content**}}"
    const tree = fromMarkdown(input, {
      extensions: [sidenote()],
      mdastExtensions: [sidenoteFromMarkdown()],
    })

    // @ts-ignore
    const paragraph = tree.children[0]
    // @ts-ignore
    const node = paragraph.children[0]

    assert.equal(node.type, "sidenote")
    assert.equal(node.data.sidenoteParsed.label, "label")
    assert.deepEqual(node.data.sidenoteParsed.properties, { key: "val" })
    // Content should be parsed as markdown children
    assert.equal(node.children[0].type, "strong")
  })
})
