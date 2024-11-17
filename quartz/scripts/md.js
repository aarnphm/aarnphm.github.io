import fs from "node:fs/promises"
import { fromMarkdown } from "mdast-util-from-markdown"
import { toMarkdown } from "mdast-util-to-markdown"
import { gfmFootnote } from "micromark-extension-gfm-footnote"
import { gfmFootnoteFromMarkdown, gfmFootnoteToMarkdown } from "mdast-util-gfm-footnote"

const doc = await fs.readFile("./content/thoughts/sparse autoencoder.md")

const tree = fromMarkdown(doc, {
  extensions: [gfmFootnote()],
  mdastExtensions: [gfmFootnoteFromMarkdown()],
})

console.log(tree)
