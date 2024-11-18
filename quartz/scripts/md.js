import fs from "node:fs/promises"
import { fromMarkdown } from "mdast-util-from-markdown"
import { gfm } from "micromark-extension-gfm"
import { frontmatter } from "micromark-extension-frontmatter"
import { frontmatterFromMarkdown } from "mdast-util-frontmatter"
import { gfmFromMarkdown } from "mdast-util-gfm"

const doc = await fs.readFile("./content/thoughts/sparse autoencoder.md")

const tree = fromMarkdown(doc, {
  extensions: [frontmatter(["yaml", "toml"]), gfm()],
  mdastExtensions: [frontmatterFromMarkdown(["yaml", "toml"]), gfmFromMarkdown()],
})

console.log(tree)
