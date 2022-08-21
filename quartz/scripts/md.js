import fs from "node:fs/promises"
import { fromMarkdown } from "mdast-util-from-markdown"
import { gfm } from "micromark-extension-gfm"
import { frontmatter } from "micromark-extension-frontmatter"
import { frontmatterFromMarkdown } from "mdast-util-frontmatter"
import { gfmFromMarkdown } from "mdast-util-gfm"
import remarkMath from "remark-math"
import { read } from "to-vfile"
import { unified } from "unified"
import remarkParse from "remark-parse"
import { toString } from "mdast-util-to-string"

const doc = await fs.readFile("./content/thoughts/sparse autoencoder.md")

const parsed = fromMarkdown(doc, {
  extensions: [frontmatter(["yaml", "toml"]), gfm()],
  mdastExtensions: [frontmatterFromMarkdown(["yaml", "toml"]), gfmFromMarkdown()],
})

const tree = unified()
  .use(remarkParse)
  .use(remarkMath)
  .parse(
    await read(
      "./content/thoughts/university/twenty-four-twenty-five/sfwr-4ml3/nearest neighbour.md",
    ),
  )
console.log(tree.children[tree.children.length - 3])
console.log(toString(tree.children[tree.children.length - 3]))
