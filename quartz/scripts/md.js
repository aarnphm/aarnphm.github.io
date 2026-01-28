import { fromMarkdown } from "mdast-util-from-markdown"
import { frontmatterFromMarkdown } from "mdast-util-frontmatter"
import { gfmFromMarkdown } from "mdast-util-gfm"
import { toString } from "mdast-util-to-string"
import { frontmatter } from "micromark-extension-frontmatter"
import { gfm } from "micromark-extension-gfm"
import fs from "node:fs/promises"
import remarkMath from "remark-math"
import remarkParse from "remark-parse"
import { read } from "to-vfile"
import { unified } from "unified"

const doc = await fs.readFile("./content/thoughts/sparse autoencoder.md")

const parsed = fromMarkdown(doc, {
  extensions: [frontmatter(["yaml", "toml"]), gfm()],
  mdastExtensions: [frontmatterFromMarkdown(["yaml", "toml"]), gfmFromMarkdown()],
})
console.log(parsed)

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
