/**
 * example usage of micromark wikilink extension.
 * demonstrates parsing, transformation, and serialization with unified ecosystem.
 */

import { fromMarkdown } from "mdast-util-from-markdown"
import { toMarkdown } from "mdast-util-to-markdown"
import { wikilink } from "./syntax"
import { wikilinkFromMarkdown, Wikilink } from "./fromMarkdown"
import { wikilinkToMarkdown } from "./toMarkdown"
import { visit } from "unist-util-visit"
import type { Root } from "mdast"

// example 1: basic parsing
console.log("=== example 1: parsing wikilinks ===\n")

const markdown1 = `
Here is a [[simple link]].

And a link with [[page#section|custom display text]].

Embed an image: ![[diagram.png|300x200]]

Block reference: [[note#^key-insight]]

Multiple links in one line: [[first]] and [[second|alias]].

Same-file anchor: [[#heading-in-this-file]]

multiple anchors: [[test#h1#h2#h3]]
`.trim()

// parse with wikilink extension
const tree1: Root = fromMarkdown(markdown1, {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})

// extract wikilink nodes from AST
function extractWikilinks(node: any): Wikilink[] {
  const wikilinks: Wikilink[] = []

  function visitNode(n: any) {
    if (n.type === "wikilink") {
      wikilinks.push(n as Wikilink)
    }
    if (n.children) {
      n.children.forEach(visitNode)
    }
  }

  visitNode(node)
  return wikilinks
}

const wikilinks1 = extractWikilinks(tree1)

console.log(`found ${wikilinks1.length} wikilinks:\n`)

wikilinks1.forEach((link, i) => {
  const { target, anchor, alias, embed } = link.data!.wikilink
  console.log(`${i + 1}. ${link.value}`)
  console.log(`   target: "${target}"`)
  if (anchor) console.log(`   anchor: "${anchor}"`)
  if (alias) console.log(`   alias: "${alias}"`)
  if (embed) console.log(`   embed: true`)
  console.log()
})

// example 2: round-trip serialization
console.log("\n=== example 2: round-trip serialization ===\n")

const markdown2 = "check [[page|link]] and [[file#section]]."

const tree2 = fromMarkdown(markdown2, {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})

const serialized = toMarkdown(tree2, {
  extensions: [wikilinkToMarkdown()],
})

console.log("input:  ", markdown2)
console.log("output: ", serialized.trim())
console.log("match:  ", markdown2 === serialized.trim())

// example 3: transform wikilinks
console.log("\n=== example 3: transforming wikilinks ===\n")

const markdown3 = "see [[important-page]] and [[another-page|alias]]."
const tree3 = fromMarkdown(markdown3, {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})

// uppercase all wikilink targets
visit(tree3, "wikilink", (node: Wikilink) => {
  if (node.data?.wikilink) {
    node.data.wikilink.target = node.data.wikilink.target.toUpperCase()
  }
})

const result = toMarkdown(tree3, {
  extensions: [wikilinkToMarkdown()],
})

console.log("input:       ", markdown3)
console.log("transformed: ", result.trim())

// example 4: handling escaping
console.log("\n=== example 4: escaping ===\n")

const escapedMarkdown = `
[[file\\|name]] has a pipe in the target.
[[file\\#tag]] has a hash in the target.
[[file|alias\\]text]] has a bracket in the alias.
`.trim()

const escapedTree = fromMarkdown(escapedMarkdown, {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown()],
})

console.log("parsed targets:")
visit(escapedTree, "wikilink", (node: Wikilink) => {
  const wl = node.data?.wikilink
  if (wl) {
    console.log(`  target: "${wl.target}", alias: "${wl.alias ?? ""}"`)
  }
})

const escapedSerialized = toMarkdown(escapedTree, {
  extensions: [wikilinkToMarkdown()],
})

console.log("\nserialized back:")
console.log(escapedSerialized.trim())

// example 5: Obsidian mode
console.log("\n=== example 5: Obsidian mode ===\n")

const obsidianMarkdown = `
[[NVIDIA#cuda]]
[[file#Parent#Child#Grandchild]]
[[file#Section Title]]
`.trim()

console.log("without Obsidian mode (default):")
const tree5a = fromMarkdown(obsidianMarkdown, {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: false })],
})

visit(tree5a, "wikilink", (node: Wikilink) => {
  const wl = node.data?.wikilink
  if (wl) {
    console.log(`  ${wl.target} → anchor: "${wl.anchor ?? ""}"`)
  }
})

console.log("\nwith Obsidian mode (nested heading + slugification):")
const tree5b = fromMarkdown(obsidianMarkdown, {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true })],
})

visit(tree5b, "wikilink", (node: Wikilink) => {
  const wl = node.data?.wikilink
  if (wl) {
    console.log(`  ${wl.target} → anchor: "${wl.anchor ?? ""}"`)
  }
})
