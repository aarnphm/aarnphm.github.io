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
import { toHast } from "mdast-util-to-hast"
import { toHtml } from "hast-util-to-html"
import type { Root } from "mdast"

// example 1: basic parsing
console.log("=== example 1: parsing wikilinks ===\n")

const markdown1 = `
Here is a [[simple link]].

And a link with [[page#section|custom display text]].

Embed an image with dimensions: ![[diagram.png|300x200]]

Embed an image with caption: ![[diagram.png|A beautiful diagram]]

Embed an image with alt and dimensions: ![[diagram.png|A beautiful diagram|300x200]]

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

const markdown3 = "see [[important page]] and [[another-page|alias]]."
const tree3 = fromMarkdown(markdown3, {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true })],
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

console.log("\n=== automatic hast conversion demo ===\n")

// example 1: regular link with obsidian mode (default)
console.log("1. regular link (obsidian mode)")
const ex1 = fromMarkdown("[[thoughts/attention|focus]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true })],
})
const hast1 = toHast(ex1)
console.log("  markdown:", "[[thoughts/attention|focus]]")
console.log("  html:    ", toHtml(hast1!))
console.log()

// example 2: image embed with caption (figure/figcaption)
console.log("2. image embed with caption (figure/figcaption)")
const ex2 = fromMarkdown("![[photo.jpg|A beautiful sunset]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true })],
})
const hast2 = toHast(ex2)
console.log("  markdown:", "![[photo.jpg|A beautiful sunset]]")
console.log("  html:    ", toHtml(hast2!))
console.log()

// example 3: image embed with caption and dimensions
console.log("3. image embed with caption and dimensions")
const ex3 = fromMarkdown("![[photo.jpg|A beautiful sunset|400x300]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true })],
})
const hast3 = toHast(ex3)
console.log("  markdown:", "![[photo.jpg|A beautiful sunset|400x300]]")
console.log("  html:    ", toHtml(hast3!))
console.log()

// example 4: image embed with only dimensions (no caption)
console.log("4. image embed with dimensions only (no figure)")
const ex4 = fromMarkdown("![[photo.jpg|400x300]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true })],
})
const hast4 = toHast(ex4)
console.log("  markdown:", "![[photo.jpg|400x300]]")
console.log("  html:    ", toHtml(hast4!))
console.log()

// example 5: video embed
console.log("5. video embed")
const ex5 = fromMarkdown("![[demo.mp4]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true })],
})
const hast5 = toHast(ex5)
console.log("  markdown:", "![[demo.mp4]]")
console.log("  html:    ", toHtml(hast5!))
console.log()

// example 6: block transclude
console.log("6. block transclude")
const ex6 = fromMarkdown("![[notes#summary]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true })],
})
const hast6 = toHast(ex6)
console.log("  markdown:", "![[notes#summary]]")
console.log("  html:    ", toHtml(hast6!))
console.log()

// example 7: stripExtensions option
console.log("7. stripExtensions with obsidian mode")
const ex7 = fromMarkdown("[[notes.md]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: true, stripExtensions: [".md", ".base"] })],
})
const hast7 = toHast(ex7)
console.log("  markdown:", "[[notes.md]]")
console.log("  html:    ", toHtml(hast7!))
console.log()

// example 8: without obsidian (no automatic conversion)
console.log("8. without obsidian mode (no automatic hast conversion)")
const ex8 = fromMarkdown("[[target]]", {
  extensions: [wikilink()],
  mdastExtensions: [wikilinkFromMarkdown({ obsidian: false })],
})
console.log("  markdown:", "[[target]]")
console.log("  note: wikilink node created but no hName/hProperties - manual conversion needed")
console.log("  wikilink data:", (ex8.children[0] as any).children[0].data.wikilink)
