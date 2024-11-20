import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { visit } from "unist-util-visit"
import { toString } from "mdast-util-to-string"
import Slugger from "github-slugger"
import katex from "katex"
import { toHast } from "mdast-util-to-hast"
import { toString as hastToString } from "hast-util-to-string"
import { fromHtmlIsomorphic } from "hast-util-from-html-isomorphic"
import { ElementContent } from "hast"

export interface Options {
  maxDepth: 1 | 2 | 3 | 4 | 5 | 6
  minEntries: number
  showByDefault: boolean
}

const defaultOptions: Options = {
  maxDepth: 3,
  minEntries: 1,
  showByDefault: true,
}

interface TocEntry {
  depth: number
  text: string
  slug: string // this is just the anchor (#some-slug), not the canonical slug
}

const slugAnchor = new Slugger()
export const TableOfContents: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "TableOfContents",
    markdownPlugins() {
      return [
        () => {
          return async (tree: Root, file) => {
            const display = file.data.frontmatter?.enableToc ?? opts.showByDefault
            if (display) {
              slugAnchor.reset()
              const toc: TocEntry[] = []
              let highestDepth: number = opts.maxDepth
              visit(tree, "heading", (node) => {
                if (node.depth <= opts.maxDepth) {
                  // we need to parse all math in here first
                  const mathBlock: { index: number; value: string }[] = []
                  for (const [i, child] of node.children.entries()) {
                    if (child.type === "inlineMath") {
                      const value = katex.renderToString(child.value, { strict: true })
                      mathBlock.push({ index: i, value })
                      node.children.splice(i, 1, {
                        type: "html",
                        value,
                      })
                    }
                  }
                  const hastNodes = toHast(node, { allowDangerousHtml: true })
                  // support copy-tex and correct parsing of TOC
                  visit(hastNodes, "raw", (node, idx, parent) => {
                    const root = fromHtmlIsomorphic(node.value, { fragment: true })
                    visit(root, "element", (node, idx, parent) => {
                      const classNames = (node.properties?.classNames ?? []) as string[]
                      if (classNames.includes("katex-mathml")) {
                        parent?.children.splice(idx!, 1)
                      }
                    })
                    parent?.children.splice(idx!, 1, ...(root.children as ElementContent[]))
                  })
                  const slug = hastToString(hastNodes)
                  const text = toString(node)
                  highestDepth = Math.min(highestDepth, node.depth)
                  toc.push({
                    depth: node.depth,
                    text,
                    slug: slugAnchor.slug(slug),
                  })
                }
              })

              if (toc.length > 0 && toc.length > opts.minEntries) {
                file.data.toc = toc.map((entry) => ({
                  ...entry,
                  depth: entry.depth - highestDepth,
                }))
              }
            }
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    toc: TocEntry[]
  }
}
