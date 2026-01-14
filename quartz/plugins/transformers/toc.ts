import { QuartzTransformerPlugin } from "../../types/plugin"
import type { Heading, PhrasingContent, Root } from "mdast"
import { visit } from "unist-util-visit"
import { toString } from "mdast-util-to-string"
import Slugger from "github-slugger"
import { isWikilink } from "../../extensions/micromark-extension-ofm-wikilinks"
import type { Wikilink } from "../../extensions/micromark-extension-ofm-wikilinks"

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

export interface TocEntry {
  depth: number
  text: string
  slug: string // this is just the anchor (#some-slug), not the canonical slug
}

const slugAnchor = new Slugger()

function normalizeWikilinkText(node: PhrasingContent): string {
  if (!isWikilink(node)) {
    return ""
  }

  const { alias, anchor, target } = (node as Wikilink).data?.wikilink ?? {}

  const trimmedAlias = alias?.trim()
  if (trimmedAlias) {
    return trimmedAlias
  }

  const trimmedAnchor = anchor?.trim().replace(/^#\^?/, "")
  if (trimmedAnchor) {
    return trimmedAnchor
  }

  const trimmedTarget = target?.trim()
  if (trimmedTarget) {
    return trimmedTarget
  }

  return ""
}

function extractHeadingText(node: Heading): string {
  const content = node.children
    .map((child) => (isWikilink(child) ? normalizeWikilinkText(child) : toString(child)))
    .join("")
    .replace(/\s+/g, " ")
    .trim()

  return content
}

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
                  const normalizedText = extractHeadingText(node)
                  const text = normalizedText.length > 0 ? normalizedText : toString(node)
                  highestDepth = Math.min(highestDepth, node.depth)
                  toc.push({
                    depth: node.depth,
                    text,
                    slug: slugAnchor.slug(text),
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
