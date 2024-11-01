import rehypeCitation from "rehype-citation"
import { PluggableList } from "unified"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
import { Element, Text, Root as HtmlRoot } from "hast"

export interface Options {
  bibliographyFile: string[] | string
  suppressBibliography: boolean
  linkCitations: boolean
  csl: string
}

const defaultOptions: Options = {
  bibliographyFile: ["./bibliography.bib"],
  suppressBibliography: false,
  linkCitations: false,
  csl: "apa",
}

const URL_PATTERN = /https?:\/\/[^\s<>)"]+/g

function extractArxivId(url: string): string {
  // Match patterns like 1234.56789 or hep-th/9912345
  const match = url.match(/(?:arxiv.org\/abs\/|arxiv.org\/pdf\/)([\w.-]+)/i)
  return match ? match[1] : ""
}

function processTextNode(node: Text): (Element | Text)[] {
  const text = node.value
  const matches = Array.from(text.matchAll(URL_PATTERN))

  if (matches.length === 0) {
    return [node]
  }

  const result: (Element | Text)[] = []
  let lastIndex = 0

  matches.forEach((match) => {
    const url = match[0]
    const startIndex = match.index!

    // Add text before URL if exists
    if (startIndex > lastIndex) {
      result.push({
        type: "text",
        value: text.slice(lastIndex, startIndex),
      })
    }

    const isArxiv = url.toLowerCase().includes("arxiv.org")
    const isTransformerCircuit = url.toLowerCase().includes("transformer-circuits.pub")
    const isAF = url.toLowerCase().includes("alignmentforum.org")

    if (isArxiv) {
      // Extract arXiv ID
      const arxivId = extractArxivId(url)

      // Add the formatted text
      result.push({
        type: "text",
        value: `arXiv preprint arXiv:${arxivId} `,
      })
    }

    // Add anchor element for URL
    result.push({
      type: "element",
      tagName: "a",
      properties: {
        href: url,
        target: "_blank",
        rel: "noopener noreferrer",
        className: ["csl-external-link"],
      },
      children: [
        {
          type: "text",
          value: isArxiv ? "[arXiv]" : isTransformerCircuit ? "[link]" : isAF ? "[post]" : url,
        },
      ],
    })
    lastIndex = startIndex + url.length
  })

  // Add remaining text after last URL if exists
  if (lastIndex < text.length) {
    result.push({
      type: "text",
      value: text.slice(lastIndex),
    })
  }

  return result
}

// Function to process a list of nodes
function processNodes(nodes: (Element | Text)[]): (Element | Text)[] {
  return nodes.flatMap((node) => {
    if (node.type === "text") {
      return processTextNode(node)
    }
    if (node.type === "element") {
      return {
        ...node,
        children: processNodes(node.children as (Element | Text)[]),
      }
    }
    return [node]
  })
}

export const Citations: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "Citations",
    htmlPlugins() {
      const plugins: PluggableList = []

      // Add rehype-citation to the list of plugins
      plugins.push([
        rehypeCitation,
        {
          bibliography: opts.bibliographyFile,
          suppressBibliography: opts.suppressBibliography,
          linkCitations: opts.linkCitations,
          csl: opts.csl,
        },
      ])

      // Transform the HTML of the citattions; add data-no-popover property to the citation links
      // using https://github.com/syntax-tree/unist-util-visit as they're just anochor links
      plugins.push(() => {
        return (tree, _file) => {
          visit(tree, "element", (node, _index, _parent) => {
            if (node.tagName === "a" && node.properties?.href?.startsWith("#bib")) {
              node.properties["data-no-popover"] = true
            }
          })
        }
      })

      // Format external links correctly
      plugins.push(() => {
        return (tree: HtmlRoot, _file) => {
          visit(tree, "element", (node) => {
            if ((node.properties?.className as string[])?.includes("references")) {
              visit(node, "element", (entry) => {
                if ((entry.properties?.className as string[])?.includes("csl-entry")) {
                  entry.children = processNodes(entry.children as (Element | Text)[])
                }
              })
            }
          })
        }
      })

      return plugins
    },
  }
}
