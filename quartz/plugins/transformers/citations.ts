import rehypeCitation from "rehype-citation"
import { PluggableList } from "unified"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
import { Element, Text, Root as HtmlRoot } from "hast"
import { extractArxivId } from "./links"

export interface Options {
  bibliographyFile: string[] | string
  suppressBibliography: boolean
  linkCitations: boolean
  csl: string
  prettyLinks?: boolean
}

const defaultOptions: Options = {
  bibliographyFile: ["./bibliography.bib"],
  suppressBibliography: false,
  linkCitations: false,
  csl: "apa",
  prettyLinks: true,
}

const URL_PATTERN = /https?:\/\/[^\s<>)"]+/g

function processTextNode(node: Text, prettyLinks: boolean): (Element | Text)[] {
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

    const isArxiv = extractArxivId(url)
    const isTransformerCircuit = url.toLowerCase().includes("transformer-circuits.pub")
    const isAF = url.toLowerCase().includes("alignmentforum.org")
    const isGitHub = url.toLowerCase().includes("github.com")

    if (prettyLinks) {
      if (isArxiv !== null) {
        // Add the formatted text
        result.push({
          type: "text",
          value: `arXiv preprint arXiv:${isArxiv} `,
        })
      }
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
          value: prettyLinks
            ? isArxiv
              ? "arxiv"
              : isGitHub
                ? "[GitHub]"
                : isTransformerCircuit
                  ? "[link]"
                  : isAF
                    ? "[post]"
                    : url
            : url,
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
function processNodes(nodes: (Element | Text)[], prettyLinks: boolean): (Element | Text)[] {
  return nodes.flatMap((node) => {
    if (node.type === "text") {
      return processTextNode(node, prettyLinks)
    }
    if (node.type === "element") {
      return {
        ...node,
        children: processNodes(node.children as (Element | Text)[], prettyLinks),
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
          visit(tree, "element", (node, _index, parent) => {
            if (node.tagName === "a" && node.properties?.href?.startsWith("#bib")) {
              node.properties["data-no-popover"] = true
              node.properties["data-bib"] = true
              // update citation to be semantically correct
              parent.tagName = "cite"
            }
          })
        }
      })

      // Format external links correctly
      plugins.push(() => {
        return (tree: HtmlRoot, _file) => {
          visit(tree, "element", (node: Element, index, parent) => {
            if ((node.properties?.className as string[])?.includes("references")) {
              const sectionChildren: Element[] = []
              visit(node, "element", (entry) => {
                if ((entry.properties?.className as string[])?.includes("csl-entry")) {
                  sectionChildren.push({
                    type: "element",
                    tagName: "li",
                    properties: {
                      ...entry.properties,
                    },
                    children: processNodes(entry.children as (Element | Text)[], opts.prettyLinks!),
                  })
                }
              })
              parent!.children.splice(index as number, 1, {
                type: "element",
                tagName: "section",
                properties: {
                  "data-references": true,
                  className: ["bibliography"],
                },
                children: [
                  {
                    type: "element",
                    tagName: "h2",
                    properties: { id: "reference-label" },
                    children: [{ type: "text", value: "References" }],
                  },
                  {
                    type: "element",
                    tagName: "ul",
                    properties: {},
                    children: sectionChildren,
                  },
                ],
              })
            }
          })
        }
      })

      return plugins
    },
  }
}
