import rehypeCitation from "rehype-citation"
import { PluggableList } from "unified"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
import { Element, Text, Root as HtmlRoot } from "hast"
import { extractArxivId } from "./links"
import { h } from "hastscript"

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

interface LinkType {
  type: string
  pattern: (url: string) => boolean | string | null
  label: string
}

const LINK_TYPES: LinkType[] = [
  {
    type: "arxiv",
    pattern: extractArxivId,
    label: "[arxiv]",
  },
  {
    type: "github",
    pattern: (url: string) => url.toLowerCase().includes("github.com"),
    label: "[GitHub]",
  },
  {
    type: "transformer",
    pattern: (url: string) => url.toLowerCase().includes("transformer-circuits.pub"),
    label: "[link]",
  },
  {
    type: "alignment",
    pattern: (url: string) => url.toLowerCase().includes("alignmentforum.org"),
    label: "[post]",
  },
]

function createTextNode(value: string): Text {
  return { type: "text", value }
}

function getLinkType(url: string): LinkType | undefined {
  return LINK_TYPES.find((type) => type.pattern(url))
}

function createLinkElement(href: string, prettyLinks: boolean): Element {
  const linkType = getLinkType(href)
  const displayText = prettyLinks && linkType ? linkType.label : href

  return h(
    "a.csl-external-link",
    { href, target: "_blank", rel: "noopener noreferrer" },
    createTextNode(displayText),
  )
}

function processTextNode(node: Text, prettyLinks: boolean): (Element | Text)[] {
  const text = node.value
  const matches = Array.from(text.matchAll(URL_PATTERN))

  if (matches.length === 0) {
    return [node]
  }

  const result: (Element | Text)[] = []
  let lastIndex = 0

  matches.forEach((match) => {
    const href = match[0]
    const startIndex = match.index!

    // Add text before URL if exists
    if (startIndex > lastIndex) {
      result.push(createTextNode(text.slice(lastIndex, startIndex)))
    }

    // Add arXiv prefix if applicable
    if (prettyLinks) {
      const arxivId = extractArxivId(href)
      if (arxivId) {
        result.push(createTextNode(`arXiv preprint arXiv:${arxivId} `))
      }
    }

    // Add link element
    result.push(createLinkElement(href, prettyLinks))
    lastIndex = startIndex + href.length
  })

  // Add remaining text after last URL if exists
  if (lastIndex < text.length) {
    result.push(createTextNode(text.slice(lastIndex)))
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

export const checkBib = ({ tagName, properties }: Element) =>
  tagName === "a" &&
  Boolean(properties.href) &&
  typeof properties.href === "string" &&
  properties.href.startsWith("#bib")

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

      plugins.push(
        // Transform the HTML of the citattions; add data-no-popover property to the citation links
        // using https://github.com/syntax-tree/unist-util-visit as they're just anochor links
        () => {
          return (tree, _file) => {
            visit(
              tree,
              (node) => checkBib(node as Element),
              (node, _index, parent) => {
                node.properties["data-no-popover"] = true
                node.properties["data-bib"] = true
                // update citation to be semantically correct
                parent.tagName = "cite"
              },
            )
          }
        },
        // Format external links correctly
        () => {
          return (tree: HtmlRoot, _file) => {
            const checkReferences = ({ properties }: Element): boolean => {
              const className = properties?.className
              return Array.isArray(className) && className.includes("references")
            }
            const checkEntries = ({ properties }: Element): boolean => {
              const className = properties?.className
              return Array.isArray(className) && className.includes("csl-entry")
            }

            visit(
              tree,
              (node) => checkReferences(node as Element),
              (node, index, parent) => {
                const entries: Element[] = []
                visit(
                  node,
                  (node) => checkEntries(node as Element),
                  (node) => {
                    const { properties, children } = node as Element
                    entries.push(
                      h("li", properties, processNodes(children as Element[], opts.prettyLinks!)),
                    )
                  },
                )

                parent!.children.splice(
                  index!,
                  1,
                  h(
                    "section.bibliography",
                    { "data-references": true },
                    h("h2#reference-label", [{ type: "text", value: "Bibliographie" }]),
                    h("ul", ...entries),
                  ),
                )
              },
            )
          }
        },
      )

      return plugins
    },
  }
}
