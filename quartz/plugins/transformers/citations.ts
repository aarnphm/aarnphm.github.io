import rehypeCitation from "rehype-citation"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
import { Element, Text } from "hast"
import { extractArxivId } from "./links"
import { h } from "hastscript"

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
    label: "[arXiv]",
  },
  {
    type: "github",
    pattern: (url: string) => url.toLowerCase().includes("github.com"),
    label: "[GitHub]",
  },
  {
    type: "transformer",
    pattern: (url: string) => url.toLowerCase().includes("transformer-circuits.pub"),
    label: "[transformer circuit]",
  },
  {
    type: "alignment",
    pattern: (url: string) => url.toLowerCase().includes("alignmentforum.org"),
    label: "[alignment forum]",
  },
]

function createTextNode(value: string): Text {
  return { type: "text", value }
}

function getLinkType(url: string): LinkType | undefined {
  return LINK_TYPES.find((type) => type.pattern(url))
}

function createLinkElement(href: string): Element {
  const linkType = getLinkType(href)
  const displayText = linkType ? linkType.label : href

  return h(
    "a.csl-external-link",
    { href, target: "_blank", rel: "noopener noreferrer" },
    createTextNode(displayText),
  )
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
    const href = match[0]
    const startIndex = match.index!

    // Add text before URL if exists
    if (startIndex > lastIndex) {
      result.push(createTextNode(text.slice(lastIndex, startIndex)))
    }

    // Add arXiv prefix if applicable
    const arxivId = extractArxivId(href)
    if (arxivId) {
      result.push(createTextNode(`arXiv preprint arXiv:${arxivId} `))
    }

    // Add link element
    result.push(createLinkElement(href))
    lastIndex = startIndex + href.length
  })

  // Add remaining text after last URL if exists
  if (lastIndex < text.length) {
    result.push(createTextNode(text.slice(lastIndex)))
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

export const checkBib = ({ tagName, properties }: Element) =>
  tagName === "a" &&
  Boolean(properties.href) &&
  typeof properties.href === "string" &&
  properties.href.startsWith("#bib")

export const checkBibSection = ({ type, tagName, properties }: Element) =>
  type === "element" && tagName === "section" && properties.dataReferences == ""

interface Options {
  bibliography: string
}

export const Citations: QuartzTransformerPlugin<Options> = ({ bibliography }: Options) => ({
  name: "Citations",
  htmlPlugins: () => [
    [
      rehypeCitation,
      {
        bibliography,
        suppressBibliography: false,
        linkCitations: true,
        csl: "apa",
      },
    ],
    // Transform the HTML of the citattions; add data-no-popover property to the citation links
    // using https://github.com/syntax-tree/unist-util-visit as they're just anochor links
    () => (tree) => {
      visit(
        tree,
        (node) => checkBib(node as Element),
        (node, _index, parent) => {
          node.properties["data-bib"] = true
          // update citation to be semantically correct
          parent.tagName = "cite"
        },
      )
    },
    // Format external links correctly
    () => (tree) => {
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
              entries.push(h("li", properties, processNodes(children as Element[])))
            },
          )

          parent!.children.splice(
            index!,
            1,
            h(
              "section.bibliography",
              { dataReferences: true },
              h("h2#reference-label", [{ type: "text", value: "Bibliographie" }]),
              h("ul", ...entries),
            ),
          )
        },
      )
    },
  ],
})
