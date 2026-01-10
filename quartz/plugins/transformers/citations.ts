import rehypeCitation from "rehype-citation"
import { visit } from "unist-util-visit"
import { Element, Text as HastText } from "hast"
import { QuartzTransformerPlugin } from "../types"
import { extractArxivId } from "./links"
import { h } from "hastscript"
import { XMLParser } from "fast-xml-parser"
import { Root, Link, Text } from "mdast"
import "@citation-js/plugin-bibtex"
import "@citation-js/plugin-doi"
import { cacheState, CachedCitationEntry, makeBibKey, normalizeArxivId } from "../stores/citations"

const URL_PATTERN = /https?:\/\/[^\s<>)"]+/g

class RateLimiter {
  private lastRequest = 0
  private readonly minInterval = 3000

  async wait(): Promise<void> {
    const now = Date.now()
    const elapsed = now - this.lastRequest
    if (elapsed < this.minInterval) {
      await new Promise((resolve) => setTimeout(resolve, this.minInterval - elapsed))
    }
    this.lastRequest = Date.now()
  }
}

const arxivRateLimiter = new RateLimiter()

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
    type: "lesswrong",
    pattern: (url: string) => url.toLowerCase().includes("lesswrong.com"),
    label: "[lesswrong]",
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

function createTextNode(value: string): HastText {
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

function processTextNode(node: HastText): (Element | HastText)[] {
  const text = node.value
  const matches = Array.from(text.matchAll(URL_PATTERN))

  if (matches.length === 0) {
    return [node]
  }

  const result: (Element | HastText)[] = []
  let lastIndex = 0

  matches.forEach((match) => {
    const href = match[0]
    const startIndex = match.index!

    if (startIndex > lastIndex) {
      result.push(createTextNode(text.slice(lastIndex, startIndex)))
    }

    const arxivId = extractArxivId(href)
    if (arxivId) {
      result.push(createTextNode(`arXiv preprint arXiv:${arxivId} `))
    }

    result.push(createLinkElement(href))
    lastIndex = startIndex + href.length
  })

  if (lastIndex < text.length) {
    result.push(createTextNode(text.slice(lastIndex)))
  }

  return result
}

function processNodes(nodes: (Element | HastText)[]): (Element | HastText)[] {
  return nodes.flatMap((node) => {
    if (node.type === "text") {
      return processTextNode(node)
    }
    if (node.type === "element") {
      return {
        ...node,
        children: processNodes(node.children as (Element | HastText)[]),
      }
    }
    return [node]
  })
}

export const checkBib = ({ tagName, properties }: Element) =>
  tagName === "a" && typeof properties?.href === "string" && properties.href.startsWith("#bib")

export const checkBibSection = ({ type, tagName, properties }: Element) =>
  type === "element" && tagName === "section" && properties.dataReferences == ""

interface Options {
  bibliography: string
}

interface ArxivMeta {
  id: string
  title: string
  authors: string[]
  year: string
  category: string
  url: string
}

declare module "vfile" {
  interface DataMap {
    citations?: {
      arxivIds: string[]
    }
  }
}

async function fetchArxivMetadata(id: string): Promise<ArxivMeta> {
  await arxivRateLimiter.wait()
  const res = await fetch(`http://export.arxiv.org/api/query?id_list=${id}`, {
    headers: {
      "User-Agent": "QuartzArxivTransformer/1.0 (+https://github.com/aarnphm)",
    },
  })

  if (!res.ok) throw new Error(`arXiv API error for ${id}: ${res.statusText}`)

  const xml = await res.text()
  const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" })
  const result = parser.parse(xml) as any
  const entry = result.feed.entry
  if (!entry) throw new Error(`No entry returned for arXiv id ${id}`)

  return {
    id,
    title: (entry.title as string).trim().replace(/\s+/g, " "),
    authors: Array.isArray(entry.author)
      ? entry.author.map((a: any) => a.name)
      : [entry.author.name],
    year: (entry.published as string).slice(0, 4),
    category: entry["arxiv:primary_category"]["@_term"],
    url: `https://arxiv.org/abs/${id}`,
  }
}

export const Citations: QuartzTransformerPlugin<Options> = (opts?: Options) => {
  const bibliography = opts?.bibliography ?? "content/References.bib"
  return {
    name: "Citations",
    markdownPlugins: () => [
      () => async (tree: Root, file: any) => {
        const arxivNodes: { node: Link; index: number; parent: any; id: string }[] = []

        visit(tree, "link", (node: Link, index: number | undefined, parent: any) => {
          if (index === undefined || !parent) return
          const arxivId = extractArxivId(node.url)
          if (!arxivId) return

          arxivNodes.push({ node, index, parent, id: normalizeArxivId(arxivId) })
        })

        const docIds = Array.from(new Set(arxivNodes.map((entry) => entry.id))).sort()
        if (docIds.length > 0) {
          file.data.citations = { arxivIds: docIds }
        } else {
          delete file.data.citations
        }

        if (arxivNodes.length === 0) return

        const VERIFICATION_TTL = 24 * 60 * 60 * 1000
        const now = Date.now()

        for (const { node, index, parent, id } of arxivNodes) {
          const cachedEntry = cacheState.papers.get(id)

          const isFresh =
            !!cachedEntry &&
            cachedEntry.inBibFile &&
            now - cachedEntry.lastVerified <= VERIFICATION_TTL

          let entry: CachedCitationEntry
          if (isFresh) {
            entry = cachedEntry
          } else {
            const title = cachedEntry?.title ?? (await fetchArxivMetadata(id)).title
            const bibkey = cachedEntry?.bibkey ?? makeBibKey(id)
            entry = {
              title,
              bibkey,
              lastVerified: cachedEntry?.lastVerified ?? 0,
              inBibFile: cachedEntry?.inBibFile ?? false,
            }
            cacheState.papers.set(id, entry)
            cacheState.dirty = true
          }

          node.children = [{ type: "text", value: entry.title } as Text]
          parent.children.splice(index, 1, node, {
            type: "text",
            value: ` [@${entry.bibkey}] `,
          } as Text)
        }
      },
    ],
    htmlPlugins: ({ cfg }) => [
      [
        rehypeCitation,
        {
          bibliography,
          suppressBibliography: false,
          linkCitations: true,
          csl: "apa",
          lang:
            cfg.configuration.locale !== "en-US"
              ? `https://raw.githubusercontent.com/citation-style-language/locales/refs/heads/master/locales-${cfg.configuration.locale}.xml`
              : "en-US",
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
            parent.tagName = "cite"
          },
        )
      },
      // Format external links correctly
      () => (tree) => {
        visit(
          tree,
          (node) => {
            const className = (node as Element).properties?.className
            return Array.isArray(className) && className.includes("references")
          },
          (node, index, parent) => {
            const entries: Element[] = []
            visit(
              node,
              (node) => {
                const className = (node as Element).properties?.className
                return Array.isArray(className) && className.includes("csl-entry")
              },
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
                h("h2#reference-label", [{ type: "text", value: "bibliographie" }]),
                h("ul", ...entries),
              ),
            )
          },
        )
      },
    ],
  }
}
