import { XMLParser } from "fast-xml-parser"
import { Element, Text as HastText } from "hast"
import { h } from "hastscript"
import { Root, Link, Text } from "mdast"
import rehypeCitation from "rehype-citation"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../../types/plugin"
import { cacheState, CachedCitationEntry, makeBibKey, normalizeArxivId } from "../stores/citations"
import "@citation-js/plugin-bibtex"
import "@citation-js/plugin-doi"
import { extractArxivId } from "./links"

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
const ARXIV_BATCH_SIZE = 50

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
    citationsDisabled?: boolean
  }
}

function chunkIds(ids: string[], size: number): string[][] {
  const chunks: string[][] = []
  for (let i = 0; i < ids.length; i += size) {
    chunks.push(ids.slice(i, i + size))
  }
  return chunks
}

function extractIdFromEntry(entry: any): string | null {
  const raw = typeof entry?.id === "string" ? entry.id : ""
  if (!raw) return null
  const tail = raw.includes("/") ? (raw.split("/").pop() ?? raw) : raw
  const normalized = normalizeArxivId(tail)
  return normalized || null
}

function parseArxivEntry(entry: any): ArxivMeta | null {
  const id = extractIdFromEntry(entry)
  if (!id) return null
  const titleRaw = typeof entry?.title === "string" ? entry.title : ""
  const published = typeof entry?.published === "string" ? entry.published : ""
  const authors = Array.isArray(entry?.author)
    ? entry.author.map((a: any) => a?.name).filter(Boolean)
    : entry?.author?.name
      ? [entry.author.name]
      : []
  const category =
    entry?.["arxiv:primary_category"] && entry["arxiv:primary_category"]["@_term"]
      ? entry["arxiv:primary_category"]["@_term"]
      : ""

  return {
    id,
    title: titleRaw.trim().replace(/\s+/g, " "),
    authors,
    year: published.slice(0, 4),
    category,
    url: `https://arxiv.org/abs/${id}`,
  }
}

async function fetchArxivMetadataBatch(ids: string[]): Promise<Map<string, ArxivMeta>> {
  const unique = Array.from(new Set(ids.map((id) => normalizeArxivId(id)).filter(Boolean)))
  const result = new Map<string, ArxivMeta>()
  if (unique.length === 0) return result

  const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" })
  const batches = chunkIds(unique, ARXIV_BATCH_SIZE)

  for (const batch of batches) {
    await arxivRateLimiter.wait()
    const res = await fetch(`http://export.arxiv.org/api/query?id_list=${batch.join(",")}`, {
      headers: {
        "User-Agent": "QuartzArxivTransformer/1.0 (+https://github.com/aarnphm)",
      },
    })

    if (!res.ok) throw new Error(`arXiv API error for ${batch.join(",")}: ${res.statusText}`)

    const xml = await res.text()
    const parsed = parser.parse(xml) as any
    const entries = Array.isArray(parsed?.feed?.entry)
      ? parsed.feed.entry
      : parsed?.feed?.entry
        ? [parsed.feed.entry]
        : []

    for (const entry of entries) {
      const meta = parseArxivEntry(entry)
      if (!meta) continue
      result.set(meta.id, meta)
    }
  }

  return result
}

export const Citations: QuartzTransformerPlugin<Options> = (opts?: Options) => {
  const bibliography = opts?.bibliography ?? "content/References.bib"
  return {
    name: "Citations",
    markdownPlugins: () => [
      () => async (tree: Root, file: any) => {
        const frontmatter = file.data?.frontmatter ?? {}
        const disableCitations = frontmatter.citations === false || frontmatter.noCitations === true
        if (disableCitations) {
          file.data.citationsDisabled = true
          delete file.data.citations
          return
        }
        file.data.citationsDisabled = false
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
        const missingIds: string[] = []
        for (const id of docIds) {
          const cachedEntry = cacheState.papers.get(id)
          const isFresh =
            !!cachedEntry &&
            cachedEntry.inBibFile &&
            now - cachedEntry.lastVerified <= VERIFICATION_TTL
          if (!isFresh) {
            missingIds.push(id)
          }
        }

        if (missingIds.length > 0) {
          const metas = await fetchArxivMetadataBatch(missingIds)
          for (const id of missingIds) {
            const cachedEntry = cacheState.papers.get(id)
            const meta = metas.get(id)
            const title = cachedEntry?.title ?? meta?.title ?? id
            const bibkey = cachedEntry?.bibkey ?? makeBibKey(id)
            const entry: CachedCitationEntry = {
              title,
              bibkey,
              lastVerified: cachedEntry?.lastVerified ?? 0,
              inBibFile: cachedEntry?.inBibFile ?? false,
            }
            cacheState.papers.set(id, entry)
            cacheState.dirty = true
          }
        }

        for (const { node, index, parent, id } of arxivNodes) {
          const entry = cacheState.papers.get(id)
          if (!entry) continue

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
      () => (tree, file: any) => {
        if (file?.data?.citationsDisabled) return
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
      () => (tree, file: any) => {
        if (file?.data?.citationsDisabled) return
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
