import rehypeCitation from "rehype-citation"
import { visit } from "unist-util-visit"
import { Element, Text as HastText } from "hast"
import { QuartzTransformerPlugin } from "../types"
import { extractArxivId } from "./links"
import { h } from "hastscript"
import { XMLParser } from "fast-xml-parser"
import { Root, Link, Text } from "mdast"
import { Cite } from "@citation-js/core"
import "@citation-js/plugin-bibtex"
import "@citation-js/plugin-doi"
import fs from "node:fs"
import path from "node:path"

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
  tagName === "a" &&
  Boolean(properties.href) &&
  typeof properties.href === "string" &&
  properties.href.startsWith("#bib")

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

// In-memory cache so each id is processed at most once per build.
const cache = new Map<string, { title: string; bibkey: string }>()

// Prevent concurrent writes for the same arXiv id which can lead to duplicated
// entries. Each arXiv id will map to a single in-flight promise.
const bibEntryTasks = new Map<string, Promise<{ key: string; entry: string }>>()

function normalizeArxivId(id: string): string {
  return id.replace(/^arxiv:/i, "").replace(/v\d+$/i, "")
}

async function fetchArxivMetadata(id: string): Promise<ArxivMeta> {
  const ARXIV_API_BASE = "http://export.arxiv.org/api/query"
  const queryUrl = `${ARXIV_API_BASE}?id_list=${id}`

  const res = await fetch(queryUrl, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (compatible; QuartzArxivTransformer/1.0; +https://github.com/aarnphm)",
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

async function fetchBibtex(id: string): Promise<string> {
  const res = await fetch(`https://arxiv.org/bibtex/${id}`)
  if (!res.ok) throw new Error(`Failed to fetch BibTeX for ${id}: ${res.statusText}`)
  const html = await res.text()
  // The response is HTML with <pre> block containing the bibtex. Extract first @...}
  const match = html.match(/@.*?\n}/s)
  if (!match) throw new Error(`Unable to extract BibTeX for ${id}`)
  return match[0]
}

async function ensureBibEntry(
  bibPath: string,
  id: string,
): Promise<{ key: string; entry: string }> {
  if (bibEntryTasks.has(id)) {
    return bibEntryTasks.get(id)!
  }

  const task = (async (): Promise<{ key: string; entry: string }> => {
    const absPath = path.isAbsolute(bibPath) ? bibPath : path.join(process.env.PWD!, bibPath)

    const fileContent = fs.existsSync(absPath) ? fs.readFileSync(absPath, "utf8") : ""
    const references = new Cite(fileContent, { generateGraph: false })

    const normId = normalizeArxivId(id)

    const existing = (references.data as any[]).find((d: any) => {
      const eprintNorm = normalizeArxivId(String(d.eprint ?? ""))
      if (eprintNorm && eprintNorm === normId) return true
      const url = String(d.url ?? "")
      if (url.includes(`/abs/${normId}`)) return true
      const journal = String(d.journal ?? "")
      if (journal.includes(`arXiv:${normId}`)) return true
      return false
    }) as any | undefined

    if (existing) {
      return { key: existing.id, entry: "" }
    }

    const bibtex = (await fetchBibtex(id)).trim()
    const newCite = new Cite(bibtex, { generateGraph: false })
    const newKey = newCite.data[0].id

    if ((references.data as any[]).some((d: any) => d.id === newKey)) {
      return { key: newKey, entry: "" }
    }

    const prefix = fileContent.trim().length ? "\n\n" : ""
    fs.appendFileSync(absPath, `${prefix}${bibtex}\n`)

    return { key: newKey, entry: bibtex }
  })()

  bibEntryTasks.set(id, task)
  try {
    return await task
  } finally {
    bibEntryTasks.delete(id)
  }
}

export const Citations: QuartzTransformerPlugin<Options> = (opts?: Options) => {
  const bibliography = opts?.bibliography ?? "content/References.bib"
  return {
    name: "Citations",
    markdownPlugins: () => [
      () => async (tree: Root) => {
        const tasks: Promise<void>[] = []

        visit(tree, "link", (node: Link, index: number | undefined, parent: any) => {
          if (index === undefined || !parent) return
          const arxivId = extractArxivId(node.url)
          if (!arxivId) return

          tasks.push(
            (async () => {
              let cached = cache.get(arxivId)
              if (!cached) {
                const meta = await fetchArxivMetadata(arxivId)
                const { key: bibkey } = await ensureBibEntry(bibliography, arxivId)
                cached = { title: meta.title, bibkey }
                cache.set(arxivId, cached)
              }

              node.children = [{ type: "text", value: cached.title } as Text]

              parent.children.splice(index, 1, node, {
                type: "text",
                value: ` [@${cached.bibkey}] `,
              } as Text)
            })(),
          )
        })

        if (tasks.length) await Promise.all(tasks)
      },
    ],
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
  }
}
