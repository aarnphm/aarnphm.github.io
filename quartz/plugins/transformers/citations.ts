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

// Replace your fetchBibtex with this Atom->BibTeX path
async function fetchBibtex(id: string): Promise<string> {
  const norm = normalizeArxivId(id)
  const ARXIV_API_BASE = "http://export.arxiv.org/api/query"
  const url = `${ARXIV_API_BASE}?id_list=${encodeURIComponent(norm)}`

  const res = await fetch(url, {
    headers: {
      // arXiv asks for a descriptive UA for automated access
      // https://info.arxiv.org/help/api/user-manual.html
      "User-Agent": "QuartzArxivTransformer/1.0 (+https://github.com/aarnphm)",
      Accept: "application/atom+xml",
    },
  })
  if (!res.ok) throw new Error(`arXiv API error for ${norm}: ${res.status} ${res.statusText}`)

  const xml = await res.text()
  const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" })
  const feed = parser.parse(xml)
  const entry = feed?.feed?.entry
  if (!entry) throw new Error(`No entry returned for arXiv id ${norm}`)

  // Normalize "entry" to object even when only one result is returned
  const e = Array.isArray(entry) ? entry[0] : entry

  // title, authors, published year
  const title: string = String(e.title || "")
    .trim()
    .replace(/\s+/g, " ")
  const authorsRaw = Array.isArray(e.author) ? e.author : [e.author]
  const authors: string[] = authorsRaw.map((a: any) => String(a.name || "").trim()).filter(Boolean)
  const year: string = String(e.published || "").slice(0, 4)

  // primary category
  const primaryClass: string = e["arxiv:primary_category"]?.["@_term"] || ""

  // DOI (if supplied) and journal_ref (if supplied)
  const doi: string | undefined = e["arxiv:doi"] ? String(e["arxiv:doi"]).trim() : undefined
  const journalRef: string | undefined = e["arxiv:journal_ref"]
    ? String(e["arxiv:journal_ref"]).trim()
    : undefined

  // Construct a stable key: firstAuthorLastNameYYYYarXivID (no version)
  const firstLast = (authors[0] || "unknown")
    .split(/\s+/)
    .pop()!
    .toLowerCase()
    .replace(/[^a-z]/g, "")
  const key = `${firstLast}${year || "n.d."}${norm.replace(/\./g, "")}`

  // Author list "First Last and First Last"
  const authorBib = authors.join(" and ")

  // Prefer @misc for preprints; include eprint & archivePrefix per biblatex guidance
  // See: arXiv bib/eprint guidance and BibLaTeX usage notes.
  const fields: Record<string, string> = {
    title,
    author: authorBib,
    year: year || new Date().getUTCFullYear().toString(),
    eprint: norm,
    archivePrefix: "arXiv",
  }
  if (primaryClass) fields.primaryClass = primaryClass
  if (doi) fields.doi = doi
  // If journal_ref present (post-publication), include it as 'journal'
  if (journalRef) fields.journal = journalRef
  fields.url = `https://arxiv.org/abs/${norm}`

  // Emit formatted BibTeX
  const body = Object.entries(fields)
    .map(([k, v]) => `  ${k} = {${v}}`)
    .join(",\n")

  return `@misc{${key},\n${body}\n}`
}

function normEprint(x?: unknown): string | null {
  if (!x) return null
  return String(x)
    .replace(/^arxiv:/i, "")
    .replace(/v\d+$/i, "")
    .trim()
}
function arxivIdFromUrl(url?: unknown): string | null {
  if (!url) return null
  const m = String(url).match(/arxiv\.org\/(?:abs|pdf)\/([0-9]+\.[0-9]+)(?:v\d+)?/i)
  return m ? m[1] : null
}
/** same work iff same versionless eprint OR same arXiv id in URL */
function isSameWorkMinimal(a: any, b: any): boolean {
  const ae = normEprint(a.eprint),
    be = normEprint(b.eprint)
  if (ae && be && ae === be) return true
  const au = arxivIdFromUrl(a.url),
    bu = arxivIdFromUrl(b.url)
  if (au && bu && au === bu) return true
  return false
}

async function ensureBibEntry(
  bibPath: string,
  id: string,
): Promise<{ key: string; entry: string }> {
  // de-dupe concurrent calls per arXiv id
  if (bibEntryTasks.has(id)) return bibEntryTasks.get(id)!

  const task = (async (): Promise<{ key: string; entry: string }> => {
    const absPath = path.isAbsolute(bibPath) ? bibPath : path.join(process.env.PWD!, bibPath)

    // load current library
    const fileContent = fs.existsSync(absPath) ? fs.readFileSync(absPath, "utf8") : ""
    const lib = new Cite(fileContent, { generateGraph: false })
    const libItems = lib.data as any[]

    // candidate identity from the *id* you were asked to insert
    const eprint = normEprint(id)
    const candidate = {
      eprint, // e.g., "2405.16444"
      url: `https://arxiv.org/abs/${eprint}`, // matches future/old entries
    }

    // 1) If anything in lib matches by (eprint || url-id), reuse it
    const found = libItems.find((x: any) => isSameWorkMinimal(x, candidate))
    if (found) return { key: found.id, entry: "" }

    // 2) Otherwise synthesize a BibTeX entry from the Atom API (captcha-free)
    const bibtex = (await fetchBibtex(eprint!)).trim() // your Atom->BibTeX impl
    const newCite = new Cite(bibtex, { generateGraph: false })
    let newEntry = newCite.data[0]

    // 3) If the generated key collides, namespace deterministically with arXiv id
    if (libItems.some((x: any) => x.id === newEntry.id)) {
      const suffix = eprint!.replace(/\./g, "")
      newEntry = { ...newEntry, id: `${newEntry.id}-${suffix}` }
    }

    // 4) Append atomically
    const prefix = fileContent.trim().length ? "\n\n" : ""
    const rendered = new Cite(newEntry).format("bibtex")
    fs.appendFileSync(absPath, `${prefix}${rendered}\n`)

    return { key: newEntry.id, entry: rendered }
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

          const cacheKey = `arxiv:${arxivId}`
          tasks.push(
            (async () => {
              let cached = cache.get(cacheKey)
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
