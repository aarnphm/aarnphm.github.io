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
import { QUARTZ } from "../../util/path"

const URL_PATTERN = /https?:\/\/[^\s<>)"]+/g

function normalizeArxivId(id: string): string {
  return id.replace(/^arxiv:/i, "").replace(/v\d+$/i, "")
}

// Rate limiter for arXiv API compliance (3 second minimum between requests)
class RateLimiter {
  private lastRequest = 0
  private readonly minInterval = 3000 // 3 seconds per arXiv Terms of Use

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

interface CachedCitationEntry {
  title: string
  bibkey: string
  lastVerified: number
  inBibFile: boolean
}

interface CitationsCachePayload {
  papers: Record<string, CachedCitationEntry>
  documents: Record<string, string[]>
}

interface CacheState {
  documents: Map<string, string[]>
  papers: Map<string, CachedCitationEntry>
  dirty: boolean
}

const repoRoot = process.env.PWD ?? process.cwd()
const citationsCacheDir = path.join(repoRoot, QUARTZ, ".quartz-cache")
const citationsCacheFile = path.join(citationsCacheDir, "citations.json")

function sanitizeLinks(links: string[] | undefined): string[] {
  if (!Array.isArray(links)) return []
  return Array.from(new Set(links.filter((link) => typeof link === "string"))).sort((a, b) =>
    a.localeCompare(b),
  )
}

function loadCachePayload(): CitationsCachePayload {
  if (!fs.existsSync(citationsCacheFile)) {
    return { papers: {}, documents: {} }
  }

  try {
    const raw = fs.readFileSync(citationsCacheFile, "utf8")
    const parsed = JSON.parse(raw) as any

    return {
      papers: parsed.papers ?? {},
      documents: parsed.documents ?? {},
    }
  } catch {
    return { papers: {}, documents: {} }
  }
}

const cacheState: CacheState = (() => {
  const payload = loadCachePayload()

  const documents = new Map<string, string[]>()
  for (const [doc, links] of Object.entries(payload.documents)) {
    documents.set(doc, sanitizeLinks(links))
  }

  const papers = new Map<string, CachedCitationEntry>()
  for (const [id, value] of Object.entries(payload.papers)) {
    if (
      value &&
      typeof value.title === "string" &&
      typeof value.bibkey === "string" &&
      typeof value.lastVerified === "number" &&
      typeof value.inBibFile === "boolean"
    ) {
      papers.set(normalizeArxivId(id), value)
    }
  }

  return {
    documents,
    papers,
    dirty: false,
  }
})()

function persistCacheState() {
  if (!cacheState.dirty) return
  const documentsEntries = Array.from(cacheState.documents.entries())
    .map(([doc, links]) => [doc, sanitizeLinks(links)] as const)
    .sort(([a], [b]) => a.localeCompare(b))
  const papersEntries = Array.from(cacheState.papers.entries()).sort(([a], [b]) =>
    a.localeCompare(b),
  )

  const payload: CitationsCachePayload = {
    papers: Object.fromEntries(papersEntries),
    documents: Object.fromEntries(documentsEntries),
  }

  fs.writeFileSync(citationsCacheFile, JSON.stringify(payload, null, 2), "utf8")
  cacheState.dirty = false
}

function pruneMetadata() {
  const activeIds = new Set<string>()
  for (const links of cacheState.documents.values()) {
    for (const link of links) {
      const id = extractArxivId(link)
      if (id) {
        activeIds.add(normalizeArxivId(id))
      }
    }
  }

  let removed = false
  for (const id of Array.from(cacheState.papers.keys())) {
    if (!activeIds.has(id)) {
      cacheState.papers.delete(id)
      removed = true
    }
  }

  if (removed) {
    cacheState.dirty = true
  }
}

function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}

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

// Prevent concurrent writes for the same arXiv id which can lead to duplicated
// entries. Each arXiv id will map to a single in-flight promise.
const bibEntryTasks = new Map<string, Promise<{ key: string; entry: string }>>()

// File-level mutex to prevent concurrent writes to References.bib
class FileMutex {
  private locked = false
  private queue: (() => void)[] = []

  async acquire(): Promise<void> {
    if (!this.locked) {
      this.locked = true
      return
    }

    return new Promise((resolve) => {
      this.queue.push(resolve)
    })
  }

  release(): void {
    const next = this.queue.shift()
    if (next) {
      next()
    } else {
      this.locked = false
    }
  }
}

const bibFileLock = new FileMutex()

async function fetchArxivMetadata(id: string): Promise<ArxivMeta> {
  const ARXIV_API_BASE = "http://export.arxiv.org/api/query"
  const queryUrl = `${ARXIV_API_BASE}?id_list=${id}`

  const res = await fetch(queryUrl, {
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

// Fetch BibTeX directly from arXiv's /bibtex/ endpoint
async function fetchBibtex(id: string): Promise<string> {
  const norm = normalizeArxivId(id)
  const url = `https://arxiv.org/bibtex/${norm}`

  // Respect arXiv rate limits
  await arxivRateLimiter.wait()

  const res = await fetch(url, {
    headers: {
      "User-Agent": "QuartzArxivTransformer/1.0 (+https://github.com/aarnphm)",
    },
  })

  if (!res.ok) {
    throw new Error(`arXiv BibTeX fetch failed for ${norm}: ${res.status} ${res.statusText}`)
  }

  const bibtex = await res.text()

  // Validate we got BibTeX
  if (!bibtex.trim().startsWith("@")) {
    throw new Error(`Invalid BibTeX response for ${norm}`)
  }

  return bibtex.trim()
}

// Extract normalized arXiv ID from various sources
function extractArxivIdFromEntry(entry: any): string | null {
  // Try eprint field (varying capitalizations)
  const eprint = entry?.eprint ?? entry?.EPRINT ?? entry?.Eprint
  if (eprint) {
    return normalizeArxivId(String(eprint))
  }

  // Try URL field
  const url = entry?.url ?? entry?.URL
  if (url) {
    const match = String(url).match(/arxiv\.org\/(?:abs|pdf)\/([0-9]+\.[0-9]+)(?:v\d+)?/i)
    if (match) {
      return normalizeArxivId(match[1])
    }
  }

  return null
}

// Simplified ensureBibEntry with proper deduplication
async function ensureBibEntry(
  bibPath: string,
  id: string,
): Promise<{ key: string; entry: string }> {
  // De-dupe concurrent calls per arXiv id
  if (bibEntryTasks.has(id)) return bibEntryTasks.get(id)!

  const task = (async (): Promise<{ key: string; entry: string }> => {
    const absPath = path.isAbsolute(bibPath) ? bibPath : path.join(process.env.PWD!, bibPath)
    const targetArxivId = normalizeArxivId(id)

    // Acquire file lock for atomic read-check-write
    await bibFileLock.acquire()
    try {
      // Load current library using citation-js
      const fileContent = fs.existsSync(absPath) ? fs.readFileSync(absPath, "utf8") : ""
      const lib = fileContent.trim() ? new Cite(fileContent, { generateGraph: false }) : null
      const libItems = (lib?.data as any[]) ?? []

      // Check if this arXiv paper already exists
      const existingEntry = libItems.find((entry: any) => {
        const entryArxivId = extractArxivIdFromEntry(entry)
        return entryArxivId === targetArxivId
      })

      if (existingEntry) {
        return { key: existingEntry.id, entry: "" }
      }

      // Fetch new entry from arXiv (release lock during fetch to avoid blocking)
      bibFileLock.release()
      const bibtex = await fetchBibtex(targetArxivId)
      await bibFileLock.acquire()

      // Re-check file after fetching (in case another task added it)
      const updatedContent = fs.existsSync(absPath) ? fs.readFileSync(absPath, "utf8") : ""
      const updatedLib = updatedContent.trim()
        ? new Cite(updatedContent, { generateGraph: false })
        : null
      const updatedItems = (updatedLib?.data as any[]) ?? []

      const recheckEntry = updatedItems.find((entry: any) => {
        const entryArxivId = extractArxivIdFromEntry(entry)
        return entryArxivId === targetArxivId
      })

      if (recheckEntry) {
        return { key: recheckEntry.id, entry: "" }
      }

      // Parse fetched entry
      const newCite = new Cite(bibtex, { generateGraph: false })
      let newEntry = newCite.data[0]

      // Handle key collisions by appending arXiv ID
      if (updatedItems.some((x: any) => x.id === newEntry.id)) {
        const suffix = targetArxivId.replace(/\./g, "")
        newEntry = { ...newEntry, id: `${newEntry.id}-${suffix}` }
      }

      // Append to file
      const prefix = updatedContent.trim().length ? "\n\n" : ""
      const rendered = new Cite(newEntry).format("bibtex")
      fs.appendFileSync(absPath, `${prefix}${rendered}\n`)

      return { key: newEntry.id, entry: rendered }
    } finally {
      bibFileLock.release()
    }
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
      () => async (tree: Root, file: any) => {
        const arxivNodes: { node: Link; index: number; parent: any; id: string }[] = []

        visit(tree, "link", (node: Link, index: number | undefined, parent: any) => {
          if (index === undefined || !parent) return
          const arxivId = extractArxivId(node.url)
          if (!arxivId) return

          arxivNodes.push({ node, index, parent, id: normalizeArxivId(arxivId) })
        })

        const fileKey: string | undefined =
          (file?.data?.filePath as string | undefined) ?? (file?.path as string | undefined)

        const docIds = Array.from(new Set(arxivNodes.map((entry) => entry.id))).sort()
        const canonicalLinks = docIds.map((id) => `https://arxiv.org/abs/${id}`)
        const previousLinks = fileKey ? cacheState.documents.get(fileKey) : undefined
        const previousLinksSanitized = previousLinks ? sanitizeLinks(previousLinks) : []

        let docChanged = false
        if (fileKey) {
          if (canonicalLinks.length === 0) {
            if (cacheState.documents.has(fileKey)) {
              cacheState.documents.delete(fileKey)
              docChanged = true
            }
          } else {
            docChanged = !previousLinks || !arraysEqual(previousLinksSanitized, canonicalLinks)
            cacheState.documents.set(fileKey, canonicalLinks)
          }
        } else if (canonicalLinks.length > 0) {
          docChanged = true
        }

        if (docChanged) {
          cacheState.dirty = true
        }

        if (docChanged) {
          pruneMetadata()
        }

        if (arxivNodes.length === 0) {
          if (cacheState.dirty) {
            persistCacheState()
          }
          return
        }

        const VERIFICATION_TTL = 24 * 60 * 60 * 1000 // 24 hours
        const now = Date.now()

        for (const { node, index, parent, id } of arxivNodes) {
          const cachedEntry = cacheState.papers.get(id)

          // check if cached entry is fresh and verified
          const needsVerification =
            !cachedEntry ||
            !cachedEntry.inBibFile ||
            now - cachedEntry.lastVerified > VERIFICATION_TTL

          let entry: CachedCitationEntry
          if (needsVerification) {
            // verify against References.bib
            const result = await ensureBibEntry(bibliography, id)
            const bibkey = result.key

            // fetch metadata if this is a new entry
            if (!cachedEntry) {
              const meta = await fetchArxivMetadata(id)
              entry = {
                title: meta.title,
                bibkey,
                lastVerified: now,
                inBibFile: true,
              }
            } else {
              // update verification timestamp
              entry = {
                ...cachedEntry,
                bibkey,
                lastVerified: now,
                inBibFile: true,
              }
            }

            cacheState.papers.set(id, entry)
            cacheState.dirty = true
          } else {
            // trust cached entry (we know it exists because needsVerification is false)
            entry = cachedEntry
          }

          node.children = [{ type: "text", value: entry.title } as Text]
          parent.children.splice(index, 1, node, {
            type: "text",
            value: ` [@${entry.bibkey}] `,
          } as Text)
        }

        if (cacheState.dirty) {
          persistCacheState()
        }
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
