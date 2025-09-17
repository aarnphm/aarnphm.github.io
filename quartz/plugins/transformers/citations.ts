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
import { createHash } from "node:crypto"
import { QUARTZ } from "../../util/path"

const URL_PATTERN = /https?:\/\/[^\s<>)"]+/g

function normalizeArxivId(id: string): string {
  return id.replace(/^arxiv:/i, "").replace(/v\d+$/i, "")
}

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
}

interface CitationsCachePayload {
  hash: string
  documents: Record<string, string[]>
  metadata: Record<string, CachedCitationEntry>
}

interface CacheState {
  documents: Map<string, string[]>
  metadata: Map<string, CachedCitationEntry>
  hash: string
  dirty: boolean
}

const repoRoot = process.env.PWD ?? process.cwd()
const citationsCacheDir = path.join(repoRoot, QUARTZ, ".quartz-cache")
const citationsCacheFile = path.join(citationsCacheDir, "citations.json")

function ensureCacheDir() {
  if (!fs.existsSync(citationsCacheDir)) {
    fs.mkdirSync(citationsCacheDir, { recursive: true })
  }
}

function sanitizeLinks(links: string[] | undefined): string[] {
  if (!Array.isArray(links)) return []
  return Array.from(new Set(links.filter((link) => typeof link === "string"))).sort((a, b) =>
    a.localeCompare(b),
  )
}

function loadCachePayload(): CitationsCachePayload {
  ensureCacheDir()
  if (!fs.existsSync(citationsCacheFile)) {
    return { hash: "", documents: {}, metadata: {} }
  }

  try {
    const raw = fs.readFileSync(citationsCacheFile, "utf8")
    const parsed = JSON.parse(raw) as Partial<CitationsCachePayload>
    return {
      hash: typeof parsed.hash === "string" ? parsed.hash : "",
      documents: parsed.documents ?? {},
      metadata: parsed.metadata ?? {},
    }
  } catch {
    return { hash: "", documents: {}, metadata: {} }
  }
}

function computeDocumentsHash(documents: Map<string, string[]>): string {
  const ordered = Array.from(documents.entries())
    .map(([doc, links]) => [doc, sanitizeLinks(links)] as const)
    .sort(([a], [b]) => a.localeCompare(b))
  return createHash("sha256").update(JSON.stringify(ordered)).digest("hex")
}

const cacheState: CacheState = (() => {
  const payload = loadCachePayload()

  const documents = new Map<string, string[]>()
  for (const [doc, links] of Object.entries(payload.documents)) {
    documents.set(doc, sanitizeLinks(links))
  }

  const metadata = new Map<string, CachedCitationEntry>()
  for (const [id, value] of Object.entries(payload.metadata)) {
    if (value && typeof value.title === "string" && typeof value.bibkey === "string") {
      metadata.set(normalizeArxivId(id), { title: value.title, bibkey: value.bibkey })
    }
  }

  const initialHash =
    payload.hash && typeof payload.hash === "string" && payload.hash.length > 0
      ? payload.hash
      : computeDocumentsHash(documents)

  return {
    documents,
    metadata,
    hash: initialHash,
    dirty: false,
  }
})()

const memoryCache = new Map<string, CachedCitationEntry>()

function persistCacheState() {
  if (!cacheState.dirty) return

  ensureCacheDir()
  const documentsEntries = Array.from(cacheState.documents.entries())
    .map(([doc, links]) => [doc, sanitizeLinks(links)] as const)
    .sort(([a], [b]) => a.localeCompare(b))
  const metadataEntries = Array.from(cacheState.metadata.entries()).sort(([a], [b]) =>
    a.localeCompare(b),
  )

  const payload: CitationsCachePayload = {
    hash: cacheState.hash,
    documents: Object.fromEntries(documentsEntries),
    metadata: Object.fromEntries(metadataEntries),
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
  for (const id of Array.from(cacheState.metadata.keys())) {
    if (!activeIds.has(id)) {
      cacheState.metadata.delete(id)
      memoryCache.delete(id)
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

    // Helper: extract a versionless arXiv id from various shapes of a CSL item
    const itemArxivId = (x: any): string | null => {
      // try eprint field (varying capitalizations)
      const e = normEprint(x?.eprint ?? x?.EPRINT ?? x?.Eprint)
      if (e) return e
      // try URL field (varying capitalizations)
      const url = x?.url ?? x?.URL
      const fromUrl = arxivIdFromUrl(url)
      if (fromUrl) return fromUrl
      return null
    }

    // Helper: parse bib file text directly to find a key by contained arXiv id/link
    const findKeyInRawFile = (content: string, eprint: string): string | null => {
      if (!content.trim()) return null
      const entryRegex = /@\w+\{\s*([^,]+)\s*,([\s\S]*?)\}\s*(?=@|$)/g
      let m: RegExpExecArray | null
      const idNoVersion = eprint.replace(/v\d+$/i, "")
      // patterns we consider a match for the same work
      const urlRe = new RegExp(String.raw`arxiv\.org\/(?:abs|pdf)\/${idNoVersion}(?:v\d+)?`, "i")
      const eprintRe = new RegExp(
        String.raw`eprint\s*=\s*[{\"]\s*${idNoVersion}(?:v\d+)?\s*[}\"]`,
        "i",
      )
      while ((m = entryRegex.exec(content)) !== null) {
        const key = m[1].trim()
        const body = m[2]
        if (urlRe.test(body) || eprintRe.test(body)) return key
      }
      return null
    }

    // candidate identity from the *id* you were asked to insert
    const eprint = normEprint(id)
    const candidate = {
      eprint, // e.g., "2405.16444"
      url: `https://arxiv.org/abs/${eprint}`, // matches future/old entries
    }

    // 1) If anything in lib matches by (eprint || url-id), reuse it
    const found = libItems.find((x: any) => {
      // First: try structured minimal check (handles URL/eprint on typical items)
      if (isSameWorkMinimal(x, candidate)) return true
      // Second: if parser used different capitalizations, compare via derived ids
      const xa = itemArxivId(x)
      return xa !== null && xa === eprint
    })
    if (found) return { key: (found as any).id, entry: "" }

    // 1.5) If not found via structured data, scan raw file for a matching entry by link/eprint
    const rawKey = fileContent ? findKeyInRawFile(fileContent, eprint!) : null
    if (rawKey) return { key: rawKey, entry: "" }

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

        const nextHash = computeDocumentsHash(cacheState.documents)
        if (nextHash !== cacheState.hash) {
          cacheState.hash = nextHash
          cacheState.dirty = true
        }

        const previousDocIds = new Set<string>(
          previousLinksSanitized
            .map((link) => extractArxivId(link))
            .filter((id): id is string => Boolean(id))
            .map((id) => normalizeArxivId(id)),
        )

        const idsToFetch = new Set<string>()
        if (docChanged) {
          for (const id of docIds) {
            if (!previousDocIds.has(id) && !cacheState.metadata.has(id)) {
              idsToFetch.add(id)
            }
          }
        }
        for (const id of docIds) {
          if (!cacheState.metadata.has(id)) {
            idsToFetch.add(id)
          }
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

        const refetched = new Set<string>()
        for (const { node, index, parent, id } of arxivNodes) {
          let entry = memoryCache.get(id) ?? cacheState.metadata.get(id)

          let needsFetch = idsToFetch.has(id) && !refetched.has(id)
          if (!entry) {
            needsFetch = true
          }

          let meta: ArxivMeta | undefined
          if (needsFetch) {
            refetched.add(id)
            meta = await fetchArxivMetadata(id)
          }

          const { key: bibkey } = await ensureBibEntry(bibliography, id)

          if (!entry || needsFetch) {
            const title = meta ? meta.title : (entry?.title ?? `arXiv:${id}`)
            entry = { title, bibkey }
            memoryCache.set(id, entry)
            cacheState.metadata.set(id, entry)
            cacheState.dirty = true
          } else if (entry.bibkey !== bibkey) {
            entry = { ...entry, bibkey }
            memoryCache.set(id, entry)
            cacheState.metadata.set(id, entry)
            cacheState.dirty = true
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
