import fs from "node:fs"
import { QuartzEmitterPlugin } from "../../types/plugin"
import { joinSegments, QUARTZ } from "../../util/path"
import { extractArxivId } from "../transformers/links"
import { Cite } from "@citation-js/core"
import "@citation-js/plugin-bibtex"
import "@citation-js/plugin-doi"
import {
  arraysEqual,
  buildCachePayload,
  cacheState,
  CitationsCachePayload,
  hydrateCache,
  makeBibKey,
  normalizeArxivId,
  pruneMetadata,
  sanitizeLinks,
} from "../stores/citations"
import { write } from "./helpers"

interface Options {
  bibliography: string
}

const citationsCacheFile = joinSegments(QUARTZ, ".quartz-cache", "citations.json")

function readCachePayload(): CitationsCachePayload {
  if (!fs.existsSync(citationsCacheFile)) {
    return { papers: {}, documents: {} }
  }

  const raw = fs.readFileSync(citationsCacheFile, "utf8")
  return JSON.parse(raw) as CitationsCachePayload
}

hydrateCache(readCachePayload())

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

function documentsEqual(a: Map<string, string[]>, b: Map<string, string[]>): boolean {
  if (a.size !== b.size) return false
  for (const [key, value] of a) {
    const other = b.get(key)
    if (!other) return false
    if (!arraysEqual(value, other)) return false
  }
  return true
}

function extractArxivIdFromEntry(entry: any): string | null {
  const eprint = entry?.eprint ?? entry?.EPRINT ?? entry?.Eprint
  if (eprint) return normalizeArxivId(String(eprint))

  const url = entry?.url ?? entry?.URL
  if (url) {
    const match = String(url).match(/arxiv\.org\/(?:abs|pdf)\/([0-9]+\.[0-9]+)(?:v\d+)?/i)
    if (match) {
      return normalizeArxivId(match[1])
    }
  }

  return null
}

async function fetchBibtex(id: string): Promise<string> {
  const norm = normalizeArxivId(id)
  const url = `https://arxiv.org/bibtex/${norm}`

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

  if (!bibtex.trim().startsWith("@")) {
    throw new Error(`Invalid BibTeX response for ${norm}`)
  }

  return bibtex.trim()
}

async function ensureBibEntry(bibPath: string, id: string, desiredKey: string): Promise<string> {
  const targetArxivId = normalizeArxivId(id)
  const fileContent = fs.existsSync(bibPath) ? fs.readFileSync(bibPath, "utf8") : ""
  const libItems = fileContent.trim()
    ? (new Cite(fileContent, { generateGraph: false }).data as any[])
    : []

  const existingEntry = libItems.find((entry: any) => {
    const entryArxivId = extractArxivIdFromEntry(entry)
    return entryArxivId === targetArxivId
  })

  if (existingEntry) {
    return existingEntry.id
  }

  const bibtex = await fetchBibtex(targetArxivId)
  const newCite = new Cite(bibtex, { generateGraph: false })
  let newEntry = newCite.data[0]
  const key = desiredKey || newEntry.id

  if (libItems.some((x: any) => x.id === key)) {
    const suffix = targetArxivId.replace(/\./g, "")
    newEntry = { ...newEntry, id: `${key}-${suffix}` }
  } else {
    newEntry = { ...newEntry, id: key }
  }

  const prefix = fileContent.trim().length ? "\n\n" : ""
  const rendered = new Cite(newEntry).format("bibtex")
  fs.appendFileSync(bibPath, `${prefix}${rendered}\n`)

  return newEntry.id
}

export async function ensureBibEntries(ids: Iterable<string>, bibliography: string) {
  const VERIFICATION_TTL = 24 * 60 * 60 * 1000
  const now = Date.now()

  for (const id of ids) {
    const cachedEntry = cacheState.papers.get(id)
    if (!cachedEntry) continue
    const needsVerification =
      !cachedEntry.inBibFile || now - cachedEntry.lastVerified > VERIFICATION_TTL
    if (!needsVerification) continue

    const bibkey = await ensureBibEntry(bibliography, id, cachedEntry.bibkey ?? makeBibKey(id))
    cacheState.papers.set(id, {
      ...cachedEntry,
      bibkey,
      lastVerified: now,
      inBibFile: true,
    })
    cacheState.dirty = true
  }
}

export const Bibliography: QuartzEmitterPlugin<Partial<Options>> = (opts) => {
  const bibliography = opts?.bibliography ?? "content/References.bib"
  return {
    name: "Bibliography",
    async *emit(ctx, content) {
      const documents = new Map<string, string[]>()
      for (const [, file] of content) {
        const ids = file.data.citations?.arxivIds
        const fileKey = file.data.slug!
        if (!fileKey || !ids || ids.length === 0) continue
        const links = sanitizeLinks(
          ids.map((id) => `https://arxiv.org/abs/${normalizeArxivId(id)}`),
        )
        documents.set(fileKey, links)
      }

      if (!documentsEqual(cacheState.documents, documents)) {
        cacheState.documents.clear()
        for (const [doc, links] of documents) {
          cacheState.documents.set(doc, links)
        }
        cacheState.dirty = true
      }

      pruneMetadata()

      const activeIds = new Set<string>()
      for (const links of cacheState.documents.values()) {
        for (const link of links) {
          const id = extractArxivId(link)
          if (id) activeIds.add(normalizeArxivId(id))
        }
      }

      await ensureBibEntries(activeIds, bibliography)

      if (cacheState.dirty) {
        const cacheCtx = { ...ctx, argv: { ...ctx.argv, output: process.env.PWD ?? process.cwd() } }
        yield write({
          ctx: cacheCtx,
          slug: `${QUARTZ}/.quartz-cache/citations`,
          ext: ".json",
          content: JSON.stringify(buildCachePayload(), null, 2),
        })
        cacheState.dirty = false
      }
    },
  }
}
