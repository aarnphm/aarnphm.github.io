import { Cite } from '@citation-js/core'
import { XMLParser } from 'fast-xml-parser'
import fs from 'node:fs'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { joinSegments, QUARTZ } from '../../util/path'
import {
  ArxivMeta,
  ARXIV_HEADERS,
  arxivRateLimiter,
  arraysEqual,
  buildCachePayload,
  cacheState,
  CitationsCachePayload,
  fetchWithRetry,
  hydrateCache,
  makeBibKey,
  normalizeArxivId,
  pruneMetadata,
  RETRY_COOLDOWN,
  sanitizeLinks,
  synthesizeBibtex,
  UNVERIFIED_TTL,
  VERIFIED_TTL,
} from '../stores/citations'
import '@citation-js/plugin-bibtex'
import '@citation-js/plugin-doi'
import { extractArxivId } from '../transformers/links'
import { write } from './helpers'

interface Options {
  bibliography: string
}

const citationsCacheFile = joinSegments(QUARTZ, '.quartz-cache', 'citations.json')

function readCachePayload(): CitationsCachePayload {
  if (!fs.existsSync(citationsCacheFile)) {
    return { papers: {}, documents: {} }
  }

  const raw = fs.readFileSync(citationsCacheFile, 'utf8')
  return JSON.parse(raw) as CitationsCachePayload
}

hydrateCache(readCachePayload())

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

const ARXIV_BATCH_SIZE = 50

function parseArxivEntry(entry: any): ArxivMeta | null {
  const raw = typeof entry?.id === 'string' ? entry.id : ''
  if (!raw) return null
  const tail = raw.includes('/') ? (raw.split('/').pop() ?? raw) : raw
  const id = normalizeArxivId(tail)
  if (!id) return null

  const titleRaw = typeof entry?.title === 'string' ? entry.title : ''
  const published = typeof entry?.published === 'string' ? entry.published : ''
  const authors = Array.isArray(entry?.author)
    ? entry.author.map((a: any) => a?.name).filter(Boolean)
    : entry?.author?.name
      ? [entry.author.name]
      : []
  const category = entry?.['arxiv:primary_category']?.['@_term'] ?? ''

  return {
    id,
    title: titleRaw.trim().replace(/\s+/g, ' '),
    authors,
    year: published.slice(0, 4),
    category,
    url: `https://arxiv.org/abs/${id}`,
  }
}

async function fetchMetadataBatch(ids: string[]): Promise<Map<string, ArxivMeta>> {
  const result = new Map<string, ArxivMeta>()
  if (ids.length === 0) return result

  const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: '@_' })

  for (let i = 0; i < ids.length; i += ARXIV_BATCH_SIZE) {
    const batch = ids.slice(i, i + ARXIV_BATCH_SIZE)
    try {
      const url = `https://export.arxiv.org/api/query?id_list=${batch.join(',')}`
      const res = await fetchWithRetry(url, { headers: ARXIV_HEADERS }, arxivRateLimiter)
      if (!res) {
        console.warn(`[citations] emitter metadata fetch failed for batch of ${batch.length}`)
        continue
      }
      const xml = await res.text()
      const parsed = parser.parse(xml) as any
      const entries = Array.isArray(parsed?.feed?.entry)
        ? parsed.feed.entry
        : parsed?.feed?.entry
          ? [parsed.feed.entry]
          : []
      for (const entry of entries) {
        const meta = parseArxivEntry(entry)
        if (meta) result.set(meta.id, meta)
      }
    } catch (e) {
      console.warn(`[citations] emitter batch error: ${e instanceof Error ? e.message : e}`)
    }
  }
  return result
}

async function fetchBibtexFallback(id: string): Promise<string | null> {
  const norm = normalizeArxivId(id)
  const url = `https://arxiv.org/bibtex/${norm}`
  const res = await fetchWithRetry(url, { headers: ARXIV_HEADERS }, arxivRateLimiter)
  if (!res) return null
  const bibtex = await res.text()
  if (!bibtex.trim().startsWith('@')) return null
  return bibtex.trim()
}

export async function ensureBibEntries(ids: Iterable<string>, bibliography: string) {
  const now = Date.now()
  const fileContent = fs.existsSync(bibliography) ? fs.readFileSync(bibliography, 'utf8') : ''
  const libItems = fileContent.trim()
    ? (new Cite(fileContent, { generateGraph: false }).data as any[])
    : []

  const existingArxivIds = new Set<string>()
  const existingKeys = new Set<string>()
  for (const item of libItems) {
    existingKeys.add(item.id)
    const arxivId = extractArxivIdFromEntry(item)
    if (arxivId) existingArxivIds.add(arxivId)
  }

  const needsMetadata: string[] = []
  for (const id of ids) {
    const entry = cacheState.papers.get(id)
    if (!entry) continue
    if (existingArxivIds.has(id)) continue
    if (entry.authors && entry.year) continue
    if (entry.failedAt && now - entry.failedAt < RETRY_COOLDOWN) continue
    needsMetadata.push(id)
  }

  if (needsMetadata.length > 0) {
    console.log(
      `[citations] fetching metadata for ${needsMetadata.length} papers missing from transformer cache`,
    )
    const metas = await fetchMetadataBatch(needsMetadata)
    for (const id of needsMetadata) {
      const meta = metas.get(id)
      if (!meta) continue
      const existing = cacheState.papers.get(id)!
      cacheState.papers.set(id, {
        ...existing,
        title: meta.title,
        authors: meta.authors,
        year: meta.year,
        category: meta.category,
        failedAt: undefined,
      })
      cacheState.dirty = true
    }
  }

  const newEntries: string[] = []

  for (const id of ids) {
    const cachedEntry = cacheState.papers.get(id)
    if (!cachedEntry) continue

    if (existingArxivIds.has(id)) {
      if (!cachedEntry.inBibFile || !cachedEntry.bibtex) {
        const existingItem = libItems.find(item => extractArxivIdFromEntry(item) === id)
        cacheState.papers.set(id, {
          ...cachedEntry,
          bibkey: existingItem?.id ?? cachedEntry.bibkey,
          inBibFile: true,
          lastVerified: now,
        })
        cacheState.dirty = true
      }
      continue
    }

    const ttl = cachedEntry.inBibFile && cachedEntry.bibtex ? VERIFIED_TTL : UNVERIFIED_TTL
    const needsVerification = !cachedEntry.inBibFile || now - cachedEntry.lastVerified > ttl
    if (!needsVerification) continue

    if (cachedEntry.failedAt && now - cachedEntry.failedAt < RETRY_COOLDOWN) continue

    try {
      let bibtex = cachedEntry.bibtex
      if (!bibtex && cachedEntry.authors && cachedEntry.year) {
        bibtex = synthesizeBibtex(id, cachedEntry)
      }
      if (!bibtex) {
        bibtex = (await fetchBibtexFallback(id)) ?? undefined
      }
      if (!bibtex) {
        console.warn(`[citations] could not generate bibtex for ${id}`)
        cacheState.papers.set(id, { ...cachedEntry, failedAt: now })
        cacheState.dirty = true
        continue
      }

      let key = cachedEntry.bibkey ?? makeBibKey(id)
      if (existingKeys.has(key)) {
        key = `${key}-${normalizeArxivId(id).replace(/\./g, '')}`
      }

      const keyedBibtex = bibtex.replace(/^@article\{[^,]*,/, `@article{${key},`)
      newEntries.push(keyedBibtex)
      existingKeys.add(key)
      existingArxivIds.add(id)

      cacheState.papers.set(id, {
        ...cachedEntry,
        bibkey: key,
        bibtex: keyedBibtex,
        lastVerified: now,
        inBibFile: true,
        failedAt: undefined,
      })
      cacheState.dirty = true
    } catch (e) {
      console.warn(`[citations] error processing ${id}: ${e instanceof Error ? e.message : e}`)
      cacheState.papers.set(id, { ...cachedEntry, failedAt: now })
      cacheState.dirty = true
    }
  }

  if (newEntries.length > 0) {
    const prefix = fileContent.trim().length ? '\n\n' : ''
    fs.appendFileSync(bibliography, `${prefix}${newEntries.join('\n\n')}\n`)
    console.log(`[citations] appended ${newEntries.length} new entries to ${bibliography}`)
  }
}

export const Bibliography: QuartzEmitterPlugin<Partial<Options>> = opts => {
  const bibliography = opts?.bibliography ?? 'content/References.bib'
  return {
    name: 'Bibliography',
    async *emit(ctx, content) {
      const documents = new Map<string, string[]>()
      for (const [, file] of content) {
        const ids = file.data.citations?.arxivIds
        const fileKey = file.data.slug!
        if (!fileKey || !ids || ids.length === 0) continue
        const links = sanitizeLinks(ids.map(id => `https://arxiv.org/abs/${normalizeArxivId(id)}`))
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
          ext: '.json',
          content: JSON.stringify(buildCachePayload(), null, 2),
        })
        cacheState.dirty = false
      }
    },
  }
}
