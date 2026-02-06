import { extractArxivId } from '../transformers/links'

export interface CachedCitationEntry {
  title: string
  bibkey: string
  lastVerified: number
  inBibFile: boolean
}

export interface CitationsCachePayload {
  papers: Record<string, CachedCitationEntry>
  documents: Record<string, string[]>
}

export interface CacheState {
  documents: Map<string, string[]>
  papers: Map<string, CachedCitationEntry>
  dirty: boolean
}

export const cacheState: CacheState = { documents: new Map(), papers: new Map(), dirty: false }

export function normalizeArxivId(id: string): string {
  return id.replace(/^arxiv:/i, '').replace(/v\d+$/i, '')
}

export function makeBibKey(id: string): string {
  return `arxiv-${normalizeArxivId(id).replace(/\./g, '')}`
}

export function sanitizeLinks(links: string[]): string[] {
  return Array.from(new Set(links)).sort((a, b) => a.localeCompare(b))
}

export function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}

export function pruneMetadata() {
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

export function hydrateCache(payload: CitationsCachePayload) {
  cacheState.documents.clear()
  for (const [doc, links] of Object.entries(payload.documents)) {
    cacheState.documents.set(doc, sanitizeLinks(links))
  }

  cacheState.papers.clear()
  for (const [id, value] of Object.entries(payload.papers)) {
    cacheState.papers.set(normalizeArxivId(id), value)
  }

  cacheState.dirty = false
}

export function buildCachePayload(): CitationsCachePayload {
  const documentsEntries = Array.from(cacheState.documents.entries())
    .map(([doc, links]) => [doc, sanitizeLinks(links)] as const)
    .sort(([a], [b]) => a.localeCompare(b))
  const papersEntries = Array.from(cacheState.papers.entries()).sort(([a], [b]) =>
    a.localeCompare(b),
  )

  return {
    papers: Object.fromEntries(papersEntries),
    documents: Object.fromEntries(documentsEntries),
  }
}
