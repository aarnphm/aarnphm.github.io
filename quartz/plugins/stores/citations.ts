import { extractArxivId } from '../transformers/links'

export interface ArxivMeta {
  id: string
  title: string
  authors: string[]
  year: string
  category: string
  url: string
}

export interface CachedCitationEntry {
  title: string
  bibkey: string
  lastVerified: number
  inBibFile: boolean
  bibtex?: string
  authors?: string[]
  year?: string
  category?: string
  failedAt?: number
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

export const RETRY_COOLDOWN = 60 * 60 * 1000
export const VERIFIED_TTL = 7 * 24 * 60 * 60 * 1000
export const UNVERIFIED_TTL = 24 * 60 * 60 * 1000

export const ARXIV_HEADERS = { 'User-Agent': 'curl/8.7.1' }

export const cacheState: CacheState = { documents: new Map(), papers: new Map(), dirty: false }

export class AdaptiveRateLimiter {
  private lastRequest = 0
  private interval: number

  constructor(
    private readonly baseInterval = 3000,
    private readonly maxInterval = 30000,
  ) {
    this.interval = baseInterval
  }

  async wait(): Promise<void> {
    const elapsed = Date.now() - this.lastRequest
    if (elapsed < this.interval) {
      await new Promise(r => setTimeout(r, this.interval - elapsed))
    }
    this.lastRequest = Date.now()
  }

  onSuccess() {
    this.interval = Math.max(this.baseInterval, Math.floor(this.interval * 0.8))
  }

  onRateLimit() {
    this.interval = Math.min(this.maxInterval, this.interval * 2)
  }
}

export const arxivRateLimiter = new AdaptiveRateLimiter()

export async function fetchWithRetry(
  url: string,
  opts: RequestInit,
  rateLimiter: AdaptiveRateLimiter,
  maxRetries = 3,
): Promise<Response | null> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    await rateLimiter.wait()
    try {
      const res = await fetch(url, opts)
      if (res.ok) {
        rateLimiter.onSuccess()
        return res
      }
      if (res.status === 429 || res.status === 503) {
        rateLimiter.onRateLimit()
        const retryAfter = res.headers.get('Retry-After')
        if (retryAfter) {
          const seconds = parseInt(retryAfter, 10)
          if (!isNaN(seconds)) await new Promise(r => setTimeout(r, seconds * 1000))
        }
        continue
      }
      return null
    } catch {
      continue
    }
  }
  return null
}

export function synthesizeBibtex(id: string, entry: CachedCitationEntry): string {
  const authors = (entry.authors ?? []).join(' and ')
  const norm = normalizeArxivId(id)
  return [
    `@article{${entry.bibkey},`,
    `  title = {${entry.title}},`,
    `  author = {${authors}},`,
    `  year = {${entry.year ?? ''}},`,
    `  eprint = {${norm}},`,
    `  archiveprefix = {arXiv},`,
    `  primaryclass = {${entry.category ?? ''}},`,
    `  url = {https://arxiv.org/abs/${norm}}`,
    `}`,
  ].join('\n')
}

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
