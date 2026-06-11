import { readPreviewTocEntries, type PreviewTocEntry } from './preview'
import { isRecord, readString } from './type-guards'
import { hostnameMatches } from './url'

export interface SepTarget {
  entry: string
  archive?: string
}

export interface SepPreview {
  entry: string
  title: string
  pageUrl: string
  extract: string
  authors?: string[]
  pubInfo?: string
  toc?: PreviewTocEntry[]
}

const sepEntryId = /^[a-z0-9][a-z0-9-]*$/
const sepArchiveId = /^(?:spr|sum|fall|win)\d{4}$/

function decodePathSegment(value: string): string {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

function targetFromParts(entry: string | undefined, archive?: string): SepTarget | undefined {
  if (!entry || !sepEntryId.test(entry)) return undefined
  if (archive !== undefined && !sepArchiveId.test(archive)) return undefined
  return { entry, ...(archive ? { archive } : {}) }
}

export function parseSepTarget(href: string): SepTarget | undefined {
  let url: URL
  try {
    url = new URL(href)
  } catch {
    return undefined
  }

  if (!hostnameMatches(url, 'plato.stanford.edu')) return undefined

  const parts = url.pathname
    .split('/')
    .filter(part => part.length > 0)
    .map(decodePathSegment)

  if (parts[0] === 'entries') return targetFromParts(parts[1])
  if (parts[0] === 'archives' && parts[2] === 'entries') return targetFromParts(parts[3], parts[1])
  return undefined
}

export function sepEntryUrl({ entry, archive }: SepTarget): string {
  const prefix = archive ? `/archives/${archive}` : ''
  return `https://plato.stanford.edu${prefix}/entries/${entry}/`
}

export function sepPreviewApiUrl(target: SepTarget, baseUrl: string): URL {
  const url = new URL('/api/sep', baseUrl)
  url.searchParams.set('entry', target.entry)
  if (target.archive) url.searchParams.set('archive', target.archive)
  return url
}

export function sepTargetFromSearchParams(params: URLSearchParams): SepTarget | null {
  const target = targetFromParts(
    params.get('entry') ?? undefined,
    params.get('archive') ?? undefined,
  )
  return target ?? null
}

export function readSepPreview(value: unknown): SepPreview | undefined {
  if (!isRecord(value)) return undefined

  const entry = readString(value, 'entry')
  const title = readString(value, 'title')?.trim()
  const pageUrl = readString(value, 'pageUrl')
  const extract = readString(value, 'extract')?.trim()
  if (!entry || !sepEntryId.test(entry) || !title || !pageUrl || !extract) return undefined

  const rawAuthors = Array.isArray(value.authors) ? value.authors : []
  const authors = rawAuthors
    .filter((item): item is string => typeof item === 'string')
    .map(item => item.trim())
    .filter(item => item.length > 0)
  const pubInfo = readString(value, 'pubInfo')?.trim()
  const toc = readPreviewTocEntries(value.toc)

  return {
    entry,
    title,
    pageUrl,
    extract,
    ...(authors.length > 0 ? { authors } : {}),
    ...(pubInfo ? { pubInfo } : {}),
    ...(toc.length > 0 ? { toc } : {}),
  }
}
