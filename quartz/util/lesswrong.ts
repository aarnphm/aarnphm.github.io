import { isRecord, readNumber, readString } from './type-guards'
import { hostnameMatches } from './url'

export interface LessWrongTarget {
  postId: string
  slug?: string
}

export interface LessWrongPreviewTocEntry {
  text: string
  href: string
  depth: number
}

export interface LessWrongPreview {
  postId: string
  title: string
  pageUrl: string
  extract: string
  author?: string
  postedAt?: string
  score?: number
  commentCount?: number
  readTimeMinutes?: number
  tags?: string[]
  toc?: LessWrongPreviewTocEntry[]
}

const lessWrongPostId = /^[A-Za-z0-9]+$/

function decodePathSegment(value: string): string {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

function normalizeSlug(value: string | undefined): string | undefined {
  if (!value) return undefined
  const slug = decodePathSegment(value)
    .trim()
    .replace(/^\/+|\/+$/g, '')
  return slug.length > 0 ? slug : undefined
}

function parsePostPath(parts: string[]): LessWrongTarget | undefined {
  if (parts[0] !== 'posts') return undefined
  const postId = parts[1]
  if (!postId || !lessWrongPostId.test(postId)) return undefined
  return { postId, ...withOptionalSlug(parts[2]) }
}

function parseSequencePostPath(parts: string[]): LessWrongTarget | undefined {
  if (parts[0] !== 's' || parts[2] !== 'p') return undefined
  const postId = parts[3]
  if (!postId || !lessWrongPostId.test(postId)) return undefined
  return { postId, ...withOptionalSlug(parts[4]) }
}

function withOptionalSlug(value: string | undefined): Pick<LessWrongTarget, 'slug'> {
  const slug = normalizeSlug(value)
  return slug ? { slug } : {}
}

export function parseLessWrongTarget(href: string): LessWrongTarget | undefined {
  let url: URL
  try {
    url = new URL(href)
  } catch {
    return undefined
  }

  if (!hostnameMatches(url, 'lesswrong.com')) return undefined

  const parts = url.pathname
    .split('/')
    .filter(part => part.length > 0)
    .map(decodePathSegment)

  return parsePostPath(parts) ?? parseSequencePostPath(parts)
}

export function lessWrongPostUrl({ postId, slug }: LessWrongTarget): string {
  const path = slug ? `/posts/${postId}/${encodeURIComponent(slug)}` : `/posts/${postId}`
  return `https://www.lesswrong.com${path}`
}

export function greaterWrongPostUrl({ postId, slug }: LessWrongTarget): string {
  const path = slug ? `/posts/${postId}/${encodeURIComponent(slug)}` : `/posts/${postId}`
  return `https://www.greaterwrong.com${path}`
}

export function lessWrongPreviewApiUrl(target: LessWrongTarget, baseUrl: string): URL {
  const url = new URL('/api/lesswrong', baseUrl)
  url.searchParams.set('postId', target.postId)
  if (target.slug) url.searchParams.set('slug', target.slug)
  return url
}

export function lessWrongTargetFromSearchParams(params: URLSearchParams): LessWrongTarget | null {
  const postId = params.get('postId')
  if (!postId || !lessWrongPostId.test(postId)) return null
  return { postId, ...withOptionalSlug(params.get('slug') ?? undefined) }
}

export function readLessWrongPreview(value: unknown): LessWrongPreview | undefined {
  if (!isRecord(value)) return undefined

  const postId = readString(value, 'postId')
  const title = readString(value, 'title')?.trim()
  const pageUrl = readString(value, 'pageUrl')
  const extract = readString(value, 'extract')?.trim()
  if (!postId || !lessWrongPostId.test(postId) || !title || !pageUrl || !extract) return undefined

  const author = readString(value, 'author')?.trim()
  const postedAt = readString(value, 'postedAt')?.trim()
  const score = readNumber(value, 'score')
  const commentCount = readNumber(value, 'commentCount')
  const readTimeMinutes = readNumber(value, 'readTimeMinutes')
  const rawTags = Array.isArray(value.tags) ? value.tags : []
  const tags = rawTags
    .filter((item): item is string => typeof item === 'string')
    .map(item => item.trim())
    .filter(item => item.length > 0)
  const toc = readLessWrongTocEntries(value.toc)

  return {
    postId,
    title,
    pageUrl,
    extract,
    ...(author ? { author } : {}),
    ...(postedAt ? { postedAt } : {}),
    ...(score !== undefined ? { score } : {}),
    ...(commentCount !== undefined ? { commentCount } : {}),
    ...(readTimeMinutes !== undefined ? { readTimeMinutes } : {}),
    ...(tags.length > 0 ? { tags } : {}),
    ...(toc.length > 0 ? { toc } : {}),
  }
}

function readLessWrongTocEntries(value: unknown): LessWrongPreviewTocEntry[] {
  if (!Array.isArray(value)) return []

  return value.flatMap(item => {
    if (!isRecord(item)) return []
    const text = readString(item, 'text')?.trim()
    const href = readString(item, 'href')?.trim()
    const depth = readNumber(item, 'depth')
    if (!text || !href || !href.startsWith('#') || depth === undefined) return []
    return [{ text, href, depth: normalizeTocDepth(depth) }]
  })
}

function normalizeTocDepth(value: number): number {
  if (!Number.isInteger(value)) return 1
  return Math.min(Math.max(value, 1), 6)
}
