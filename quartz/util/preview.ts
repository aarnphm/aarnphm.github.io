import { isRecord, readNumber, readString } from './type-guards'

export interface PreviewTocEntry {
  text: string
  href: string
  depth: number
}

export function normalizeTocDepth(value: number): number {
  if (!Number.isInteger(value)) return 1
  return Math.min(Math.max(value, 1), 6)
}

export function readPreviewTocEntries(value: unknown): PreviewTocEntry[] {
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

export function truncateText(value: string, limit: number): string {
  if (value.length <= limit) return value
  const slice = value.slice(0, limit + 1)
  const boundary = slice.lastIndexOf(' ')
  return `${slice.slice(0, boundary > limit * 0.7 ? boundary : limit).trimEnd()}...`
}
