import type { Data } from 'vfile'
import { canonicalTag, getAllSegmentPrefixes, legacyTagSlug } from './path'

export { canonicalTag }

export function tagMetadata(values: readonly string[]) {
  const tags: string[] = []
  const display = new Map<string, string>()
  const aliases = new Map<string, string>()
  for (const value of values) {
    const tag = canonicalTag(value)
    if (!tag) continue
    if (!tags.includes(tag)) tags.push(tag)
    if (!display.has(tag)) display.set(tag, value.trim().replace(/^#/, ''))
    const legacy = legacyTagSlug(value)
    const canonicalPrefixes = getAllSegmentPrefixes(tag)
    for (const [index, prefix] of getAllSegmentPrefixes(legacy).entries()) {
      const destination = canonicalPrefixes[index]
      if (destination && prefix !== destination) aliases.set(prefix, destination)
    }
  }
  return { tags, display: Object.fromEntries(display), aliases: Object.fromEntries(aliases) }
}

export function registerTag(data: Data, value: string): string {
  const next = tagMetadata([value])
  const tag = next.tags[0] ?? ''
  if (data.frontmatter && tag) {
    data.frontmatter.tags = [...new Set([...(data.frontmatter.tags ?? []), tag])]
    data.tagDisplay = { ...next.display, ...data.tagDisplay }
    data.tagAliases = { ...data.tagAliases, ...next.aliases }
  }
  return tag
}

export function tagMatches(candidate: string, query: string): boolean {
  const tag = canonicalTag(candidate)
  const parent = canonicalTag(query)
  return parent.length > 0 && (tag === parent || tag.startsWith(`${parent}/`))
}

export function tagLabel(tag: string, source: { tagDisplay?: Record<string, string> }): string {
  const key = canonicalTag(tag)
  return source.tagDisplay && Object.hasOwn(source.tagDisplay, key) ? source.tagDisplay[key] : tag
}

export function tagLabels(
  sources: readonly {
    tagDisplay?: Record<string, string>
    fileName?: string
    filePath?: string
    slug?: string
  }[],
): Record<string, string> {
  const labels = new Map<string, string>()
  const ordered = [...sources].sort((a, b) =>
    (a.fileName ?? a.filePath ?? a.slug ?? '').localeCompare(
      b.fileName ?? b.filePath ?? b.slug ?? '',
    ),
  )
  for (const source of ordered) {
    for (const [tag, label] of Object.entries(source.tagDisplay ?? {})) {
      const tagSegments = tag.split('/')
      const labelSegments = label.split('/')
      for (let length = 1; length <= tagSegments.length; length++) {
        const prefix = tagSegments.slice(0, length).join('/')
        if (!labels.has(prefix)) labels.set(prefix, labelSegments.slice(0, length).join('/'))
      }
    }
  }
  return Object.fromEntries(labels)
}

declare module 'vfile' {
  interface DataMap {
    tagDisplay?: Record<string, string>
    tagAliases?: Record<string, string>
  }
}

/** Preserve historical tag routes through the redirect emitter, without case-colliding files. */
export function tagRouteAliases(
  sources: readonly {
    frontmatter?: { tags?: readonly string[] }
    tagAliases?: Record<string, string>
    slug?: string
  }[],
): Map<string, string> {
  const published = new Set(
    sources.flatMap(source =>
      (source.frontmatter?.tags ?? []).map(canonicalTag).flatMap(getAllSegmentPrefixes),
    ),
  )
  published.add('index')
  const aliases = new Map<string, string>()
  for (const source of sources) {
    for (const [legacy, destination] of Object.entries(source.tagAliases ?? {}))
      if (legacy !== destination && published.has(destination)) aliases.set(legacy, destination)
    if (source.slug?.startsWith('tags/')) {
      const legacy = source.slug.slice('tags/'.length)
      const destination = canonicalTag(legacy)
      if (legacy !== destination && published.has(destination)) aliases.set(legacy, destination)
    }
  }
  return aliases
}

export function canonicalTagRoute(slug: string): string {
  return slug.startsWith('tags/') ? `tags/${canonicalTag(slug.slice('tags/'.length))}` : slug
}

export function parseTagSearchQuery(
  source: string,
): { tag: string; term: string; exact: boolean } | undefined {
  const value = source.trim()
  const explicit = /^tag:\s*#?([^\s]*)(?:\s+(.*))?$/i.exec(value)
  const match = explicit ?? /^#([^\s]*)(?:\s+(.*))?$/.exec(value)
  return match
    ? { tag: canonicalTag(match[1]), term: match[2]?.trim() ?? '', exact: Boolean(explicit) }
    : undefined
}
