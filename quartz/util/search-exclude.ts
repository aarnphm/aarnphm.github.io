export const SEARCH_EXCLUDE_PREFIXES = ['hinterland', 'embeddings', 'description.json'] as const

function contentRelativePath(value: string): string {
  const normalized = value.replace(/\\/g, '/').replace(/^\.?\//, '')
  return normalized.startsWith('content/') ? normalized.slice('content/'.length) : normalized
}

export function isSearchExcluded(slugOrPath: string): boolean {
  const value = contentRelativePath(slugOrPath)
  return SEARCH_EXCLUDE_PREFIXES.some(prefix => value === prefix || value.startsWith(`${prefix}/`))
}
