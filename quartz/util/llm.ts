import type { FullSlug } from './path'

export function isWatchMarkdownSlug(slug: FullSlug | undefined): boolean {
  return slug === 'triathlon' || slug === 'are.na'
}
