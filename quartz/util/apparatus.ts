import type { QuartzPluginData } from '../plugins/vfile'
import { isSlidePage } from './section-icon'

/**
 * Whether a page carries the margin apparatus of technical notes (renderPage/technical-apparatus.ts):
 * notes with `layout: technical` in their frontmatter (frontmatter.ts reads it as `pageLayout`).
 * Protected pages, slides, flashcard decks and poems keep their own form; /fr has its own apparatus
 * (renderPage/fr-margins.ts).
 */
export function usesApparatus(data: QuartzPluginData): boolean {
  const slug = data.slug
  const frontmatter = data.frontmatter
  if (!slug || slug.startsWith('fr/')) return false
  if (data.protectedPassword || frontmatter?.protected) return false
  if (data.flashcards || isSlidePage(slug, frontmatter?.slides)) return false
  if (frontmatter?.poem) return false
  return frontmatter?.pageLayout === 'technical'
}
