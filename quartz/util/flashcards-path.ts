import { FullSlug, joinSegments } from './path'

const flashcardPathRe = /\.(flashcards|fc)(\.md)?$/i

export function isFlashcardPath(fp: string): boolean {
  return flashcardPathRe.test(fp)
}

export function sourceSlugForDeck(slug: string): string {
  return slug.replace(flashcardPathRe, '')
}

export function deckPathsForSource(sourcePath: string): string[] {
  const base = sourcePath.replace(/\.[^./]+$/, '')
  return [`${base}.fc`, `${base}.flashcards`, `${base}.fc.md`, `${base}.flashcards.md`]
}

export function flashcardsSlug(sourceSlug: string): FullSlug {
  return joinSegments(sourceSlug, 'flashcards') as FullSlug
}
