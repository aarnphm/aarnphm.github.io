import { decodeStackedNoteHash, hashStackedNoteSlug } from './stacked-notes'

export interface NoteStack {
  readonly slugs: readonly string[]
  readonly valid: boolean
}

/** The notes embed stores the stack URL after the wikilink's fragment separator. */
export function parseNoteStack(raw: string): NoteStack | undefined {
  const separator = raw.indexOf('?')
  const hash = separator < 0 ? raw : raw.slice(0, separator)
  const parameters = new URLSearchParams(separator < 0 ? '' : raw.slice(separator + 1))
  const root = decodeStackedNoteHash(hash)
  if (!root && !parameters.has('stackedNotes')) return undefined
  const members = parameters.getAll('stackedNotes').map(decodeStackedNoteHash)
  const valid =
    root !== null &&
    members.every(slug => slug !== null) &&
    [...parameters.keys()].every(key => key === 'stackedNotes')
  return {
    slugs: valid ? [...new Set([root, ...members].filter(slug => slug !== null))] : [],
    valid,
  }
}

export function noteStackQuery(slugs: readonly string[]): string {
  const parameters = new URLSearchParams()
  for (const slug of slugs.slice(1)) parameters.append('stackedNotes', hashStackedNoteSlug(slug))
  return parameters.size ? `?${parameters}` : ''
}
