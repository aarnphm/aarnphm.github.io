import {
  PDF_PASSAGE_TEXT_LIMIT,
  type PdfMark,
  type PdfMarkInput,
  type PdfMarkKind,
  type PdfQuote,
} from '../../util/pdf-marks'
import { isRecord } from '../../util/type-guards'

export interface Scratchpad {
  body: string
  revision: number
}

export interface LectureNote {
  id: string
  kind: PdfMarkKind
  quote: PdfQuote | null
  quoteText?: string
  heading: string | null
  body: string
  revision: number
  createdAt: number
  deletedAt: number | null
}

export class DraftConflict extends Error {
  constructor() {
    super(
      'A newer version was saved in another tab. Your draft is preserved here. Export it before loading the saved version.',
    )
  }
}

export const scratchpadLimit = 100_000
export const noteLimit = 20_000

export function storagePrefix(slug: string): string {
  return `course-workspace:v1:${encodeURIComponent(slug)}:`
}

function isRevision(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

function parseScratchpad(value: unknown): Scratchpad | null {
  if (!isRecord(value) || typeof value.body !== 'string' || !isRevision(value.revision)) return null
  return { body: value.body, revision: value.revision }
}

function parseNote(value: unknown): LectureNote | null {
  if (!isRecord(value)) return null
  if (
    typeof value.id !== 'string' ||
    !/^[\w-]+$/.test(value.id) ||
    (value.kind !== 'mark' && value.kind !== 'question' && value.kind !== 'contra') ||
    typeof value.body !== 'string' ||
    (value.quoteText !== undefined && typeof value.quoteText !== 'string') ||
    !isRevision(value.revision) ||
    typeof value.createdAt !== 'number' ||
    !Number.isFinite(value.createdAt) ||
    (value.deletedAt !== null && typeof value.deletedAt !== 'number') ||
    (value.heading !== null && typeof value.heading !== 'string')
  )
    return null
  let quote: PdfQuote | null = null
  if (value.quote !== null) {
    if (
      !isRecord(value.quote) ||
      typeof value.quote.exact !== 'string' ||
      !value.quote.exact.trim() ||
      typeof value.quote.prefix !== 'string' ||
      typeof value.quote.suffix !== 'string'
    )
      return null
    quote = { exact: value.quote.exact, prefix: value.quote.prefix, suffix: value.quote.suffix }
  }
  return {
    id: value.id,
    kind: value.kind,
    quote,
    quoteText: typeof value.quoteText === 'string' ? value.quoteText : quote?.exact,
    heading: value.heading,
    body: value.body,
    revision: value.revision,
    createdAt: value.createdAt,
    deletedAt: value.deletedAt,
  }
}

function readStored<T>(key: string, parse: (value: unknown) => T | null): T | null {
  const stored = localStorage.getItem(key)
  if (stored === null) return null
  const value: unknown = JSON.parse(stored)
  const result = parse(value)
  if (!result)
    throw new Error('A saved draft could not be read. The stored copy has been preserved.')
  return result
}

export function loadScratchpad(slug: string): Scratchpad {
  return (
    readStored(`${storagePrefix(slug)}scratchpad`, parseScratchpad) ?? { body: '', revision: 0 }
  )
}

export function saveScratchpad(slug: string, draft: Scratchpad): Scratchpad {
  if (draft.body.length > scratchpadLimit) throw new Error('The scratchpad is too long to save.')
  const key = `${storagePrefix(slug)}scratchpad`
  const current = readStored(key, parseScratchpad)
  if ((current?.revision ?? 0) !== draft.revision) throw new DraftConflict()
  const saved = { ...draft, revision: draft.revision + 1 }
  localStorage.setItem(key, JSON.stringify(saved))
  return saved
}

export function loadNotes(slug: string): LectureNote[] {
  const prefix = `${storagePrefix(slug)}note:`
  const notes: LectureNote[] = []
  for (let index = 0; index < localStorage.length; index++) {
    const key = localStorage.key(index)
    if (!key?.startsWith(prefix)) continue
    const note = readStored(key, parseNote)
    if (note && key === `${prefix}${note.id}`) notes.push(note)
    else throw new Error('A saved note could not be read. The stored copy has been preserved.')
  }
  return notes.sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id))
}

export function saveNote(slug: string, draft: LectureNote): LectureNote {
  if (draft.body.length > noteLimit) throw new Error('The note is too long to save.')
  const key = `${storagePrefix(slug)}note:${draft.id}`
  const current = readStored(key, parseNote)
  if ((current?.revision ?? 0) !== draft.revision) throw new DraftConflict()
  const saved = { ...draft, revision: draft.revision + 1 }
  localStorage.setItem(key, JSON.stringify(saved))
  return saved
}

export function removeNote(slug: string, id: string): void {
  localStorage.removeItem(`${storagePrefix(slug)}note:${id}`)
}

/** A passage mark from the marks API as a workspace note; marks on PDF pages are skipped. */
export function noteFromMark(mark: PdfMark): LectureNote | null {
  if (mark.target.type !== 'passage') return null
  return {
    id: mark.id,
    kind: mark.kind,
    quote: mark.target.quote,
    quoteText: mark.target.text ?? mark.target.quote?.exact,
    heading: mark.target.heading,
    body: mark.body,
    revision: mark.revision,
    createdAt: mark.createdAt,
    deletedAt: null,
  }
}

/** The PUT body for a note: `src` is the unit slug, notes stay private. */
export function markInputOf(src: string, doc: string, note: LectureNote): PdfMarkInput {
  const text = note.quoteText?.slice(0, PDF_PASSAGE_TEXT_LIMIT)
  return {
    src,
    doc,
    revision: note.revision,
    kind: note.kind,
    target: {
      type: 'passage',
      page: 0,
      quote: note.quote,
      heading: note.heading,
      ...(note.quote && text ? { text } : {}),
    },
    body: note.body,
    visibility: 'private',
  }
}

export function markdownExport(
  title: string,
  slug: string,
  pdf: string | undefined,
  scratchpad: Scratchpad,
  notes: LectureNote[],
): string {
  const lines = [`# ${title}`, '', `Lecture: [[${slug}]]`]
  if (pdf) lines.push(`Source: [PDF](/${encodeURI(pdf)})`)
  lines.push('', '## scratchpad', '', scratchpad.body, '', '## marks and notes', '')
  for (const note of notes.filter(note => note.deletedAt === null)) {
    const label = note.kind === 'contra' ? 'disagree' : note.kind
    const target = note.heading ? `${slug}#${note.heading}` : slug
    lines.push(`### ${label}`, '', `[[${target}|lecture]] · note ${note.id}`, '')
    if (note.quote)
      lines.push(
        (note.quoteText ?? note.quote.exact)
          .split('\n')
          .map(line => `> ${line}`)
          .join('\n'),
        '',
      )
    lines.push(note.body, '')
  }
  return `${lines.join('\n')}\n`
}
