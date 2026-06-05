import type { ContentDetails } from '../plugins/emitters/contentIndex'
import { FullSlug, isFullSlug } from './path'
import { isRecord, readString } from './type-guards'

export interface RecentNote {
  slug: FullSlug
  name: string
  title: string
  aliases: string[]
}

export interface RecentNoteStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
}

export const recentNotesStorageKey = 'recent-notes'
export const recentNotesLimit = 32

function fallbackRecentNote(slug: FullSlug): RecentNote {
  return { slug, name: slug, title: '', aliases: [] }
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter(item => typeof item === 'string') : []
}

function readRecentNote(value: unknown): RecentNote | null {
  if (typeof value === 'string') {
    return isFullSlug(value) ? fallbackRecentNote(value) : null
  }
  if (!isRecord(value)) return null

  const rawSlug = readString(value, 'slug')
  if (!rawSlug || !isFullSlug(rawSlug)) return null

  const name = readString(value, 'name') ?? rawSlug
  const title = readString(value, 'title') ?? ''
  return {
    slug: rawSlug,
    name: name.length > 0 ? name : rawSlug,
    title,
    aliases: stringArray(value.aliases),
  }
}

export function readRecentNotes(storage: RecentNoteStorage, key = recentNotesStorageKey) {
  try {
    const raw = storage.getItem(key)
    if (!raw) return []

    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []

    return parsed.flatMap(item => {
      const note = readRecentNote(item)
      return note ? [note] : []
    })
  } catch {
    return []
  }
}

export function writeRecentNotes(
  storage: RecentNoteStorage,
  notes: RecentNote[],
  key = recentNotesStorageKey,
) {
  try {
    storage.setItem(key, JSON.stringify(notes))
  } catch {
    return
  }
}

export function recentNoteFromContent(slug: FullSlug, content: ContentDetails): RecentNote {
  return { slug, name: content.fileName, title: content.title ?? '', aliases: content.aliases }
}

export function upsertRecentNote(notes: RecentNote[], note: RecentNote, limit = recentNotesLimit) {
  const withoutCurrent = notes.filter(item => item.slug !== note.slug)
  return [...withoutCurrent, note].slice(-limit)
}

function isSubsequence(needle: string, haystack: string) {
  let index = 0
  for (const char of haystack) {
    if (char === needle[index]) index++
    if (index === needle.length) return true
  }
  return false
}

function tokenScore(token: string, value: string) {
  if (value === token) return 80
  if (value.startsWith(token)) return 60
  if (value.split('/').some(segment => segment.startsWith(token))) return 48
  if (value.includes(token)) return 36
  if (isSubsequence(token, value)) return 8
  return 0
}

function noteScore(note: RecentNote, tokens: string[]) {
  const values = [note.name, note.title, note.slug, ...note.aliases]
    .filter(value => value.length > 0)
    .map(value => value.toLowerCase())

  let score = 0
  for (const token of tokens) {
    const best = Math.max(...values.map(value => tokenScore(token, value)))
    if (best === 0) return 0
    score += best
  }
  return score
}

export function queryRecentNotes(notes: RecentNote[], query: string, limit: number) {
  const tokens = query
    .toLowerCase()
    .split(/\s+/)
    .filter(token => token.length > 0)

  if (tokens.length === 0) return notes.slice(-limit).reverse()

  return notes
    .map((note, index) => ({ note, index, score: noteScore(note, tokens) }))
    .filter(result => result.score > 0)
    .sort((a, b) => b.score - a.score || b.index - a.index)
    .slice(0, limit)
    .map(result => result.note)
}
