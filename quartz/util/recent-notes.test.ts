import assert from 'node:assert/strict'
import test from 'node:test'
import { FilePath, FullSlug, isFilePath, isFullSlug } from './path'
import {
  queryRecentNotes,
  readRecentNotes,
  recentNoteFromContent,
  upsertRecentNote,
  writeRecentNotes,
  type RecentNoteStorage,
} from './recent-notes'

class MemoryStorage implements RecentNoteStorage {
  private readonly data = new Map<string, string>()

  getItem(key: string) {
    return this.data.get(key) ?? null
  }

  setItem(key: string, value: string) {
    this.data.set(key, value)
  }
}

function slug(value: string): FullSlug {
  if (!isFullSlug(value)) throw new Error(`invalid full slug: ${value}`)
  return value
}

function filePath(value: string): FilePath {
  if (!isFilePath(value)) throw new Error(`invalid file path: ${value}`)
  return value
}

test('recent notes parse old slug lists and cached entries', () => {
  const storage = new MemoryStorage()
  storage.setItem(
    'recent-notes',
    JSON.stringify([
      'thoughts/llama-3',
      { slug: 'thoughts/search', name: 'thoughts/search.md', title: 'Search', aliases: ['find'] },
      '../bad',
    ]),
  )

  assert.deepEqual(readRecentNotes(storage), [
    { slug: 'thoughts/llama-3', name: 'thoughts/llama-3', title: '', aliases: [] },
    { slug: 'thoughts/search', name: 'thoughts/search.md', title: 'Search', aliases: ['find'] },
  ])
})

test('recent notes keep newest order and cap size', () => {
  const oldNote = { slug: slug('a'), name: 'a.md', title: 'A', aliases: [] }
  const movedNote = { slug: slug('b'), name: 'b.md', title: 'B', aliases: [] }
  const notes = upsertRecentNote([oldNote, movedNote], oldNote, 2)

  assert.deepEqual(notes, [movedNote, oldNote])
  assert.deepEqual(
    upsertRecentNote(notes, { slug: slug('c'), name: 'c.md', title: 'C', aliases: [] }, 2),
    [oldNote, { slug: slug('c'), name: 'c.md', title: 'C', aliases: [] }],
  )
})

test('recent notes query matches title, filename, slug, alias, and recency ties', () => {
  const notes = [
    { slug: slug('thoughts/older'), name: 'older.md', title: 'Older Search', aliases: [] },
    {
      slug: slug('thoughts/llama-3'),
      name: 'llama.md',
      title: 'Llama Three',
      aliases: ['language model'],
    },
    { slug: slug('thoughts/search'), name: 'search.md', title: 'Search', aliases: ['find'] },
  ]

  assert.deepEqual(
    queryRecentNotes(notes, '', 2).map(note => note.slug),
    ['thoughts/search', 'thoughts/llama-3'],
  )
  assert.deepEqual(
    queryRecentNotes(notes, 'find', 10).map(note => note.slug),
    ['thoughts/search'],
  )
  assert.deepEqual(
    queryRecentNotes(notes, 'lng mdl', 10).map(note => note.slug),
    ['thoughts/llama-3'],
  )
  assert.deepEqual(
    queryRecentNotes(notes, 'search', 10).map(note => note.slug),
    ['thoughts/search', 'thoughts/older'],
  )
})

test('recent notes round-trip hydrated content entries', () => {
  const storage = new MemoryStorage()
  const noteSlug = slug('thoughts/cache')
  const note = recentNoteFromContent(noteSlug, {
    slug: noteSlug,
    title: 'Cache',
    filePath: filePath('thoughts/cache.md'),
    links: [],
    aliases: ['ready'],
    tags: [],
    layout: 'default',
    fileName: filePath('thoughts/cache.md'),
  })

  writeRecentNotes(storage, [note])

  assert.deepEqual(readRecentNotes(storage), [note])
})
