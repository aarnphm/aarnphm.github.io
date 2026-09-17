import type { ArenaNote, ArenaNoteQuote, ArenaReadLink } from '../../util/arena-reader'
import { arenaFeedSourceNames, type ArenaFeedEntry } from '../../util/arena-feed'

export type FeedFilter = 'unread' | 'read' | 'all' | 'curius'
export type NoteStage = 'draft' | 'ready' | 'backfilled'

export interface ReaderPass {
  seed: string
  current: string | null
  visited: string[]
}

export interface NoteDraft {
  subject: string
  note: ArenaNote
  dirty: boolean
  ready: boolean
  localVersion: number
  conflict: ArenaNote | null
}

export function eligibleEntries(
  entries: ArenaFeedEntry[],
  readLinks: ArenaReadLink[],
  filter: FeedFilter,
  query = '',
): ArenaFeedEntry[] {
  const read = new Set(readLinks.filter(link => link.readAt !== null).map(link => link.articleId))
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)
  return entries.filter(entry => {
    if (filter === 'unread' && read.has(entry.articleId)) return false
    if (filter === 'read' && !read.has(entry.articleId)) return false
    if (filter === 'curius' && !entry.curius?.length) return false
    const text = [entry.title, entry.sourceUrl, ...entry.tags, ...arenaFeedSourceNames(entry)]
      .join(' ')
      .toLocaleLowerCase()
    return terms.every(term => text.includes(term))
  })
}

export function nextEntry(entries: ArenaFeedEntry[], visited: string[], current: string | null) {
  const skipped = new Set(visited)
  return entries.find(entry => entry.articleId !== current && !skipped.has(entry.articleId)) ?? null
}

export function noteStage(note: ArenaNote): NoteStage {
  if (note.exportedRevision === note.revision) return 'backfilled'
  if (note.readyRevision === note.revision) return 'ready'
  return 'draft'
}

export function mergeReadLinks(local: ArenaReadLink[], remote: ArenaReadLink[]): ArenaReadLink[] {
  const links = new Map(local.map(link => [link.articleId, link]))
  for (const link of remote) {
    if ((links.get(link.articleId)?.revision ?? -1) <= link.revision)
      links.set(link.articleId, link)
  }
  return [...links.values()]
}

export function draftFromNote(subject: string, note: ArenaNote): NoteDraft {
  return {
    subject,
    note,
    dirty: false,
    ready: noteStage(note) === 'ready',
    localVersion: 0,
    conflict: null,
  }
}

export function editDraft(draft: NoteDraft, body: string): NoteDraft {
  return {
    ...draft,
    note: { ...draft.note, body },
    dirty: true,
    ready: false,
    localVersion: draft.localVersion + 1,
  }
}

export function acknowledgeDraft(current: NoteDraft, sent: NoteDraft, note: ArenaNote): NoteDraft {
  if (current.localVersion === sent.localVersion)
    return { ...draftFromNote(current.subject, note), localVersion: current.localVersion }
  // A response acknowledges only the version sent; keystrokes during the request remain pending.
  return {
    ...current,
    note: { ...current.note, revision: note.revision, createdAt: note.createdAt },
    conflict: null,
  }
}

export function mergeRemoteDraft(draft: NoteDraft, remote: ArenaNote): NoteDraft {
  if (remote.revision < draft.note.revision) return draft
  if (!draft.dirty) return draftFromNote(draft.subject, remote)
  if (draft.note.revision !== remote.revision) return { ...draft, conflict: remote }
  return draft
}

export function boundedQuote(text: string, before: string, after: string): ArenaNoteQuote | null {
  const selected = text.trim()
  const exact = selected.slice(0, 4096)
  if (!exact) return null
  return {
    exact,
    prefix: before.slice(-100),
    suffix: (selected.length > 4096 ? selected.slice(4096) : after).slice(0, 100),
  }
}

export function quoteFromSelection(root: HTMLElement): ArenaNoteQuote | null {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null
  const range = selection.getRangeAt(0)
  if (!root.contains(range.commonAncestorContainer)) return null
  const before = range.cloneRange()
  before.selectNodeContents(root)
  before.setEnd(range.startContainer, range.startOffset)
  const after = range.cloneRange()
  after.selectNodeContents(root)
  after.setStart(range.endContainer, range.endOffset)
  return boundedQuote(range.toString(), before.toString(), after.toString())
}
