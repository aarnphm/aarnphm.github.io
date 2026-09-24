import { marked } from 'marked'
import { useLayoutEffect, useRef, useState } from 'preact/hooks'
import type { ArenaFeedEntry } from '../../util/arena-feed'
import type { ArenaNote } from '../../util/arena-reader'
import { sanitizeReaderHtml } from './content'
import { DeleteNote } from './delete-note'
import { ReaderFilter } from './filter'
import { noteStage, type NoteDraft } from './model'

export type DraftStatus = 'pending' | 'sync-failed' | 'synced' | 'unsaved' | 'conflict'

export const draftStatusLabel: Record<DraftStatus, string> = {
  pending: 'Pending sync',
  'sync-failed': 'Sync failed',
  synced: 'Synced',
  unsaved: 'Not saved on this device',
  conflict: 'Two versions need review',
}

interface NotesProps {
  drafts: NoteDraft[]
  entries: ArenaFeedEntry[]
  article: ArenaFeedEntry | null
  editing: string | null
  status: (draft: NoteDraft) => DraftStatus
  onAdd: () => void
  onEdit: (id: string | null) => void
  onChange: (draft: NoteDraft, body: string) => void
  onReady: (draft: NoteDraft) => void
  onRetry: () => void
  onDelete: (draft: NoteDraft) => void
  onResolve: (draft: NoteDraft, choice: 'remote' | 'copy') => void
  onOpenArticle: (id: string) => void
  onOpenSnapshot: (note: ArenaNote) => void
  onExport?: () => void
  snapshotId: string | null
  inbox?: boolean
}

const stages = { draft: 'Draft', ready: 'Ready to backfill', backfilled: 'Backfilled' }
type NoteFilter = 'all' | 'draft' | 'ready' | 'backfilled'
const noteFilters: { value: NoteFilter; label: string }[] = [
  { value: 'all', label: 'all notes' },
  { value: 'draft', label: 'drafts' },
  { value: 'ready', label: 'ready to backfill' },
  { value: 'backfilled', label: 'backfilled' },
]

export function NotesPanel(props: NotesProps) {
  const [stage, setStage] = useState<NoteFilter>('all')
  const [preview, setPreview] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)
  const listScroll = useRef(0)
  const draft = props.drafts.find(item => item.note.id === props.editing)
  const listContext = props.inbox ? 'inbox' : props.article?.articleId
  useLayoutEffect(() => {
    listScroll.current = 0
  }, [listContext])
  useLayoutEffect(() => {
    if (!draft && listRef.current) listRef.current.scrollTop = listScroll.current
  }, [draft?.note.id, listContext])
  const visible = props.drafts.filter(
    item => stage === 'all' || (item.dirty ? 'draft' : noteStage(item.note)) === stage,
  )
  const toolbar = props.inbox && (
    <div class="arena-notes-controls">
      <button
        type="button"
        class="arena-notes-export"
        aria-label="export ready notes"
        title="export ready notes"
        onClick={props.onExport}
      >
        <svg
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          stroke-width="1.25"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
          focusable="false"
        >
          <path d="M8 2v8m-3-3 3 3 3-3M2 10v4h12v-4" />
        </svg>
        export
      </button>
      <ReaderFilter
        label="show notes"
        options={noteFilters}
        value={stage}
        onChange={value => {
          setStage(value)
          props.onEdit(null)
          setPreview(false)
        }}
      />
    </div>
  )
  if (draft) {
    const title =
      props.entries.find(entry => entry.articleId === draft.note.articleId)?.title ?? 'Saved link'
    const status = props.status(draft)
    return (
      <div class="arena-notes-editor">
        {toolbar}
        <div class="arena-notes-toolbar">
          <button
            type="button"
            aria-label="Back to notes list"
            onClick={() => {
              props.onEdit(null)
              setPreview(false)
            }}
          >
            ← notes
          </button>
          <span role="status" class={`arena-note-save-status ${status}`}>
            {draftStatusLabel[status]}
          </span>
        </div>
        <p class="arena-notes-article-title">{title}</p>
        {draft.note.quote && (
          <blockquote class="arena-note-quote">
            <p>{draft.note.quote.exact}</p>
            {draft.note.snapshotId && draft.note.snapshotId !== props.snapshotId && (
              <button type="button" onClick={() => props.onOpenSnapshot(draft.note)}>
                Open captured copy
              </button>
            )}
          </blockquote>
        )}
        {draft.conflict && (
          <section class="arena-note-conflict" aria-label="Conflicting note versions">
            <p>A newer version was saved on another device. Your draft is preserved below.</p>
            <details>
              <summary>View synced version</summary>
              <p>{draft.conflict.body || '(Empty note)'}</p>
            </details>
            <div class="arena-notes-toolbar">
              <button type="button" onClick={() => props.onResolve(draft, 'copy')}>
                Keep both notes
              </button>
              <button type="button" onClick={() => props.onResolve(draft, 'remote')}>
                Use synced version
              </button>
            </div>
          </section>
        )}
        <div class="arena-notes-toolbar">
          <label for="arena-note-body">Your note</label>
          <button type="button" aria-pressed={preview} onClick={() => setPreview(!preview)}>
            {preview ? 'Edit text' : 'Preview'}
          </button>
        </div>
        {preview ? (
          <div
            class="arena-note-preview"
            dangerouslySetInnerHTML={{
              __html: sanitizeReaderHtml(marked.parse(draft.note.body, { async: false })),
            }}
          />
        ) : (
          <textarea
            id="arena-note-body"
            class="arena-note-input"
            value={draft.note.body}
            onInput={event => props.onChange(draft, event.currentTarget.value)}
            placeholder="Write a thought, question, or connection…"
            maxLength={20000}
            autoFocus
          />
        )}
        <div class="arena-note-actions">
          {status === 'sync-failed' && (
            <button type="button" onClick={props.onRetry}>
              sync
            </button>
          )}
          <button
            type="button"
            disabled={!draft.note.body.trim() || Boolean(draft.conflict)}
            onClick={() => props.onReady(draft)}
          >
            {draft.ready ? 'return to draft' : 'ready to backfill'}
          </button>
          <DeleteNote key={draft.note.id} onDelete={() => props.onDelete(draft)} />
        </div>
      </div>
    )
  }
  return (
    <div
      ref={listRef}
      class="arena-notes-list"
      onScroll={event => {
        listScroll.current = event.currentTarget.scrollTop
      }}
    >
      {toolbar}
      {!props.inbox && (
        <>
          <p class="arena-reader-status">{props.article?.title}</p>
          <button
            type="button"
            class="arena-note-add"
            disabled={!props.article}
            onClick={props.onAdd}
          >
            Add note
          </button>
        </>
      )}
      {props.inbox && visible.length === 0 && (
        <p class="arena-reader-empty-hint">Notes you write while reading will appear here.</p>
      )}
      {visible.map(item => {
        const entry = props.entries.find(entry => entry.articleId === item.note.articleId)
        const noteStatus = props.status(item)
        return (
          <section class="arena-note-card" key={item.note.id}>
            <div class="arena-note-card-meta">
              <span>{item.dirty ? stages.draft : stages[noteStage(item.note)]}</span>
              <span>{draftStatusLabel[noteStatus]}</span>
            </div>
            {props.inbox && (
              <button
                type="button"
                class="arena-note-article-link"
                onClick={() => props.onOpenArticle(item.note.articleId)}
              >
                {entry?.title ?? item.note.sourceUrl}
              </button>
            )}
            {item.note.quote && <blockquote>{item.note.quote.exact}</blockquote>}
            <p class="arena-note-card-body">{item.note.body || 'Empty draft'}</p>
            <button type="button" onClick={() => props.onEdit(item.note.id)}>
              Edit note
            </button>
          </section>
        )
      })}
    </div>
  )
}
