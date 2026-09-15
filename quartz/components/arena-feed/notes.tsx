import { marked } from 'marked'
import { useState } from 'preact/hooks'
import type { ArenaFeedEntry } from '../../util/arena-feed'
import type { ArenaNote } from '../../util/arena-reader'
import { sanitizeReaderHtml } from './content'
import { noteStage, type NoteDraft } from './model'

export type DraftStatus = 'device' | 'saving' | 'synced' | 'unsaved' | 'conflict'

export const draftStatusLabel: Record<DraftStatus, string> = {
  device: 'Saved on device',
  saving: 'Saving…',
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
  snapshotId: string | null
  inbox?: boolean
}

const stages = { draft: 'Draft', ready: 'Ready to backfill', backfilled: 'Backfilled' }

export function NotesPanel(props: NotesProps) {
  const [stage, setStage] = useState<'all' | 'draft' | 'ready' | 'backfilled'>('all')
  const [preview, setPreview] = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)
  const draft = props.drafts.find(item => item.note.id === props.editing)
  const visible = props.drafts.filter(
    item => stage === 'all' || (item.dirty ? 'draft' : noteStage(item.note)) === stage,
  )
  if (draft) {
    const title =
      props.entries.find(entry => entry.articleId === draft.note.articleId)?.title ?? 'Saved link'
    const status = props.status(draft)
    return (
      <div class="arena-notes-editor">
        <div class="arena-notes-toolbar">
          <button
            type="button"
            onClick={() => {
              props.onEdit(null)
              setPreview(false)
            }}
          >
            All notes
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
        <div class="arena-notes-editor-footer">
          <span class="arena-reader-status">
            {draft.note.body.trim() ? 'Markdown supported' : 'Write text to sync this draft.'}
          </span>
          <div class="arena-notes-toolbar">
            {status === 'device' && (
              <button type="button" onClick={props.onRetry}>
                Sync now
              </button>
            )}
            <button
              type="button"
              disabled={!draft.note.body.trim() || Boolean(draft.conflict) || status === 'saving'}
              onClick={() => props.onReady(draft)}
            >
              {draft.ready ? 'Return to draft' : 'Ready to backfill'}
            </button>
            <button type="button" onClick={() => setDeleting(draft.note.id)}>
              Delete
            </button>
          </div>
          {deleting === draft.note.id && (
            <div class="arena-note-delete">
              <p>Delete this reader note?</p>
              <button
                type="button"
                onClick={() => {
                  props.onDelete(draft)
                  setDeleting(null)
                }}
              >
                Delete note
              </button>
              <button type="button" onClick={() => setDeleting(null)}>
                Keep note
              </button>
            </div>
          )}
        </div>
      </div>
    )
  }
  return (
    <div class="arena-notes-list">
      {props.inbox ? (
        <label class="arena-reader-select">
          Show notes
          <select
            value={stage}
            onChange={event => {
              const value = event.currentTarget.value
              if (
                value === 'all' ||
                value === 'draft' ||
                value === 'ready' ||
                value === 'backfilled'
              )
                setStage(value)
            }}
          >
            <option value="all">All notes</option>
            <option value="draft">Drafts</option>
            <option value="ready">Ready to backfill</option>
            <option value="backfilled">Backfilled</option>
          </select>
        </label>
      ) : (
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
      {visible.length === 0 && (
        <p class="arena-reader-empty-hint">
          {props.inbox
            ? 'Notes you write while reading will appear here.'
            : 'Keep a thought here. Select a passage in the article to attach a quote.'}
        </p>
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
