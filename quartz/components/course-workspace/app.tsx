import { render } from 'preact'
import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks'
import { isUnitMarkSrc, newPdfMarkId, PDF_BODY_LIMIT } from '../../util/pdf-marks'
import { renderBody } from '../pdf-reader/body'
import { conflictMark, deleteMark, fetchMarks, MarkApiError, putMark } from '../pdf-reader/marks'
import { kindMeta } from '../pdf-reader/parts'
import { mountTranscriptPlayer } from '../scripts/transcript-player'
import { lectureSelection, noteRange, showNoteRange, type LectureSelection } from './anchors'
import { DraftEditor } from './editor'
import { initialWorkspaceTab, WorkspaceHeader, type WorkspaceTab } from './Header'
import { noteKinds, SelectionActions, type SelectionAction } from './selection'
import { WorkspaceSourceLinks } from './SourceLinks'
import {
  DraftConflict,
  loadNotes,
  loadScratchpad,
  markdownExport,
  markInputOf,
  noteFromMark,
  noteLimit,
  removeNote,
  saveNote,
  saveScratchpad,
  scratchpadLimit,
  storagePrefix,
  type LectureNote,
  type Scratchpad,
} from './store'

interface WorkspaceProps {
  slug: string
  title: string
  pdf?: string
  reader?: string
  /** The server-rendered video tab body: player slot and transcript. */
  watch?: HTMLElement
  article: HTMLElement
  root: HTMLElement
  signal: AbortSignal
}

/** Signed in: notes live on the marks API under the unit slug. */
interface Account {
  doc: string
}

const SAVE_DELAY = 600
const RETRY_DELAY = 4000

function accountMessage(error: unknown): string {
  if (error instanceof MarkApiError) {
    if (error.status === 409)
      return conflictMark(error)
        ? 'This note changed in another tab. Reload to see the saved version.'
        : 'This note was deleted elsewhere.'
    if (error.status === 401) return 'Signed out. Sign in again to save notes.'
    if (error.retryable) return 'Not saved yet; retrying.'
    return error.message
  }
  return 'Not saved to your account.'
}

function storageMessage(error: unknown): string {
  if (error instanceof DraftConflict) return error.message
  return 'Not saved in this browser. Your draft is still here; export Markdown to keep it.'
}

function Workspace(props: WorkspaceProps) {
  const hasSource = Boolean(props.reader)
  const hasVideo = Boolean(props.watch)
  const [tab, setTab] = useState<WorkspaceTab>(initialWorkspaceTab(hasSource, hasVideo))
  const [scratchPreview, setScratchPreview] = useState(false)
  const [notePreview, setNotePreview] = useState(false)
  const [scratchIssue, setScratchIssue] = useState('')
  const [notesIssue, setNotesIssue] = useState('')
  const [scratch, setScratch] = useState<Scratchpad>({ body: '', revision: 0 })
  const [notes, setNotes] = useState<LectureNote[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [selection, setSelection] = useState<LectureSelection | null>(null)
  const [sourceFailed, setSourceFailed] = useState(false)
  const videoRef = useRef<HTMLElement>(null)
  const watchMounted = useRef(false)
  const scratchRef = useRef(scratch)
  const notesRef = useRef(notes)
  const activeRef = useRef(activeId)
  activeRef.current = activeId
  const [account, setAccount] = useState<Account | null>(null)
  const accountRef = useRef(account)
  const [signInUrl, setSignInUrl] = useState<string | null>(null)
  const [browserNotes, setBrowserNotes] = useState<LectureNote[]>([])
  const [saving, setSaving] = useState(0)
  const timers = useRef(new Map<string, number>())
  const chains = useRef(new Map<string, Promise<void>>())
  const revisions = useRef(new Map<string, number>())

  const replaceScratch = (draft: Scratchpad) => {
    scratchRef.current = draft
    setScratch(draft)
  }
  const replaceNotes = (drafts: LectureNote[]) => {
    notesRef.current = drafts
    setNotes(drafts)
  }

  useLayoutEffect(() => {
    try {
      replaceScratch(loadScratchpad(props.slug))
    } catch (error) {
      setScratchIssue(storageMessage(error))
    }
    try {
      replaceNotes(loadNotes(props.slug))
    } catch (error) {
      setNotesIssue(storageMessage(error))
    }
    const onStorage = (event: StorageEvent) => {
      if (event.key !== null && !event.key.startsWith(storagePrefix(props.slug))) return
      const editing = document.activeElement?.closest('.course-draft-editor')
      if (event.key === `${storagePrefix(props.slug)}scratchpad` || event.key === null) {
        if (editing?.closest('#course-workspace-scratchpad'))
          setScratchIssue(new DraftConflict().message)
        else {
          try {
            replaceScratch(loadScratchpad(props.slug))
            setScratchIssue('')
          } catch (error) {
            setScratchIssue(storageMessage(error))
          }
        }
      }
      if (!accountRef.current && (event.key?.includes(':note:') || event.key === null)) {
        const activeDraft = editing?.closest('#course-workspace-notes')
          ? notesRef.current.find(note => note.id === activeRef.current)
          : undefined
        if (activeDraft && (event.key === null || event.key.endsWith(`:note:${activeDraft.id}`)))
          setNotesIssue(new DraftConflict().message)
        else {
          try {
            const saved = loadNotes(props.slug)
            replaceNotes(
              activeDraft
                ? [...saved.filter(note => note.id !== activeDraft.id), activeDraft].sort(
                    (a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id),
                  )
                : saved,
            )
            if (!activeDraft) setNotesIssue('')
          } catch (error) {
            setNotesIssue(storageMessage(error))
          }
        }
      }
    }
    window.addEventListener('storage', onStorage, { signal: props.signal })
    return () => window.removeEventListener('storage', onStorage)
  }, [props.slug, props.signal])

  useLayoutEffect(() => {
    const header = document.querySelector<HTMLElement>('#quartz-root > .header')
    const owner = props.root.closest<HTMLElement>('.course-workspace')
    const offset = () =>
      owner?.style.setProperty('--course-workspace-top', `${(header?.offsetHeight ?? 0) + 12}px`)
    offset()
    const observer = new ResizeObserver(offset)
    if (header) observer.observe(header)
    return () => {
      observer.disconnect()
    }
  }, [props.root])

  useLayoutEffect(() => {
    if (props.watch) videoRef.current?.append(props.watch)
  }, [props.watch])

  // the player iframe loads when the tab first opens; it then stays for seeking and follow
  useEffect(() => {
    const watch = props.watch
    if (tab !== 'video' || !watch || watchMounted.current) return
    watchMounted.current = true
    watch.dataset.courseWatch = watch.dataset.workspaceWatch
    mountTranscriptPlayer(watch, props.signal)
  }, [tab, props.watch, props.signal])

  const selectionRef = useRef(selection)
  selectionRef.current = selection
  const rangeCacheRef = useRef(
    new Map<string, { signature: string; article: HTMLElement; range: Range | null }>(),
  )
  const resolveNoteRange = (note: LectureNote): Range | null => {
    if (!note.quote) return null
    const sig = `${note.heading ?? ''}\0${note.quote.prefix}\0${note.quote.exact}\0${note.quote.suffix}`
    const cached = rangeCacheRef.current.get(note.id)
    if (cached && cached.article === props.article && cached.signature === sig) {
      return cached.range
    }
    const range = noteRange(props.article, note)
    rangeCacheRef.current.set(note.id, { signature: sig, article: props.article, range })
    return range
  }

  useEffect(() => {
    let frame = 0
    const capture = () => {
      const sel = window.getSelection()
      if ((!sel || sel.isCollapsed) && selectionRef.current === null) return
      if (sel?.anchorNode && props.root.contains(sel.anchorNode)) return
      window.cancelAnimationFrame(frame)
      frame = window.requestAnimationFrame(() => setSelection(lectureSelection(props.article)))
    }
    document.addEventListener('selectionchange', capture, { signal: props.signal })
    const dismiss = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !props.root.contains(document.activeElement)) setSelection(null)
    }
    document.addEventListener('keydown', dismiss, { signal: props.signal })
    return () => {
      window.cancelAnimationFrame(frame)
      document.removeEventListener('selectionchange', capture)
      document.removeEventListener('keydown', dismiss)
    }
  }, [props.article, props.root, props.signal])

  const highlightSignature = notes
    .filter(note => note.deletedAt === null && note.quote !== null)
    .map(
      note =>
        `${note.id}:${note.kind}:${note.heading ?? ''}:${note.quote?.prefix ?? ''}:${note.quote?.exact ?? ''}:${note.quote?.suffix ?? ''}`,
    )
    .join('\n')

  useLayoutEffect(() => {
    if (!('highlights' in CSS) || typeof Highlight === 'undefined') return
    for (const kind of noteKinds) {
      const ranges = notesRef.current
        .filter(note => note.deletedAt === null && note.kind === kind)
        .map(note => resolveNoteRange(note))
        .filter((range): range is Range => range !== null)
      CSS.highlights.set(`course-${kind}`, new Highlight(...ranges))
    }
    return () => {
      for (const kind of noteKinds) CSS.highlights.delete(`course-${kind}`)
    }
  }, [highlightSignature, props.article])

  useEffect(() => {
    if (!isUnitMarkSrc(props.slug)) return
    fetchMarks(props.slug, props.signal)
      .then(response => {
        if (!response.canWrite) {
          setSignInUrl(response.signInUrl)
          return
        }
        const next = { doc: response.doc }
        accountRef.current = next
        setAccount(next)
        setBrowserNotes(notesRef.current.filter(note => note.deletedAt === null))
        const remote = response.marks
          .map(noteFromMark)
          .filter((note): note is LectureNote => note !== null)
        for (const note of remote) revisions.current.set(note.id, note.revision)
        replaceNotes(remote.sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id)))
        setNotesIssue('')
      })
      // offline or no marks API (static preview): the browser copy stays in charge
      .catch(() => undefined)
    const pending = timers.current
    return () => {
      // a navigation flushes debounced edits; the requests outlive the page
      for (const [id, timer] of pending) {
        window.clearTimeout(timer)
        void sendNote(id)
      }
      pending.clear()
    }
  }, [props.slug, props.signal])

  const enqueue = (id: string, task: () => Promise<void>) => {
    setSaving(count => count + 1)
    const chain = (chains.current.get(id) ?? Promise.resolve())
      .then(task)
      .catch(error => {
        setNotesIssue(accountMessage(error))
        if (error instanceof MarkApiError && error.retryable) schedule(id, RETRY_DELAY)
      })
      .finally(() => setSaving(count => count - 1))
    chains.current.set(id, chain)
    return chain
  }

  const sendNote = (id: string) =>
    enqueue(id, async () => {
      const current = accountRef.current
      const note = notesRef.current.find(entry => entry.id === id)
      if (!current || !note) return
      const revision = revisions.current.get(id) ?? 0
      const saved = await putMark(id, markInputOf(props.slug, current.doc, { ...note, revision }))
      revisions.current.set(id, saved.revision)
      replaceNotes(
        notesRef.current.map(entry =>
          entry.id === id ? { ...entry, revision: saved.revision } : entry,
        ),
      )
      setNotesIssue('')
    })

  const schedule = (id: string, delay = SAVE_DELAY) => {
    window.clearTimeout(timers.current.get(id))
    timers.current.set(
      id,
      window.setTimeout(() => {
        timers.current.delete(id)
        void sendNote(id)
      }, delay),
    )
  }

  const updateRemote = (draft: LectureNote) => {
    const current = notesRef.current
    if (draft.deletedAt !== null) {
      window.clearTimeout(timers.current.get(draft.id))
      timers.current.delete(draft.id)
      replaceNotes(current.filter(note => note.id !== draft.id))
      void enqueue(draft.id, async () => {
        const revision = revisions.current.get(draft.id)
        if (revision === undefined) return
        await deleteMark(draft.id, revision)
        revisions.current.delete(draft.id)
      })
      return
    }
    replaceNotes(
      current.some(note => note.id === draft.id)
        ? current.map(note => (note.id === draft.id ? draft : note))
        : [...current, draft],
    )
    schedule(draft.id)
  }

  const moveBrowserNotes = async () => {
    const current = accountRef.current
    if (!current) return
    let moved = 0
    for (const note of browserNotes) {
      const id = newPdfMarkId()
      const copy = { ...note, id, revision: 0, body: note.body.slice(0, PDF_BODY_LIMIT) }
      try {
        const saved = await putMark(id, markInputOf(props.slug, current.doc, copy))
        revisions.current.set(id, saved.revision)
        replaceNotes([...notesRef.current, { ...copy, revision: saved.revision }])
        removeNote(props.slug, note.id)
        moved++
      } catch (error) {
        setNotesIssue(accountMessage(error))
        break
      }
    }
    const left = browserNotes.slice(moved)
    setBrowserNotes(left)
    if (left.length === 0) setNotesIssue('')
  }

  const editScratch = (body: string) => {
    const draft = { ...scratchRef.current, body }
    try {
      replaceScratch(saveScratchpad(props.slug, draft))
      setScratchIssue('')
    } catch (error) {
      replaceScratch(draft)
      setScratchIssue(storageMessage(error))
    }
  }

  const updateNote = (draft: LectureNote) => {
    if (accountRef.current) return updateRemote(draft)
    let updated = draft
    try {
      updated = saveNote(props.slug, draft)
      setNotesIssue('')
    } catch (error) {
      setNotesIssue(storageMessage(error))
    }
    const current = notesRef.current
    replaceNotes(
      current.some(note => note.id === updated.id)
        ? current.map(note => (note.id === updated.id ? updated : note))
        : [...current, updated],
    )
  }

  const addNote = (action: SelectionAction, selected: LectureSelection | null = null) => {
    if (action === 'scratchpad' && selected) {
      const quote = selected.text
        .split('\n')
        .map(line => `> ${line}`)
        .join('\n')
      const target = selected.heading ? `${props.slug}#${selected.heading}` : props.slug
      const separator = scratchRef.current.body ? '\n\n' : ''
      editScratch(`${scratchRef.current.body}${separator}${quote}\n\n[[${target}|lecture]]\n`)
      setTab('scratchpad')
      setScratchPreview(false)
    } else {
      const kind = action === 'note' || action === 'scratchpad' ? 'mark' : action
      const note: LectureNote = {
        id: accountRef.current ? newPdfMarkId() : crypto.randomUUID(),
        kind,
        quote: selected?.quote ?? null,
        quoteText: selected?.text,
        heading: selected?.heading ?? null,
        body: '',
        revision: 0,
        createdAt: Date.now(),
        deletedAt: null,
      }
      updateNote(note)
      setActiveId(action === 'note' || !selected ? note.id : null)
      setNotePreview(false)
      setTab('notes')
    }
    setSelection(null)
    window.getSelection()?.removeAllRanges()
    props.root.scrollIntoView({ block: 'nearest' })
  }

  const focusTab = (value: WorkspaceTab) =>
    document.getElementById(`course-workspace-tab-${value}`)?.focus()
  const active = notes.find(note => note.id === activeId && note.deletedAt === null)
  const visible = notes.filter(note => note.deletedAt === null)
  const deleted = notes.filter(note => note.deletedAt !== null)
  const exportDrafts = () => {
    const markdown = markdownExport(
      props.title,
      props.slug,
      props.pdf,
      scratchRef.current,
      notesRef.current,
    )
    const url = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `${props.slug.replaceAll('/', '-')}-notes.md`
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const loadSaved = () => {
    try {
      replaceScratch(loadScratchpad(props.slug))
      replaceNotes(loadNotes(props.slug))
      setScratchIssue('')
      setNotesIssue('')
    } catch (error) {
      setScratchIssue(storageMessage(error))
      setNotesIssue(storageMessage(error))
    }
  }

  return (
    <div class="course-workspace-app">
      <div class="course-workspace-panel">
        <WorkspaceHeader
          hasSource={hasSource}
          hasVideo={hasVideo}
          tab={tab}
          noteCount={visible.length}
          onSelect={setTab}
          onExport={exportDrafts}
          onLoadSaved={scratchIssue || notesIssue ? loadSaved : undefined}
        />
        {props.reader && (
          <section
            role="tabpanel"
            id="course-workspace-source"
            aria-labelledby="course-workspace-tab-source"
            class="course-workspace-pane course-workspace-source"
            hidden={tab !== 'source'}
          >
            <WorkspaceSourceLinks pdf={props.pdf} reader={props.reader} />
            {sourceFailed && (
              <p role="status">The source could not load. Open the reader to retry.</p>
            )}
            <iframe
              src={`${props.reader}?embed=course`}
              title={`PDF source: ${props.title}`}
              class="course-source-frame"
              onError={() => setSourceFailed(true)}
            />
          </section>
        )}
        {props.watch && (
          <section
            ref={videoRef}
            role="tabpanel"
            id="course-workspace-video"
            aria-labelledby="course-workspace-tab-video"
            class="course-workspace-pane course-workspace-video"
            hidden={tab !== 'video'}
          />
        )}
        <section
          role="tabpanel"
          id="course-workspace-scratchpad"
          aria-labelledby="course-workspace-tab-scratchpad"
          class="course-workspace-pane course-workspace-draft"
          hidden={tab !== 'scratchpad'}
        >
          <div class="course-draft-toolbar">
            <span role="status">{scratchIssue || 'private · saved in this browser'}</span>
            <button
              type="button"
              aria-pressed={scratchPreview}
              onClick={() => setScratchPreview(!scratchPreview)}
            >
              {scratchPreview ? 'edit' : 'preview'}
            </button>
          </div>
          {scratchPreview ? (
            <div
              class="course-draft-preview"
              dangerouslySetInnerHTML={{ __html: renderBody(scratch.body) }}
            />
          ) : (
            <DraftEditor
              label="Lecture scratchpad"
              value={scratch.body}
              limit={scratchpadLimit}
              onChange={editScratch}
              onExit={() => focusTab('scratchpad')}
            />
          )}
        </section>
        <section
          role="tabpanel"
          id="course-workspace-notes"
          aria-labelledby="course-workspace-tab-notes"
          class="course-workspace-pane course-workspace-notes"
          hidden={tab !== 'notes'}
        >
          <div class="course-draft-toolbar">
            <span role="status">
              {notesIssue ||
                (account
                  ? saving > 0
                    ? 'private · saving to your account…'
                    : 'private · saved to your account'
                  : 'private · saved in this browser')}
            </span>
            {!account && signInUrl && (
              <a href={signInUrl} data-router-ignore>
                sign in to keep notes
              </a>
            )}
            {account && browserNotes.length > 0 && (
              <button type="button" onClick={() => void moveBrowserNotes()}>
                move {browserNotes.length} browser {browserNotes.length === 1 ? 'note' : 'notes'} to
                account
              </button>
            )}
            <button type="button" onClick={() => addNote('note')}>
              add note
            </button>
          </div>
          {active ? (
            <article class="course-note-card" data-kind={active.kind}>
              <div class="course-draft-toolbar">
                <button
                  type="button"
                  onClick={() => {
                    setActiveId(null)
                    setNotePreview(false)
                  }}
                >
                  ← notes
                </button>
                <button
                  type="button"
                  aria-pressed={notePreview}
                  onClick={() => setNotePreview(!notePreview)}
                >
                  {notePreview ? 'edit' : 'preview'}
                </button>
              </div>
              {active.quote && (
                <blockquote
                  dangerouslySetInnerHTML={{
                    __html: renderBody(active.quoteText ?? active.quote.exact),
                  }}
                />
              )}
              {notePreview ? (
                <div
                  class="course-draft-preview"
                  dangerouslySetInnerHTML={{ __html: renderBody(active.body) }}
                />
              ) : (
                <DraftEditor
                  key={active.id}
                  label="Lecture note"
                  autoFocus
                  value={active.body}
                  limit={account ? PDF_BODY_LIMIT : noteLimit}
                  onChange={body => {
                    const current = notesRef.current.find(note => note.id === active.id)
                    if (current) updateNote({ ...current, body })
                  }}
                  onExit={() => {
                    setActiveId(null)
                    focusTab('notes')
                  }}
                />
              )}
              <div class="course-note-actions" role="group" aria-label="Note kind">
                {noteKinds.map(kind => (
                  <button
                    type="button"
                    aria-pressed={active.kind === kind}
                    onClick={() => updateNote({ ...active, kind })}
                  >
                    {kindMeta[kind].glyph} {kindMeta[kind].label}
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => {
                    updateNote({ ...active, deletedAt: Date.now() })
                    setActiveId(null)
                  }}
                >
                  delete note
                </button>
              </div>
            </article>
          ) : (
            <>
              {visible.length === 0 && (
                <p class="course-notes-empty">
                  Select a passage to mark it, ask a question, or add a note. You can also add a
                  note about the whole lecture.
                </p>
              )}
              <ol class="course-notes-list">
                {visible.map(note => {
                  const range = resolveNoteRange(note)
                  return (
                    <li
                      key={note.id}
                      class="course-note-card"
                      data-kind={note.kind}
                      data-note-id={note.id}
                    >
                      <div class="course-note-meta">
                        <span>
                          {kindMeta[note.kind].glyph} {kindMeta[note.kind].label}
                        </span>
                        {note.quote && (
                          <button
                            type="button"
                            disabled={!range}
                            onClick={() => {
                              if (!range) return
                              showNoteRange(props.article, range)
                            }}
                          >
                            {range ? 'show passage' : 'passage changed'}
                          </button>
                        )}
                      </div>
                      {note.quote && (
                        <blockquote
                          dangerouslySetInnerHTML={{
                            __html: renderBody(note.quoteText ?? note.quote.exact),
                          }}
                        />
                      )}
                      {note.body && (
                        <div
                          class="course-draft-preview"
                          dangerouslySetInnerHTML={{ __html: renderBody(note.body) }}
                        />
                      )}
                      <button
                        type="button"
                        onClick={() => {
                          setActiveId(note.id)
                          setNotePreview(false)
                        }}
                      >
                        edit note
                      </button>
                    </li>
                  )
                })}
              </ol>
              {deleted.length > 0 && (
                <details class="course-deleted-notes">
                  <summary>deleted notes ({deleted.length})</summary>
                  {deleted.map(note => (
                    <div key={note.id}>
                      <span>
                        {(note.body || note.quoteText || note.quote?.exact || 'empty note').slice(
                          0,
                          100,
                        )}
                      </span>
                      <button
                        type="button"
                        onClick={() => updateNote({ ...note, deletedAt: null })}
                      >
                        restore
                      </button>
                    </div>
                  ))}
                </details>
              )}
            </>
          )}
        </section>
      </div>
      {selection && (
        <SelectionActions range={selection.range} onAction={action => addNote(action, selection)} />
      )}
    </div>
  )
}

export function mountCourseWorkspace(root: HTMLElement, signal: AbortSignal): () => void {
  const slug = root.dataset.courseWorkspaceSlug
  const container = root.closest<HTMLElement>('.course-reading-split')
  const article = container?.querySelector<HTMLElement>(':scope > article.course-lecture')
  if (!slug || !article) return () => undefined
  const watch = root.querySelector<HTMLElement>('.course-workspace-watch') ?? undefined
  root.replaceChildren()
  render(
    <Workspace
      slug={slug}
      title={root.dataset.courseWorkspaceTitle ?? 'lecture'}
      pdf={root.dataset.courseWorkspacePdf}
      reader={root.dataset.courseWorkspaceReader}
      watch={watch}
      article={article}
      root={root}
      signal={signal}
    />,
    root,
  )
  return () => render(null, root)
}
