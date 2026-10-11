import { render } from 'preact'
import { useEffect, useLayoutEffect, useRef, useState } from 'preact/hooks'
import { renderBody } from '../pdf-reader/body'
import { kindMeta } from '../pdf-reader/parts'
import { lectureSelection, noteRange, showNoteRange, type LectureSelection } from './anchors'
import { DraftEditor } from './editor'
import { noteKinds, SelectionActions, type SelectionAction } from './selection'
import { SplitDivider } from './split-divider'
import {
  DraftConflict,
  loadNotes,
  loadScratchpad,
  markdownExport,
  noteLimit,
  saveNote,
  saveScratchpad,
  scratchpadLimit,
  storagePrefix,
  type LectureNote,
  type Scratchpad,
} from './store'

type Tab = 'source' | 'scratchpad' | 'notes'
interface WorkspaceProps {
  slug: string
  title: string
  pdf?: string
  reader?: string
  video?: string
  article: HTMLElement
  container: HTMLElement
  workspace: HTMLElement
  root: HTMLElement
  signal: AbortSignal
}

function storageMessage(error: unknown): string {
  if (error instanceof DraftConflict) return error.message
  return 'Not saved in this browser. Your draft is still here; export Markdown to keep it.'
}

function Workspace(props: WorkspaceProps) {
  const compactQuery = useRef(window.matchMedia('(max-width: 1400px)'))
  const [compact, setCompact] = useState(compactQuery.current.matches)
  const [open, setOpen] = useState(!compactQuery.current.matches)
  const [tab, setTab] = useState<Tab>(props.reader ? 'source' : 'scratchpad')
  const [sourceVisited, setSourceVisited] = useState(false)
  const [scratchPreview, setScratchPreview] = useState(false)
  const [notePreview, setNotePreview] = useState(false)
  const [scratchIssue, setScratchIssue] = useState('')
  const [notesIssue, setNotesIssue] = useState('')
  const [scratch, setScratch] = useState<Scratchpad>({ body: '', revision: 0 })
  const [notes, setNotes] = useState<LectureNote[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [selection, setSelection] = useState<LectureSelection | null>(null)
  const [sourceFailed, setSourceFailed] = useState(false)
  const dialog = useRef<HTMLDialogElement>(null)
  const launcher = useRef<HTMLButtonElement>(null)
  const wasOpen = useRef(open)
  const scratchRef = useRef(scratch)
  const notesRef = useRef(notes)
  const activeRef = useRef(activeId)
  activeRef.current = activeId

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
      if (event.key?.includes(':note:') || event.key === null) {
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
    const query = compactQuery.current
    const change = () => {
      dialog.current?.close()
      setCompact(query.matches)
      setOpen(!query.matches)
    }
    query.addEventListener('change', change, { signal: props.signal })
    const header = document.querySelector<HTMLElement>('#quartz-root > .header')
    const owner = props.root.closest<HTMLElement>('.course-workspace')
    const offset = () =>
      owner?.style.setProperty('--course-workspace-top', `${(header?.offsetHeight ?? 0) + 12}px`)
    offset()
    const observer = new ResizeObserver(offset)
    if (header) observer.observe(header)
    return () => {
      query.removeEventListener('change', change)
      observer.disconnect()
    }
  }, [props.signal, props.root])

  useLayoutEffect(() => {
    const panel = dialog.current
    if (!panel) return
    if (open && !panel.open) {
      if (compact) panel.showModal()
      else panel.show()
    } else if (!open && panel.open) panel.close()
    if (open && tab === 'source') setSourceVisited(true)
    if (wasOpen.current && !open) launcher.current?.focus({ preventScroll: true })
    wasOpen.current = open
  }, [open, compact, tab])

  useEffect(() => {
    let frame = 0
    const capture = () => {
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

  useLayoutEffect(() => {
    if (!('highlights' in CSS) || typeof Highlight === 'undefined') return
    for (const kind of noteKinds) {
      const ranges = notes
        .filter(note => note.deletedAt === null && note.kind === kind)
        .map(note => noteRange(props.article, note))
        .filter((range): range is Range => range !== null)
      CSS.highlights.set(`course-${kind}`, new Highlight(...ranges))
    }
    return () => {
      for (const kind of noteKinds) CSS.highlights.delete(`course-${kind}`)
    }
  }, [notes, props.article])

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
        id: crypto.randomUUID(),
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
    setOpen(true)
  }

  const close = () => {
    setOpen(false)
    launcher.current?.focus({ preventScroll: true })
  }
  const focusTab = (value: Tab) => document.getElementById(`course-workspace-tab-${value}`)?.focus()
  const active = notes.find(note => note.id === activeId && note.deletedAt === null)
  const visible = notes.filter(note => note.deletedAt === null)
  const deleted = notes.filter(note => note.deletedAt !== null)
  const tabs: Tab[] = props.reader ? ['source', 'scratchpad', 'notes'] : ['scratchpad', 'notes']
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
    <div class="course-workspace-app" data-open={open} data-compact={compact}>
      <SplitDivider
        container={props.container}
        article={props.article}
        workspace={props.workspace}
        enabled={!compact && open}
      />
      <button
        ref={launcher}
        type="button"
        class="course-workspace-open"
        aria-haspopup={compact ? 'dialog' : undefined}
        aria-expanded={open}
        aria-controls="course-workspace-dialog"
        onClick={() => setOpen(true)}
      >
        source · scratchpad · notes
      </button>
      <dialog
        ref={dialog}
        id="course-workspace-dialog"
        class="course-workspace-dialog"
        aria-labelledby="course-workspace-heading"
        onCancel={close}
        onKeyDown={event => {
          if (event.key !== 'Escape' || event.defaultPrevented) return
          event.preventDefault()
          event.stopPropagation()
          close()
        }}
      >
        <header class="course-workspace-header">
          <h2 id="course-workspace-heading">lecture workspace</h2>
          <button type="button" aria-label="Close lecture workspace" onClick={close}>
            ×
          </button>
        </header>
        <div
          role="tablist"
          aria-label="Lecture workspace"
          class="course-workspace-tabs"
          onKeyDown={event => {
            const at = tabs.indexOf(tab)
            const next =
              event.key === 'ArrowRight'
                ? tabs[(at + 1) % tabs.length]
                : event.key === 'ArrowLeft'
                  ? tabs[(at + tabs.length - 1) % tabs.length]
                  : event.key === 'Home'
                    ? tabs[0]
                    : event.key === 'End'
                      ? tabs[tabs.length - 1]
                      : null
            if (!next) return
            event.preventDefault()
            setTab(next)
            focusTab(next)
          }}
        >
          {tabs.map(value => (
            <button
              type="button"
              role="tab"
              id={`course-workspace-tab-${value}`}
              aria-selected={tab === value}
              aria-controls={`course-workspace-${value}`}
              tabIndex={tab === value ? 0 : -1}
              onClick={() => setTab(value)}
            >
              {value}
              {value === 'notes' ? ` ${visible.length}` : ''}
            </button>
          ))}
        </div>
        {props.reader && (
          <section
            role="tabpanel"
            id="course-workspace-source"
            aria-labelledby="course-workspace-tab-source"
            class="course-workspace-pane course-workspace-source"
            hidden={tab !== 'source'}
          >
            <div class="course-source-links">
              <a
                href={`${props.reader}?view=reader`}
                target="_blank"
                rel="noopener noreferrer"
                data-router-ignore
              >
                open reader ↗
              </a>
              <a href={`/${props.pdf}?raw=1`} target="_blank" rel="noopener noreferrer">
                pdf ↗
              </a>
              {props.video && (
                <a href={props.video} target="_blank" rel="noopener noreferrer">
                  video ↗
                </a>
              )}
              <span>mit ocw · cc by-nc-sa</span>
            </div>
            {sourceFailed && (
              <p role="status">The source could not load. Open the reader to retry.</p>
            )}
            {sourceVisited && (
              <iframe
                src={`${props.reader}?embed=course`}
                title={`PDF source: ${props.title}`}
                class="course-source-frame"
                onError={() => setSourceFailed(true)}
              />
            )}
          </section>
        )}
        <section
          role="tabpanel"
          id="course-workspace-scratchpad"
          aria-labelledby="course-workspace-tab-scratchpad"
          class="course-workspace-pane course-workspace-draft"
          hidden={tab !== 'scratchpad'}
        >
          <div class="course-draft-toolbar">
            <span>private · this browser</span>
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
          <p class="course-draft-status" role="status">
            {scratchIssue || 'saved in this browser'}
          </p>
        </section>
        <section
          role="tabpanel"
          id="course-workspace-notes"
          aria-labelledby="course-workspace-tab-notes"
          class="course-workspace-pane course-workspace-notes"
          hidden={tab !== 'notes'}
        >
          <div class="course-draft-toolbar">
            <span>private · this browser</span>
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
                  limit={noteLimit}
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
                  const range = noteRange(props.article, note)
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
                              if (compact) close()
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
          <p class="course-draft-status" role="status">
            {notesIssue || 'saved in this browser'}
          </p>
        </section>
        <footer class="course-workspace-footer">
          {(scratchIssue || notesIssue) && (
            <button type="button" onClick={loadSaved}>
              load saved version
            </button>
          )}
          <button type="button" onClick={exportDrafts}>
            export Markdown
          </button>
        </footer>
      </dialog>
      {selection && (
        <SelectionActions range={selection.range} onAction={action => addNote(action, selection)} />
      )}
    </div>
  )
}

export function mountCourseWorkspace(root: HTMLElement, signal: AbortSignal): () => void {
  const slug = root.dataset.courseWorkspaceSlug
  const container = root.closest<HTMLElement>('.course-reading-split')
  const workspace = root.closest<HTMLElement>('.course-workspace')
  const article = container?.querySelector<HTMLElement>(':scope > article.course-lecture')
  if (!slug || !article || !container || !workspace) return () => undefined
  root.replaceChildren()
  render(
    <Workspace
      slug={slug}
      title={root.dataset.courseWorkspaceTitle ?? 'lecture'}
      pdf={root.dataset.courseWorkspacePdf}
      reader={root.dataset.courseWorkspaceReader}
      video={root.dataset.courseWorkspaceVideo}
      article={article}
      container={container}
      workspace={workspace}
      root={root}
      signal={signal}
    />,
    root,
  )
  return () => render(null, root)
}
