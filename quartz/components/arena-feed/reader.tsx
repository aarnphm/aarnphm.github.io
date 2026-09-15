import { useCallback, useEffect, useMemo, useRef, useState } from 'preact/hooks'
import type { ArenaFeedEntry } from '../../util/arena-feed'
import type {
  ArenaFeedResponse,
  ArenaNote,
  ArenaNoteQuote,
  ArenaReaderRenderResult,
} from '../../util/arena-reader'
import { isNote, isReadLink, isRecord, ReaderApiError, readerApi } from './api'
import { ArticleContent, safeHref } from './content'
import {
  acknowledgeDraft,
  draftFromNote,
  editDraft,
  eligibleEntries,
  mergeRemoteDraft,
  mergeReadLinks,
  nextEntry,
  quoteFromSelection,
  type FeedFilter,
  type NoteDraft,
  type ReaderPass,
} from './model'
import { NotesPanel, type DraftStatus } from './notes'
import {
  loadDrafts,
  loadPass,
  loadPosition,
  openReaderStore,
  removeDraft,
  saveDraft,
  savePass,
  savePosition,
} from './store'

type Panel = 'queue' | 'notes' | null

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : 'The reader request failed. Your drafts remain on this device.'
}

function delay(seconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason)
    const abort = () => {
      clearTimeout(timer)
      reject(signal.reason)
    }
    const timer = window.setTimeout(
      () => {
        signal.removeEventListener('abort', abort)
        resolve()
      },
      Math.min(15, Math.max(1, seconds)) * 1000,
    )
    signal.addEventListener('abort', abort, { once: true })
  })
}

function reconcileNotes(
  subject: string,
  local: Record<string, NoteDraft>,
  notes: ArenaNote[],
): Record<string, NoteDraft> {
  const merged = { ...local }
  for (const note of notes) {
    const draft = merged[note.id]
    merged[note.id] = draft ? mergeRemoteDraft(draft, note) : draftFromNote(subject, note)
  }
  return merged
}

export function ArenaReader({ signal }: { signal: AbortSignal }) {
  const [database] = useState(() => {
    const pending = openReaderStore()
    void pending.catch(() => undefined)
    return pending
  })
  const [feed, setFeed] = useState<ArenaFeedResponse | null>(null)
  const [pendingFeed, setPendingFeed] = useState<ArenaFeedResponse | null>(null)
  const [pass, setPass] = useState<ReaderPass | null>(null)
  const [fatal, setFatal] = useState<unknown>(null)
  const [retry, setRetry] = useState(0)
  const [notice, setNotice] = useState('')
  const [filter, setFilter] = useState<FeedFilter>('unread')
  const [query, setQuery] = useState('')
  const [limit, setLimit] = useState(50)
  const [panel, setPanel] = useState<Panel>(null)
  const [wide, setWide] = useState(() => matchMedia('(min-width: 68rem)').matches)
  const [expanded, setExpanded] = useState(false)
  const [inbox, setInbox] = useState(
    () => new URL(location.href).searchParams.get('view') === 'notes',
  )
  const [editing, setEditing] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, NoteDraft>>({})
  const draftRef = useRef(drafts)
  const [savedVersions, setSavedVersions] = useState<Record<string, number>>({})
  const [saving, setSaving] = useState<Set<string>>(new Set())
  const savingRef = useRef(new Set<string>())
  const [syncTick, setSyncTick] = useState(0)
  const [readBusy, setReadBusy] = useState(false)
  const [result, setResult] = useState<ArenaReaderRenderResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [renderVersion, setRenderVersion] = useState(0)
  const [snapshot, setSnapshot] = useState<string | null>(null)
  const [undo, setUndo] = useState<{ articleId: string; title: string } | null>(null)
  const [viewport, setViewport] = useState({
    height: window.visualViewport?.height ?? window.innerHeight,
    inset: 0,
  })
  const dialog = useRef<HTMLDialogElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const pendingQuote = useRef<ArenaNoteQuote | null>(null)
  const articleRef = useRef<HTMLDivElement>(null)
  const returnFocus = useRef<HTMLElement | null>(null)
  const panelHistory = useRef(false)
  const ownerRef = useRef<string | null>(null)
  const lastRendered = useRef<string | null>(null)
  const forceRefresh = useRef(false)
  const passRef = useRef(pass)
  const feedRef = useRef(feed)
  passRef.current = pass
  feedRef.current = feed

  const setAllDrafts = useCallback((next: Record<string, NoteDraft>) => {
    draftRef.current = next
    setDrafts(next)
  }, [])

  const persist = useCallback(
    (draft: NoteDraft) => {
      void saveDraft(database, draft)
        .then(() => {
          if (
            !signal.aborted &&
            ownerRef.current === draft.subject &&
            draftRef.current[draft.note.id]?.localVersion === draft.localVersion
          ) {
            setSavedVersions(previous => ({ ...previous, [draft.note.id]: draft.localVersion }))
          }
        })
        .catch(() => {
          if (!signal.aborted)
            setNotice(
              'Local note storage failed. Keep this page open and copy your draft until it syncs.',
            )
        })
    },
    [database, signal],
  )

  const updateDraft = useCallback(
    (draft: NoteDraft) => {
      if (ownerRef.current !== draft.subject) return
      setAllDrafts({ ...draftRef.current, [draft.note.id]: draft })
      persist(draft)
    },
    [persist, setAllDrafts],
  )

  useEffect(() => {
    let active = true
    setFatal(null)
    void (async () => {
      const initial = await readerApi.feed(crypto.randomUUID(), signal)
      const restoredPass = loadPass(initial.subject)
      const response = await readerApi.feed(restoredPass.seed, signal)
      if (!active || signal.aborted) return
      if (response.subject !== initial.subject)
        throw new ReaderApiError(
          401,
          'The signed-in identity changed. Reload the reader to open that account.',
        )
      ownerRef.current = response.subject
      const [local, notes] = await Promise.all([
        loadDrafts(database, response.subject).catch(() => {
          setNotice(
            'Local note storage is unavailable. Notes can sync while connected; keep unsynced text on this page.',
          )
          return []
        }),
        readerApi.notes(null, signal),
      ])
      if (!active || signal.aborted) return
      const localDrafts = Object.fromEntries(local.map(draft => [draft.note.id, draft]))
      const merged = reconcileNotes(response.subject, localDrafts, notes)
      setAllDrafts(merged)
      setSavedVersions(Object.fromEntries(local.map(draft => [draft.note.id, draft.localVersion])))
      for (const draft of Object.values(merged)) persist(draft)
      const selected = new URL(location.href).searchParams.get('article')
      const unread = eligibleEntries(response.entries, response.readLinks, 'unread')
      const current = response.entries.some(entry => entry.articleId === selected)
        ? selected
        : unread.some(entry => entry.articleId === restoredPass.current)
          ? restoredPass.current
          : (nextEntry(unread, restoredPass.visited, null)?.articleId ?? null)
      setFeed(response)
      setPass({ ...restoredPass, current })
      if (selected && !response.entries.some(entry => entry.articleId === selected))
        setNotice('This link is no longer in the saved catalogue. Your notes remain in the inbox.')
    })().catch(error => {
      if (active && !signal.aborted) setFatal(error)
    })
    return () => {
      active = false
    }
  }, [retry, database, persist, setAllDrafts, signal])

  useEffect(() => {
    if (!feed || !pass) return
    try {
      savePass(feed.subject, pass)
    } catch {
      setNotice('This browser could not save the current reading position.')
    }
    const url = new URL(location.href)
    if (pass.current) url.searchParams.set('article', pass.current)
    else url.searchParams.delete('article')
    if (inbox) url.searchParams.set('view', 'notes')
    else url.searchParams.delete('view')
    history.replaceState(history.state, '', url)
  }, [feed?.subject, pass, inbox, panel])

  useEffect(() => {
    const media = matchMedia('(min-width: 68rem)')
    const onMedia = () => setWide(media.matches)
    const onViewport = () => {
      const height = window.visualViewport?.height ?? innerHeight
      const top = window.visualViewport?.offsetTop ?? 0
      setViewport({ height, inset: Math.max(0, innerHeight - height - top) })
    }
    media.addEventListener('change', onMedia)
    window.visualViewport?.addEventListener('resize', onViewport)
    window.visualViewport?.addEventListener('scroll', onViewport)
    window.addEventListener('resize', onViewport)
    return () => {
      media.removeEventListener('change', onMedia)
      window.visualViewport?.removeEventListener('resize', onViewport)
      window.visualViewport?.removeEventListener('scroll', onViewport)
      window.removeEventListener('resize', onViewport)
    }
  }, [])

  const closePanel = useCallback(() => {
    setSyncTick(value => value + 1)
    if (panelHistory.current) history.back()
    else setPanel(null)
  }, [])

  const openPanel = useCallback((next: Exclude<Panel, null>) => {
    if (next === 'notes' && contentRef.current)
      pendingQuote.current = quoteFromSelection(contentRef.current)
    if (!panelHistory.current) {
      if (document.activeElement instanceof HTMLElement)
        returnFocus.current = document.activeElement
      history.pushState(
        { ...(isRecord(history.state) ? history.state : {}), arenaReaderPanel: true },
        '',
        location.href,
      )
      panelHistory.current = true
    }
    setPanel(next)
  }, [])

  useEffect(() => {
    const onPop = (event: PopStateEvent) => {
      if (!panelHistory.current || location.pathname.replace(/\/$/, '') !== '/arena/feed') return
      event.stopImmediatePropagation()
      panelHistory.current = false
      setPanel(null)
      setSyncTick(value => value + 1)
    }
    window.addEventListener('popstate', onPop, true)
    return () => {
      window.removeEventListener('popstate', onPop, true)
    }
  }, [])

  useEffect(() => {
    const element = dialog.current
    if (!element) return
    const focused = document.activeElement
    if (element.open) element.close()
    if (panel) {
      if (wide) element.show()
      else element.showModal()
      if (focused instanceof HTMLElement && element.contains(focused))
        focused.focus({ preventScroll: true })
    } else returnFocus.current?.focus({ preventScroll: true })
    if (!panel || wide) return
    const original = document.documentElement.style.overflow
    document.documentElement.style.overflow = 'hidden'
    return () => {
      document.documentElement.style.overflow = original
    }
  }, [panel, wide])

  useEffect(() => {
    if (!feed) return
    let running = false
    const refresh = async () => {
      const currentPass = passRef.current
      if (running || !currentPass || signal.aborted) return
      running = true
      try {
        const fresh = await readerApi.feed(currentPass.seed, signal)
        if (fresh.subject !== ownerRef.current) {
          ownerRef.current = null
          setAllDrafts({})
          setFeed(null)
          setPass(null)
          setResult(null)
          setFatal(
            new ReaderApiError(
              401,
              'The signed-in identity changed. Reload the reader to open that account.',
            ),
          )
          return
        }
        if (fresh.revision !== feedRef.current?.revision) {
          setPendingFeed(fresh)
          setFeed(
            current =>
              current && {
                ...current,
                readLinks: mergeReadLinks(current.readLinks, fresh.readLinks),
              },
          )
        } else
          setFeed(current => ({
            ...fresh,
            readLinks: mergeReadLinks(current?.readLinks ?? [], fresh.readLinks),
          }))
        const notes = await readerApi.notes(null, signal)
        if (signal.aborted || ownerRef.current !== fresh.subject) return
        const merged = reconcileNotes(fresh.subject, draftRef.current, notes)
        setAllDrafts(merged)
        for (const draft of Object.values(merged)) persist(draft)
        setSyncTick(value => value + 1)
      } catch (error) {
        if (!signal.aborted) setNotice(errorMessage(error))
      } finally {
        running = false
      }
    }
    const onFocus = () => {
      void refresh()
    }
    window.addEventListener('focus', onFocus)
    window.addEventListener('online', onFocus)
    return () => {
      window.removeEventListener('focus', onFocus)
      window.removeEventListener('online', onFocus)
    }
  }, [feed?.subject, persist, setAllDrafts, signal])

  useEffect(() => {
    if (!feed) return
    const timers: number[] = []
    for (const draft of Object.values(drafts)) {
      if (
        !draft.dirty ||
        draft.conflict ||
        !draft.note.body.trim() ||
        draft.note.deletedAt !== null ||
        savingRef.current.has(draft.note.id)
      )
        continue
      const timer = window.setTimeout(() => {
        const current = draftRef.current[draft.note.id]
        if (
          !current ||
          current.localVersion !== draft.localVersion ||
          ownerRef.current !== draft.subject
        )
          return
        savingRef.current.add(draft.note.id)
        setSaving(new Set(savingRef.current))
        let acknowledged = false
        void readerApi
          .save(draft.note, draft.ready, draft.subject, signal)
          .then(note => {
            const latest = draftRef.current[draft.note.id]
            if (signal.aborted || ownerRef.current !== draft.subject || !latest) return
            updateDraft(acknowledgeDraft(latest, draft, note))
            acknowledged = true
          })
          .catch(error => {
            if (signal.aborted || ownerRef.current !== draft.subject) return
            const latest = draftRef.current[draft.note.id]
            if (
              error instanceof ReaderApiError &&
              error.status === 409 &&
              isNote(error.current) &&
              latest
            )
              updateDraft({ ...latest, conflict: error.current })
            else setNotice(`${errorMessage(error)} Unsynced notes remain on this device.`)
          })
          .finally(() => {
            savingRef.current.delete(draft.note.id)
            if (!signal.aborted) {
              setSaving(new Set(savingRef.current))
              if (acknowledged && draftRef.current[draft.note.id]?.dirty)
                setSyncTick(value => value + 1)
            }
          })
      }, 800)
      timers.push(timer)
    }
    return () => {
      for (const timer of timers) clearTimeout(timer)
    }
  }, [drafts, feed?.subject, signal, syncTick, updateDraft])

  const selected = feed?.entries.find(entry => entry.articleId === pass?.current) ?? null
  const eligible = useMemo(
    () => eligibleEntries(feed?.entries ?? [], feed?.readLinks ?? [], filter, query),
    [feed, filter, query],
  )
  const remaining = useMemo(
    () => eligibleEntries(feed?.entries ?? [], feed?.readLinks ?? [], 'unread'),
    [feed],
  )
  const activeRead = feed?.readLinks.find(link => link.articleId === selected?.articleId)
  const artifact = result?.status === 'ready' ? result.artifact : null

  useEffect(() => {
    if (!selected || inbox) return
    const controller = new AbortController()
    const requestSignal = AbortSignal.any([signal, controller.signal])
    const changed = lastRendered.current !== selected.articleId
    const explicitRefresh = forceRefresh.current
    forceRefresh.current = false
    lastRendered.current = selected.articleId
    if (changed) setResult(null)
    setLoading(true)
    void (async () => {
      let response = snapshot
        ? await readerApi.snapshot(selected.articleId, snapshot, requestSignal)
        : await readerApi.render(selected.articleId, requestSignal, explicitRefresh)
      let polls = 0
      while (response.status === 'pending' && polls < 24) {
        setResult(response)
        await delay(response.retryAfter, requestSignal)
        response = await readerApi.renderStatus(response.statusUrl, requestSignal)
        polls += 1
      }
      if (!requestSignal.aborted) {
        setResult(previous =>
          response.status === 'unavailable' && previous?.status === 'ready'
            ? { ...previous, warning: response.message }
            : response,
        )
        if (response.status === 'pending')
          setNotice('This copy is still being prepared. Reopen the link to check its status.')
      }
    })()
      .catch(error => {
        if (!requestSignal.aborted) setNotice(errorMessage(error))
      })
      .finally(() => {
        if (!requestSignal.aborted) setLoading(false)
      })
    return () => {
      controller.abort()
    }
  }, [selected?.articleId, inbox, renderVersion, signal, snapshot])

  useEffect(() => {
    if (!artifact || !feed || inbox) return
    let mounted = true
    let timer = 0
    const root = articleRef.current
    if (!root) return
    const fraction = () => {
      const start = root.getBoundingClientRect().top + scrollY
      return Math.min(
        1,
        Math.max(0, (scrollY - start) / Math.max(1, root.scrollHeight - innerHeight)),
      )
    }
    const save = () => {
      void savePosition(
        database,
        feed.subject,
        artifact.articleId,
        artifact.fingerprint,
        fraction(),
      ).catch(() => undefined)
    }
    const onScroll = () => {
      clearTimeout(timer)
      timer = window.setTimeout(save, 500)
    }
    void loadPosition(database, feed.subject, artifact.articleId, artifact.fingerprint)
      .then(position => {
        if (!mounted) return
        const top = root.getBoundingClientRect().top + scrollY
        window.scrollTo({
          top: top + position * Math.max(0, root.scrollHeight - innerHeight),
          behavior: 'instant',
        })
        window.addEventListener('scroll', onScroll, { passive: true })
      })
      .catch(() => {
        if (mounted) window.addEventListener('scroll', onScroll, { passive: true })
      })
    return () => {
      mounted = false
      clearTimeout(timer)
      window.removeEventListener('scroll', onScroll)
      save()
    }
  }, [artifact?.snapshotId, database, feed?.subject, inbox])

  useEffect(
    () => () => {
      void database.then(db => db.close()).catch(() => undefined)
    },
    [database],
  )

  function choose(entry: ArenaFeedEntry) {
    setSyncTick(value => value + 1)
    setSnapshot(null)
    pendingQuote.current = null
    setEditing(null)
    setInbox(false)
    setPass(current => current && { ...current, current: entry.articleId })
    if (panel === 'queue' || (!wide && panel)) closePanel()
  }

  function advance() {
    if (!pass || !selected) return
    const visited = [...new Set([...pass.visited, selected.articleId])]
    const next = nextEntry(eligible, visited, selected.articleId)
    setPass({ ...pass, visited, current: next?.articleId ?? null })
    setSnapshot(null)
    setEditing(null)
    setSyncTick(value => value + 1)
  }

  async function markRead(entry: ArenaFeedEntry, read: boolean, goNext: boolean) {
    if (!feed || readBusy) return
    const subject = feed.subject
    setReadBusy(true)
    try {
      const revision =
        feed.readLinks.find(link => link.articleId === entry.articleId)?.revision ?? 0
      const readLink = await readerApi.read(entry.articleId, read, revision, subject, signal)
      if (signal.aborted || ownerRef.current !== subject) return
      setFeed(
        current =>
          current && { ...current, readLinks: mergeReadLinks(current.readLinks, [readLink]) },
      )
      if (read) setUndo({ articleId: entry.articleId, title: entry.title })
      else {
        setUndo(null)
        setPass(
          current =>
            current && {
              ...current,
              visited: current.visited.filter(id => id !== entry.articleId),
            },
        )
      }
      if (goNext) advance()
    } catch (error) {
      if (error instanceof ReaderApiError && error.status === 409 && isReadLink(error.current)) {
        const currentLink = error.current
        setFeed(
          current =>
            current && {
              ...current,
              readLinks: [
                ...current.readLinks.filter(link => link.articleId !== entry.articleId),
                currentLink,
              ],
            },
        )
      }
      if (!signal.aborted) setNotice(errorMessage(error))
    } finally {
      if (!signal.aborted) setReadBusy(false)
    }
  }

  function addNote() {
    if (!feed || !selected) return
    const now = Date.now()
    const quote =
      (contentRef.current ? quoteFromSelection(contentRef.current) : null) ?? pendingQuote.current
    pendingQuote.current = null
    const occurrence = selected.occurrences[0]
    const note: ArenaNote = {
      id: crypto.randomUUID(),
      articleId: selected.articleId,
      sourceUrl: selected.sourceUrl,
      body: '',
      snapshotId: artifact?.snapshotId ?? null,
      quote,
      occurrence: occurrence
        ? { channelSlug: occurrence.channelSlug, blockId: occurrence.blockId }
        : null,
      createdAt: now,
      updatedAt: now,
      revision: 0,
      readyRevision: null,
      exportedRevision: null,
      exportReceipt: null,
      deletedAt: null,
    }
    const draft = { ...draftFromNote(feed.subject, note), dirty: true, localVersion: 1 }
    updateDraft(draft)
    setEditing(note.id)
    openPanel('notes')
  }

  async function deleteNote(draft: NoteDraft) {
    if (savingRef.current.has(draft.note.id)) {
      setNotice('Wait for the current save before deleting this note.')
      return
    }
    try {
      if (draft.note.revision > 0) await readerApi.delete(draft.note, draft.subject, signal)
      if (signal.aborted || ownerRef.current !== draft.subject) return
      const remaining = { ...draftRef.current }
      delete remaining[draft.note.id]
      setAllDrafts(remaining)
      await removeDraft(database, draft.subject, draft.note.id)
      setEditing(null)
    } catch (error) {
      if (!signal.aborted) setNotice(errorMessage(error))
    }
  }

  function resolveConflict(draft: NoteDraft, choice: 'remote' | 'copy') {
    if (!draft.conflict) return
    if (choice === 'copy') {
      const copy = {
        ...draft,
        note: {
          ...draft.note,
          id: crypto.randomUUID(),
          revision: 0,
          readyRevision: null,
          exportedRevision: null,
          exportReceipt: null,
          deletedAt: null,
        },
        conflict: null,
        dirty: true,
        ready: false,
      }
      updateDraft(copy)
      setEditing(copy.note.id)
    }
    updateDraft(draftFromNote(draft.subject, draft.conflict))
  }

  const status = (draft: NoteDraft): DraftStatus =>
    draft.conflict
      ? 'conflict'
      : saving.has(draft.note.id)
        ? 'saving'
        : !draft.dirty
          ? 'synced'
          : savedVersions[draft.note.id] === draft.localVersion
            ? 'device'
            : 'unsaved'
  const visibleDrafts = Object.values(drafts)
    .filter(draft => draft.note.deletedAt === null || draft.dirty)
    .sort((left, right) => right.note.updatedAt - left.note.updatedAt)
  const selectedDrafts = visibleDrafts.filter(draft => draft.note.articleId === selected?.articleId)

  const notesProps = {
    drafts: inbox ? visibleDrafts : selectedDrafts,
    entries: feed?.entries ?? [],
    article: selected,
    editing,
    status,
    onAdd: addNote,
    onEdit: setEditing,
    onChange: (draft: NoteDraft, body: string) => updateDraft(editDraft(draft, body)),
    onReady: (draft: NoteDraft) =>
      updateDraft({
        ...draft,
        dirty: true,
        ready: !draft.ready,
        localVersion: draft.localVersion + 1,
      }),
    onRetry: () => setSyncTick(value => value + 1),
    onDelete: (draft: NoteDraft) => {
      void deleteNote(draft)
    },
    onResolve: resolveConflict,
    onOpenArticle: (id: string) => {
      const entry = feed?.entries.find(entry => entry.articleId === id)
      if (entry) choose(entry)
    },
    onOpenSnapshot: (note: ArenaNote) => {
      const entry = feed?.entries.find(entry => entry.articleId === note.articleId)
      if (entry) {
        choose(entry)
        setSnapshot(note.snapshotId)
        if (wide && panel) closePanel()
      }
    },
    snapshotId: artifact?.snapshotId ?? null,
    inbox,
  }

  async function shuffle() {
    if (!feed) return
    const seed = crypto.randomUUID()
    try {
      const response = await readerApi.feed(seed, signal)
      if (response.subject !== ownerRef.current)
        throw new Error('The signed-in identity changed. Reload the reader.')
      setFeed(response)
      setPendingFeed(null)
      const next = eligibleEntries(response.entries, response.readLinks, filter, query)[0]
      setPass({ seed, current: next?.articleId ?? null, visited: [] })
      setSnapshot(null)
      setInbox(false)
      if (panel) closePanel()
    } catch (error) {
      if (!signal.aborted) setNotice(errorMessage(error))
    }
  }

  async function exportNotes() {
    try {
      const notes = await readerApi.notes(null, signal, true)
      const blob = new Blob(
        [
          JSON.stringify(
            { schemaVersion: 1, exportedAt: new Date().toISOString(), notes },
            null,
            2,
          ),
        ],
        { type: 'application/json' },
      )
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `arena-notes-${new Date().toISOString().slice(0, 10)}.json`
      link.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setNotice(
        `Exported ${notes.length} ready notes. Backfill receipts are recorded by the later Markdown workflow.`,
      )
    } catch (error) {
      if (!signal.aborted) setNotice(errorMessage(error))
    }
  }

  return (
    <div class="arena-reader" data-panel={panel ?? 'closed'} data-wide={wide}>
      <header class="arena-reader-header">
        <div>
          <a href="/arena" class="internal">
            Arena
          </a>
          <span aria-hidden="true"> / </span>
          <span>Reader</span>
        </div>
        <button
          type="button"
          aria-pressed={inbox}
          onClick={() => {
            setInbox(!inbox)
            setEditing(null)
            if (panel) closePanel()
          }}
        >
          {inbox ? 'Back to reading' : 'Notes inbox'}
        </button>
      </header>
      {notice && (
        <div class="arena-reader-notice" role="status">
          <span>{notice}</span>
          <button type="button" aria-label="Dismiss message" onClick={() => setNotice('')}>
            ×
          </button>
        </div>
      )}
      {pendingFeed && (
        <div class="arena-reader-notice">
          <span>The saved catalogue has changed.</span>
          <button
            type="button"
            onClick={() => {
              setFeed(current => ({
                ...pendingFeed,
                readLinks: current?.readLinks ?? pendingFeed.readLinks,
              }))
              setPendingFeed(null)
              setPass(current => current && { ...current, visited: [] })
              setNotice('Queue refreshed. New Later links are first in the next pass.')
            }}
          >
            Refresh queue
          </button>
        </div>
      )}
      {undo && feed && (
        <div class="arena-reader-undo" role="status">
          <span>Marked read: {undo.title}</span>
          <button
            type="button"
            disabled={readBusy}
            onClick={() => {
              const entry = feed.entries.find(entry => entry.articleId === undo.articleId)
              if (entry) void markRead(entry, false, false)
            }}
          >
            Undo
          </button>
        </div>
      )}
      {fatal ? (
        <div class="arena-reader-empty" role="alert">
          <h1>
            {fatal instanceof ReaderApiError && fatal.status === 401
              ? 'Sign in to your reader'
              : 'Reader unavailable'}
          </h1>
          <p>{errorMessage(fatal)}</p>
          {fatal instanceof ReaderApiError && fatal.loginUrl && (
            <a href={safeHref(fatal.loginUrl)} data-router-ignore>
              Sign in
            </a>
          )}
          <button type="button" onClick={() => setRetry(value => value + 1)}>
            Retry
          </button>
        </div>
      ) : !feed ? (
        <div class="arena-reader-empty" role="status">
          <h1>Your reading queue</h1>
          <p>Loading your saved links and notes…</p>
        </div>
      ) : (
        <>
          <div class="arena-reader-layout">
            <main class="arena-reader-main" ref={articleRef}>
              {inbox ? (
                <section class="arena-reader-inbox">
                  <header>
                    <h1>Notes inbox</h1>
                    <p>
                      Keep drafts here. Mark a revision ready when you want to bring it back into
                      your Garden.
                    </p>
                    <button
                      type="button"
                      onClick={() => {
                        void exportNotes()
                      }}
                    >
                      Export ready notes
                    </button>
                  </header>
                  <NotesPanel {...notesProps} />
                </section>
              ) : selected ? (
                <>
                  <ArticleContent
                    entry={selected}
                    result={result}
                    loading={loading}
                    contentRef={contentRef}
                    onRetry={() => setRenderVersion(value => value + 1)}
                  />
                  <div class="arena-reader-article-actions">
                    <button type="button" onClick={addNote}>
                      Add note from selection
                    </button>
                    <button
                      type="button"
                      disabled={loading || !artifact}
                      onClick={() => {
                        setSnapshot(null)
                        forceRefresh.current = true
                        setRenderVersion(value => value + 1)
                      }}
                    >
                      Refresh from source
                    </button>
                    {activeRead?.readAt != null && (
                      <button
                        type="button"
                        disabled={readBusy}
                        onClick={() => {
                          void markRead(selected, false, false)
                        }}
                      >
                        Mark unread
                      </button>
                    )}
                  </div>
                </>
              ) : (
                <div class="arena-reader-empty">
                  <h1>
                    {remaining.length ? 'You reached the end of this pass' : 'You’re all caught up'}
                  </h1>
                  <p>
                    {remaining.length
                      ? `${remaining.length.toLocaleString()} unread links remain. A new shuffle brings skipped links back.`
                      : 'Read links and your notes are still available in the queue.'}
                  </p>
                  <button
                    type="button"
                    onClick={() => {
                      void shuffle()
                    }}
                  >
                    Start another pass
                  </button>
                </div>
              )}
            </main>
            <dialog
              ref={dialog}
              class="arena-reader-panel"
              aria-labelledby="arena-reader-panel-title"
              data-editing={Boolean(editing) || expanded}
              style={{
                '--arena-visible-height': `${viewport.height}px`,
                '--arena-keyboard-inset': `${viewport.inset}px`,
              }}
              onCancel={event => {
                event.preventDefault()
                closePanel()
              }}
              onClick={event => {
                if (event.target === event.currentTarget && !wide) closePanel()
              }}
            >
              <div class="arena-reader-panel-inner">
                <div
                  class="arena-reader-drawer-handle"
                  aria-hidden="true"
                  onPointerDown={event => {
                    event.currentTarget.dataset.startY = String(event.clientY)
                    event.currentTarget.setPointerCapture(event.pointerId)
                  }}
                  onPointerUp={event => {
                    const start = Number(event.currentTarget.dataset.startY)
                    const movement = event.clientY - start
                    if (movement > 70) closePanel()
                    if (movement < -50) setExpanded(true)
                  }}
                />
                <header class="arena-reader-panel-header">
                  <h2 id="arena-reader-panel-title">
                    {panel === 'queue' ? 'Reading queue' : 'Notes'}
                  </h2>
                  <div>
                    <button
                      type="button"
                      class="arena-reader-expand"
                      aria-pressed={expanded}
                      onClick={() => setExpanded(!expanded)}
                    >
                      {expanded ? 'Collapse' : 'Expand'}
                    </button>
                    <button
                      type="button"
                      aria-label={`Close ${panel ?? 'panel'}`}
                      onClick={closePanel}
                    >
                      Close
                    </button>
                  </div>
                </header>
                {panel === 'queue' ? (
                  <div class="arena-reader-queue">
                    <p class="arena-reader-status">
                      {remaining.filter(entry => entry.later).length.toLocaleString()} Later ·{' '}
                      {remaining.length.toLocaleString()} unread
                    </p>
                    <div class="arena-reader-queue-controls">
                      <label>
                        Show
                        <select
                          value={filter}
                          onChange={event => {
                            const value = event.currentTarget.value
                            if (value === 'unread' || value === 'read' || value === 'all') {
                              setFilter(value)
                              setLimit(50)
                            }
                          }}
                        >
                          <option value="unread">Unread</option>
                          <option value="read">Read</option>
                          <option value="all">All saved links</option>
                        </select>
                      </label>
                      <button
                        type="button"
                        onClick={() => {
                          void shuffle()
                        }}
                      >
                        Shuffle
                      </button>
                    </div>
                    <label class="arena-reader-search">
                      Search saved links
                      <input
                        type="search"
                        value={query}
                        onInput={event => {
                          setQuery(event.currentTarget.value)
                          setLimit(50)
                        }}
                        placeholder="Title, source, or channel"
                      />
                    </label>
                    <ol>
                      {eligible.slice(0, limit).map(entry => (
                        <li key={entry.articleId}>
                          <button
                            type="button"
                            aria-current={
                              entry.articleId === selected?.articleId ? 'page' : undefined
                            }
                            onClick={() => choose(entry)}
                          >
                            <span class="arena-reader-queue-title">{entry.title}</span>
                            <span class="arena-reader-queue-meta">
                              {entry.later && <span>Later · </span>}
                              {entry.occurrences[0]?.channelName}
                              {pass?.visited.includes(entry.articleId) && ' · skipped this pass'}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ol>
                    {eligible.length === 0 && <p>No links match these filters.</p>}
                    {eligible.length > limit && (
                      <button type="button" onClick={() => setLimit(value => value + 50)}>
                        Show 50 more ({eligible.length.toLocaleString()} total)
                      </button>
                    )}
                  </div>
                ) : (
                  <NotesPanel {...notesProps} inbox={false} drafts={selectedDrafts} />
                )}
              </div>
            </dialog>
          </div>
          {!inbox && (
            <nav class="arena-reader-bottom-bar" aria-label="Reader actions">
              <button
                type="button"
                aria-expanded={panel === 'queue'}
                aria-controls="arena-reader-panel-title"
                onClick={() => (panel === 'queue' ? closePanel() : openPanel('queue'))}
              >
                Queue<span class="arena-reader-counter">{remaining.length.toLocaleString()}</span>
              </button>
              <button
                type="button"
                aria-expanded={panel === 'notes'}
                aria-controls="arena-reader-panel-title"
                onClick={() => (panel === 'notes' ? closePanel() : openPanel('notes'))}
              >
                Notes<span class="arena-reader-counter">{selectedDrafts.length}</span>
              </button>
              <button
                type="button"
                disabled={!selected || readBusy}
                onClick={() => {
                  if (selected) void markRead(selected, true, true)
                }}
              >
                {readBusy ? 'Saving…' : activeRead?.readAt != null ? 'Read · Next' : 'Mark read'}
              </button>
              <button
                type="button"
                disabled={!selected}
                onClick={advance}
                aria-label="Next link, keep current link unread"
              >
                Next →
              </button>
            </nav>
          )}
        </>
      )}
    </div>
  )
}
