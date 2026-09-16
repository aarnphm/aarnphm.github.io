import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'preact/hooks'
import type { ArenaFeedEntry } from '../../util/arena-feed'
import type {
  ArenaFeedResponse,
  ArenaNote,
  ArenaNoteQuote,
  ArenaReaderRenderResult,
} from '../../util/arena-reader'
import { isNote, isReadLink, isRecord, ReaderApiError, readerApi } from './api'
import { ArticleContent, safeHref } from './content'
import { ReaderFilter } from './filter'
import { ReaderLoading } from './loading'
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
type BackgroundNotice = 'storage' | 'position' | 'refresh' | 'sync'

const queueFilters: { value: FeedFilter; label: string }[] = [
  { value: 'unread', label: 'unread' },
  { value: 'read', label: 'read' },
  { value: 'all', label: 'all saved links' },
]

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
  const queueRef = useRef<HTMLDivElement>(null)
  const returnFocus = useRef<HTMLElement | null>(null)
  const panelHistory = useRef(false)
  const ownerRef = useRef<string | null>(null)
  const lastRendered = useRef<string | null>(null)
  const forceRefresh = useRef(false)
  const passRef = useRef(pass)
  const feedRef = useRef(feed)
  const lastBackgroundNotice = useRef(new Map<BackgroundNotice, number>())
  passRef.current = pass
  feedRef.current = feed

  const notify = useCallback(
    (message: string, background?: BackgroundNotice) => {
      if (signal.aborted) return
      if (background) {
        const now = Date.now()
        const last = lastBackgroundNotice.current.get(background)
        if (last !== undefined && now - last < 30_000) return
        lastBackgroundNotice.current.set(background, now)
      }
      const event: CustomEventMap['toast'] = new CustomEvent('toast', {
        detail: {
          message,
          durationMs: 6000,
          containerHost: dialog.current?.open ? dialog.current : undefined,
        },
      })
      document.dispatchEvent(event)
    },
    [signal],
  )

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
          notify(
            'Local note storage failed. Keep this page open and copy your draft until it syncs.',
            'storage',
          )
        })
    },
    [database, notify, signal],
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
          if (active)
            notify(
              'Local note storage is unavailable. Notes can sync while connected; keep unsynced text on this page.',
              'storage',
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
        notify('This link is outside the reading queue. Your notes remain in the inbox.')
    })().catch(error => {
      if (active && !signal.aborted) setFatal(error)
    })
    return () => {
      active = false
    }
  }, [retry, database, notify, persist, setAllDrafts, signal])

  useEffect(() => {
    if (!feed || !pass) return
    try {
      savePass(feed.subject, pass)
    } catch {
      notify('This browser could not save the current reading position.', 'position')
    }
    const url = new URL(location.href)
    if (pass.current) url.searchParams.set('article', pass.current)
    else url.searchParams.delete('article')
    if (inbox) url.searchParams.set('view', 'notes')
    else url.searchParams.delete('view')
    history.replaceState(history.state, '', url)
  }, [feed?.subject, pass, inbox, panel, notify])

  useLayoutEffect(() => {
    queueRef.current?.scrollTo({ top: 0 })
  }, [query, filter, pass?.seed])

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
    const onPop = (event: CustomEventMap['beforepopstate']) => {
      if (!panelHistory.current || location.pathname.replace(/\/$/, '') !== '/arena/feed') return
      event.preventDefault()
      panelHistory.current = false
      setPanel(null)
      setSyncTick(value => value + 1)
    }
    document.addEventListener('beforepopstate', onPop)
    return () => {
      document.removeEventListener('beforepopstate', onPop)
    }
  }, [])

  // Keep the native dialog and its grid columns in the same frame.
  useLayoutEffect(() => {
    const element = dialog.current
    if (!element) return
    const focused = document.activeElement
    const modal = Boolean(panel && !wide)
    if (element.open && (!panel || element.matches(':modal') !== modal)) element.close()
    if (panel) {
      if (!element.open) {
        if (wide) element.show()
        else element.showModal()
      }
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
        notify(errorMessage(error), 'refresh')
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
  }, [feed?.subject, notify, persist, setAllDrafts, signal])

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
            else notify(`${errorMessage(error)} Unsynced notes remain on this device.`, 'sync')
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
  }, [drafts, feed?.subject, notify, signal, syncTick, updateDraft])

  const selected = feed?.entries.find(entry => entry.articleId === pass?.current) ?? null
  const originalUrl = selected ? safeHref(selected.sourceUrl) : undefined
  const eligible = useMemo(
    () => eligibleEntries(feed?.entries ?? [], feed?.readLinks ?? [], filter, query),
    [feed, filter, query],
  )
  const remaining = useMemo(
    () => eligibleEntries(feed?.entries ?? [], feed?.readLinks ?? [], 'unread'),
    [feed],
  )
  const activeRead = feed?.readLinks.find(link => link.articleId === selected?.articleId)
  const isRead = activeRead?.readAt != null
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
    if (explicitRefresh) notify('refreshing from the source…')
    void (async () => {
      let response = snapshot
        ? await readerApi.snapshot(selected.articleId, snapshot, requestSignal)
        : await readerApi.render(selected.articleId, requestSignal, explicitRefresh)
      let polls = 0
      while (response.status === 'pending' && polls < 24) {
        const pending = response
        setResult(previous => (previous?.status === 'ready' ? previous : pending))
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
          notify('This copy is still being prepared. Reopen the link to check its status.')
      }
    })()
      .catch(error => {
        if (!requestSignal.aborted) notify(errorMessage(error))
      })
      .finally(() => {
        if (!requestSignal.aborted) setLoading(false)
      })
    return () => {
      controller.abort()
    }
  }, [selected?.articleId, inbox, notify, renderVersion, signal, snapshot])

  useEffect(() => {
    if (!artifact || !feed || inbox) return
    let mounted = true
    let timer = 0
    const root = articleRef.current
    if (!root) return
    const fraction = () =>
      Math.min(1, Math.max(0, root.scrollTop / Math.max(1, root.scrollHeight - root.clientHeight)))
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
        root.scrollTo({
          top: position * Math.max(0, root.scrollHeight - root.clientHeight),
          behavior: 'instant',
        })
        root.addEventListener('scroll', onScroll, { passive: true })
      })
      .catch(() => {
        if (mounted) root.addEventListener('scroll', onScroll, { passive: true })
      })
    return () => {
      mounted = false
      clearTimeout(timer)
      root.removeEventListener('scroll', onScroll)
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
    if (!wide && panel && panel !== 'queue') closePanel()
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
      notify(errorMessage(error))
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
      notify('Wait for the current save before deleting this note.')
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
      notify(errorMessage(error))
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
      notify(errorMessage(error))
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
      notify(
        `Exported ${notes.length} ready notes. Backfill receipts are recorded by the later Markdown workflow.`,
      )
    } catch (error) {
      notify(errorMessage(error))
    }
  }

  return (
    <div class="arena-reader" data-panel={panel ?? 'closed'} data-wide={wide}>
      <header class="arena-reader-header">
        <div>
          <a href="/arena" class="internal">
            arena
          </a>
          <span aria-hidden="true"> / </span>
          <span>reader</span>
        </div>
        <button
          type="button"
          class={inbox ? 'arena-reader-icon-button' : undefined}
          aria-label={inbox ? 'back to reading' : 'inbox'}
          title={inbox ? 'back to reading' : 'inbox'}
          onClick={() => {
            setInbox(!inbox)
            setEditing(null)
            if (panel) closePanel()
          }}
        >
          {inbox ? (
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
              <path d="m7 3-5 5 5 5M2 8h12" />
            </svg>
          ) : (
            'inbox'
          )}
        </button>
      </header>
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
              notify('Queue refreshed. New Later links are first in the next pass.')
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
        <ReaderLoading />
      ) : (
        <div class="arena-reader-layout">
          <main class="arena-reader-main" ref={articleRef}>
            {inbox ? (
              <section class="arena-reader-inbox">
                <header>
                  <h1>notes</h1>
                </header>
                <NotesPanel {...notesProps} onExport={() => void exportNotes()} />
              </section>
            ) : selected ? (
              <ArticleContent
                entry={selected}
                result={result}
                loading={loading}
                contentRef={contentRef}
                onRetry={() => setRenderVersion(value => value + 1)}
              />
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
                <h2 id="arena-reader-panel-title">{panel === 'queue' ? 'queue' : 'notes'}</h2>
                <div>
                  <button
                    type="button"
                    class="arena-reader-expand"
                    aria-pressed={expanded}
                    onClick={() => setExpanded(!expanded)}
                  >
                    {expanded ? 'collapse' : 'expand'}
                  </button>
                  <button
                    type="button"
                    class="arena-reader-icon-button"
                    aria-label={`Close ${panel ?? 'panel'}`}
                    title={`Close ${panel ?? 'panel'}`}
                    onClick={closePanel}
                  >
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      aria-hidden="true"
                      focusable="false"
                    >
                      <path d="m6 6 12 12M6 18 18 6" />
                    </svg>
                  </button>
                </div>
              </header>
              {panel === 'queue' ? (
                <div class="arena-reader-queue">
                  <div class="arena-reader-queue-entries" ref={queueRef}>
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
                              {entry.later && <span>later · </span>}
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
                  <div class="arena-reader-queue-footer">
                    <div class="arena-reader-search">
                      <input
                        type="search"
                        aria-label="Search saved links"
                        value={query}
                        onInput={event => {
                          setQuery(event.currentTarget.value)
                          setLimit(50)
                        }}
                        placeholder="search saved links…"
                      />
                      <button
                        type="button"
                        class="arena-reader-icon-button"
                        aria-label="Shuffle queue"
                        title="Shuffle queue"
                        onClick={() => {
                          void shuffle()
                        }}
                      >
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          stroke-width="1.5"
                          stroke-linecap="round"
                          stroke-linejoin="round"
                          aria-hidden="true"
                          focusable="false"
                        >
                          <path d="m17 3 4 4-4 4M17 13l4 4-4 4M3 7h3c5 0 7 10 12 10h3M3 17h3c2 0 3.5-1.6 5-4M13 9c1.5-1.4 3-2 5-2h3" />
                        </svg>
                      </button>
                    </div>
                    <div class="arena-reader-queue-controls">
                      <ReaderFilter
                        label="show"
                        options={queueFilters}
                        value={filter}
                        onChange={value => {
                          setFilter(value)
                          setLimit(50)
                        }}
                      />
                      <p class="arena-reader-status">
                        {remaining.filter(entry => entry.later).length.toLocaleString()} later ·{' '}
                        {remaining.length.toLocaleString()} unread
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <NotesPanel {...notesProps} inbox={false} drafts={selectedDrafts} />
              )}
            </div>
          </dialog>
          {!inbox && (
            <div class="arena-reader-footer">
              <nav class="arena-reader-bottom-bar" aria-label="Reader actions">
                <button
                  type="button"
                  aria-expanded={panel === 'queue'}
                  aria-controls="arena-reader-panel-title"
                  onClick={() => (panel === 'queue' ? closePanel() : openPanel('queue'))}
                >
                  queue<span class="arena-reader-counter">{remaining.length.toLocaleString()}</span>
                </button>
                <button
                  type="button"
                  aria-expanded={panel === 'notes'}
                  aria-controls="arena-reader-panel-title"
                  onClick={() => {
                    pendingQuote.current = contentRef.current
                      ? quoteFromSelection(contentRef.current)
                      : null
                    if (pendingQuote.current) addNote()
                    else if (panel === 'notes') closePanel()
                    else openPanel('notes')
                  }}
                >
                  notes<span class="arena-reader-counter">{selectedDrafts.length}</span>
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
                  refresh
                </button>
                <button
                  type="button"
                  disabled={!selected || readBusy}
                  aria-label={isRead ? 'Mark unread' : 'Read and go to next link'}
                  onClick={() => {
                    if (selected) void markRead(selected, !isRead, !isRead)
                  }}
                >
                  {readBusy ? 'saving…' : isRead ? 'unread' : 'read'}
                </button>
                {originalUrl && (
                  <a
                    class="arena-reader-icon-button"
                    href={originalUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="open original"
                    title="open original in a new tab"
                  >
                    <svg viewBox="0 0 15 15" fill="none" aria-hidden="true" focusable="false">
                      <path
                        fill-rule="evenodd"
                        clip-rule="evenodd"
                        d="M12 13C12.5523 13 13 12.5523 13 12V3C13 2.44771 12.5523 2 12 2H3C2.44771 2 2 2.44771 2 3V6.5C2 6.77614 2.22386 7 2.5 7C2.77614 7 3 6.77614 3 6.5V3H12V12H8.5C8.22386 12 8 12.2239 8 12.5C8 12.7761 8.22386 13 8.5 13H12ZM9 6.5C9 6.5001 9 6.50021 9 6.50031V6.50035V9.5C9 9.77614 8.77614 10 8.5 10C8.22386 10 8 9.77614 8 9.5V7.70711L2.85355 12.8536C2.65829 13.0488 2.34171 13.0488 2.14645 12.8536C1.95118 12.6583 1.95118 12.3417 2.14645 12.1464L7.29289 7H5.5C5.22386 7 5 6.77614 5 6.5C5 6.22386 5.22386 6 5.5 6H8.5C8.56779 6 8.63244 6.01349 8.69139 6.03794C8.74949 6.06198 8.80398 6.09744 8.85143 6.14433C8.94251 6.23434 8.9992 6.35909 8.99999 6.49708L8.99999 6.49738"
                        fill="currentColor"
                      />
                    </svg>
                  </a>
                )}
                <button
                  type="button"
                  class="arena-reader-icon-button"
                  disabled={!selected}
                  onClick={advance}
                  aria-label="Next link, keep current link unread"
                  title="Next link, keep current link unread"
                >
                  <span aria-hidden="true">→</span>
                </button>
              </nav>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
