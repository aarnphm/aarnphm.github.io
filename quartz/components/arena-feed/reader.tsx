import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'preact/hooks'
import type {
  ArenaFeedResponse,
  ArenaNote,
  ArenaNoteQuote,
  ArenaReaderRenderResult,
} from '../../util/arena-reader'
import type { ToastShowOptions } from '../scripts/toast'
import {
  arenaFeedSourceNames,
  orderArenaFeedEntries,
  type ArenaFeedEntry,
} from '../../util/arena-feed'
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
import { rangesForQuotes } from './quote-highlights'
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
const noteSyncIntervalMs = 5_000

const queueFilters: { value: FeedFilter; label: string }[] = [
  { value: 'unread', label: 'unread' },
  { value: 'read', label: 'read' },
  { value: 'all', label: 'all' },
  { value: 'curius', label: 'curius' },
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
  const [queueFillsViewport, setQueueFillsViewport] = useState(false)
  const [panel, setPanel] = useState<Panel>(null)
  const lastPanel = useRef<Exclude<Panel, null>>('queue')
  const panelContent = panel ?? lastPanel.current
  const [wide, setWide] = useState(() => matchMedia('(min-width: 68rem)').matches)
  const [expanded, setExpanded] = useState(false)
  const [inbox, setInbox] = useState(
    () => new URL(location.href).searchParams.get('view') === 'notes',
  )
  const [editing, setEditing] = useState<string | null>(null)
  const [selectionAction, setSelectionAction] = useState<{
    quote: ArenaNoteQuote
    left: number
    top: number
  } | null>(null)
  const [drafts, setDrafts] = useState<Record<string, NoteDraft>>({})
  const draftRef = useRef(drafts)
  const [localFailures, setLocalFailures] = useState<Set<string>>(new Set())
  const [syncFailures, setSyncFailures] = useState<Set<string>>(new Set())
  const savingTasks = useRef(new Map<string, Promise<void>>())
  const deleting = useRef(new Set<string>())
  const syncNowRef = useRef<() => void>(() => undefined)
  const [readBusy, setReadBusy] = useState(false)
  const [result, setResult] = useState<ArenaReaderRenderResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [renderVersion, setRenderVersion] = useState(0)
  const [snapshot, setSnapshot] = useState<string | null>(null)
  const markReadRef = useRef<
    (entry: ArenaFeedEntry, read: boolean, goNext: boolean) => Promise<void>
  >(async () => undefined)
  const [viewport, setViewport] = useState({
    height: window.visualViewport?.height ?? window.innerHeight,
    inset: 0,
  })
  const readerRef = useRef<HTMLDivElement>(null)
  const dialog = useRef<HTMLDialogElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const pendingQuote = useRef<ArenaNoteQuote | null>(null)
  const articleRef = useRef<HTMLDivElement>(null)
  const queueRef = useRef<HTMLDivElement>(null)
  const queueListRef = useRef<HTMLOListElement>(null)
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
    (message: string, background?: BackgroundNotice, action?: ToastShowOptions['action']) => {
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
          containerHost: dialog.current?.matches(':modal') ? dialog.current : undefined,
          action,
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
          )
            setLocalFailures(previous => {
              if (!previous.has(draft.note.id)) return previous
              const next = new Set(previous)
              next.delete(draft.note.id)
              return next
            })
        })
        .catch(() => {
          if (
            !signal.aborted &&
            ownerRef.current === draft.subject &&
            draftRef.current[draft.note.id]?.localVersion === draft.localVersion
          )
            setLocalFailures(previous => new Set(previous).add(draft.note.id))
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

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const root = readerRef.current
      if (
        !root ||
        signal.aborted ||
        event.defaultPrevented ||
        event.isComposing ||
        event.repeat ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey
      )
        return

      const target = event.target instanceof Element ? event.target : document.activeElement
      if (target && target !== document.body && !root.contains(target)) return
      if (
        (target instanceof HTMLElement && target.isContentEditable) ||
        target?.closest(
          'input, textarea, select, [role="textbox"], [role="combobox"], [role="listbox"], [role="menu"], .cm-editor',
        ) ||
        document.querySelector(
          '.link-hint-marker, #palette-container.active, #shortcut-container.active, .search-container.active',
        )
      )
        return
      const modal = document.querySelector(':modal')
      if (modal && modal !== dialog.current) return

      const key = event.key.toLowerCase()
      const shortcut = event.shiftKey
        ? key === 'n'
          ? 'Shift+N'
          : key === 'o'
            ? 'Shift+O'
            : null
        : key === 'q' || key === 'n' || key === 'r'
          ? key
          : null
      if (!shortcut) return
      const control = (modal ?? root).querySelector<HTMLButtonElement | HTMLAnchorElement>(
        `[aria-keyshortcuts="${shortcut}"]`,
      )
      if (!control || (control instanceof HTMLButtonElement && control.disabled)) return
      event.preventDefault()
      control.click()
    }

    // Let focused controls and the site's keyboard handlers consume their keys first.
    window.addEventListener('keydown', onKey, { signal })
    return () => window.removeEventListener('keydown', onKey)
  }, [signal])

  const closePanel = useCallback(() => {
    syncNowRef.current()
    if (panelHistory.current) history.back()
    else setPanel(null)
  }, [])

  const openPanel = useCallback((next: Exclude<Panel, null>) => {
    lastPanel.current = next
    if (next === 'queue') {
      const seed = crypto.randomUUID()
      setPass(current => (current ? { ...current, seed } : current))
    }
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
      syncNowRef.current()
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

  useLayoutEffect(() => {
    const queue = queueRef.current
    const list = queueListRef.current
    if (panel !== 'queue' || !queue || !list) return

    const update = () => {
      setQueueFillsViewport(queue.clientHeight > 0 && list.offsetHeight >= queue.clientHeight)
    }
    const observer = new ResizeObserver(update)
    observer.observe(queue)
    observer.observe(list, { box: 'border-box' })
    update()
    return () => observer.disconnect()
  }, [panel, feed?.subject, fatal])

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
        syncNowRef.current()
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

  const syncDirtyDrafts = useCallback(() => {
    if (signal.aborted || !feedRef.current) return
    for (const draft of Object.values(draftRef.current)) {
      if (
        ownerRef.current !== draft.subject ||
        !draft.dirty ||
        draft.conflict ||
        !draft.note.body.trim() ||
        draft.note.deletedAt !== null ||
        savingTasks.current.has(draft.note.id) ||
        deleting.current.has(draft.note.id)
      )
        continue
      const task = readerApi
        .save(draft.note, draft.ready, draft.subject, signal)
        .then(note => {
          const latest = draftRef.current[draft.note.id]
          if (signal.aborted || ownerRef.current !== draft.subject || !latest) return
          updateDraft(acknowledgeDraft(latest, draft, note))
          setSyncFailures(previous => {
            if (!previous.has(draft.note.id)) return previous
            const next = new Set(previous)
            next.delete(draft.note.id)
            return next
          })
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
          else {
            setSyncFailures(previous =>
              previous.has(draft.note.id) ? previous : new Set(previous).add(draft.note.id),
            )
            notify(`${errorMessage(error)} Unsynced notes remain on this device.`, 'sync')
          }
        })
        .finally(() => {
          savingTasks.current.delete(draft.note.id)
        })
      savingTasks.current.set(draft.note.id, task)
    }
  }, [notify, signal, updateDraft])
  syncNowRef.current = syncDirtyDrafts

  useEffect(() => {
    if (!feed) return
    syncDirtyDrafts()
    // Keep device recovery immediate while server saves sample the latest draft on a fixed cadence.
    const timer = window.setInterval(syncDirtyDrafts, noteSyncIntervalMs)
    return () => {
      clearInterval(timer)
    }
  }, [feed?.subject, syncDirtyDrafts])

  const selected = feed?.entries.find(entry => entry.articleId === pass?.current) ?? null
  const originalUrl = selected ? safeHref(selected.sourceUrl) : undefined
  const eligible = useMemo(
    () =>
      orderArenaFeedEntries(
        eligibleEntries(feed?.entries ?? [], feed?.readLinks ?? [], filter, query),
        pass?.seed ?? '',
      ),
    [feed, filter, query, pass?.seed],
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
    syncDirtyDrafts()
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
    syncDirtyDrafts()
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
      if (read) {
        notify(`Marked read: ${entry.title}`, undefined, {
          label: 'Undo',
          onClick: () => {
            const current = feedRef.current?.entries.find(
              current => current.articleId === entry.articleId,
            )
            if (current) void markReadRef.current(current, false, false)
          },
        })
      } else {
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
  markReadRef.current = markRead

  function createNote(selectedQuote: ArenaNoteQuote | null = null) {
    if (!feed || !selected) return
    const now = Date.now()
    const quote =
      selectedQuote ??
      (contentRef.current ? quoteFromSelection(contentRef.current) : null) ??
      pendingQuote.current
    pendingQuote.current = null
    setSelectionAction(null)
    window.getSelection()?.removeAllRanges()
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

  function addNote() {
    createNote()
  }

  async function deleteNote(draft: NoteDraft) {
    if (deleting.current.has(draft.note.id)) return
    deleting.current.add(draft.note.id)
    try {
      await savingTasks.current.get(draft.note.id)
      const latest = draftRef.current[draft.note.id]
      if (!latest || ownerRef.current !== draft.subject) return
      if (latest.note.revision > 0) await readerApi.delete(latest.note, latest.subject, signal)
      if (signal.aborted || ownerRef.current !== draft.subject) return
      const remaining = { ...draftRef.current }
      delete remaining[draft.note.id]
      setAllDrafts(remaining)
      await removeDraft(database, draft.subject, draft.note.id)
      setLocalFailures(previous => {
        const next = new Set(previous)
        next.delete(draft.note.id)
        return next
      })
      setSyncFailures(previous => {
        const next = new Set(previous)
        next.delete(draft.note.id)
        return next
      })
      setEditing(null)
    } catch (error) {
      notify(errorMessage(error))
    } finally {
      deleting.current.delete(draft.note.id)
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
      : !draft.dirty
        ? 'synced'
        : localFailures.has(draft.note.id)
          ? 'unsaved'
          : syncFailures.has(draft.note.id)
            ? 'sync-failed'
            : 'pending'
  const visibleDrafts = Object.values(drafts)
    .filter(draft => draft.note.deletedAt === null || draft.dirty)
    .sort((left, right) => right.note.updatedAt - left.note.updatedAt)
  const selectedDrafts = visibleDrafts.filter(draft => draft.note.articleId === selected?.articleId)
  const quoteKey = selectedDrafts
    .filter(draft => draft.note.quote)
    .map(draft => `${draft.note.id}:${JSON.stringify(draft.note.quote)}`)
    .join('|')

  useEffect(() => {
    const name = 'arena-reader-notes'
    if (!('highlights' in CSS) || typeof Highlight === 'undefined') return
    CSS.highlights.delete(name)
    const root = contentRef.current
    if (!artifact || inbox || !root || !quoteKey) return
    let frame = 0
    const paint = () => {
      frame = 0
      const quotes = Object.values(draftRef.current).flatMap(draft =>
        draft.note.articleId === artifact.articleId &&
        draft.note.deletedAt === null &&
        draft.note.quote
          ? [draft.note.quote]
          : [],
      )
      const ranges = rangesForQuotes(root, quotes)
      if (ranges.length) CSS.highlights.set(name, new Highlight(...ranges))
      else CSS.highlights.delete(name)
    }
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(paint)
    }
    paint()
    const observer = artifact.kind === 'pdf' ? new MutationObserver(schedule) : null
    observer?.observe(root, { childList: true, characterData: true, subtree: true })
    return () => {
      observer?.disconnect()
      if (frame) cancelAnimationFrame(frame)
      CSS.highlights.delete(name)
    }
  }, [artifact?.snapshotId, inbox, quoteKey, selected?.articleId])

  useEffect(() => {
    if (!artifact || inbox) {
      setSelectionAction(null)
      return
    }
    let frame = 0
    const update = () => {
      frame = 0
      const root = contentRef.current
      const selection = window.getSelection()
      const quote = root ? quoteFromSelection(root) : null
      const rects = selection?.rangeCount ? selection.getRangeAt(0).getClientRects() : null
      const rect = rects
        ? Array.from(rects)
            .reverse()
            .find(item => item.width && item.height)
        : null
      if (!quote || !rect || !rect.width || !rect.height) {
        setSelectionAction(null)
        return
      }
      const left = Math.min(window.innerWidth - 48, Math.max(48, rect.left + rect.width / 2))
      const top = Math.min(window.innerHeight - 40, Math.max(8, rect.bottom + 6))
      setSelectionAction(previous =>
        previous?.quote.exact === quote.exact && previous.left === left && previous.top === top
          ? previous
          : { quote, left, top },
      )
    }
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update)
    }
    const scroller = articleRef.current
    document.addEventListener('selectionchange', schedule)
    scroller?.addEventListener('scroll', schedule, { passive: true })
    window.addEventListener('resize', schedule)
    return () => {
      document.removeEventListener('selectionchange', schedule)
      scroller?.removeEventListener('scroll', schedule)
      window.removeEventListener('resize', schedule)
      if (frame) cancelAnimationFrame(frame)
    }
  }, [artifact?.snapshotId, inbox, selected?.articleId])

  const notesProps = {
    drafts: inbox ? visibleDrafts : selectedDrafts,
    entries: feed?.entries ?? [],
    article: selected,
    editing,
    status,
    onAdd: addNote,
    onEdit: setEditing,
    onChange: (draft: NoteDraft, body: string) => {
      const latest = draftRef.current[draft.note.id]
      if (latest) updateDraft(editDraft(latest, body))
    },
    onReady: (draft: NoteDraft) => {
      const latest = draftRef.current[draft.note.id]
      if (!latest) return
      updateDraft({
        ...latest,
        dirty: true,
        ready: !latest.ready,
        localVersion: latest.localVersion + 1,
      })
    },
    onRetry: syncDirtyDrafts,
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

  const readerControls = (
    <nav class="arena-reader-bottom-bar" aria-label="Reader actions">
      <button
        type="button"
        class={panel === 'queue' ? undefined : 'arena-reader-icon-button'}
        aria-label={`Queue, ${remaining.length.toLocaleString()} unread links`}
        aria-expanded={panel === 'queue'}
        aria-controls="arena-reader-panel"
        aria-keyshortcuts="q"
        title="Toggle queue (Q)"
        onClick={() => (panel === 'queue' ? closePanel() : openPanel('queue'))}
      >
        {panel === 'queue' ? (
          <>
            queue
            <span class="arena-reader-counter">{remaining.length.toLocaleString()}</span>
          </>
        ) : (
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            aria-hidden="true"
            focusable="false"
          >
            <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
          </svg>
        )}
      </button>
      <button
        type="button"
        class={panel === 'notes' ? undefined : 'arena-reader-icon-button'}
        aria-label={`Notes, ${selectedDrafts.length} for this article`}
        aria-expanded={panel === 'notes'}
        aria-controls="arena-reader-panel"
        aria-keyshortcuts="Shift+N"
        title="Toggle notes (Shift+N)"
        onClick={() => {
          pendingQuote.current = contentRef.current ? quoteFromSelection(contentRef.current) : null
          if (pendingQuote.current) addNote()
          else if (panel === 'notes') closePanel()
          else openPanel('notes')
        }}
      >
        {panel === 'notes' ? (
          <>
            notes<span class="arena-reader-counter">{selectedDrafts.length}</span>
          </>
        ) : (
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
            <path d="M14 3H5v18h14V8ZM14 3v5h5M8 12h8M8 16h6" />
          </svg>
        )}
      </button>
      <button
        type="button"
        class="arena-reader-icon-button"
        disabled={loading || !artifact}
        aria-label="Refresh saved copy"
        title="Refresh saved copy"
        onClick={() => {
          setSnapshot(null)
          forceRefresh.current = true
          setRenderVersion(value => value + 1)
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
          <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8M21 3v5h-5M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16M8 16H3v5" />
        </svg>
      </button>
      <button
        type="button"
        class="arena-reader-icon-button"
        disabled={!feed || readBusy}
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
      <button
        type="button"
        class="arena-reader-icon-button"
        disabled={!selected || readBusy}
        aria-label={
          readBusy ? 'Saving read status' : isRead ? 'Mark unread' : 'Read and go to next link'
        }
        aria-busy={readBusy}
        aria-keyshortcuts={isRead ? undefined : 'r'}
        title={isRead ? 'Mark unread' : 'Read and go to next link (R)'}
        onClick={() => {
          if (selected) void markRead(selected, !isRead, !isRead)
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
          <path d={isRead ? 'm9 6-6 6 6 6M3 12h11a6 6 0 0 1 6 6' : 'm5 12 4 4L19 6'} />
        </svg>
      </button>
      {originalUrl && (
        <a
          class="arena-reader-icon-button"
          href={originalUrl}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="open original"
          aria-keyshortcuts="Shift+O"
          title="Open original in a new tab (Shift+O)"
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
        disabled={!selected || readBusy}
        onClick={advance}
        aria-label="Next link, keep current link unread"
        aria-keyshortcuts="n"
        title="Next link, keep current link unread (N)"
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
          <path d="M5 12h14m-5-5 5 5-5 5" />
        </svg>
      </button>
    </nav>
  )

  return (
    <div class="arena-reader" ref={readerRef} data-panel={panel ?? 'closed'} data-wide={wide}>
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
          aria-label={inbox ? 'back to reading' : 'inbox'}
          title={inbox ? 'back to reading' : 'inbox'}
          onClick={() => {
            setInbox(!inbox)
            setEditing(null)
            if (panel) closePanel()
          }}
        >
          {inbox ? 'back' : 'inbox'}
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
            {selectionAction && !inbox && artifact && (
              <button
                type="button"
                class="arena-reader-selection-action"
                style={{ left: `${selectionAction.left}px`, top: `${selectionAction.top}px` }}
                aria-label="Add selected text to a new note"
                onPointerDown={event => event.preventDefault()}
                onClick={() => createNote(selectionAction.quote)}
              >
                add note
              </button>
            )}
          </main>
          <dialog
            ref={dialog}
            id="arena-reader-panel"
            class="arena-reader-panel"
            aria-label={panelContent}
            data-panel={panelContent}
            inert={!panel}
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
              {panelContent === 'queue' ? (
                <div class="arena-reader-queue">
                  <div class="arena-reader-queue-entries" ref={queueRef}>
                    <ol ref={queueListRef} data-fills-viewport={queueFillsViewport}>
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
                              {arenaFeedSourceNames(entry).join(' / ')}
                              {pass?.visited.includes(entry.articleId) && ' · skipped this pass'}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ol>
                    {eligible.length === 0 && <p>No links match these filters.</p>}
                  </div>
                  <div class="arena-reader-queue-footer">
                    <div class="arena-reader-queue-controls">
                      <input
                        class="arena-reader-search-input"
                        type="search"
                        aria-label="Search saved links"
                        value={query}
                        onInput={event => {
                          setQuery(event.currentTarget.value)
                          setLimit(50)
                        }}
                        placeholder="search saved links…"
                      />
                      <ReaderFilter
                        label="Filter links"
                        hideLabel
                        options={queueFilters}
                        value={filter}
                        onChange={value => {
                          setFilter(value)
                          setLimit(50)
                        }}
                      />
                      {eligible.length > limit && (
                        <button
                          type="button"
                          class="arena-reader-icon-button"
                          aria-label={`Show ${Math.min(50, eligible.length - limit)} more links`}
                          title="Show more links"
                          onClick={() => setLimit(value => value + 50)}
                        >
                          <svg
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="1.5"
                            aria-hidden="true"
                            focusable="false"
                          >
                            <circle cx="5" cy="12" r="1" />
                            <circle cx="12" cy="12" r="1" />
                            <circle cx="19" cy="12" r="1" />
                          </svg>
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <NotesPanel {...notesProps} inbox={false} drafts={selectedDrafts} />
              )}
              {!wide && panel && !inbox && <div class="arena-reader-footer">{readerControls}</div>}
            </div>
          </dialog>
          {!inbox && <div class="arena-reader-footer">{readerControls}</div>}
        </div>
      )}
    </div>
  )
}
