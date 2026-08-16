import { streamHostPathname, STREAM_HOSTNAME } from '../../util/stream-host'
import {
  parseStreamManifest,
  type StreamManifestEntry,
  type StreamManifestGroup,
} from '../../util/stream-manifest'
import { currentNavSignal } from './nav-lifecycle'
import { setupStreamSearch } from './stream-search.inline'
import { setupStreamVirtualizer, type StreamVirtualSource } from './stream-virtualizer'

const formatEntryTimes = (scope: ParentNode): void => {
  const formatter = new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: false,
    timeZoneName: 'shortOffset',
  })
  for (const time of scope.querySelectorAll<HTMLTimeElement>(
    '.stream-entry-date[datetime], .stream-entry-date time[datetime]',
  )) {
    const isoDate = time.getAttribute('datetime')
    if (!isoDate) continue
    const date = new Date(isoDate)
    if (!Number.isNaN(date.getTime())) time.textContent = formatter.format(date)
  }
}

const fetchStreamManifest = async (signal: AbortSignal): Promise<StreamManifestGroup[]> => {
  const response = await fetch('/streams.jsonl', { signal })
  if (!response.ok) throw new Error(`stream manifest request failed with ${response.status}`)
  return parseStreamManifest(await response.text())
}

const notifyContentMounted = (entry: HTMLElement, slug?: string): void => {
  const event: CustomEventMap['contentdecrypted'] = new CustomEvent('contentdecrypted', {
    detail: { article: entry, content: entry, slug },
  })
  document.dispatchEvent(event)
}

const fetchGroupDocument = async (
  group: StreamManifestGroup,
  canonicalizePath: (path: string) => string,
  signal: AbortSignal,
): Promise<Document> => {
  if (!group.path) throw new Error(`stream group ${group.groupId} has no path`)
  const response = await fetch(canonicalizePath(group.path), { signal })
  if (!response.ok) throw new Error(`stream group request failed with ${response.status}`)
  return new DOMParser().parseFromString(await response.text(), 'text/html')
}

const entriesForGroup = (parsed: Document, group: StreamManifestGroup): HTMLElement[] => {
  const sourceEntries = new Map<string, HTMLElement>()
  for (const entry of parsed.querySelectorAll<HTMLElement>('.stream-entry[data-entry-id]')) {
    const id = entry.dataset.entryId
    if (id) sourceEntries.set(id, entry)
  }

  return group.entries.map(entry => {
    const source = sourceEntries.get(entry.id)
    if (!source) throw new Error(`stream group ${group.groupId} is missing ${entry.id}`)
    return document.importNode(source, true)
  })
}

const entryForGroup = (
  parsed: Document,
  group: StreamManifestGroup,
  manifestEntry: StreamManifestEntry,
): HTMLElement => {
  for (const source of parsed.querySelectorAll<HTMLElement>('.stream-entry[data-entry-id]')) {
    if (source.dataset.entryId === manifestEntry.id) return document.importNode(source, true)
  }
  throw new Error(`stream group ${group.groupId} is missing ${manifestEntry.id}`)
}

const setupLazyFeed = (
  root: HTMLElement,
  feed: HTMLOListElement,
  sentinel: HTMLElement,
  getManifest: () => Promise<StreamManifestGroup[]>,
  getGroupDocument: (group: StreamManifestGroup) => Promise<Document>,
  canonicalizePath: (path: string) => string,
  signal: AbortSignal,
  appendEntry: (entry: HTMLElement, group: StreamManifestGroup) => void,
): (() => void) => {
  const loadedEntryIds = new Set(
    Array.from(feed.querySelectorAll<HTMLElement>('.stream-entry[data-entry-id]')).flatMap(entry =>
      entry.dataset.entryId ? [entry.dataset.entryId] : [],
    ),
  )
  let nextGroupIndex = 0
  let loading = false
  const isLoaded = (entryId: string): boolean =>
    loadedEntryIds.has(entryId) ||
    Array.from(feed.children).some(
      child => child instanceof HTMLElement && child.dataset.entryId === entryId,
    )

  const showArchiveFallback = (): void => {
    const link = document.createElement('a')
    link.href = canonicalizePath('/stream/on')
    link.className = 'internal'
    link.textContent = 'open the stream archive'
    sentinel.replaceChildren('older entries are unavailable. ', link)
    sentinel.dataset.streamLoading = 'error'
  }

  const updateSentinel = (state: 'idle' | 'loading'): void => {
    sentinel.dataset.streamLoading = state
    sentinel.textContent = state === 'loading' ? 'loading older entries…' : ''
  }

  const observer = new IntersectionObserver(
    entries => {
      if (
        entries.some(entry => entry.isIntersecting) &&
        root.dataset.streamSearchActive !== 'true'
      ) {
        void loadNextGroups()
      }
    },
    { rootMargin: '1600px 0px' },
  )

  const loadNextGroups = async (): Promise<void> => {
    if (loading || signal.aborted) return
    loading = true
    updateSentinel('loading')
    const batchStart = nextGroupIndex
    try {
      const groups = await getManifest()
      const nextGroups: StreamManifestGroup[] = []
      while (nextGroupIndex < groups.length && nextGroups.length < 2) {
        const group = groups[nextGroupIndex]
        nextGroupIndex += 1
        const loaded = group.entries.every(entry => isLoaded(entry.id))
        if (!loaded && group.path) nextGroups.push(group)
      }

      if (nextGroups.length === 0) {
        observer.disconnect()
        sentinel.remove()
        return
      }

      const batches = await Promise.all(
        nextGroups.map(async group => entriesForGroup(await getGroupDocument(group), group)),
      )
      nextGroups.forEach((group, index) => {
        for (const entry of batches[index]) {
          const entryId = entry.dataset.entryId
          if (!entryId || isLoaded(entryId)) continue
          loadedEntryIds.add(entryId)
          feed.append(entry)
          appendEntry(entry, group)
        }
      })

      updateSentinel('idle')
      observer.unobserve(sentinel)
      if (nextGroupIndex >= groups.length) {
        observer.disconnect()
        sentinel.remove()
      } else {
        observer.observe(sentinel)
      }
    } catch (error) {
      if (signal.aborted) return
      nextGroupIndex = batchStart
      console.error(error)
      observer.disconnect()
      showArchiveFallback()
    } finally {
      loading = false
    }
  }

  observer.observe(sentinel)
  return () => observer.disconnect()
}

const hydrateStream = (): void => {
  const root = document.querySelector<HTMLElement>('.stream')
  const feed = root?.querySelector<HTMLOListElement>('.stream-feed')
  if (!root || !feed) return
  const signal = currentNavSignal()
  const isRoot = root.dataset.streamView === 'root'
  const isStreamHost = window.location.hostname === STREAM_HOSTNAME
  const canonicalizePath = (path: string): string =>
    isStreamHost ? streamHostPathname(path) : path
  const canonicalPath = canonicalizePath(root.dataset.streamCanonical ?? '/stream')
  const sentinel = root.querySelector<HTMLElement>('[data-stream-feed-sentinel]')
  const originalUrl = new URL(window.location.href)
  const originalSearch = originalUrl.search
  const originalHash = originalUrl.hash
  let activeTimestamp: string | null = null
  let manifestRequest: Promise<StreamManifestGroup[]> | null = null
  const groupDocumentRequests = new Map<string, Promise<Document>>()
  let refreshVirtualFeed = (): void => {}

  const getManifest = (): Promise<StreamManifestGroup[]> => {
    manifestRequest ??= fetchStreamManifest(signal)
    return manifestRequest
  }

  const getGroupDocument = (group: StreamManifestGroup): Promise<Document> => {
    if (!group.path) return Promise.reject(new Error(`stream group ${group.groupId} has no path`))
    const existing = groupDocumentRequests.get(group.path)
    if (existing) return existing
    const request = fetchGroupDocument(group, canonicalizePath, signal).catch(error => {
      groupDocumentRequests.delete(group.path ?? '')
      throw error
    })
    groupDocumentRequests.set(group.path, request)
    return request
  }

  const loadEntry = async (
    entry: StreamManifestEntry,
    group: StreamManifestGroup,
  ): Promise<HTMLElement> => entryForGroup(await getGroupDocument(group), group, entry)

  const streamEntries = (): HTMLElement[] =>
    Array.from(feed.querySelectorAll<HTMLElement>(':scope > [data-entry-id]'))

  const streamLinks = (): HTMLAnchorElement[] =>
    Array.from(
      feed.querySelectorAll<HTMLAnchorElement>(
        '.stream-entry-date[data-stream-link][data-stream-timestamp]',
      ),
    )

  const notifyProtectedContent = (): void => {
    document.dispatchEvent(
      new CustomEvent('protectedcontentloaded', { detail: { container: root } }),
    )
  }

  const targetEntryId = (): string | null => {
    const url = new URL(window.location.href)
    return (
      url.searchParams.get('entry')?.trim() || decodeURIComponent(url.hash.slice(1)).trim() || null
    )
  }

  const applyFilters = (): void => {
    const entries = streamEntries()
    const entryId = targetEntryId()
    const hasEntryTarget =
      entryId !== null && entries.some(entry => entry.dataset.entryId === entryId)
    root.toggleAttribute('data-stream-active-entry', hasEntryTarget)
    if (hasEntryTarget && entryId) root.dataset.streamActiveEntry = entryId
    else root.removeAttribute('data-stream-active-entry')

    if (activeTimestamp) root.dataset.streamActiveTimestamp = activeTimestamp
    else root.removeAttribute('data-stream-active-timestamp')

    for (const entry of entries) {
      const matchesEntry = hasEntryTarget && entry.dataset.entryId === entryId
      const matchesTimestamp =
        activeTimestamp !== null && entry.dataset.streamTimestamp === activeTimestamp
      const visible = hasEntryTarget ? matchesEntry : activeTimestamp === null || matchesTimestamp
      entry.hidden = !visible
      entry.classList.toggle('stream-entry-active', matchesEntry || matchesTimestamp)
    }
    for (const link of streamLinks()) {
      const active = activeTimestamp !== null && link.dataset.streamTimestamp === activeTimestamp
      link.classList.toggle('is-active', active)
      if (active) link.setAttribute('aria-current', 'page')
      else link.removeAttribute('aria-current')
    }
    if (sentinel?.isConnected) {
      sentinel.hidden = activeTimestamp !== null || root.dataset.streamSearchActive === 'true'
    }
    refreshVirtualFeed()
    notifyProtectedContent()
  }

  const applyHistory = (path: string): void => {
    const url = new URL(window.location.href)
    url.pathname = path
    url.search = originalSearch
    url.hash = originalHash
    window.history.replaceState(window.history.state, '', url)
  }

  const timestampForPath = (path: string): string | null => {
    for (const entry of streamEntries()) {
      const href = entry.dataset.streamHref
      if (href && canonicalizePath(href) === path) return entry.dataset.streamTimestamp ?? null
    }
    for (const link of streamLinks()) {
      const href = link.dataset.streamHref
      if (href && canonicalizePath(href) === path) return link.dataset.streamTimestamp ?? null
    }
    return null
  }

  const mountEntry = (entry: HTMLElement, group: StreamManifestGroup): void => {
    formatEntryTimes(entry)
    applyFilters()
    notifyContentMounted(entry, group.path?.replace(/^\//, ''))
  }

  formatEntryTimes(root)
  applyFilters()

  const cleanups: (() => void)[] = []
  const virtualizer = isRoot ? setupStreamVirtualizer(feed, signal) : null
  if (virtualizer) {
    refreshVirtualFeed = virtualizer.refresh
    cleanups.push(virtualizer.destroy)
    const sourceForEntry = (
      entryId: string,
      knownGroup?: StreamManifestGroup,
    ): StreamVirtualSource => {
      let resolvedGroup = knownGroup
      return {
        entryId,
        load: async () => {
          if (!resolvedGroup) {
            const groups = await getManifest()
            resolvedGroup = groups.find(group => group.entries.some(entry => entry.id === entryId))
          }
          const group = resolvedGroup
          const manifestEntry = group?.entries.find(entry => entry.id === entryId)
          if (!group || !manifestEntry) throw new Error(`stream manifest is missing ${entryId}`)
          return loadEntry(manifestEntry, group)
        },
        mount: entry => {
          if (!resolvedGroup) throw new Error(`stream manifest is missing group for ${entryId}`)
          mountEntry(entry, resolvedGroup)
        },
      }
    }
    for (const entry of streamEntries()) {
      const entryId = entry.dataset.entryId
      if (entryId) virtualizer.registerMounted(entry, sourceForEntry(entryId))
    }

    const appendEntry = (entry: HTMLElement, group: StreamManifestGroup): void => {
      mountEntry(entry, group)
      const entryId = entry.dataset.entryId
      if (entryId) virtualizer.registerMounted(entry, sourceForEntry(entryId, group))
    }
    if (sentinel) {
      cleanups.push(
        setupLazyFeed(
          root,
          feed,
          sentinel,
          getManifest,
          getGroupDocument,
          canonicalizePath,
          signal,
          appendEntry,
        ),
      )
    }
    const ensureTargetEntry = async (): Promise<void> => {
      const entryId = targetEntryId()
      if (!entryId || streamEntries().some(entry => entry.dataset.entryId === entryId)) return
      const groups = await getManifest()
      const group = groups.find(candidate => candidate.entries.some(entry => entry.id === entryId))
      const manifestEntry = group?.entries.find(entry => entry.id === entryId)
      if (!group || !manifestEntry || signal.aborted || targetEntryId() !== entryId) return
      const entry = await loadEntry(manifestEntry, group)
      if (signal.aborted || targetEntryId() !== entryId) return
      const existing = streamEntries().find(candidate => candidate.dataset.entryId === entryId)
      if (existing) {
        existing.scrollIntoView({ block: 'start' })
        return
      }
      feed.append(entry)
      appendEntry(entry, group)
      window.requestAnimationFrame(() => entry.scrollIntoView({ block: 'start' }))
    }
    void ensureTargetEntry().catch(error => {
      if (!signal.aborted) console.error(error)
    })
  } else if (isRoot && sentinel) {
    cleanups.push(
      setupLazyFeed(
        root,
        feed,
        sentinel,
        getManifest,
        getGroupDocument,
        canonicalizePath,
        signal,
        mountEntry,
      ),
    )
  }
  if (isRoot) {
    const searchCleanup = setupStreamSearch({
      root,
      feed,
      sentinel,
      getManifest,
      canonicalizePath,
      loadEntry,
      mountEntry,
      signal,
    })
    if (searchCleanup) cleanups.push(searchCleanup)
  }

  const onClick = (event: MouseEvent): void => {
    if (!isRoot || root.dataset.streamSearchActive === 'true') return
    const target = event.target
    if (!(target instanceof Element)) return
    const link = target.closest<HTMLAnchorElement>(
      '.stream-entry-date[data-stream-link][data-stream-timestamp]',
    )
    if (!link || !feed.contains(link)) return
    const timestamp = link.dataset.streamTimestamp
    const href = link.dataset.streamHref
    if (!timestamp || !href) return
    event.preventDefault()
    activeTimestamp = activeTimestamp === timestamp ? null : timestamp
    applyHistory(activeTimestamp ? canonicalizePath(href) : canonicalPath)
    applyFilters()
    link.focus()
  }

  const onKeydown = (event: KeyboardEvent): void => {
    if (event.key !== 'Escape' || activeTimestamp === null) return
    event.preventDefault()
    activeTimestamp = null
    applyHistory(canonicalPath)
    applyFilters()
  }

  const onFocusout = (): void => {
    window.setTimeout(() => {
      if (activeTimestamp === null) return
      const active = document.activeElement
      if (active instanceof Node && root.contains(active)) return
      activeTimestamp = null
      applyHistory(canonicalPath)
      applyFilters()
    }, 0)
  }

  const onPopstate = (): void => {
    activeTimestamp = isRoot ? timestampForPath(window.location.pathname) : null
    applyFilters()
  }

  root.addEventListener('click', onClick, { signal })
  root.addEventListener('keydown', onKeydown, { signal })
  root.addEventListener('focusout', onFocusout, { capture: true, signal })
  window.addEventListener('popstate', onPopstate, { signal })

  if (isRoot) activeTimestamp = timestampForPath(window.location.pathname)
  applyFilters()

  window.addCleanup(() => {
    for (let index = cleanups.length - 1; index >= 0; index -= 1) cleanups[index]()
  })
}

document.addEventListener('nav', () => {
  hydrateStream()
})
