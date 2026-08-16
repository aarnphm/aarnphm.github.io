import type { StreamManifestEntry, StreamManifestGroup } from '../../util/stream-manifest'
import { tokenizeTerm } from '../../util/search-text'
import {
  buildStreamSearchData,
  matchStreamEntries,
  tagTokens,
  type IndexedStreamEntry,
  type StreamSearchData,
} from './stream-search-index'
import { setupStreamVirtualizer, type StreamVirtualizer } from './stream-virtualizer'

interface StreamSearchSetup {
  root: HTMLElement
  feed: HTMLOListElement
  sentinel: HTMLElement | null
  getManifest: () => Promise<StreamManifestGroup[]>
  canonicalizePath: (path: string) => string
  loadEntry: (entry: StreamManifestEntry, group: StreamManifestGroup) => Promise<HTMLElement>
  mountEntry: (entry: HTMLElement, group: StreamManifestGroup) => void
  signal: AbortSignal
}

const appendHighlightedText = (target: ParentNode, value: string, rawTokens: string[]): void => {
  const tokens = Array.from(new Set(rawTokens.map(token => token.toLowerCase()).filter(Boolean)))
  if (tokens.length === 0) {
    target.textContent = value
    return
  }

  const lowerValue = value.toLowerCase()
  let cursor = 0
  while (cursor < value.length) {
    let matchIndex = -1
    let matchToken = ''
    for (const token of tokens) {
      const index = lowerValue.indexOf(token, cursor)
      if (index === -1) continue
      if (
        matchIndex === -1 ||
        index < matchIndex ||
        (index === matchIndex && token.length > matchToken.length)
      ) {
        matchIndex = index
        matchToken = token
      }
    }
    if (matchIndex === -1) {
      target.append(value.slice(cursor))
      return
    }
    if (matchIndex > cursor) target.append(value.slice(cursor, matchIndex))
    const mark = document.createElement('mark')
    mark.className = 'search-highlight'
    mark.textContent = value.slice(matchIndex, matchIndex + matchToken.length)
    target.append(mark)
    cursor = matchIndex + matchToken.length
  }
}

const highlightEntry = (entry: HTMLElement, rawTokens: string[]): void => {
  const tokens = Array.from(new Set(rawTokens.map(token => token.toLowerCase()).filter(Boolean)))
  if (tokens.length === 0) return
  const matchingNodes: Text[] = []
  const walker = document.createTreeWalker(entry, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const value = node.textContent
      const parent = node.parentElement
      if (
        !value ||
        !parent ||
        parent.closest('script, style, template, svg, mark') ||
        !tokens.some(token => value.toLowerCase().includes(token))
      ) {
        return NodeFilter.FILTER_REJECT
      }
      return NodeFilter.FILTER_ACCEPT
    },
  })
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node instanceof Text) matchingNodes.push(node)
  }
  for (const node of matchingNodes) {
    const value = node.textContent
    if (!value) continue
    const fragment = document.createDocumentFragment()
    appendHighlightedText(fragment, value, tokens)
    node.replaceWith(fragment)
  }
}

const entryHref = (
  entry: StreamManifestEntry,
  group: StreamManifestGroup,
  canonicalizePath: (path: string) => string,
): string => {
  const path = canonicalizePath(group.path ?? '/stream')
  const url = new URL(path, window.location.origin)
  url.searchParams.set('entry', entry.id)
  url.hash = entry.id
  return `${url.pathname}${url.search}${url.hash}`
}

const renderLoadingResult = (
  indexed: IndexedStreamEntry,
  canonicalizePath: (path: string) => string,
): HTMLLIElement => {
  const { entry, group } = indexed
  const item = document.createElement('li')
  item.className = 'stream-entry stream-search-loading'
  item.setAttribute('aria-busy', 'true')
  item.dataset.entryId = entry.id
  item.dataset.streamGroupId = group.groupId
  if (group.timestamp) item.dataset.streamTimestamp = group.timestamp.toString()

  const meta = document.createElement('div')
  meta.className = 'stream-entry-meta'
  if (group.path) {
    const date = document.createElement('a')
    date.className = 'stream-entry-date internal'
    date.href = canonicalizePath(group.path)
    date.dataset.slug = group.path.replace(/^\//, '')
    date.dataset.noPopover = ''
    const time = document.createElement('time')
    if (entry.isoDate) time.dateTime = entry.isoDate
    time.textContent = entry.displayDate ?? group.isoDate ?? 'undated'
    date.append(time)
    meta.append(date)
  }

  const body = document.createElement('div')
  body.className = 'stream-entry-body'
  const message = document.createElement('p')
  message.className = 'stream-search-loading-label'
  message.textContent = `loading “${entry.title ?? entry.description ?? 'entry'}”…`
  body.append(message)

  item.append(meta, body)
  return item
}

const renderLoadFailure = (
  indexed: IndexedStreamEntry,
  canonicalizePath: (path: string) => string,
): HTMLLIElement => {
  const item = document.createElement('li')
  item.className = 'stream-entry stream-search-load-error'
  const meta = document.createElement('div')
  meta.className = 'stream-entry-meta'
  meta.textContent = indexed.entry.displayDate ?? indexed.group.isoDate ?? 'undated'
  const body = document.createElement('div')
  body.className = 'stream-entry-body'
  body.append('could not load this entry. ')
  const link = document.createElement('a')
  link.className = 'internal'
  link.href = entryHref(indexed.entry, indexed.group, canonicalizePath)
  link.textContent = 'open its daily page'
  body.append(link)
  item.append(meta, body)
  return item
}

const updateStatus = (form: HTMLFormElement, message: string): void => {
  let status = form.parentElement?.querySelector<HTMLElement>('.stream-search-status') ?? null
  if (!status && message) {
    status = document.createElement('div')
    status.className = 'stream-search-status'
    status.setAttribute('role', 'status')
    form.after(status)
  }
  if (!status) return
  status.textContent = message
  status.hidden = message.length === 0
}

export const setupStreamSearch = ({
  root,
  feed,
  sentinel,
  getManifest,
  canonicalizePath,
  loadEntry,
  mountEntry,
  signal,
}: StreamSearchSetup): (() => void) | null => {
  const form = root.querySelector<HTMLFormElement>('.stream-search-form')
  const input = root.querySelector<HTMLInputElement>('.stream-search-input')
  if (!form || !input) return null

  const resultsFeed = document.createElement('ol')
  resultsFeed.className = 'stream-feed stream-search-results'
  resultsFeed.hidden = true
  feed.after(resultsFeed)

  let searchData: Promise<StreamSearchData> | null = null
  let searchTimeout: number | null = null
  let searchVersion = 0
  let browseScrollY = 0
  let browseScrollCaptured = false
  let searchActive = false
  let resultVirtualizer: StreamVirtualizer | null = null

  const prepare = (): Promise<StreamSearchData> => {
    searchData ??= getManifest().then(buildStreamSearchData)
    return searchData
  }

  const activateSearch = (): void => {
    if (!searchActive) {
      if (!browseScrollCaptured) browseScrollY = window.scrollY
      searchActive = true
      feed.remove()
    }
    root.dataset.streamSearchActive = 'true'
    if (sentinel) sentinel.hidden = true
    resultsFeed.hidden = false
  }

  const clearResults = (): void => {
    resultVirtualizer?.destroy()
    resultVirtualizer = null
    resultsFeed.replaceChildren()
  }

  const restoreBrowse = (): void => {
    if (!searchActive) return
    searchActive = false
    browseScrollCaptured = false
    root.removeAttribute('data-stream-search-active')
    resultsFeed.hidden = true
    clearResults()
    resultsFeed.before(feed)
    if (sentinel) sentinel.hidden = false
    resultsFeed.removeAttribute('aria-busy')
    updateStatus(form, '')
    input.blur()
    window.requestAnimationFrame(() => window.scrollTo({ top: browseScrollY, behavior: 'auto' }))
  }

  const search = async (query: string, version: number): Promise<void> => {
    const trimmed = query.trim()
    if (!trimmed) {
      restoreBrowse()
      return
    }

    activateSearch()
    clearResults()
    resultsFeed.setAttribute('aria-busy', 'true')
    updateStatus(form, 'searching…')
    const data = await prepare()
    if (signal.aborted || version !== searchVersion) return
    const matches = await matchStreamEntries(data, trimmed)
    if (signal.aborted || version !== searchVersion) return

    const tokens = trimmed.startsWith('#') ? tagTokens(trimmed) : tokenizeTerm(trimmed)
    if (matches.length === 0) {
      resultsFeed.removeAttribute('aria-busy')
      if (trimmed.startsWith('#')) {
        const readableTags = tagTokens(trimmed)
          .map(tag => `#${tag}`)
          .join(' ')
        updateStatus(
          form,
          readableTags ? `no entries tagged ${readableTags}` : "type a tag name after '#'",
        )
      } else {
        updateStatus(form, `no results for “${trimmed}”`)
      }
      return
    }

    const placeholders = matches.map(entry => renderLoadingResult(entry, canonicalizePath))
    resultsFeed.replaceChildren(...placeholders)
    resultsFeed.removeAttribute('aria-busy')
    const virtualizer = setupStreamVirtualizer(resultsFeed, signal)
    resultVirtualizer = virtualizer
    matches.forEach((indexed, index) => {
      virtualizer.registerPlaceholder(placeholders[index], {
        entryId: indexed.entry.id,
        load: async () => {
          const element = await loadEntry(indexed.entry, indexed.group)
          if (signal.aborted || version !== searchVersion) {
            throw new DOMException('stale search result', 'AbortError')
          }
          element.classList.add('stream-entry-search-result')
          highlightEntry(element, tokens)
          return element
        },
        mount: element => mountEntry(element, indexed.group),
        failure: error => {
          if (!(error instanceof DOMException && error.name === 'AbortError')) console.error(error)
          return renderLoadFailure(indexed, canonicalizePath)
        },
      })
    })

    if (trimmed.startsWith('#')) {
      const readableTags = tagTokens(trimmed)
        .map(tag => `#${tag}`)
        .join(' ')
      updateStatus(
        form,
        `showing ${matches.length} ${matches.length === 1 ? 'entry' : 'entries'} tagged ${readableTags}`,
      )
      return
    }
    updateStatus(form, `showing ${matches.length} ${matches.length === 1 ? 'entry' : 'entries'}`)
  }

  const onInput = (): void => {
    searchVersion += 1
    const version = searchVersion
    if (searchTimeout !== null) window.clearTimeout(searchTimeout)
    searchTimeout = window.setTimeout(() => {
      void search(input.value, version).catch(error => {
        if (signal.aborted || version !== searchVersion) return
        console.error(error)
        updateStatus(form, 'search unavailable')
      })
    }, 300)
  }

  const onShortcut = (event: KeyboardEvent): void => {
    const key = event.key.toLowerCase()
    const shortcut = event.metaKey && key === '.'
    const commandSearch = (event.metaKey || event.ctrlKey) && key === 'k'
    if (!shortcut && !commandSearch) return
    const target = event.target
    if (target instanceof HTMLElement) {
      const editable =
        target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable
      if (editable && target !== input) return
    }
    event.preventDefault()
    event.stopPropagation()
    event.stopImmediatePropagation()
    if (!searchActive) {
      browseScrollY = window.scrollY
      browseScrollCaptured = true
    }
    input.focus()
    input.select()
    void prepare().catch(() => {})
  }

  const onSubmit = (event: SubmitEvent): void => event.preventDefault()
  const onPointerdown = (): void => {
    if (searchActive) return
    browseScrollY = window.scrollY
    browseScrollCaptured = true
  }
  const onFocus = (): void => {
    void prepare().catch(() => {})
  }

  input.addEventListener('pointerdown', onPointerdown, { signal })
  input.addEventListener('focus', onFocus, { signal })
  input.addEventListener('input', onInput, { signal })
  form.addEventListener('submit', onSubmit, { signal })
  document.addEventListener('keydown', onShortcut, { capture: true, signal })

  return () => {
    searchVersion += 1
    if (searchTimeout !== null) window.clearTimeout(searchTimeout)
    resultVirtualizer?.destroy()
    if (!feed.isConnected && resultsFeed.isConnected) resultsFeed.before(feed)
    resultsFeed.remove()
    if (sentinel) sentinel.hidden = false
    void searchData?.then(data => data.index.destroy()).catch(() => {})
  }
}
