import FlexSearch, { type Document as FlexSearchDocument } from 'flexsearch'
import type { StreamManifestEntry, StreamManifestGroup } from '../../util/stream-manifest'
import { encode, tokenizeTerm } from '../../util/search-text'

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

interface IndexedEntry {
  id: number
  entry: StreamManifestEntry
  group: StreamManifestGroup
}

interface SearchData {
  index: FlexSearchDocument
  entries: IndexedEntry[]
}

const tagsForEntry = (entry: StreamManifestEntry): string[] => {
  if (!entry.metadata || typeof entry.metadata !== 'object' || Array.isArray(entry.metadata))
    return []
  const tags = Reflect.get(entry.metadata, 'tags')
  if (!Array.isArray(tags)) return []
  return tags.map(tag => String(tag).trim()).filter(tag => tag.length > 0)
}

const buildSearchData = async (groups: StreamManifestGroup[]): Promise<SearchData> => {
  const entries = groups.flatMap(group => group.entries.map(entry => ({ id: 0, entry, group })))
  entries.forEach((entry, index) => {
    entry.id = index
  })

  const index = new FlexSearch.Document({
    tokenize: 'forward',
    encode,
    document: { id: 'id', index: ['content', 'metadata', 'isoDate', 'displayDate', 'tags'] },
  })

  for (const indexed of entries) {
    const tags = tagsForEntry(indexed.entry)
    await index.addAsync({
      id: indexed.id,
      content: [indexed.entry.title, indexed.entry.description, indexed.entry.content]
        .filter(value => value !== null && value.length > 0)
        .join(' '),
      metadata: JSON.stringify(indexed.entry.metadata ?? {}),
      isoDate: indexed.entry.isoDate ?? indexed.group.isoDate ?? '',
      displayDate: indexed.entry.displayDate ?? indexed.group.isoDate ?? '',
      tags: tags.flatMap(tag => [tag, `#${tag}`]).join(' '),
    })
  }

  return { index, entries }
}

const tagTokens = (query: string): string[] =>
  Array.from(
    new Set(
      query
        .trim()
        .split(/\s+/)
        .map(token => (token.startsWith('#') ? token.slice(1) : ''))
        .filter(token => token.length > 0),
    ),
  )

const matchedEntries = async (data: SearchData, query: string): Promise<IndexedEntry[]> => {
  const normalizedQuery = query.trim().toLowerCase()
  if (normalizedQuery.startsWith('#')) {
    const queryTags = tagTokens(normalizedQuery)
    if (queryTags.length === 0) return []
    return data.entries.filter(({ entry }) => {
      const tags = tagsForEntry(entry).map(tag => tag.toLowerCase())
      return queryTags.every(queryTag => tags.some(tag => tag.startsWith(queryTag)))
    })
  }

  const results = await data.index.searchAsync({
    query,
    limit: 500,
    index: ['content', 'metadata', 'isoDate', 'displayDate', 'tags'],
  })
  const ids = new Set<number>()
  for (const result of results) {
    for (const id of result.result) {
      const numericId = Number(id)
      if (Number.isInteger(numericId)) ids.add(numericId)
    }
  }
  return data.entries.filter(entry => ids.has(entry.id))
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
  indexed: IndexedEntry,
  canonicalizePath: (path: string) => string,
): HTMLLIElement => {
  const { entry, group } = indexed
  const item = document.createElement('li')
  item.className = 'stream-entry stream-search-loading'
  item.setAttribute('aria-busy', 'true')

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
  indexed: IndexedEntry,
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

export const mapWithConcurrency = async <Input, Output>(
  values: readonly Input[],
  concurrency: number,
  transform: (value: Input, index: number) => Promise<Output>,
): Promise<Output[]> => {
  const results: Output[] = []
  let cursor = 0
  const worker = async (): Promise<void> => {
    while (cursor < values.length) {
      const index = cursor
      cursor += 1
      results[index] = await transform(values[index], index)
    }
  }
  const workerCount = Math.min(values.length, Math.max(1, Math.floor(concurrency)))
  await Promise.all(Array.from({ length: workerCount }, worker))
  return results
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

  let searchData: Promise<SearchData> | null = null
  let searchTimeout: number | null = null
  let searchVersion = 0
  let browseScrollY = 0
  let browseScrollCaptured = false
  let searchActive = false

  const prepare = (): Promise<SearchData> => {
    searchData ??= getManifest().then(buildSearchData)
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

  const restoreBrowse = (): void => {
    if (!searchActive) return
    searchActive = false
    browseScrollCaptured = false
    root.removeAttribute('data-stream-search-active')
    resultsFeed.hidden = true
    resultsFeed.replaceChildren()
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
    resultsFeed.replaceChildren()
    resultsFeed.setAttribute('aria-busy', 'true')
    updateStatus(form, 'searching…')
    const data = await prepare()
    if (signal.aborted || version !== searchVersion) return
    const matches = await matchedEntries(data, trimmed)
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
    updateStatus(form, `loading ${matches.length} ${matches.length === 1 ? 'entry' : 'entries'}…`)

    const loaded = await mapWithConcurrency(matches, 4, async (indexed, index) => {
      try {
        const element = await loadEntry(indexed.entry, indexed.group)
        if (signal.aborted || version !== searchVersion) return false
        element.classList.add('stream-entry-search-result')
        highlightEntry(element, tokens)
        placeholders[index].replaceWith(element)
        mountEntry(element, indexed.group)
        return true
      } catch (error) {
        if (signal.aborted || version !== searchVersion) return false
        console.error(error)
        placeholders[index].replaceWith(renderLoadFailure(indexed, canonicalizePath))
        return false
      }
    })
    if (signal.aborted || version !== searchVersion) return
    resultsFeed.removeAttribute('aria-busy')
    const loadedCount = loaded.filter(Boolean).length
    if (loadedCount !== matches.length) {
      updateStatus(form, `loaded ${loadedCount} of ${matches.length} entries`)
      return
    }

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
    if (!feed.isConnected && resultsFeed.isConnected) resultsFeed.before(feed)
    resultsFeed.remove()
    if (sentinel) sentinel.hidden = false
    void searchData?.then(data => data.index.destroy()).catch(() => {})
  }
}
