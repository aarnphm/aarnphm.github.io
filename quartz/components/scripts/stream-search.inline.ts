import FlexSearch, { type Document as FlexSearchDocument } from 'flexsearch'
import type { StreamManifestEntry, StreamManifestGroup } from '../../util/stream-manifest'
import { encode, tokenizeTerm } from '../../util/search-text'

interface StreamSearchSetup {
  root: HTMLElement
  feed: HTMLOListElement
  sentinel: HTMLElement | null
  getManifest: () => Promise<StreamManifestGroup[]>
  canonicalizePath: (path: string) => string
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
      content: indexed.entry.text,
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

const appendHighlightedText = (target: HTMLElement, value: string, rawTokens: string[]): void => {
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

const renderSearchResult = (
  indexed: IndexedEntry,
  tokens: string[],
  canonicalizePath: (path: string) => string,
): HTMLLIElement => {
  const { entry, group } = indexed
  const item = document.createElement('li')
  item.className = 'stream-entry stream-entry-search-result'
  item.dataset.entryId = entry.id
  item.dataset.streamGroupId = group.groupId
  if (group.timestamp !== null) item.dataset.streamTimestamp = String(group.timestamp)

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

  const tags = tagsForEntry(entry)
  if (tags.length > 0) {
    const tagList = document.createElement('div')
    tagList.className = 'stream-entry-tags'
    for (const tag of tags) {
      const tagElement = document.createElement('span')
      tagElement.className = 'stream-entry-tag'
      appendHighlightedText(tagElement, tag, tokens)
      tagList.append(tagElement)
    }
    meta.append(tagList)
  }

  const body = document.createElement('div')
  body.className = 'stream-entry-body'
  const title = document.createElement('h2')
  title.className = 'stream-entry-title'
  const link = document.createElement('a')
  link.className = 'internal stream-entry-search-link'
  link.href = entryHref(entry, group, canonicalizePath)
  link.dataset.slug = (group.path ?? '/stream').replace(/^\//, '')
  appendHighlightedText(link, entry.title ?? entry.description ?? 'entry', tokens)
  title.append(link)
  body.append(title)

  if (entry.title && entry.description && entry.description !== entry.title) {
    const description = document.createElement('p')
    description.className = 'stream-entry-description stream-entry-search-description'
    appendHighlightedText(description, entry.description, tokens)
    body.append(description)
  }
  if (entry.wordCount > 0) {
    const wordCount = document.createElement('div')
    wordCount.className = 'stream-entry-wordcount'
    const emphasis = document.createElement('em')
    emphasis.textContent = entry.wordCount === 1 ? '1 word' : `${entry.wordCount} words`
    wordCount.append(emphasis)
    body.append(wordCount)
  }

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

  const restoreBrowse = (): void => {
    if (!searchActive) return
    searchActive = false
    browseScrollCaptured = false
    root.removeAttribute('data-stream-search-active')
    resultsFeed.hidden = true
    resultsFeed.replaceChildren()
    feed.hidden = false
    if (sentinel) sentinel.hidden = false
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

    updateStatus(form, 'searching…')
    const data = await prepare()
    if (signal.aborted || version !== searchVersion) return
    const matches = await matchedEntries(data, trimmed)
    if (signal.aborted || version !== searchVersion) return

    if (!searchActive) {
      if (!browseScrollCaptured) browseScrollY = window.scrollY
      searchActive = true
    }
    root.dataset.streamSearchActive = 'true'
    feed.hidden = true
    if (sentinel) sentinel.hidden = true
    resultsFeed.hidden = false
    const tokens = trimmed.startsWith('#') ? tagTokens(trimmed) : tokenizeTerm(trimmed)
    resultsFeed.replaceChildren(
      ...matches.map(entry => renderSearchResult(entry, tokens, canonicalizePath)),
    )

    if (trimmed.startsWith('#')) {
      const readableTags = tagTokens(trimmed)
        .map(tag => `#${tag}`)
        .join(' ')
      if (!readableTags) {
        updateStatus(form, "type a tag name after '#'")
        return
      }
      updateStatus(
        form,
        matches.length > 0
          ? `showing ${matches.length} ${matches.length === 1 ? 'entry' : 'entries'} tagged ${readableTags}`
          : `no entries tagged ${readableTags}`,
      )
      return
    }
    updateStatus(
      form,
      matches.length > 0
        ? `showing ${matches.length} ${matches.length === 1 ? 'entry' : 'entries'}`
        : `no results for “${trimmed}”`,
    )
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
    resultsFeed.remove()
    feed.hidden = false
    if (sentinel) sentinel.hidden = false
    void searchData?.then(data => data.index.destroy()).catch(() => {})
  }
}
