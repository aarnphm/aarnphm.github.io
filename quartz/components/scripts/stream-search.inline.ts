import FlexSearch from 'flexsearch'
import { tokenizeTerm, encode, isStreamHost } from './util'

interface StreamEntry {
  id: string
  html: string
  metadata: unknown
  isoDate: string | null
  displayDate: string | null
}

interface StreamGroup {
  groupId: string
  timestamp: number | null
  isoDate: string | null
  groupSize: number
  path: string | null
  entries: StreamEntry[]
}

interface IndexedEntry {
  id: number
  entryId: string
  groupId: string
  content: string
  metadata: string
  isoDate: string
  displayDate: string
  tags: string[]
}

function extractMetadata(raw: unknown): { tags: string[]; metadataString: string } {
  let metadataObj: Record<string, unknown> = {}
  let metadataString = '{}'

  if (typeof raw === 'string') {
    metadataObj = JSON.parse(raw)
  } else if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    metadataObj = raw as Record<string, unknown>
    metadataString = JSON.stringify(metadataObj)
  }

  const rawTags = Array.isArray(metadataObj.tags) ? metadataObj.tags : []
  const tags = rawTags.map(tag => String(tag).trim()).filter(tag => tag.length > 0)

  return { tags, metadataString }
}

let searchIndex: any | null = null
let indexedEntries: IndexedEntry[] = []
let isIndexBuilt = false
let searchTimeout: number | null = null

function stripHtml(html: string): string {
  const tmp = document.createElement('div')
  tmp.innerHTML = html
  return tmp.textContent || tmp.innerText || ''
}

async function buildSearchIndex() {
  if (isIndexBuilt) return

  const endpoint = isStreamHost() ? `${window.location.origin}/streams.jsonl` : '/streams.jsonl'
  const response = await fetch(endpoint)

  const text = await response.text()
  const lines = text.trim().split('\n')

  let entryIndex = 0
  for (const line of lines) {
    if (!line.trim()) continue

    const group: StreamGroup = JSON.parse(line)

    for (const entry of group.entries) {
      const { tags, metadataString } = extractMetadata(entry.metadata)

      const indexedEntry: IndexedEntry = {
        id: entryIndex++,
        entryId: entry.id,
        groupId: group.groupId,
        content: stripHtml(entry.html),
        metadata: metadataString,
        isoDate: entry.isoDate || group.isoDate || '',
        displayDate: entry.displayDate || group.isoDate || '',
        tags,
      }
      indexedEntries.push(indexedEntry)
    }
  }

  searchIndex = new FlexSearch.Document({
    tokenize: 'forward',
    encode,
    document: { id: 'id', index: ['content', 'metadata', 'isoDate', 'displayDate', 'tags'] },
  })

  for (const entry of indexedEntries) {
    const tagsField = entry.tags
      .flatMap(tag => [tag, `#${tag}`])
      .join(' ')
      .trim()

    await searchIndex.addAsync({ ...entry, tags: tagsField })
  }

  isIndexBuilt = true
}

function highlightTextNodes(element: HTMLElement, searchTerm: string) {
  const tokens = tokenizeTerm(searchTerm)
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null)

  const nodesToReplace: { node: Text; parent: Node }[] = []
  let currentNode: Node | null

  while ((currentNode = walker.nextNode())) {
    const textNode = currentNode as Text
    const text = textNode.nodeValue || ''

    let hasMatch = false
    for (const token of tokens) {
      if (text.toLowerCase().includes(token.toLowerCase())) {
        hasMatch = true
        break
      }
    }

    if (hasMatch && textNode.parentNode) {
      nodesToReplace.push({ node: textNode, parent: textNode.parentNode })
    }
  }

  for (const { node, parent } of nodesToReplace) {
    const text = node.nodeValue || ''
    const fragment = document.createDocumentFragment()
    let lastIndex = 0
    let modified = false

    for (const token of tokens) {
      const regex = new RegExp(token, 'gi')
      let match: RegExpExecArray | null

      while ((match = regex.exec(text))) {
        modified = true

        if (match.index > lastIndex) {
          fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.index)))
        }

        const mark = document.createElement('mark')
        mark.className = 'search-highlight'
        mark.textContent = match[0]
        fragment.appendChild(mark)

        lastIndex = match.index + match[0].length
      }
    }

    if (modified) {
      if (lastIndex < text.length) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex)))
      }
      parent.replaceChild(fragment, node)
    }
  }
}

function clearHighlights() {
  const highlights = document.querySelectorAll('.stream-entry .search-highlight')
  highlights.forEach(mark => {
    const text = mark.textContent || ''
    const textNode = document.createTextNode(text)
    mark.parentNode?.replaceChild(textNode, mark)
  })
}

function parseTagTokens(query: string): string[] {
  return Array.from(
    new Set(
      query
        .trim()
        .split(/\s+/)
        .map(token => (token.startsWith('#') ? token.slice(1) : ''))
        .filter(token => token.length > 0),
    ),
  )
}

async function filterStreamEntries(query: string) {
  if (!searchIndex || !isIndexBuilt) {
    await buildSearchIndex()
  }

  const trimmedQuery = query.trim()
  const streamEntries = document.querySelectorAll<HTMLElement>('.stream-entry')

  clearHighlights()

  if (!trimmedQuery) {
    streamEntries.forEach(entry => {
      entry.style.display = ''
    })
    updateSearchStatus('')
    return
  }

  const lowerQuery = trimmedQuery.toLowerCase()
  const isTagQuery = lowerQuery.startsWith('#')
  const tagTokens = isTagQuery ? parseTagTokens(lowerQuery) : []
  const matchedEntryIds = new Set<string>()
  let highlightTerm = trimmedQuery

  if (isTagQuery) {
    if (tagTokens.length === 0) {
      streamEntries.forEach(entry => {
        entry.style.display = ''
      })
      updateSearchStatus("type a tag name after '#'")
      return
    }

    for (const entry of indexedEntries) {
      const normalizedTags = entry.tags.map(tag => tag.toLowerCase())
      const matchesAll = tagTokens.every(token =>
        normalizedTags.some(entryTag => entryTag.startsWith(token)),
      )
      if (matchesAll) {
        matchedEntryIds.add(entry.entryId)
      }
    }

    highlightTerm = tagTokens.join(' ')
  } else {
    try {
      const results = await searchIndex.searchAsync({
        query: trimmedQuery,
        limit: 500,
        index: ['content', 'metadata', 'isoDate', 'displayDate', 'tags'],
      })

      for (const fieldResult of Object.values(results)) {
        if (fieldResult && (fieldResult as any).result) {
          for (const id of (fieldResult as any).result) {
            const entry = indexedEntries[Number(id)]
            if (entry) {
              matchedEntryIds.add(entry.entryId)
            }
          }
        }
      }
    } catch (err) {
      console.error('[StreamSearch] Search failed:', err)
      updateSearchStatus('search error')
      return
    }
  }

  let visibleCount = 0
  streamEntries.forEach(entry => {
    const entryId = entry.dataset.entryId
    if (entryId && matchedEntryIds.has(entryId)) {
      entry.style.display = ''

      const contentEl = entry.querySelector('.stream-entry-content') as HTMLElement
      if (contentEl && highlightTerm) {
        highlightTextNodes(contentEl, highlightTerm)
      }

      if (isTagQuery && highlightTerm) {
        const tagElements = entry.querySelectorAll('.stream-entry-tag')
        tagElements.forEach(tagEl => {
          highlightTextNodes(tagEl as HTMLElement, highlightTerm)
        })
      }
      visibleCount++
    } else {
      entry.style.display = 'none'
    }
  })

  if (isTagQuery) {
    const readableTags = tagTokens.map(tag => `#${tag}`).join(' ')
    updateSearchStatus(
      visibleCount > 0
        ? `showing ${visibleCount} ${visibleCount === 1 ? 'entry' : 'entries'} tagged ${readableTags}`
        : `no entries tagged ${readableTags}`,
    )
  } else {
    updateSearchStatus(
      visibleCount > 0
        ? `showing ${visibleCount} ${visibleCount === 1 ? 'entry' : 'entries'}`
        : `no results for "${trimmedQuery}"`,
    )
  }
}

function updateSearchStatus(message: string) {
  let statusEl = document.querySelector('.stream-search-status') as HTMLElement
  if (!statusEl && message) {
    statusEl = document.createElement('div')
    statusEl.className = 'stream-search-status'
    const form = document.querySelector('.stream-search-form')
    if (form) {
      form.after(statusEl)
    }
  }

  if (statusEl) {
    if (message) {
      statusEl.textContent = message
      statusEl.style.display = 'block'
    } else {
      statusEl.style.display = 'none'
    }
  }
}

document.addEventListener('nav', async () => {
  const currentPath = window.location.pathname
  const isStreamPage =
    currentPath === '/stream' || currentPath.startsWith('/stream/') || isStreamHost()
  if (!isStreamPage) return

  await buildSearchIndex()

  const form = document.querySelector<HTMLFormElement>('.stream-search-form')
  const searchInput = document.querySelector<HTMLInputElement>('.stream-search-input')

  if (!form || !searchInput) return

  const focusShortcutHandler = (event: KeyboardEvent) => {
    const key = event.key.toLowerCase()
    const isMetaDot = event.metaKey && key === '.'
    const isCommandK = (event.metaKey || event.ctrlKey) && key === 'k'
    if (!isMetaDot && !isCommandK) return

    const target = event.target as Element | null
    if (target) {
      const tag = target.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || (target as HTMLElement).isContentEditable) return
    }

    event.preventDefault()
    event.stopPropagation()
    event.stopImmediatePropagation()

    searchInput.focus()
    searchInput.select()
  }

  const focusListenerOptions: AddEventListenerOptions = { capture: true }
  document.addEventListener('keydown', focusShortcutHandler, focusListenerOptions)

  const handleInput = () => {
    if (searchTimeout !== null) window.clearTimeout(searchTimeout)
    searchTimeout = window.setTimeout(async () => {
      await filterStreamEntries(searchInput.value)
    }, 300)
  }

  const handleSubmit = (e: Event) => e.preventDefault()

  searchInput.addEventListener('input', handleInput)
  form.addEventListener('submit', handleSubmit)

  window.addCleanup(() => {
    searchInput.removeEventListener('input', handleInput)
    form.removeEventListener('submit', handleSubmit)
    document.removeEventListener('keydown', focusShortcutHandler, focusListenerOptions)
    if (searchTimeout !== null) {
      window.clearTimeout(searchTimeout)
      searchTimeout = null
    }
  })
})
