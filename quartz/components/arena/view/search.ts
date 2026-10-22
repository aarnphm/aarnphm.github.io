import type { SearchResultOptions, SearchScope } from '../model'
import { tokenizeTerm } from '../../../util/search-text'
import { addCleanup, session, type ArenaSearchIndex, type SearchIndexItem } from './session-state'

function appendHighlightedText(parent: HTMLElement, searchTerm: string, text: string) {
  const terms = tokenizeTerm(searchTerm)
    .map(term => term.toLowerCase())
    .filter(term => term.length > 0)
    .sort((a, b) => b.length - a.length)
  if (terms.length === 0) {
    parent.textContent = text
    return
  }

  const lowerText = text.toLowerCase()
  let cursor = 0
  while (cursor < text.length) {
    let nextIndex = -1
    let nextTerm = ''
    for (const term of terms) {
      const index = lowerText.indexOf(term, cursor)
      if (index === -1) continue
      if (
        nextIndex === -1 ||
        index < nextIndex ||
        (index === nextIndex && term.length > nextTerm.length)
      ) {
        nextIndex = index
        nextTerm = term
      }
    }

    if (nextIndex === -1) {
      parent.append(document.createTextNode(text.slice(cursor)))
      return
    }
    if (nextIndex > cursor) {
      parent.append(document.createTextNode(text.slice(cursor, nextIndex)))
    }
    const highlight = document.createElement('span')
    highlight.className = 'highlight'
    highlight.textContent = text.slice(nextIndex, nextIndex + nextTerm.length)
    parent.append(highlight)
    cursor = nextIndex + nextTerm.length
  }
}

export const getSearchIndex = () => session.searchIndex
export const setSearchIndex = (index: SearchIndexItem[]) => {
  session.searchIndex = index
}

// Fetch arena search index JSON
export async function fetchArenaSearchIndex(): Promise<ArenaSearchIndex | null> {
  if (session.arenaSearchData) return session.arenaSearchData
  const request = session.searchRequest

  try {
    const response = await fetch(
      session.channelViews?.searchSource ?? '/static/arena-search.json',
      { signal: request?.signal },
    )
    if (!response.ok) {
      console.warn(`Failed to fetch arena search index: ${response.status}`)
      return null
    }
    const data: ArenaSearchIndex = await response.json()
    if (request?.signal.aborted) return null
    session.arenaSearchData = data
    return session.arenaSearchData
  } catch (error) {
    if (request?.signal.aborted) return null
    console.error('Error fetching arena search index:', error)
    return null
  }
}

export async function buildSearchIndex(scope: SearchScope): Promise<SearchIndexItem[] | null> {
  const index: SearchIndexItem[] = []

  if (scope === 'channel') {
    // For channel pages, fetch JSON and filter by current channel
    const data = await fetchArenaSearchIndex()
    if (!data) return null

    const currentSlug = document.body?.dataset.slug || ''
    const channelSlug = currentSlug.replace(/^arena\//, '')

    data.blocks
      .filter(block => block.channelSlug === channelSlug)
      .forEach(block => {
        index.push({
          blockId: block.id,
          channelSlug: block.channelSlug,
          channelName: block.channelName,
          title: block.title || block.content,
          content: block.content,
          highlighted: block.highlighted,
          hasModalInDom: block.hasModalInDom,
          tags: block.tags,
        })
      })
  } else {
    // For index page, use all blocks from JSON
    const data = await fetchArenaSearchIndex()
    if (!data) return null

    data.blocks.forEach(block => {
      index.push({
        blockId: block.id,
        channelSlug: block.channelSlug,
        channelName: block.channelName,
        title: block.title || block.content,
        content: block.content,
        highlighted: block.highlighted,
        hasModalInDom: block.hasModalInDom,
        tags: block.tags,
      })
    })
  }

  return index
}

function parseSearchQuery(query: string): { tagQueries: string[]; regularTerms: string[] } {
  const tagQueries: string[] = []
  let remainingQuery = query

  const tagPatternRegex =
    /#([^#]+?)(?=\s*#|\s*r\/|$)|r\/([^\s#]+(?:\s+[^\s#]+)*?)(?=\s*#|\s*r\/|$)/g
  let match

  while ((match = tagPatternRegex.exec(query)) !== null) {
    const tagText = (match[1] || match[2] || '').trim()
    if (tagText.length > 0) {
      tagQueries.push(tagText)
      remainingQuery = remainingQuery.replace(match[0], ' ')
    }
  }

  const regularTerms = remainingQuery
    .trim()
    .split(/\s+/)
    .filter(term => term.length > 0)

  return { tagQueries, regularTerms }
}

export function performSearch(query: string, index: SearchIndexItem[]): SearchIndexItem[] {
  const lowerQuery = query.toLowerCase().trim()
  if (lowerQuery.length < 2) return []

  const { tagQueries, regularTerms } = parseSearchQuery(lowerQuery)
  const tokens = tokenizeTerm(regularTerms.join(' '))

  const scoredResults = index
    .map(item => {
      const lowerTitle = item.title.toLowerCase()
      const lowerContent = item.content.toLowerCase()
      const itemTags = item.tags || []
      const matchedTags: string[] = []

      let score = 0
      let titleMatchCount = 0
      let contentMatchCount = 0
      let tagMatchCount = 0

      // Tag-specific queries (# or r/ prefix)
      for (const tagQuery of tagQueries) {
        const lowerTagQuery = tagQuery.toLowerCase()
        for (const tag of itemTags) {
          const lowerTag = tag.toLowerCase()
          if (lowerTag === lowerTagQuery) {
            score += 25
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          } else if (lowerTag.includes(lowerTagQuery)) {
            score += 20
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          }
        }
      }

      // Regular terms match tags
      for (const term of regularTerms) {
        const lowerTerm = term.toLowerCase()
        for (const tag of itemTags) {
          const lowerTag = tag.toLowerCase()
          if (lowerTag === lowerTerm) {
            score += 18
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          } else if (lowerTag.includes(lowerTerm)) {
            score += 15
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          }
        }
      }

      // Check each token for title and content matches
      for (const token of tokens) {
        const tokenLower = token.toLowerCase()

        if (lowerTitle.includes(tokenLower)) {
          titleMatchCount++
          score += 10
        }

        if (lowerContent.includes(tokenLower)) {
          contentMatchCount++
          score += 5
        }
      }

      // Boost for exact phrase match in title/content
      if (regularTerms.length > 0) {
        const regularQuery = regularTerms.join(' ').toLowerCase()
        if (lowerTitle.includes(regularQuery)) {
          score += 20
        }
        if (lowerContent.includes(regularQuery)) {
          score += 10
        }
      }

      // Boost for highlighted items
      if (item.highlighted) {
        score += 3
      }

      // Must have at least one match
      if (score === 0) return null

      const resultItem = { ...item }
      if (matchedTags.length > 0) {
        resultItem.matchedTags = matchedTags
      }

      return { item: resultItem, score, titleMatchCount, contentMatchCount, tagMatchCount }
    })
    .filter((result): result is NonNullable<typeof result> => result !== null)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score

      if (b.tagMatchCount !== a.tagMatchCount) {
        return b.tagMatchCount - a.tagMatchCount
      }

      if (b.titleMatchCount !== a.titleMatchCount) {
        return b.titleMatchCount - a.titleMatchCount
      }

      return b.contentMatchCount - a.contentMatchCount
    })

  return scoredResults.map(result => result.item)
}

export function renderSearchResults(
  results: SearchIndexItem[],
  scope: SearchScope,
  searchQuery: string = '',
) {
  const container = document.getElementById('arena-search-container')
  if (!container) return

  if (results.length === 0) {
    container.innerHTML = '<div class="arena-search-no-results">no results found</div>'
    container.classList.add('active')
    return
  }

  const fragment = document.createDocumentFragment()
  results.forEach((result, idx) => {
    const resultItem = document.createElement('div')
    resultItem.className = 'arena-search-result-item'
    resultItem.setAttribute('data-block-id', result.blockId)
    if (result.channelSlug) {
      resultItem.setAttribute('data-channel-slug', result.channelSlug)
    }
    resultItem.dataset.index = String(idx)
    resultItem.tabIndex = -1
    resultItem.setAttribute('role', 'option')
    if (!resultItem.id) {
      const uniqueId = `arena-search-${result.blockId}`
      resultItem.id = uniqueId
    }

    const title = document.createElement('div')
    title.className = 'arena-search-result-title'
    appendHighlightedText(title, searchQuery, result.title)
    resultItem.appendChild(title)

    if (result.matchedTags && result.matchedTags.length > 0) {
      const tagsContainer = document.createElement('ul')
      tagsContainer.className = 'arena-search-result-tags'

      result.matchedTags.forEach(tag => {
        const tagBadge = document.createElement('li')
        tagBadge.className = 'arena-search-result-tag-badge'
        appendHighlightedText(tagBadge, searchQuery, tag)
        tagsContainer.appendChild(tagBadge)
      })

      resultItem.appendChild(tagsContainer)
    }

    if (scope === 'index' && result.channelName) {
      const badge = document.createElement('span')
      badge.className = 'arena-search-result-channel-badge'
      badge.textContent = result.channelName
      resultItem.appendChild(badge)
    }

    fragment.appendChild(resultItem)
  })

  container.innerHTML = ''
  container.appendChild(fragment)
  container.classList.add('active')
}

export const getSearchResults = () =>
  Array.from(
    document.querySelectorAll<HTMLElement>('#arena-search-container .arena-search-result-item'),
  )

export const resetActiveResultHighlight = () => {
  session.activeResultIndex = null
  if (session.searchInput) {
    session.searchInput.removeAttribute('aria-activedescendant')
  }
  getSearchResults().forEach(result => result.classList.remove('active'))
}

export const setActiveResult = (index: number | null, options?: SearchResultOptions) => {
  const results = getSearchResults()
  if (results.length === 0) {
    resetActiveResultHighlight()
    if (options?.focus && session.searchInput) {
      session.searchInput.focus()
    }
    return
  }

  if (index === null) {
    resetActiveResultHighlight()
    if (options?.focus && session.searchInput) {
      session.searchInput.focus()
    }
    return
  }

  const clamped = Math.max(0, Math.min(index, results.length - 1))
  results.forEach((result, idx) => {
    result.classList.toggle('active', idx === clamped)
  })

  const target = results[clamped]
  session.activeResultIndex = clamped
  if (!target.id) {
    const fallbackId = target.getAttribute('data-block-id') || `arena-result-${clamped}`
    target.id = `arena-search-${fallbackId}`
  }
  if (session.searchInput) {
    session.searchInput.setAttribute('aria-activedescendant', target.id)
  }

  if (options?.focus !== false) {
    target.focus()
  }

  if (options?.scroll !== false) {
    target.scrollIntoView({ block: 'nearest' })
  }
}

export const wireSearchResultsInteractions = () => {
  const items = getSearchResults()
  items.forEach(item => {
    const idx = Number.parseInt(item.dataset.index ?? '', 10)
    if (!Number.isInteger(idx)) return

    const onFocus = () => setActiveResult(idx, { focus: false, scroll: false })
    const onMouseEnter = () => setActiveResult(idx, { focus: false, scroll: false })

    item.addEventListener('focus', onFocus)
    item.addEventListener('mouseenter', onMouseEnter)

    addCleanup(() => {
      item.removeEventListener('focus', onFocus)
      item.removeEventListener('mouseenter', onMouseEnter)
    })
  })
}

export const focusSearchInput = (prefill?: string) => {
  if (!session.searchInput) return
  if (typeof prefill === 'string') {
    session.searchInput.value = prefill
    session.searchInput.dispatchEvent(new Event('input', { bubbles: true }))
  }
  resetActiveResultHighlight()
  const valueLength = session.searchInput.value.length
  session.searchInput.focus()
  try {
    if (valueLength > 0) {
      session.searchInput.select()
    } else {
      session.searchInput.setSelectionRange(valueLength, valueLength)
    }
  } catch {}
}

export const clearSearchState = (options?: { blur?: boolean }) => {
  if (session.searchInput) {
    session.searchInput.value = ''
    if (options?.blur !== false) {
      session.searchInput.blur()
    }
    session.searchInput.removeAttribute('aria-activedescendant')
  }
  resetActiveResultHighlight()
  closeSearchDropdown()
}

export function closeSearchDropdown() {
  const container = document.getElementById('arena-search-container')
  if (container) {
    container.classList.remove('active')
    container.innerHTML = ''
  }
}
