import type { LexicalIndexPriority } from '../lexical-search-types'
import { FullSlug } from '../../../util/path'
import { isSearchExcluded } from '../../../util/search-exclude'
import { parseSearchQuery } from '../../../util/search-scope'
import { createImageSemantic } from '../image-semantic'
import { getLexicalSearchClient } from '../lexical-search'
import { siteSemantic } from '../site-semantic'
import { shared, type SearchImage, type SearchMode, type SearchSession } from './state'

const SEARCH_MODE_STORAGE_KEY = 'quartz:search:mode'

export const loadStoredSearchMode = (): SearchMode | null => {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    const stored = window.localStorage.getItem(SEARCH_MODE_STORAGE_KEY)
    return stored === 'lexical' || stored === 'semantic' ? stored : null
  } catch (err) {
    console.warn('[Search] failed to read stored search mode:', err)
    return null
  }
}

export const persistSearchMode = (mode: SearchMode) => {
  if (typeof window === 'undefined') {
    return
  }

  try {
    window.localStorage.setItem(SEARCH_MODE_STORAGE_KEY, mode)
  } catch (err) {
    console.warn('[Search] failed to persist search mode:', err)
  }
}

let searchDataPromise: Promise<ContentIndex> | null = null

async function loadSearchData(): Promise<ContentIndex> {
  if (typeof fetchSearchData === 'undefined') {
    return fetchData
  }

  try {
    return await fetchSearchData
  } catch (err) {
    console.warn('[Search] failed to load search index:', err)
    return fetchData
  }
}

export function getSearchData(): Promise<ContentIndex> {
  searchDataPromise ??= loadSearchData()
  return searchDataPromise
}

export function ensureImageSearch(image: SearchImage) {
  image.ready ??= createImageSemantic()
    .then(service => {
      if (!image.active) {
        service.dispose()
        throw new Error('image search page closed')
      }
      image.semantic = service
      return service
    })
    .catch(error => {
      image.ready = null
      throw error
    })
  return image.ready
}

export function requireData(s: SearchSession) {
  if (!s.data) throw new Error('search data is not loaded')
  return s.data
}

// Publish `data` only after the index is filled: the opener and the first query both call this,
// and a query against a half-filled index returns whichever folders were added first.
export function ensureSearchData(
  s: SearchSession,
  priority: LexicalIndexPriority = 'user-visible',
) {
  const baseReady = (s.searchDataReady ??= (async () => {
    const loaded = await getSearchData()
    await fillDocument(loaded, priority)
    s.idDataMap = Object.keys(loaded).filter(
      slug => !isSearchExcluded(slug) && !isSearchExcluded(loaded[slug].fileName),
    ) as FullSlug[]
    s.slugToIndex = new Map<string, number>()
    s.idDataMap.forEach((slug, idx) => s.slugToIndex.set(slug, idx))
    s.data = loaded
    return loaded
  })().catch(err => {
    s.searchDataReady = null
    throw err
  }))
  if (priority === 'user-visible') {
    return baseReady.then(async loaded => {
      await fillDocument(loaded, 'user-visible')
      return loaded
    })
  }
  return baseReady
}

export function updateModeUI(s: SearchSession, mode: SearchMode) {
  const { modeButtons, modeToggle, searchLayout } = s
  modeButtons.forEach(button => {
    const btnMode = (button.dataset.mode as SearchMode) ?? 'lexical'
    const isActive = btnMode === mode
    button.classList.toggle('active', isActive)
    button.setAttribute('aria-pressed', String(isActive))
  })
  if (modeToggle) {
    modeToggle.dataset.mode = mode
  }
  searchLayout.dataset.mode = mode
  setSemanticState(
    s,
    s.semanticBooting ? 'loading' : shared.semanticInitFailed ? 'unavailable' : 'ready',
  )
}

export function setSemanticState(
  s: SearchSession,
  state: 'idle' | 'loading' | 'ready' | 'unavailable',
) {
  const { modeButtons } = s
  const { scope } = parseSearchQuery(s.searchBar.value)
  if (scope === 'images' || scope === 'pdf') state = 'ready'
  modeButtons.forEach(button => {
    const btnMode = (button.dataset.mode as SearchMode) ?? 'lexical'
    if (btnMode !== 'semantic') return
    const disabled = state === 'loading' || state === 'unavailable'
    button.disabled = disabled
    button.setAttribute('aria-disabled', String(disabled))
    // The icon spins instead of a "loading" label beside the input.
    button.classList.toggle('is-loading', state === 'loading')
    button.setAttribute('aria-busy', state === 'loading' ? 'true' : 'false')
  })
}

async function bootSemantic(s: SearchSession): Promise<boolean> {
  setSemanticState(s, 'loading')
  try {
    shared.semantic = await siteSemantic()
    shared.semanticReady = true
    setSemanticState(s, 'ready')
    return true
  } catch (err) {
    console.warn('[SemanticClient] initialization failed:', err)
    shared.semantic = null
    shared.semanticReady = false
    shared.semanticInitFailed = true
    setSemanticState(s, 'unavailable')
    return false
  }
}

export async function ensureSemantic(s: SearchSession): Promise<boolean> {
  if (shared.semanticReady) return true
  if (shared.semanticInitFailed) return false
  s.semanticBooting ??= bootSemantic(s).finally(() => {
    s.semanticBooting = null
  })
  return s.semanticBooting
}

function fillDocument(
  data: ContentIndex,
  priority: LexicalIndexPriority = 'user-visible',
): Promise<void> {
  return getLexicalSearchClient().ensureIndexed(data, priority)
}
