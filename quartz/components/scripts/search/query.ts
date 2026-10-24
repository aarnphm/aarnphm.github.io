import type { LexicalField, LexicalFieldResult } from '../lexical-search-types'
import { parseCodeQuery } from '../../../util/code-query'
import { escapeHTML } from '../../../util/escape'
import { pdfMetadata } from '../../../util/pdf-search'
import { isSearchExcluded } from '../../../util/search-exclude'
import { parseSearchQuery, searchScopeIncludes, type SearchScope } from '../../../util/search-scope'
import { highlight, tokenizeTerm } from '../../../util/search-text'
import { fuseSearchRanks } from '../../../util/semantic-search'
import { imageSimilarityStatus, searchableImage } from '../image-search-controls'
import { getLexicalSearchClient } from '../lexical-search'
import { pdfCoverageStatus, pdfPageLabel } from '../pdf-search-controls'
import { isSemanticCancellation } from '../semantic.inline'
import { resetSiteSemantic } from '../site-semantic'
import { searchCode } from './code'
import {
  ensureImageSearch,
  ensureSearchData,
  requireData,
  setSemanticState,
  updateModeUI,
} from './engine'
import { clearPreview, setImageSearch } from './preview'
import { completeSearchProgress, resetProgressBar, startSearchProgress } from './progress'
import { commitSearchDisplay, displayResults } from './results'
import {
  shared,
  type Item,
  type SearchMode,
  type SearchSession,
  type SearchType,
  type SimilarityResult,
} from './state'

const numTagResults = 10

const formatForDisplay = (
  s: SearchSession,
  term: string,
  id: number,
  renderType: SearchType,
  scope: SearchScope,
) => {
  const slug = s.idDataMap[id]
  const fileData = requireData(s)[slug]
  if (!fileData) return null
  if (isSearchExcluded(slug) || isSearchExcluded(fileData.fileName)) return null
  if (scope === 'images' && (!searchableImage(slug) || (s.image.ids && !s.image.ids.has(slug))))
    return null
  if (
    fileData.layout === 'letter' ||
    ((scope === 'notes' || scope === 'all') &&
      shared.searchMode === 'semantic' &&
      slug.includes('arena'))
  )
    return null
  const aliases: string[] = fileData.aliases
  const target = aliases.find(alias => alias.toLowerCase().includes(term.toLowerCase())) ?? ''

  const queryTokens = tokenizeTerm(term)
  const titleTokens = tokenizeTerm(fileData.title ?? '')
  const titleMatch = titleTokens.some(t => queryTokens.includes(t))

  return {
    id,
    slug,
    title: renderType === 'tags' || target ? fileData.title : highlight(term, fileData.title ?? ''),
    target,
    content: highlight(term, fileData.content ?? fileData.fileName ?? '', true),
    tags: highlightTags(term, fileData.tags, renderType),
    aliases: aliases,
    titleMatch,
    fileName: fileData.fileName,
    protected: fileData.protected,
  }
}

function highlightTags(term: string, tags: string[], renderType: SearchType) {
  if (!tags || renderType !== 'tags') {
    return []
  }

  const tagTerm = term.toLowerCase()
  return tags
    .map(tag => {
      if (tag.toLowerCase().includes(tagTerm)) {
        return `<li><p class="match-tag">#${tag}</p></li>`
      } else {
        return `<li><p>#${tag}</p></li>`
      }
    })
    .slice(0, numTagResults)
}

export async function showSimilarImages(s: SearchSession, seed: string) {
  const { image, results, searchBar, searchLayout } = s
  const token = ++s.searchSeq
  s.resultPages?.dispose()
  s.resultPages = null
  setImageSearch(s, true)
  image.semantic?.cancel()
  shared.semantic?.cancel()
  s.pdf.cancel()
  if (s.runSearchTimer !== null) window.clearTimeout(s.runSearchTimer)
  s.runSearchTimer = null
  shared.rawSearchTerm = 'filter: images'
  shared.currentSearchTerm = ''
  searchBar.value = shared.rawSearchTerm
  searchBar.focus()
  searchLayout.classList.add('display-results')
  clearPreview(s)
  results.replaceChildren(imageSimilarityStatus('finding similar images…'))
  startSearchProgress(s)
  try {
    await ensureSearchData(s)
    if (token !== s.searchSeq || !image.active) return
    const documents = await image.similarity.search(seed, s.idDataMap.length)
    if (token !== s.searchSeq || !image.active) return
    const hits = documents.flatMap(document => {
      const id = s.idDataMap.findIndex(slug => slug === document.slug)
      const item = id >= 0 ? formatForDisplay(s, '', id, 'basic', 'images') : null
      return item ? [{ item, similarity: document.score }] : []
    })
    await commitSearchDisplay(s, token, () => {
      displayResults(s, hits)
      results.dataset.imageSeed = seed
    })
  } catch (error) {
    if (isSemanticCancellation(error) || token !== s.searchSeq || !image.active) return
    results.replaceChildren(
      imageSimilarityStatus('similar images unavailable. Type an image query to continue.'),
    )
  } finally {
    if (token === s.searchSeq) completeSearchProgress(s)
  }
}

export async function runSearch(s: SearchSession, rawTerm: string, token: number) {
  try {
    await querySearch(s, rawTerm, token)
  } catch (error) {
    if (token !== s.searchSeq) return
    console.warn('[Search] query failed:', error)
    await commitSearchDisplay(s, token, () => {
      resetProgressBar(s)
      clearPreview(s)
      const message = document.createElement('p')
      message.setAttribute('role', 'status')
      message.textContent = 'Search could not load. Reload the page to retry.'
      s.results.replaceChildren(message)
    })
  }
}

async function querySearch(s: SearchSession, rawTerm: string, token: number) {
  const codeQuery = parseCodeQuery(rawTerm)
  if (codeQuery) {
    await searchCode(s, codeQuery, token)
    return
  }
  const { image, modeButtons, results } = s
  await ensureSearchData(s)
  if (token !== s.searchSeq) return
  const { scope, term: trimmed, explicit } = parseSearchQuery(rawTerm)
  setImageSearch(s, scope === 'images')
  if (scope === 'images') {
    const ids = await image.similarity.ids().catch(() => undefined)
    if (token !== s.searchSeq) return
    image.ids = ids
  }
  if (trimmed === '' && !explicit) {
    await commitSearchDisplay(s, token, () => {
      results.replaceChildren()
      clearPreview(s)
      s.currentHover = null
      shared.currentSearchTerm = ''
      resetProgressBar(s)
    })
    return
  }

  const modeForRanking: SearchMode = shared.searchMode
  const searchLimit = s.idDataMap.length
  const initialType: SearchType = trimmed.startsWith('#') ? 'tags' : 'basic'
  let workingType: SearchType = initialType
  let highlightTerm = trimmed
  let tagTerm = ''
  let searchResults: LexicalFieldResult[] = []
  const lexicalClient = getLexicalSearchClient()
  const loadedData = requireData(s)
  const { idDataMap, slugToIndex } = s

  if (trimmed === '') {
    searchResults = [
      {
        field: 'title',
        result: idDataMap.flatMap((slug, id) =>
          searchScopeIncludes(scope, { ...loadedData[slug], slug }) ? [id] : [],
        ),
      },
    ]
  } else if (initialType === 'tags') {
    tagTerm = trimmed.substring(1).trim()
    const separatorIndex = tagTerm.indexOf(' ')
    if (separatorIndex !== -1) {
      const tag = tagTerm.substring(0, separatorIndex).trim()
      const query = tagTerm.substring(separatorIndex + 1).trim()
      searchResults = await lexicalClient.search(loadedData, {
        query,
        scope,
        limit: searchLimit,
        index: ['title', 'fileName', 'content', 'aliases'],
        tag: { tags: tag },
      })
      if (token !== s.searchSeq) return
      workingType = 'basic'
      highlightTerm = query
    } else {
      searchResults = await lexicalClient.search(loadedData, {
        query: tagTerm,
        scope,
        limit: searchLimit,
        index: ['tags'],
      })
      if (token !== s.searchSeq) return
      highlightTerm = tagTerm
    }
  } else {
    searchResults = await lexicalClient.search(loadedData, {
      query: highlightTerm,
      scope,
      limit: searchLimit,
      index: ['title', 'fileName', 'content', 'aliases'],
    })
    if (token !== s.searchSeq) return
  }

  const getByField = (field: LexicalField): number[] => {
    const hit = searchResults.find(x => x.field === field)
    return hit ? [...hit.result] : []
  }

  const allIds: Set<number> = new Set([
    ...getByField('aliases'),
    ...getByField('title'),
    ...getByField('fileName'),
    ...getByField('content'),
    ...getByField('tags'),
  ])

  const candidateItems = new Map<string, Item>()
  const ensureItem = (id: number): Item | null => {
    const slug = idDataMap[id]
    if (!slug || !searchScopeIncludes(scope, { ...loadedData[slug], slug })) return null
    const cached = candidateItems.get(slug)
    if (cached) return cached
    const item = formatForDisplay(s, highlightTerm, id, workingType, scope)
    if (item) {
      candidateItems.set(slug, item)
      return item
    }
    return null
  }

  const baseIndices: number[] = []
  for (const id of allIds) {
    const item = ensureItem(id)
    if (!item) continue
    const idx = slugToIndex.get(item.slug)
    if (typeof idx === 'number') {
      baseIndices.push(idx)
    }
  }

  let semanticIds: number[] = []
  const semanticSimilarity = new Map<number, number>()

  const orchestrator =
    modeForRanking === 'semantic' && shared.semanticReady && (scope === 'notes' || scope === 'all')
      ? shared.semantic
      : null

  const render = async () => {
    if (token !== s.searchSeq) return
    const useSemantic = semanticIds.length > 0
    const boosts = new Map<number, number>()
    for (const id of baseIndices) {
      if (ensureItem(id)?.titleMatch) boosts.set(id, 1.5)
    }
    const rankedEntries = fuseSearchRanks(
      [
        { ids: baseIndices, weight: 1, boosts },
        { ids: semanticIds, weight: useSemantic ? 1 : 0 },
      ],
      useSemantic ? 60 : 0,
    ).flatMap(({ id }) => {
      const item = ensureItem(id)
      return item ? [{ item, similarity: semanticSimilarity.get(id) ?? Number.NaN }] : []
    })

    const displayEntries: SimilarityResult[] = []
    for (const entry of rankedEntries) {
      displayEntries.push({ item: entry.item, similarity: entry.similarity })
    }

    await commitSearchDisplay(s, token, () => {
      shared.currentSearchTerm = highlightTerm
      return displayResults(s, displayEntries)
    })
  }

  await render()

  if (scope === 'pdf' && initialType === 'basic' && highlightTerm) {
    startSearchProgress(s)
    const metadata = pdfMetadata(loadedData)
    const lexical = baseIndices.map(id => idDataMap[id])
    try {
      const documents = await s.pdf.search(
        highlightTerm,
        lexical,
        metadata,
        new Set(Object.keys(metadata)),
        modeForRanking === 'semantic',
      )
      if (token !== s.searchSeq) return
      const entries = documents.flatMap(document => {
        const id = slugToIndex.get(document.slug)
        const item = id === undefined ? null : ensureItem(id)
        if (!item) return []
        if (document.passage) {
          item.passage = document.passage
          item.title = highlight(highlightTerm, escapeHTML(document.passage.title))
          item.target = ''
          item.content = `${escapeHTML(pdfPageLabel(document.passage))} · ${highlight(highlightTerm, escapeHTML(document.passage.text), true)}`
        }
        return [{ item, similarity: document.score }]
      })
      await commitSearchDisplay(s, token, () => {
        displayResults(s, entries)
        results.prepend(pdfCoverageStatus(s.pdf.coverage))
      })
    } catch (error) {
      if (isSemanticCancellation(error) || token !== s.searchSeq) return
      console.warn('[Search] PDF search unavailable:', error)
      const entries = s.pdf.fallback(highlightTerm, lexical, metadata).flatMap(document => {
        const id = slugToIndex.get(document.slug)
        const item = id === undefined ? null : ensureItem(id)
        return item ? [{ item, similarity: Number.NaN }] : []
      })
      await commitSearchDisplay(s, token, () => {
        displayResults(s, entries)
        const status = document.createElement('p')
        status.setAttribute('role', 'status')
        status.textContent = 'pdf text search unavailable; showing title and filename matches'
        results.prepend(status)
      })
    } finally {
      if (token === s.searchSeq) completeSearchProgress(s)
    }
    return
  }

  if (scope === 'images' && initialType === 'basic' && highlightTerm) {
    startSearchProgress(s)
    try {
      const service = await ensureImageSearch(image)
      if (token !== s.searchSeq || !image.active) return
      const documents = await service.search(highlightTerm, image.ids?.size ?? searchLimit)
      if (token !== s.searchSeq || !image.active) return
      for (const { slug, score } of documents) {
        const id = slugToIndex.get(slug)
        if (id === undefined || !ensureItem(id)) continue
        semanticIds.push(id)
        semanticSimilarity.set(id, score)
      }
      completeSearchProgress(s)
      await render()
    } catch (error) {
      if (isSemanticCancellation(error) || token !== s.searchSeq || !image.active) return
      image.semantic?.dispose()
      image.semantic = null
      image.ready = null
      completeSearchProgress(s)
      await commitSearchDisplay(s, token, () => {
        const status = document.createElement('p')
        status.setAttribute('role', 'status')
        status.textContent = 'Visual search unavailable. Showing image metadata matches.'
        results.prepend(status)
      })
    }
    return
  }

  if (
    initialType === 'tags' ||
    !orchestrator ||
    !shared.semanticReady ||
    highlightTerm.length < 2
  ) {
    return
  }

  const showProgress = modeForRanking === 'semantic'
  if (showProgress) {
    startSearchProgress(s)
  }

  try {
    const documents = await orchestrator.search(highlightTerm, orchestrator.index.ids.length)
    if (token !== s.searchSeq) {
      return
    }

    semanticIds = []
    semanticSimilarity.clear()
    for (const { slug, score } of documents) {
      const id = slugToIndex.get(slug)
      if (id === undefined || !ensureItem(id)) continue
      semanticIds.push(id)
      semanticSimilarity.set(id, score)
    }

    semanticIds.forEach(docId => {
      ensureItem(docId)
    })
    if (showProgress) completeSearchProgress(s)
  } catch (err) {
    if (isSemanticCancellation(err) || token !== s.searchSeq) return
    console.warn('[SemanticClient] search failed:', err)
    if (showProgress) completeSearchProgress(s)
    resetSiteSemantic(orchestrator.client)
    shared.semantic = null
    shared.semanticReady = false
    shared.semanticInitFailed = true
    setSemanticState(s, 'unavailable')
    if (shared.searchMode === 'semantic') {
      shared.searchMode = 'lexical'
      updateModeUI(s, shared.searchMode)
    }
    modeButtons.forEach(button => {
      if ((button.dataset.mode as SearchMode) === 'semantic') {
        button.disabled = true
        button.setAttribute('aria-disabled', 'true')
      }
    })
  }

  await render()
}
