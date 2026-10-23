import { createNoteRanker, type NoteRanker } from '../../../util/note-pagerank'
import { FullSlug, isFullSlug, resolveRelative } from '../../../util/path'
import { isSearchExcluded } from '../../../util/search-exclude'
import { parseSearchQuery, searchScopeIncludes, type SearchScope } from '../../../util/search-scope'
import { fuseSearchRanks, type SemanticDocument } from '../../../util/semantic-search'
import {
  imageSimilarityButton,
  imageSimilarityStatus,
  searchableImage,
} from '../image-search-controls'
import { getLexicalSearchClient } from '../lexical-search'
import { noteRankDistribution } from '../note-rank-distribution'
import { isSemanticCancellation } from '../semantic.inline'
import { getCommandItems } from './actions'
import { appendHighlightedText } from './format'
import {
  ensureAttachmentData,
  ensureContentData,
  ensureImageSearch,
  ensureQuickOpen,
  hidePalette,
  paletteMode,
  quickOpenProgress,
  quickOpenStatus,
  setFocusFirstChild,
  type PaletteState,
} from './shell'

interface Item {
  slug: FullSlug
  name: string
  similarity?: number
}

export const numSearchResults = 13
const noteRankers = new WeakMap<ContentIndex, NoteRanker>()

export async function displayNearbyNotes(s: PaletteState, generation: number) {
  try {
    const metadata = await ensureContentData(s)
    if (generation !== s.queryGeneration || paletteMode.actionType !== 'quick_open' || !s.isActive)
      return
    let ranker = noteRankers.get(metadata)
    if (!ranker) {
      ranker = createNoteRanker(metadata)
      noteRankers.set(metadata, ranker)
    }
    const items = ranker
      .related(s.currentSlug, numSearchResults)
      .map(slug => ({ slug, name: metadata[slug].title || metadata[slug].fileName }))
    s.output.removeAttribute('aria-busy')
    quickOpenProgress(s, false)
    if (items.length) displayResults(s, items, '')
    else quickOpenStatus(s, '')
  } catch (error) {
    if (generation !== s.queryGeneration || paletteMode.actionType !== 'quick_open' || !s.isActive)
      return
    console.warn('[QuickOpen] nearby notes unavailable:', error)
    s.output.removeAttribute('aria-busy')
    quickOpenProgress(s, false)
    quickOpenStatus(s, 'Nearby notes are unavailable.')
  }
}

export async function querySearch(
  s: PaletteState,
  currentSearchTerm: string,
  generation = s.queryGeneration,
) {
  const queryActionType = paletteMode.actionType
  if (paletteMode.actionType === 'quick_open') {
    const { term: query, scope, explicit } = parseSearchQuery(currentSearchTerm)
    if (!query && (!explicit || scope === 'notes')) {
      await displayNearbyNotes(s, generation)
      return
    }
    s.output.setAttribute('aria-busy', 'true')
    quickOpenProgress(s, true)
    let scopedDocuments: SemanticDocument[] = []
    let settled = false
    try {
      if (scope !== 'notes' || !query) {
        const content = await ensureContentData(s)
        const loaded = scope === 'notes' ? content : await ensureAttachmentData(s)
        if (
          generation !== s.queryGeneration ||
          queryActionType !== paletteMode.actionType ||
          !s.isActive
        )
          return
        if (scope === 'images') {
          const ids = await s.imageSimilarity.ids().catch(() => undefined)
          if (generation !== s.queryGeneration || !s.isActive) return
          s.imageIds = ids
        }
        const slugs = Object.entries(loaded)
          .filter(([slug, source]) => !isSearchExcluded(slug) && !isSearchExcluded(source.fileName))
          .map(([slug]) => slug)
        const hits = query
          ? await getLexicalSearchClient().search(loaded, {
              query,
              scope,
              limit: scope === 'images' ? slugs.length : numSearchResults,
              index: ['title', 'fileName', 'aliases'],
            })
          : [{ result: slugs.map((_, id) => id) }]
        if (generation !== s.queryGeneration || !s.isActive) return
        scopedDocuments = [...new Set(hits.flatMap(hit => hit.result))].flatMap(id => {
          const slug = slugs[id]
          return slug ? [{ slug, score: Number.NaN }] : []
        })
        displayQuickOpenResults(s, scopedDocuments, query, scope)
        if (scope === 'images' && query) {
          quickOpenProgress(s, true)
          s.output.setAttribute('aria-busy', 'true')
          const service = await ensureImageSearch(s)
          if (generation !== s.queryGeneration || !s.isActive) return
          const documents = await service.search(query, numSearchResults * 8)
          if (
            generation !== s.queryGeneration ||
            queryActionType !== paletteMode.actionType ||
            !s.isActive
          )
            return
          const scores = new Map(documents.map(document => [document.slug, document.score]))
          const combined = fuseSearchRanks([
            { ids: scopedDocuments.map(document => document.slug), weight: 1 },
            { ids: documents.map(document => document.slug), weight: 1 },
          ]).map(({ id }) => ({ slug: id, score: scores.get(id) ?? Number.NaN }))
          displayQuickOpenResults(s, combined, query, scope)
          return
        }
        if ((scope !== 'notes' && scope !== 'all') || !query) return
        s.output.setAttribute('aria-busy', 'true')
        quickOpenProgress(s, true)
      }
      void ensureContentData(s).then(
        () => {
          if (settled || generation !== s.queryGeneration || !s.isActive) return
          displayPendingTitles(s, query, undefined, scope)
        },
        () => {},
      )
      const service = await ensureQuickOpen(s)
      if (
        generation !== s.queryGeneration ||
        queryActionType !== paletteMode.actionType ||
        !s.isActive
      )
        return
      const documents = await service.search(query, numSearchResults * 8)
      if (
        generation !== s.queryGeneration ||
        queryActionType !== paletteMode.actionType ||
        !s.isActive
      )
        return
      const scores = new Map(documents.map(document => [document.slug, document.score]))
      const combined = fuseSearchRanks([
        { ids: scopedDocuments.map(document => document.slug), weight: 1 },
        { ids: documents.map(document => document.slug), weight: 1 },
      ]).map(({ id }) => ({ slug: id, score: scores.get(id) ?? Number.NaN }))
      displayQuickOpenResults(s, combined, query, scope)
    } catch (error) {
      if (isSemanticCancellation(error)) return
      if (
        generation !== s.queryGeneration ||
        queryActionType !== paletteMode.actionType ||
        !s.isActive
      )
        return
      console.warn('[QuickOpen] semantic search failed:', error)
      s.output.removeAttribute('aria-busy')
      quickOpenProgress(s, false)
      if (scope === 'images') {
        s.imageSemantic?.dispose()
        s.imageSemantic = null
        s.imageReady = null
        displayQuickOpenResults(s, scopedDocuments, query, scope)
        const status = document.createElement('p')
        status.setAttribute('role', 'status')
        status.textContent = 'Visual search unavailable. Showing image metadata matches.'
        s.output.prepend(status)
      } else if (scopedDocuments.length) displayQuickOpenResults(s, scopedDocuments, query, scope)
      else
        displayPendingTitles(
          s,
          query,
          'semantic unavailable; use ctrl+k for full-text search',
          scope,
        )
    } finally {
      settled = true
    }
  } else {
    const query = currentSearchTerm.toLowerCase().trim()
    const actions = paletteMode.actionType === 'command' ? s.commandActs : s.connectorActs
    const matchedActions = query
      ? actions.filter(
          action =>
            action.name.toLowerCase().includes(query) ||
            action.detail?.toLowerCase().includes(query) ||
            action.auxInnerHtml.toLowerCase().includes(query),
        )
      : actions

    getCommandItems(s, matchedActions)
  }
}

function displayPendingTitles(
  s: PaletteState,
  query: string,
  message?: string,
  scope: SearchScope = 'notes',
) {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
  const items = Object.entries(s.data ?? {})
    .flatMap(([slug, note]) => {
      if (
        !isFullSlug(slug) ||
        note.protected ||
        isSearchExcluded(slug) ||
        isSearchExcluded(note.fileName) ||
        !searchScopeIncludes(scope, { ...note, slug })
      )
        return []
      const title = note.title || note.fileName
      const text = [title, note.fileName, ...(note.aliases ?? [])].join(' ').toLowerCase()
      return terms.every(term => text.includes(term)) ? [{ slug, name: title }] : []
    })
    .sort((a, b) => {
      const match = (item: Item) => Number(item.name.toLowerCase() === query.toLowerCase())
      return match(b) - match(a)
    })
    .slice(0, numSearchResults)
  if (items.length) displayResults(s, items, query)
  else s.output.replaceChildren()
  if (message) {
    const status = document.createElement('p')
    status.setAttribute('role', 'status')
    status.textContent = items.length ? `${message}; title matches shown` : message
    s.output.prepend(status)
  } else {
    s.output.setAttribute('aria-busy', 'true')
  }
}

async function showSimilarImages(s: PaletteState, seed: string) {
  const generation = ++s.queryGeneration
  s.imageSemantic?.cancel()
  s.quickOpenSemantic?.cancel()
  window.clearTimeout(s.quickOpenTimer)
  paletteMode.currentSearchTerm = 'filter: images'
  s.bar.value = paletteMode.currentSearchTerm
  s.bar.focus()
  s.output.replaceChildren(imageSimilarityStatus('finding similar images…'))
  quickOpenProgress(s, true)
  s.output.setAttribute('aria-busy', 'true')
  try {
    await ensureAttachmentData(s)
    if (generation !== s.queryGeneration || !s.isActive) return
    const documents = await s.imageSimilarity.search(seed, numSearchResults * 8)
    if (generation !== s.queryGeneration || !s.isActive) return
    displayQuickOpenResults(s, documents, '', 'images')
    s.output.dataset.imageSeed = seed
  } catch (error) {
    if (isSemanticCancellation(error) || generation !== s.queryGeneration || !s.isActive) return
    s.output.replaceChildren(
      imageSimilarityStatus('similar images unavailable. Type an image query to continue.'),
    )
  } finally {
    if (generation === s.queryGeneration) {
      quickOpenProgress(s, false)
      s.output.removeAttribute('aria-busy')
    }
  }
}

export function displayQuickOpenResults(
  s: PaletteState,
  documents: SemanticDocument[],
  query: string,
  scope: SearchScope,
) {
  const items: Item[] = []
  for (const document of documents) {
    if (!isFullSlug(document.slug) || isSearchExcluded(document.slug)) continue
    if (
      scope === 'images' &&
      (!searchableImage(document.slug) || (s.imageIds && !s.imageIds.has(document.slug)))
    )
      continue
    const fileData = s.data?.[document.slug]
    if (
      !fileData ||
      fileData.protected ||
      isSearchExcluded(fileData.fileName) ||
      !searchScopeIncludes(scope, { ...fileData, slug: document.slug })
    )
      continue
    items.push({
      slug: document.slug,
      name: fileData.title || fileData.fileName,
      similarity: document.score,
    })
    if (items.length === numSearchResults) break
  }
  s.output.removeAttribute('aria-busy')
  quickOpenProgress(s, false)
  displayResults(s, items, query)
}

function displayResults(s: PaletteState, finalResults: Item[], currentSearchTerm: string) {
  if (paletteMode.actionType !== 'quick_open') return
  if (!finalResults) return

  s.output.replaceChildren()
  s.currentHover = null

  const noMatchEl = document.createElement('div')
  noMatchEl.classList.add('suggestion-item', 'no-match')
  noMatchEl.tabIndex = -1
  const noMatchContent = document.createElement('div')
  noMatchContent.className = 'suggestion-content'
  const noMatchTitle = document.createElement('div')
  noMatchTitle.className = 'suggestion-title'
  noMatchTitle.textContent = currentSearchTerm
  noMatchContent.appendChild(noMatchTitle)
  const noMatchAux = document.createElement('div')
  noMatchAux.className = 'suggestion-aux'
  const noMatchAction = document.createElement('span')
  noMatchAction.className = 'suggestion-action'
  noMatchAction.textContent = 'enter to schedule a chat'
  noMatchAux.appendChild(noMatchAction)
  noMatchEl.append(noMatchContent, noMatchAux)

  const onNoMatchClick = () => {
    window.location.href = `mailto:contact@aarnphm.xyz?subject=Chat about: ${encodeURIComponent(currentSearchTerm)}`
    hidePalette(s)
  }

  noMatchEl.addEventListener('click', onNoMatchClick)
  window.addCleanup(() => noMatchEl.removeEventListener('click', onNoMatchClick))
  if (finalResults.length === 0) {
    s.output.appendChild(noMatchEl)
  } else {
    s.output.append(...finalResults.map(result => toHtml(s, result)))
  }
  setFocusFirstChild(s)
}

function toHtml(s: PaletteState, { name, slug, similarity }: Item) {
  const item = document.createElement('div')
  item.classList.add('suggestion-item')
  item.tabIndex = -1
  item.dataset.slug = slug

  const content = document.createElement('a')
  content.href = new URL(resolveRelative(s.currentSlug, slug), location.toString()).toString()
  content.classList.add('suggestion-content')
  const title = document.createElement('div')
  title.classList.add('suggestion-title')
  appendHighlightedText(title, parseSearchQuery(paletteMode.currentSearchTerm).term, name)
  content.appendChild(title)

  const aux = document.createElement('div')
  aux.classList.add('suggestion-aux')
  if (!searchableImage(slug) && similarity !== undefined && Number.isFinite(similarity)) {
    item.dataset.similarity = String(similarity)
    const score = document.createElement('span')
    score.className = 'result-likelihood'
    score.title = 'cosine similarity'
    score.setAttribute('aria-label', `Cosine similarity: ${similarity.toFixed(3)}`)
    score.textContent = similarity.toFixed(3)
    aux.append(score)
  }
  if (s.data) {
    let ranker = noteRankers.get(s.data)
    if (!ranker) {
      ranker = createNoteRanker(s.data)
      noteRankers.set(s.data, ranker)
    }
    const rank = ranker.position(slug)
    if (rank) {
      aux.append(noteRankDistribution(ranker, rank))
    }
  }

  if (searchableImage(slug))
    aux.append(imageSimilarityButton(slug, seed => showSimilarImages(s, seed), similarity))
  item.append(content, aux)

  const onClick = (event: MouseEvent) => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    window.spaNavigate(new URL(resolveRelative(s.currentSlug, slug), location.toString()))
    hidePalette(s)
  }

  const onMouseEnter = () => {
    // Remove focus class from all other items
    s.output.querySelectorAll<HTMLDivElement>('.suggestion-item.focus').forEach(el => {
      el.classList.remove('focus')
    })
    // Add focus to current item
    item.classList.add('focus')
    s.currentHover = item
  }

  item.addEventListener('click', onClick)
  item.addEventListener('mouseenter', onMouseEnter)
  window.addCleanup(() => {
    item.removeEventListener('click', onClick)
    item.removeEventListener('mouseenter', onMouseEnter)
  })

  return item
}
