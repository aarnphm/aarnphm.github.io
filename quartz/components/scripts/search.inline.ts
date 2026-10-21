import type { LexicalField, LexicalFieldResult, LexicalIndexPriority } from './lexical-search-types'
import { documentPreviewUrl, readDocumentPreview } from '../../util/document-preview'
import { escapeHTML } from '../../util/escape'
import { fetchCanonical } from '../../util/fetch-canonical'
import {
  FullSlug,
  getFullSlug,
  isFullSlug,
  normalizeRelativeURLs,
  resolveRelative,
} from '../../util/path'
import { isSearchExcluded } from '../../util/search-exclude'
import {
  isSearchModeToggle,
  isSearchToggle,
  searchModeShortcutLabel,
} from '../../util/search-shortcuts'
import { highlight, tokenizeTerm } from '../../util/search-text'
import { fuseSearchRanks } from '../../util/semantic-search'
import { renderCoursePreview } from './course-preview-card'
import { renderDocumentPreview } from './document-preview-card'
import { registerEscapeHandler } from './escape-handler'
import { getLexicalSearchClient } from './lexical-search'
import { fitSearchPath } from './search-path'
import { isSemanticCancellation } from './semantic.inline'
import { resetSiteSemantic, siteSemantic, type SiteSemantic } from './site-semantic'

type Item = {
  id: number
  slug: FullSlug
  title: string
  content: string
  tags: string[]
  aliases: string[]
  target: string
  fileName: string
  protected?: boolean
  titleMatch?: boolean
}

type SearchType = 'basic' | 'tags'
type SearchMode = 'lexical' | 'semantic'
const SEARCH_MODE_STORAGE_KEY = 'quartz:search:mode'

const loadStoredSearchMode = (): SearchMode | null => {
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

const persistSearchMode = (mode: SearchMode) => {
  if (typeof window === 'undefined') {
    return
  }

  try {
    window.localStorage.setItem(SEARCH_MODE_STORAGE_KEY, mode)
  } catch (err) {
    console.warn('[Search] failed to persist search mode:', err)
  }
}

let searchMode: SearchMode = 'lexical'
let currentSearchTerm: string = ''
let rawSearchTerm: string = ''
let semantic: SiteSemantic | null = null
let semanticReady = false
let semanticInitFailed = false
let searchDataPromise: Promise<ContentIndex> | null = null
const configuredSearchElements = new WeakSet<HTMLDivElement>()
type SimilarityResult = { item: Item; similarity: number }

const p = new DOMParser()
const fetchContentCache: Map<FullSlug, Document> = new Map()
const numSearchResultsLexical = 10
const numSearchResultsSemantic = 60
const numTagResults = 10

function fileExtension(fileName: string): string | null {
  const name = fileName.split('/').at(-1) ?? fileName
  const index = name.lastIndexOf('.')
  if (index <= 0 || index === name.length - 1) {
    return null
  }

  return name.slice(index).toLowerCase()
}

function isDirectContentFile(fileName: string): boolean {
  const ext = fileExtension(fileName)
  return ext !== null && ext !== '.canvas' && ext !== '.base' && ext !== '.bases'
}

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

function getSearchData(): Promise<ContentIndex> {
  searchDataPromise ??= loadSearchData()
  return searchDataPromise
}

function getNumSearchResults(mode: SearchMode): number {
  return mode === 'semantic' ? numSearchResultsSemantic : numSearchResultsLexical
}
function highlightHTML(searchTerm: string, el: HTMLElement) {
  const p = new DOMParser()
  const tokenizedTerms = tokenizeTerm(searchTerm)
  const html = p.parseFromString(el.innerHTML, 'text/html')

  const createHighlightSpan = (text: string) => {
    const span = document.createElement('span')
    span.className = 'highlight'
    span.textContent = text
    return span
  }

  const highlightTextNodes = (node: Node, term: string) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const nodeText = node.nodeValue ?? ''
      const regex = new RegExp(term.toLowerCase(), 'gi')
      const matches = nodeText.match(regex)
      if (!matches || matches.length === 0) return
      const spanContainer = document.createElement('span')
      let lastIndex = 0
      for (const match of matches) {
        const matchIndex = nodeText.indexOf(match, lastIndex)
        spanContainer.appendChild(document.createTextNode(nodeText.slice(lastIndex, matchIndex)))
        spanContainer.appendChild(createHighlightSpan(match))
        lastIndex = matchIndex + match.length
      }
      spanContainer.appendChild(document.createTextNode(nodeText.slice(lastIndex)))
      node.parentNode?.replaceChild(spanContainer, node)
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      if ((node as HTMLElement).classList.contains('highlight')) return
      Array.from(node.childNodes).forEach(child => highlightTextNodes(child, term))
    }
  }

  for (const term of tokenizedTerms) {
    highlightTextNodes(html.body, term)
  }

  return html.body
}

async function setupSearch(searchElement: HTMLDivElement, currentSlug: FullSlug) {
  searchElement.dataset.currentSlug = currentSlug
  if (configuredSearchElements.has(searchElement)) {
    return
  }
  configuredSearchElements.add(searchElement)

  const containerElement = searchElement.querySelector<HTMLElement>('.search-container')
  if (!containerElement) return
  const container = containerElement

  const searchButton = searchElement.querySelector<HTMLButtonElement>('.search-button')
  if (!searchButton) return

  const searchBarElement = searchElement.querySelector<HTMLInputElement>('.search-bar')
  if (!searchBarElement) return
  const searchBar = searchBarElement

  const searchLayoutElement = searchElement.querySelector<HTMLOutputElement>('.search-layout')
  if (!searchLayoutElement) return
  const searchLayout = searchLayoutElement

  const searchSpace = searchElement.querySelector<HTMLFormElement>('.search-space')
  if (!searchSpace) return

  const progressBar = document.createElement('div')
  progressBar.className = 'semantic-search-progress'
  progressBar.setAttribute('role', 'progressbar')
  progressBar.setAttribute('aria-label', 'Loading search')
  progressBar.setAttribute('aria-hidden', 'true')
  searchBar.parentElement?.appendChild(progressBar)
  let progressResetTimer = 0
  const progressReasons = new Set<'search' | 'preview'>()

  const startSearchProgress = (reason: 'search' | 'preview' = 'search') => {
    progressReasons.add(reason)
    progressBar.setAttribute(
      'aria-label',
      progressReasons.has('search') ? 'Loading search' : 'Loading preview',
    )
    window.clearTimeout(progressResetTimer)
    progressBar.setAttribute('aria-hidden', 'false')
    progressBar.style.opacity = '1'
    progressBar.style.width = '100%'
    progressBar.style.animation = 'semantic-progress-sweep 2.5s linear infinite'
  }

  const completeSearchProgress = (reason: 'search' | 'preview' = 'search') => {
    progressReasons.delete(reason)
    if (progressReasons.size > 0) {
      progressBar.setAttribute(
        'aria-label',
        progressReasons.has('search') ? 'Loading search' : 'Loading preview',
      )
      return
    }
    window.clearTimeout(progressResetTimer)
    progressBar.setAttribute('aria-hidden', 'true')
    progressBar.style.animation = 'none'
    progressBar.style.opacity = '0'
    progressResetTimer = window.setTimeout(() => {
      progressBar.style.width = '0'
      progressBar.style.backgroundPosition = '-100% 0'
    }, 300)
  }

  const resetProgressBar = () => {
    progressReasons.clear()
    window.clearTimeout(progressResetTimer)
    progressBar.setAttribute('aria-hidden', 'true')
    progressBar.style.animation = 'none'
    progressBar.style.opacity = '0'
    progressBar.style.width = '0'
    progressBar.style.backgroundPosition = '-100% 0'
  }

  let data: ContentIndex | null = null
  let idDataMap: FullSlug[] = []
  let slugToIndex = new Map<string, number>()
  const requireData = () => {
    if (!data) throw new Error('search data is not loaded')
    return data
  }
  // Publish `data` only after the index is filled: the opener and the first query both call this,
  // and a query against a half-filled index returns whichever folders were added first.
  let searchDataReady: Promise<ContentIndex> | null = null
  const ensureSearchData = (priority: LexicalIndexPriority = 'user-visible') => {
    const baseReady = (searchDataReady ??= (async () => {
      const loaded = await getSearchData()
      await fillDocument(loaded, priority)
      idDataMap = Object.keys(loaded) as FullSlug[]
      slugToIndex = new Map<string, number>()
      idDataMap.forEach((slug, idx) => slugToIndex.set(slug, idx))
      data = loaded
      return loaded
    })().catch(err => {
      searchDataReady = null
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
  const el = searchSpace.querySelector('ul#helper')
  const modeToggle = searchSpace.querySelector('.search-mode-toggle') as HTMLDivElement | null
  const modeButtons = modeToggle
    ? Array.from(modeToggle.querySelectorAll<HTMLButtonElement>('.mode-option'))
    : []
  const semanticStatus = searchSpace.querySelector<HTMLElement>('.semantic-status')

  const appendLayout = (el: HTMLElement) => {
    searchLayout.appendChild(el)
  }

  if (!el) {
    const keys = [
      { kbd: '↑↓', description: 'pour naviguer' },
      { kbd: '↵', description: 'pour ouvrir' },
      { kbd: 'esc', description: 'pour rejeter' },
      {
        kbd: searchModeShortcutLabel(/mac/i.test(navigator.userAgent)),
        description: 'pour basculer en recherche sémantique',
      },
    ]
    const helper = document.createElement('ul')
    helper.id = 'helper'
    for (const { kbd, description } of keys) {
      const liEl = document.createElement('li')
      liEl.innerHTML = `<kbd>${escapeHTML(kbd)}</kbd>${description}`
      helper.appendChild(liEl)
    }
    searchSpace.appendChild(helper)
  }

  const updateModeUI = (mode: SearchMode) => {
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
  }

  const setSemanticState = (state: 'idle' | 'loading' | 'ready' | 'unavailable') => {
    if (semanticStatus) {
      semanticStatus.dataset.state = state
      semanticStatus.textContent = state === 'unavailable' ? 'semantic (unavailable)' : ''
    }
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
    if (state !== 'ready' && searchMode === 'semantic') {
      searchMode = 'lexical'
      updateModeUI(searchMode)
    }
  }

  const enablePreview = searchLayout.dataset.preview === 'true'
  const storedMode = loadStoredSearchMode()
  let semanticBooting: Promise<boolean> | null = null
  const bootSemantic = async (): Promise<boolean> => {
    setSemanticState('loading')
    try {
      semantic = await siteSemantic()
      semanticReady = true
      setSemanticState('ready')
      return true
    } catch (err) {
      console.warn('[SemanticClient] initialization failed:', err)
      semantic = null
      semanticReady = false
      semanticInitFailed = true
      setSemanticState('unavailable')
      return false
    }
  }

  const ensureSemantic = async (): Promise<boolean> => {
    if (semanticReady) return true
    if (semanticInitFailed) return false
    semanticBooting ??= bootSemantic().finally(() => {
      semanticBooting = null
    })
    return semanticBooting
  }

  if (semanticReady) {
    setSemanticState('ready')
  } else if (semanticInitFailed) {
    setSemanticState('unavailable')
  } else {
    setSemanticState('idle')
  }
  if (storedMode === 'semantic') {
    if (semanticReady) {
      searchMode = storedMode
    }
  } else if (storedMode === 'lexical') {
    searchMode = storedMode
  }
  if (!semanticReady && searchMode === 'semantic') {
    searchMode = 'lexical'
  }
  let searchSeq = 0
  let runSearchTimer: number | null = null
  searchLayout.dataset.mode = searchMode

  let requestedMode: SearchMode = searchMode
  let modeRequest = 0
  const triggerSearchWithMode = (mode: SearchMode) => {
    requestedMode = mode
    const request = ++modeRequest
    if (mode === 'semantic' && !semanticReady) {
      void ensureSemantic().then(ready => {
        if (request !== modeRequest) return
        if (ready) triggerSearchWithMode('semantic')
        else {
          requestedMode = 'lexical'
          updateModeUI('lexical')
        }
      })
      updateModeUI(mode)
      return
    }
    if (searchMode === mode) {
      updateModeUI(mode)
      return
    }
    semantic?.cancel()
    if (runSearchTimer !== null) {
      window.clearTimeout(runSearchTimer)
      runSearchTimer = null
    }
    searchMode = mode
    updateModeUI(mode)
    persistSearchMode(searchMode)
    if (rawSearchTerm.trim() !== '') {
      searchLayout.classList.add('display-results')
      const token = ++searchSeq
      void runSearch(rawSearchTerm, token)
    }
  }

  updateModeUI(searchMode)

  modeButtons.forEach(button => {
    const btnMode = (button.dataset.mode as SearchMode) ?? 'lexical'
    if (btnMode === 'semantic') {
      button.disabled = semanticInitFailed
      button.setAttribute('aria-disabled', String(semanticInitFailed))
    }
    const handler = () => triggerSearchWithMode(btnMode)
    button.addEventListener('click', handler)
    window.addCleanup(() => button.removeEventListener('click', handler))
  })
  let preview: HTMLDivElement | undefined = undefined
  let previewInner: HTMLDivElement | undefined = undefined
  let previewRequest: AbortController | undefined

  function clearPreview() {
    previewRequest?.abort()
    previewRequest = undefined
    preview?.replaceChildren()
  }

  let results = searchLayout.querySelector('.results-container') as HTMLDivElement
  if (!results) {
    results = document.createElement('div')
    results.className = 'results-container'
    appendLayout(results)
  }

  const fitResultPaths = () => {
    for (const label of results.querySelectorAll<HTMLElement>('.search-result-path')) {
      fitSearchPath(label, currentSearchTerm)
    }
  }
  const pathObserver = new ResizeObserver(fitResultPaths)
  pathObserver.observe(results)
  document.fonts.addEventListener('loadingdone', fitResultPaths)
  window.addCleanup(() => {
    pathObserver.disconnect()
    document.fonts.removeEventListener('loadingdone', fitResultPaths)
  })

  if (enablePreview) {
    preview = searchLayout.querySelector('.preview-container') as HTMLDivElement
    if (!preview) {
      preview = document.createElement('div')
      preview.className = 'preview-container'
      appendLayout(preview)
    }
  }

  let currentHover: HTMLElement | null = null
  let previousFocus: HTMLElement | null = null
  const pressedKeys = new Set<string>()
  let composing = false
  let pendingNavigation: -1 | 1 | null = null
  let pendingDisplay: { token: number; apply: () => void | Promise<void> } | null = null

  function commitSearchDisplay(token: number, apply: () => void | Promise<void>) {
    if (token !== searchSeq) return
    if (pressedKeys.size > 0 || composing) {
      pendingDisplay = { token, apply }
      return
    }
    pendingDisplay = null
    searchLayout.classList.toggle('display-results', rawSearchTerm.trim() !== '')
    searchLayout.removeAttribute('aria-busy')
    return apply()
  }

  function flushSearchDisplay() {
    if (pressedKeys.size > 0 || composing) return
    const pending = pendingDisplay
    pendingDisplay = null
    if (pending) void commitSearchDisplay(pending.token, pending.apply)
  }

  function hideSearch(focusButton = true) {
    modeRequest++
    requestedMode = searchMode
    updateModeUI(searchMode)
    semantic?.cancel()
    searchSeq += 1
    pressedKeys.clear()
    composing = false
    pendingNavigation = null
    pendingDisplay = null
    if (runSearchTimer !== null) {
      window.clearTimeout(runSearchTimer)
      runSearchTimer = null
    }
    container!.classList.remove('active')
    if (container!.matches(':popover-open')) container!.hidePopover()
    searchBar!.value = ''
    rawSearchTerm = ''
    currentSearchTerm = ''
    currentHover = null
    results.replaceChildren()
    clearPreview()
    searchLayout!.classList.remove('display-results')
    searchLayout.removeAttribute('aria-busy')
    if (focusButton) {
      const target = previousFocus?.isConnected ? previousFocus : searchButton
      target?.focus({ preventScroll: true })
    }
    previousFocus = null
    resetProgressBar()
  }

  function navigateSearchResult(anchor: HTMLAnchorElement) {
    const href = anchor.getAttribute('href')
    if (!href) return
    window.spaNavigate(new URL(href, window.location.toString()))
    hideSearch()
  }

  function showSearch(type: SearchType) {
    if (!container!.matches(':popover-open')) {
      previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
      container!.showPopover()
    }
    container!.classList.add('active')
    if (!data) {
      startSearchProgress()
      void ensureSearchData().then(
        () => {
          if (container.classList.contains('active')) completeSearchProgress()
        },
        error => {
          resetProgressBar()
          console.warn('[Search] initialization failed:', error)
        },
      )
    }
    if (type === 'tags') {
      searchBar!.value = '#'
      rawSearchTerm = '#'
    }
    searchBar!.focus()
  }

  async function shortcutHandler(e: HTMLElementEventMap['keydown']) {
    if (e.defaultPrevented) return
    const paletteOpen = document.querySelector('search#palette-container') as HTMLDivElement
    if (paletteOpen && paletteOpen.classList.contains('active')) return
    if (e.isComposing || composing) return

    const inlineSearch = document.querySelector<HTMLElement>('#base-search-bar')
    const key = e.key.toLowerCase()

    if (isSearchToggle(e)) {
      if (
        !container.classList.contains('active') &&
        inlineSearch?.checkVisibility({ visibilityProperty: true })
      )
        return
      if (key === 'k' && !container.classList.contains('active')) {
        const hasInlineSearch = [...document.querySelectorAll('.page-list-search-container')].some(
          element => element.checkVisibility({ visibilityProperty: true }),
        )
        if (hasInlineSearch) return

        const streamSearchInput = document.querySelector<HTMLInputElement>('.stream-search-input')
        if (streamSearchInput) {
          e.preventDefault()
          streamSearchInput.focus()
          streamSearchInput.select()
          return
        }
      }
      e.preventDefault()
      if (e.repeat) return
      const searchBarOpen = container!.classList.contains('active')
      if (searchBarOpen) {
        hideSearch()
      } else {
        showSearch('basic')
      }
      return
    } else if (e.shiftKey && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault()
      if (e.repeat) return
      const searchBarOpen = container!.classList.contains('active')
      if (searchBarOpen) {
        hideSearch()
      } else {
        showSearch('tags')
      }
      return
    }

    if (!container!.classList.contains('active')) return
    if (isSearchModeToggle(e) && container.contains(document.activeElement)) {
      if (requestedMode === 'lexical' && semanticInitFailed) return
      e.preventDefault()
      if (!e.repeat) triggerSearchWithMode(requestedMode === 'semantic' ? 'lexical' : 'semantic')
      return
    }
    if (preview?.contains(document.activeElement)) {
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        currentHover?.focus()
      }
      return
    }
    if (e.key === 'ArrowRight' && results.contains(document.activeElement)) {
      const link = preview?.querySelector<HTMLAnchorElement>('a[href]')
      if (link?.checkVisibility()) {
        e.preventDefault()
        link.focus()
      }
      return
    }
    if (e.key === 'Enter') {
      let anchor: HTMLAnchorElement | undefined
      if (results.contains(document.activeElement)) {
        anchor = document.activeElement as HTMLAnchorElement
        if (anchor.classList.contains('no-match')) return
        await displayPreview(anchor)
        e.preventDefault()
      } else {
        anchor = document.getElementsByClassName('result-card')[0] as HTMLAnchorElement
        if (!anchor || anchor.classList.contains('no-match')) return
        await displayPreview(anchor)
        e.preventDefault()
      }
      if (anchor !== undefined) navigateSearchResult(anchor)
    } else if (
      e.key === 'ArrowUp' ||
      (e.shiftKey && e.key === 'Tab') ||
      (e.ctrlKey && e.key === 'p')
    ) {
      e.preventDefault()
      pressedKeys.add(e.code || e.key)
      pendingNavigation = -1
    } else if (e.key === 'ArrowDown' || e.key === 'Tab' || (e.ctrlKey && e.key === 'n')) {
      e.preventDefault()
      pressedKeys.add(e.code || e.key)
      pendingNavigation = 1
    } else if (
      e.target === searchBar &&
      (e.key.length === 1 || e.key === 'Backspace' || e.key === 'Delete')
    ) {
      pressedKeys.add(e.code || e.key)
    }
  }

  function releaseKey(e: HTMLElementEventMap['keyup']) {
    pressedKeys.delete(e.code || e.key)
    if (pressedKeys.size > 0 || composing) return
    flushSearchDisplay()
    const direction = pendingNavigation
    pendingNavigation = null
    if (!container.classList.contains('active')) return
    if (direction === -1 && results.contains(document.activeElement)) {
      const currentResult = currentHover ?? document.activeElement
      void focusResult(currentResult?.previousElementSibling)
    } else if (direction === 1 && (document.activeElement === searchBar || currentHover !== null)) {
      const currentResult = currentHover ?? results.querySelector<HTMLElement>('.result-card')
      void focusResult(currentResult?.nextElementSibling)
    }
  }

  function releasePressedKeys() {
    pressedKeys.clear()
    pendingNavigation = null
    flushSearchDisplay()
  }

  async function focusResult(result: Element | null | undefined) {
    if (!(result instanceof HTMLElement)) return
    currentHover?.classList.remove('focus')
    currentHover = result
    result.classList.add('focus')
    result.focus()
    await displayPreview(result)
  }

  const formatForDisplay = (term: string, id: number, renderType: SearchType) => {
    const slug = idDataMap[id]
    const fileData = requireData()[slug]
    if (!fileData) return null
    if (isSearchExcluded(slug) || isSearchExcluded(fileData.fileName)) return null
    if (fileData.layout === 'letter' || (searchMode === 'semantic' && slug.includes('arena')))
      return null
    const aliases: string[] = fileData.aliases
    const target = aliases.find(alias => alias.toLowerCase().includes(term.toLowerCase())) ?? ''

    const queryTokens = tokenizeTerm(term)
    const titleTokens = tokenizeTerm(fileData.title ?? '')
    const titleMatch = titleTokens.some(t => queryTokens.includes(t))

    return {
      id,
      slug,
      title:
        renderType === 'tags' || target ? fileData.title : highlight(term, fileData.title ?? ''),
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

  function resolveUrl(slug: FullSlug): URL {
    const baseSlug = (searchElement.dataset.currentSlug as FullSlug | undefined) ?? currentSlug
    return new URL(resolveRelative(baseSlug, slug), location.toString())
  }

  const resultToHTML = ({ item, similarity }: SimilarityResult) => {
    const { slug, title, content, tags, target, fileName } = item
    const isProtected = item.protected === true
    const htmlTags = tags.length > 0 ? `<ul class="tags">${tags.join('')}</ul>` : ``
    const itemTile = document.createElement('a')
    const titleContent = target ? highlight(currentSearchTerm, target) : title
    const subscript = target ? `<b>${slug}</b>` : ``
    const similarityLabel = Number.isFinite(similarity) ? similarity.toFixed(3) : '—'
    itemTile.classList.add('result-card')
    itemTile.id = slug
    itemTile.href = resolveUrl(slug).toString()

    const fileData = requireData()[slug]
    const resolvedFileName = (fileData?.fileName || fileName || slug || '').toLowerCase()
    const extension = fileExtension(resolvedFileName)
    const isCanvas = extension === '.canvas'
    const isBases = extension === '.base' || extension === '.bases'
    const isDirectFile = isDirectContentFile(resolvedFileName)
    if (isCanvas) itemTile.dataset.canvas = 'true'
    if (isBases) itemTile.dataset.bases = 'true'
    if (isDirectFile) itemTile.dataset.directFile = 'true'

    if (isProtected) {
      itemTile.dataset.protected = 'true'
      itemTile.innerHTML = `<hgroup>
        <h3>${titleContent}</h3>
        ${subscript}${htmlTags}
        <span class="search-protected-badge" style="
          font-size: 0.75rem;
          color: var(--gray);
          font-style: italic;
          margin-left: 0.5rem;
        ">🔒 protected content</span>
        ${searchMode === 'semantic' ? `<span class="result-likelihood" title="cosine similarity">&nbsp;${similarityLabel}</span>` : ''}
      </hgroup>`
    } else {
      delete itemTile.dataset.protected
      itemTile.innerHTML = `<hgroup>
        <h3>${titleContent}</h3>
        ${subscript}${htmlTags}
        ${searchMode === 'semantic' ? `<span class="result-likelihood" title="cosine similarity">&nbsp;${similarityLabel}</span>` : ''}
        ${enablePreview && window.innerWidth > 600 ? '' : `<p>${content}</p>`}
      </hgroup>`
    }

    const accessibleName = itemTile.textContent?.trim() ?? ''
    for (const label of itemTile.querySelectorAll<HTMLElement>('h3, hgroup > b')) {
      const path = label.textContent ?? ''
      if (!path.includes('/') && label.tagName !== 'B' && path !== slug && path !== fileName)
        continue
      label.classList.add('search-result-path')
      label.title = path
      itemTile.setAttribute('aria-label', accessibleName)
    }

    if (Number.isFinite(similarity)) itemTile.dataset.similarity = String(similarity)

    const handler = (evt: MouseEvent) => {
      if (evt.altKey || evt.ctrlKey || evt.metaKey || evt.shiftKey) return
      const anchor = evt.currentTarget as HTMLAnchorElement | null
      if (!anchor) return
      evt.preventDefault()
      navigateSearchResult(anchor)
    }

    async function onMouseEnter(ev: MouseEvent) {
      if (!ev.target) return
      const target = ev.target as HTMLInputElement
      await displayPreview(target)
    }

    itemTile.addEventListener('mouseenter', onMouseEnter)
    window.addCleanup(() => itemTile.removeEventListener('mouseenter', onMouseEnter))
    itemTile.addEventListener('click', handler)
    window.addCleanup(() => itemTile.removeEventListener('click', handler))

    return itemTile
  }

  async function displayResults(finalResults: SimilarityResult[]) {
    results.replaceChildren()
    if (finalResults.length === 0) {
      results.innerHTML = `<a class="result-card no-match">
          <h3>No results.</h3>
          <p>Try another search term?</p>
      </a>`
      currentHover = null
    } else {
      results.append(...finalResults.map(resultToHTML))
      fitResultPaths()
    }

    if (finalResults.length === 0 && preview) {
      clearPreview()
    } else {
      const firstChild = results.querySelector<HTMLElement>('.result-card')
      if (!firstChild) return
      firstChild.classList.add('focus')
      currentHover = firstChild
      await displayPreview(firstChild)
    }
  }

  async function fetchContent(slug: FullSlug, signal: AbortSignal): Promise<Document> {
    const cached = fetchContentCache.get(slug)
    if (cached) return cached

    const targetUrl = resolveUrl(slug)
    const response = await fetchCanonical(targetUrl, { signal })
    if (!response.ok) throw new Error(`Could not fetch ${targetUrl}: ${response.status}`)
    const page = p.parseFromString(await response.text(), 'text/html')
    normalizeRelativeURLs(page, targetUrl)
    fetchContentCache.set(slug, page)
    return page
  }

  async function displayPreview(el: HTMLElement | null) {
    if (!enablePreview || !el || !preview) return
    const slug = el.id
    if (!isFullSlug(slug)) return
    clearPreview()
    const request = new AbortController()
    previewRequest = request
    startSearchProgress('preview')
    request.signal.addEventListener('abort', () => completeSearchProgress('preview'), {
      once: true,
    })
    try {
      const isProtected = el.dataset.protected === 'true' || requireData()[slug]?.protected === true
      if (isProtected) {
        previewInner = document.createElement('div')
        previewInner.classList.add('preview-inner', 'preview-redacted')

        const blurLayer = document.createElement('div')
        blurLayer.className = 'preview-redacted-blur'
        const label = document.createElement('span')
        label.className = 'preview-redacted-label'
        label.textContent = 'redacted'

        previewInner.append(blurLayer, label)
        preview.replaceChildren(previewInner)
        return
      }

      const documentType =
        el.dataset.canvas === 'true' ? 'canvas' : el.dataset.bases === 'true' ? 'base' : null
      if (documentType) {
        const loading = document.createElement('p')
        loading.setAttribute('role', 'status')
        loading.textContent = 'Loading preview…'
        preview.appendChild(loading)
        try {
          const response = await fetch(documentPreviewUrl(slug, resolveUrl(slug).toString()), {
            signal: request.signal,
          })
          const summary = response.ok
            ? readDocumentPreview(await response.json(), documentType, slug)
            : null
          if (request.signal.aborted) return
          if (summary) {
            const inner = document.createElement('div')
            inner.classList.add('preview-inner', 'document-popover', 'search-document-preview')
            renderDocumentPreview(summary, resolveUrl(slug).toString(), inner)
            preview.replaceChildren(inner)
            preview.scrollTop = 0
            return
          }
        } catch (error) {
          if (request.signal.aborted) return
          console.warn('[Search] document preview unavailable:', error)
        }
      }

      if (documentType || el.dataset.directFile === 'true') {
        const fileData = requireData()[slug]
        previewInner = document.createElement('div')
        previewInner.classList.add('preview-inner')

        const metaContainer = document.createElement('div')
        metaContainer.style.padding = '1rem'
        metaContainer.style.color = 'var(--gray)'
        metaContainer.style.fontSize = '0.9rem'

        const metadata = (label: string, value: string) => {
          const row = document.createElement('p')
          const heading = document.createElement('strong')
          heading.textContent = `${label}:`
          row.append(heading, ` ${value}`)
          metaContainer.appendChild(row)
        }

        metadata('File', fileData?.fileName || slug)

        if (fileData?.description) {
          metadata('Description', fileData.description)
        }

        if (fileData?.tags && fileData.tags.length > 0) {
          metadata('Tags', fileData.tags.join(', '))
        }

        previewInner.appendChild(metaContainer)
        preview.replaceChildren(previewInner)
        return
      }

      const page = await fetchContent(slug, request.signal)
      if (request.signal.aborted) return
      if (page.querySelector('.course-home, [data-course-card]')) {
        const inner = document.createElement('div')
        inner.classList.add('preview-inner', 'course-popover', 'search-course-preview')
        renderCoursePreview(page, resolveUrl(slug), inner)
        preview.replaceChildren(inner)
        preview.scrollTop = 0
        return
      }

      const innerDiv = Array.from(page.querySelectorAll<HTMLElement>('.popover-hint')).flatMap(
        el => [...highlightHTML(currentSearchTerm, el).children],
      )
      previewInner = document.createElement('div')
      previewInner.classList.add('preview-inner')
      previewInner.append(...innerDiv)
      if (previewInner.querySelector('.stream-feed')) previewInner.classList.add('stream-preview')
      preview.replaceChildren(previewInner)

      const highlights = [...preview.getElementsByClassName('highlight')].sort(
        (a, b) => b.innerHTML.length - a.innerHTML.length,
      )
      if (highlights.length > 0) {
        const highlight = highlights[0]
        const containerRect = preview.getBoundingClientRect()
        const highlightRect = highlight.getBoundingClientRect()
        const relativeTop = highlightRect.top - containerRect.top + preview.scrollTop - 20
        preview.scrollTo({ top: relativeTop, behavior: 'smooth' })
      }
    } catch (error) {
      if (request.signal.aborted) return
      console.warn('[Search] preview unavailable:', error)
      const message = document.createElement('p')
      message.setAttribute('role', 'status')
      message.textContent = 'preview unavailable. '
      const open = document.createElement('a')
      open.href = resolveUrl(slug).toString()
      open.textContent = 'open page'
      message.appendChild(open)
      preview.replaceChildren(message)
    } finally {
      if (previewRequest === request) completeSearchProgress('preview')
    }
  }

  async function runSearch(rawTerm: string, token: number) {
    try {
      await querySearch(rawTerm, token)
    } catch (error) {
      if (token !== searchSeq) return
      console.warn('[Search] query failed:', error)
      await commitSearchDisplay(token, () => {
        resetProgressBar()
        clearPreview()
        const message = document.createElement('p')
        message.setAttribute('role', 'status')
        message.textContent = 'Search could not load. Reload the page to retry.'
        results.replaceChildren(message)
      })
    }
  }

  async function querySearch(rawTerm: string, token: number) {
    await ensureSearchData()
    if (token !== searchSeq) return
    const trimmed = rawTerm.trim()
    if (trimmed === '') {
      await commitSearchDisplay(token, () => {
        results.replaceChildren()
        clearPreview()
        currentHover = null
        currentSearchTerm = ''
        resetProgressBar()
      })
      return
    }

    const modeForRanking: SearchMode = searchMode
    const initialType: SearchType = trimmed.startsWith('#') ? 'tags' : 'basic'
    let workingType: SearchType = initialType
    let highlightTerm = trimmed
    let tagTerm = ''
    let searchResults: LexicalFieldResult[] = []
    const lexicalClient = getLexicalSearchClient()
    const loadedData = requireData()

    if (initialType === 'tags') {
      tagTerm = trimmed.substring(1).trim()
      const separatorIndex = tagTerm.indexOf(' ')
      if (separatorIndex !== -1) {
        const tag = tagTerm.substring(0, separatorIndex).trim()
        const query = tagTerm.substring(separatorIndex + 1).trim()
        searchResults = await lexicalClient.search(loadedData, {
          query,
          limit: Math.max(getNumSearchResults(modeForRanking), 10000),
          index: ['title', 'fileName', 'content', 'aliases'],
          tag: { tags: tag },
        })
        if (token !== searchSeq) return
        workingType = 'basic'
        highlightTerm = query
      } else {
        searchResults = await lexicalClient.search(loadedData, {
          query: tagTerm,
          limit: getNumSearchResults(modeForRanking),
          index: ['tags'],
        })
        if (token !== searchSeq) return
        highlightTerm = tagTerm
      }
    } else {
      searchResults = await lexicalClient.search(loadedData, {
        query: highlightTerm,
        limit: getNumSearchResults(modeForRanking),
        index: ['title', 'fileName', 'content', 'aliases'],
      })
      if (token !== searchSeq) return
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
      if (!slug) return null
      const cached = candidateItems.get(slug)
      if (cached) return cached
      const item = formatForDisplay(highlightTerm, id, workingType)
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

    const orchestrator = modeForRanking === 'semantic' && semanticReady ? semantic : null

    const render = async () => {
      if (token !== searchSeq) return
      const useSemantic = semanticReady && semanticIds.length > 0
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
      )
        .flatMap(({ id }) => {
          const item = ensureItem(id)
          return item ? [{ item, similarity: semanticSimilarity.get(id) ?? Number.NaN }] : []
        })
        .slice(0, getNumSearchResults(modeForRanking))

      const displayEntries: SimilarityResult[] = []
      for (const entry of rankedEntries) {
        displayEntries.push({ item: entry.item, similarity: entry.similarity })
      }

      await commitSearchDisplay(token, () => {
        currentSearchTerm = highlightTerm
        return displayResults(displayEntries)
      })
    }

    await render()

    if (initialType === 'tags' || !orchestrator || !semanticReady || highlightTerm.length < 2) {
      return
    }

    const showProgress = modeForRanking === 'semantic'
    if (showProgress) {
      startSearchProgress()
    }

    try {
      const documents = await orchestrator.search(
        highlightTerm,
        getNumSearchResults(modeForRanking) * 8,
      )
      if (token !== searchSeq) {
        return
      }

      semanticIds = []
      semanticSimilarity.clear()
      for (const { slug, score } of documents) {
        const id = slugToIndex.get(slug)
        if (id === undefined || !ensureItem(id)) continue
        semanticIds.push(id)
        semanticSimilarity.set(id, score)
        if (semanticIds.length >= getNumSearchResults(modeForRanking)) break
      }

      semanticIds.forEach(docId => {
        ensureItem(docId)
      })
      if (showProgress) completeSearchProgress()
    } catch (err) {
      if (isSemanticCancellation(err) || token !== searchSeq) return
      console.warn('[SemanticClient] search failed:', err)
      if (showProgress) completeSearchProgress()
      resetSiteSemantic(orchestrator.client)
      semantic = null
      semanticReady = false
      semanticInitFailed = true
      setSemanticState('unavailable')
      if (searchMode === 'semantic') {
        searchMode = 'lexical'
        updateModeUI(searchMode)
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

  function onType() {
    semantic?.cancel()
    previewRequest?.abort()
    rawSearchTerm = searchBar.value
    const hasQuery = rawSearchTerm.trim() !== ''
    const term = rawSearchTerm
    const token = ++searchSeq
    pendingDisplay = null
    searchLayout.setAttribute('aria-busy', String(hasQuery))
    if (runSearchTimer !== null) {
      window.clearTimeout(runSearchTimer)
      runSearchTimer = null
    }
    if (composing) return
    if (!hasQuery || searchMode === 'lexical') {
      void runSearch(term, token)
      return
    }
    runSearchTimer = window.setTimeout(() => {
      runSearchTimer = null
      void runSearch(term, token)
    }, 60)
  }

  const startComposition = () => {
    composing = true
  }
  const finishComposition = () => {
    composing = false
    pressedKeys.clear()
    onType()
  }

  document.addEventListener('keydown', shortcutHandler)
  document.addEventListener('keyup', releaseKey)
  window.addEventListener('blur', releasePressedKeys)
  window.addCleanup(() => {
    document.removeEventListener('keydown', shortcutHandler)
    document.removeEventListener('keyup', releaseKey)
    window.removeEventListener('blur', releasePressedKeys)
  })
  const warmSearch = () => {
    if (data) return
    void ensureSearchData().catch(error => console.warn('[Search] warm failed:', error))
  }
  const openHandler = () => showSearch('basic')
  searchButton.addEventListener('pointerenter', warmSearch)
  searchButton.addEventListener('focusin', warmSearch)
  searchButton.addEventListener('click', openHandler)
  window.addCleanup(() => {
    searchButton.removeEventListener('pointerenter', warmSearch)
    searchButton.removeEventListener('focusin', warmSearch)
    searchButton.removeEventListener('click', openHandler)
  })
  searchBar.addEventListener('input', onType)
  searchBar.addEventListener('compositionstart', startComposition)
  searchBar.addEventListener('compositionend', finishComposition)
  window.addCleanup(() => {
    searchBar.removeEventListener('input', onType)
    searchBar.removeEventListener('compositionstart', startComposition)
    searchBar.removeEventListener('compositionend', finishComposition)
  })
  window.addCleanup(() => {
    hideSearch(false)
  })

  // The lexical mode button inside carries `.active` while search is closed.
  registerEscapeHandler(container, hideSearch, () => container.classList.contains('active'))
  if (currentSlug === 'index') {
    let idle = 0
    let timer = 0
    const trail = document.querySelector<HTMLElement>('.homepage-trail-scene')
    const preload = () => {
      if (!searchElement.isConnected) return
      // A closed scene can finish preparing before the queued idle callback runs.
      if (
        trail?.shadowRoot?.querySelector<HTMLElement>('.trail-scene')?.dataset.trailReveal ===
        'revealing'
      ) {
        schedulePreload()
        return
      }
      void ensureSearchData('background').catch(error =>
        console.warn('[Search] preload failed:', error),
      )
    }
    const schedulePreload = () => {
      // Trail textures and search indexing share the main thread on a fresh homepage.
      const scene = trail?.shadowRoot?.querySelector<HTMLElement>('.trail-scene')
      if (trail && scene?.dataset.trailOpening !== 'complete') {
        trail.addEventListener('trail:prepared', schedulePreload, { once: true })
        return
      }
      if (scene?.dataset.trailReveal === 'revealing') {
        trail?.addEventListener('trail:reveal-end', schedulePreload, { once: true })
        return
      }
      if (window.requestIdleCallback) idle = window.requestIdleCallback(preload, { timeout: 2000 })
      else timer = window.setTimeout(preload, 1000)
    }
    const prefetch = () => {
      void getSearchData().catch(error => console.warn('[Search] prefetch failed:', error))
      schedulePreload()
    }
    if (document.readyState === 'complete') prefetch()
    else window.addEventListener('load', prefetch, { once: true })
    window.addCleanup(() => {
      window.removeEventListener('load', prefetch)
      trail?.removeEventListener('trail:prepared', schedulePreload)
      trail?.removeEventListener('trail:reveal-end', schedulePreload)
      if (idle) window.cancelIdleCallback(idle)
      window.clearTimeout(timer)
    })
  }
  window.addCleanup(() => {
    configuredSearchElements.delete(searchElement)
  })
}

function fillDocument(
  data: ContentIndex,
  priority: LexicalIndexPriority = 'user-visible',
): Promise<void> {
  return getLexicalSearchClient().ensureIndexed(data, priority)
}

document.addEventListener('nav', async () => {
  const onPagehide = (event: PageTransitionEvent) => {
    if (!event.persisted) getLexicalSearchClient().disposeForPagehide()
  }
  const onPageshow = (event: PageTransitionEvent) => {
    if (event.persisted) getLexicalSearchClient().recoverFromBfcache()
  }
  window.addEventListener('pagehide', onPagehide)
  window.addEventListener('pageshow', onPageshow)
  window.addCleanup(() => {
    window.removeEventListener('pagehide', onPagehide)
    window.removeEventListener('pageshow', onPageshow)
  })
  const currentSlug = getFullSlug(window)
  const searchElements = Array.from(
    document.getElementsByClassName('search') as HTMLCollectionOf<HTMLDivElement>,
  )
  for (const element of searchElements) {
    await setupSearch(element, currentSlug)
  }
})
