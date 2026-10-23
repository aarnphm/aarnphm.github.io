import { escapeHTML } from '../../util/escape'
import { FullSlug, getFullSlug } from '../../util/path'
import { searchModeShortcutLabel } from '../../util/search-shortcuts'
import { registerEscapeHandler } from './escape-handler'
import { createImageSimilarity } from './image-similarity'
import { getLexicalSearchClient } from './lexical-search'
import {
  ensureSearchData,
  getSearchData,
  loadStoredSearchMode,
  setSemanticState,
  updateModeUI,
} from './search/engine'
import {
  onType,
  releaseKey,
  releasePressedKeys,
  shortcutHandler,
  triggerSearchWithMode,
} from './search/input'
import { setPreviewVisible, togglePreview } from './search/preview'
import { showSimilarImages } from './search/query'
import { fitResultPaths, openSearchResult } from './search/results'
import { shared, type SearchImage, type SearchMode, type SearchSession } from './search/state'
import { hideSearch, showSearch } from './search/visibility'

const configuredSearchElements = new WeakSet<HTMLDivElement>()

async function setupSearch(searchElement: HTMLDivElement, currentSlug: FullSlug) {
  searchElement.dataset.currentSlug = currentSlug
  if (configuredSearchElements.has(searchElement)) {
    return
  }
  configuredSearchElements.add(searchElement)

  const containerElement = searchElement.querySelector<HTMLElement>('.search-container')
  if (!containerElement) return
  const container = containerElement
  const image: SearchImage = {
    similarity: createImageSimilarity(),
    ids: undefined,
    semantic: null,
    ready: null,
    active: true,
  }
  window.addCleanup(() => image.similarity.dispose())
  window.addCleanup(() => {
    image.active = false
    image.semantic?.dispose()
  })

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

  const el = searchSpace.querySelector('ul#helper')
  const modeToggle = searchSpace.querySelector('.search-mode-toggle') as HTMLDivElement | null
  const modeButtons = modeToggle
    ? Array.from(modeToggle.querySelectorAll<HTMLButtonElement>('.mode-option'))
    : []

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
      ...(searchLayout.dataset.previewAvailable === 'true'
        ? [
            {
              kbd: /mac/i.test(navigator.userAgent) ? '⌥⇧P' : 'Alt+Shift+P',
              description: 'pour l’aperçu',
            },
          ]
        : []),
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

  const enablePreview = searchLayout.dataset.previewAvailable === 'true'
  let results = searchLayout.querySelector('.results-container') as HTMLDivElement
  if (!results) {
    results = document.createElement('div')
    results.className = 'results-container'
    appendLayout(results)
  }

  let preview: HTMLDivElement | undefined = undefined
  if (enablePreview) {
    preview = searchLayout.querySelector('.preview-container') as HTMLDivElement
    if (!preview) {
      preview = document.createElement('div')
      preview.className = 'preview-container'
      appendLayout(preview)
    }
  }

  const s: SearchSession = {
    searchElement,
    currentSlug,
    container,
    searchButton,
    searchBar,
    searchLayout,
    searchSpace,
    image,
    progressBar,
    progressResetTimer: 0,
    progressReasons: new Set<'search' | 'preview'>(),
    data: null,
    idDataMap: [],
    slugToIndex: new Map<string, number>(),
    searchDataReady: null,
    modeToggle,
    modeButtons,
    semanticStatus: searchSpace.querySelector<HTMLElement>('.semantic-status'),
    enablePreview,
    previewVisible: false,
    semanticBooting: null,
    searchSeq: 0,
    runSearchTimer: null,
    requestedMode: shared.searchMode,
    modeRequest: 0,
    preview,
    previewRequest: undefined,
    results,
    currentHover: null,
    previewToggle: searchSpace.querySelector<HTMLButtonElement>('.search-preview-toggle'),
    previousFocus: null,
    pressedKeys: new Set<string>(),
    composing: false,
    pendingNavigation: null,
    pendingOpen: null,
    pendingDisplay: null,
    showSimilarImages: seed => showSimilarImages(s, seed),
  }

  const storedMode = loadStoredSearchMode()
  if (shared.semanticReady) {
    setSemanticState(s, 'ready')
  } else if (shared.semanticInitFailed) {
    setSemanticState(s, 'unavailable')
  } else {
    setSemanticState(s, 'idle')
  }
  if (storedMode === 'semantic') {
    if (shared.semanticReady) {
      shared.searchMode = storedMode
    }
  } else if (storedMode === 'lexical') {
    shared.searchMode = storedMode
  }
  if (!shared.semanticReady && shared.searchMode === 'semantic') {
    shared.searchMode = 'lexical'
  }
  searchLayout.dataset.mode = shared.searchMode
  s.requestedMode = shared.searchMode

  updateModeUI(s, shared.searchMode)

  modeButtons.forEach(button => {
    const btnMode = (button.dataset.mode as SearchMode) ?? 'lexical'
    if (btnMode === 'semantic') {
      button.disabled = shared.semanticInitFailed
      button.setAttribute('aria-disabled', String(shared.semanticInitFailed))
    }
    const handler = () => triggerSearchWithMode(s, btnMode)
    button.addEventListener('click', handler)
    window.addCleanup(() => button.removeEventListener('click', handler))
  })

  const fitPaths = () => fitResultPaths(s)
  const pathObserver = new ResizeObserver(fitPaths)
  pathObserver.observe(results)
  document.fonts.addEventListener('loadingdone', fitPaths)
  window.addCleanup(() => {
    pathObserver.disconnect()
    document.fonts.removeEventListener('loadingdone', fitPaths)
  })

  const previewToggle = s.previewToggle
  const togglePreviewHandler = () => togglePreview(s)
  previewToggle?.addEventListener('click', togglePreviewHandler)
  window.addCleanup(() => previewToggle?.removeEventListener('click', togglePreviewHandler))
  setPreviewVisible(s, false)

  const submitSearch = (event: SubmitEvent) => {
    event.preventDefault()
    if (!s.composing && container.classList.contains('active')) openSearchResult(s)
  }
  searchSpace.addEventListener('submit', submitSearch)
  window.addCleanup(() => searchSpace.removeEventListener('submit', submitSearch))

  const typeHandler = () => onType(s)
  const startComposition = () => {
    s.composing = true
  }
  const finishComposition = () => {
    s.composing = false
    s.pressedKeys.clear()
    onType(s)
  }

  const keydownHandler = (e: HTMLElementEventMap['keydown']) => shortcutHandler(s, e)
  const keyupHandler = (e: HTMLElementEventMap['keyup']) => releaseKey(s, e)
  const blurHandler = () => releasePressedKeys(s)
  document.addEventListener('keydown', keydownHandler)
  document.addEventListener('keyup', keyupHandler)
  window.addEventListener('blur', blurHandler)
  window.addCleanup(() => {
    document.removeEventListener('keydown', keydownHandler)
    document.removeEventListener('keyup', keyupHandler)
    window.removeEventListener('blur', blurHandler)
  })
  const warmSearch = () => {
    if (s.data) return
    void ensureSearchData(s).catch(error => console.warn('[Search] warm failed:', error))
  }
  const openHandler = () => showSearch(s, 'basic')
  searchButton.addEventListener('pointerenter', warmSearch)
  searchButton.addEventListener('focusin', warmSearch)
  searchButton.addEventListener('click', openHandler)
  window.addCleanup(() => {
    searchButton.removeEventListener('pointerenter', warmSearch)
    searchButton.removeEventListener('focusin', warmSearch)
    searchButton.removeEventListener('click', openHandler)
  })
  searchBar.addEventListener('input', typeHandler)
  searchBar.addEventListener('compositionstart', startComposition)
  searchBar.addEventListener('compositionend', finishComposition)
  window.addCleanup(() => {
    searchBar.removeEventListener('input', typeHandler)
    searchBar.removeEventListener('compositionstart', startComposition)
    searchBar.removeEventListener('compositionend', finishComposition)
  })
  window.addCleanup(() => {
    hideSearch(s, false)
  })

  // The lexical mode button inside carries `.active` while search is closed.
  registerEscapeHandler(
    container,
    () => hideSearch(s),
    () => container.classList.contains('active'),
  )
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
      void ensureSearchData(s, 'background').catch(error =>
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
