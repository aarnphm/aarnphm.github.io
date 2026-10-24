import { parseCodeQuery } from '../../../util/code-query'
import { parseSearchQuery } from '../../../util/search-scope'
import { isSearchModeToggle, isSearchToggle } from '../../../util/search-shortcuts'
import { ensureSemantic, persistSearchMode, updateModeUI } from './engine'
import { isImageGridDirection } from './image-grid'
import { clearPreview, displayPreview, setImageSearch, togglePreview } from './preview'
import { resetProgressBar } from './progress'
import { runSearch } from './query'
import { flushSearchDisplay, openSearchResult } from './results'
import { shared, type SearchMode, type SearchSession } from './state'
import { hideSearch, showSearch } from './visibility'

export function triggerSearchWithMode(s: SearchSession, mode: SearchMode) {
  if (parseCodeQuery(s.searchBar.value)) return
  const { image, searchLayout } = s
  s.requestedMode = mode
  const request = ++s.modeRequest
  if (mode === 'semantic' && !shared.semanticReady) {
    void ensureSemantic(s).then(ready => {
      if (request !== s.modeRequest) return
      if (ready) triggerSearchWithMode(s, 'semantic')
      else {
        s.requestedMode = 'lexical'
        updateModeUI(s, 'lexical')
      }
    })
    updateModeUI(s, mode)
    return
  }
  if (shared.searchMode === mode) {
    updateModeUI(s, mode)
    return
  }
  image.similarity.cancel()
  image.semantic?.cancel()
  s.pdf.cancel()
  s.code.cancelPending()
  shared.semantic?.cancel()
  if (s.runSearchTimer !== null) {
    window.clearTimeout(s.runSearchTimer)
    s.runSearchTimer = null
  }
  shared.searchMode = mode
  updateModeUI(s, mode)
  persistSearchMode(shared.searchMode)
  if (shared.rawSearchTerm.trim() !== '') {
    searchLayout.classList.add('display-results')
    const token = ++s.searchSeq
    void runSearch(s, shared.rawSearchTerm, token)
  }
}

export function shortcutHandler(s: SearchSession, e: HTMLElementEventMap['keydown']) {
  const { container, enablePreview, preview, results, searchBar } = s
  if (e.defaultPrevented) return
  const paletteOpen = document.querySelector('search#palette-container') as HTMLDivElement
  if (paletteOpen && paletteOpen.classList.contains('active')) return
  if (e.isComposing || s.composing) return

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
      hideSearch(s)
    } else {
      showSearch(s, 'basic')
    }
    return
  } else if (e.shiftKey && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault()
    if (e.repeat) return
    const searchBarOpen = container!.classList.contains('active')
    if (searchBarOpen) {
      hideSearch(s)
    } else {
      showSearch(s, 'tags')
    }
    return
  }

  if (!container!.classList.contains('active')) return
  if (enablePreview && e.altKey && !e.ctrlKey && !e.metaKey && e.shiftKey && e.code === 'KeyP') {
    e.preventDefault()
    if (!e.repeat) togglePreview(s)
    return
  }
  if (isSearchModeToggle(e) && container.contains(document.activeElement)) {
    if (s.requestedMode === 'lexical' && shared.semanticInitFailed) return
    e.preventDefault()
    if (!e.repeat) triggerSearchWithMode(s, s.requestedMode === 'semantic' ? 'lexical' : 'semantic')
    return
  }
  if (preview?.contains(document.activeElement)) {
    if (document.activeElement?.closest('.pdf-embed, .search-code-preview pre')) return
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      s.currentHover?.focus()
    }
    return
  }
  if (
    isImageGridDirection(e.key) &&
    !e.altKey &&
    !e.ctrlKey &&
    !e.metaKey &&
    !e.shiftKey &&
    s.imageGrid.active &&
    (results.contains(document.activeElement?.closest('.image-search-row') ?? null) ||
      (document.activeElement === searchBar && (e.key === 'ArrowUp' || e.key === 'ArrowDown')))
  ) {
    e.preventDefault()
    s.pressedKeys.add(e.code || e.key)
    s.pendingNavigation = e.key
    return
  }
  if (e.key === 'ArrowRight' && results.contains(document.activeElement)) {
    const link =
      preview?.querySelector<HTMLElement>('.pdf-viewer-container, .search-code-preview pre') ??
      preview?.querySelector<HTMLAnchorElement>('a[href]')
    if (link?.checkVisibility()) {
      e.preventDefault()
      link.focus()
    }
    return
  }
  if (
    e.target instanceof HTMLButtonElement &&
    results.contains(e.target) &&
    (e.key === 'Enter' || e.key === ' ')
  )
    return
  if (e.key === 'Enter') {
    if (e.target !== searchBar && !results.contains(document.activeElement)) return
    // Cancel native form submission before any asynchronous work can yield to the browser.
    e.preventDefault()
    if (!e.repeat) openSearchResult(s)
  } else if (
    e.key === 'ArrowUp' ||
    (e.shiftKey && e.key === 'Tab') ||
    (e.ctrlKey && e.key === 'p')
  ) {
    e.preventDefault()
    s.pressedKeys.add(e.code || e.key)
    s.pendingNavigation = -1
  } else if (e.key === 'ArrowDown' || e.key === 'Tab' || (e.ctrlKey && e.key === 'n')) {
    e.preventDefault()
    s.pressedKeys.add(e.code || e.key)
    s.pendingNavigation = 1
  } else if (
    e.target === searchBar &&
    (e.key.length === 1 || e.key === 'Backspace' || e.key === 'Delete')
  ) {
    s.pressedKeys.add(e.code || e.key)
  }
}

export function releaseKey(s: SearchSession, e: HTMLElementEventMap['keyup']) {
  const { container, results, searchBar } = s
  s.pressedKeys.delete(e.code || e.key)
  if (s.pressedKeys.size > 0 || s.composing) return
  flushSearchDisplay(s)
  const direction = s.pendingNavigation
  s.pendingNavigation = null
  if (!container.classList.contains('active')) return
  if (typeof direction === 'string') {
    const target = s.imageGrid.target(document.activeElement, direction)
    if (target) void focusResult(s, target)
    else if (direction === 'ArrowUp') searchBar.focus()
    return
  }
  if (direction !== null) {
    const controls = [
      ...results.querySelectorAll<HTMLElement>(
        'a.result-card[href], button.image-similar-button, button.search-load-more',
      ),
    ]
    const active = document.activeElement
    const index = active instanceof HTMLElement ? controls.indexOf(active) : -1
    const next = index < 0 ? (direction === 1 ? 0 : controls.length - 1) : index + direction
    if (next < 0) searchBar.focus()
    else void focusResult(s, controls[next])
  }
}

export function releasePressedKeys(s: SearchSession) {
  s.pressedKeys.clear()
  s.pendingNavigation = null
  flushSearchDisplay(s)
}

async function focusResult(s: SearchSession, result: Element | null | undefined) {
  if (!(result instanceof HTMLElement)) return
  s.currentHover?.classList.remove('focus')
  s.currentHover = result
  result.classList.add('focus')
  result.focus()
  const anchor =
    result instanceof HTMLAnchorElement
      ? result
      : result.closest('.image-search-row')?.querySelector<HTMLAnchorElement>('a.result-card')
  await displayPreview(s, anchor ?? null)
}

export function onType(s: SearchSession) {
  const { image, results, searchBar, searchLayout } = s
  delete results.dataset.imageSeed
  s.resultPages?.dispose()
  s.resultPages = null
  setImageSearch(s, parseSearchQuery(searchBar.value).scope === 'images')
  image.similarity.cancel()
  image.semantic?.cancel()
  s.pdf.cancel()
  s.code.cancelPending()
  shared.semantic?.cancel()
  s.previewRequest?.abort()
  shared.rawSearchTerm = searchBar.value
  const hasQuery = shared.rawSearchTerm.trim() !== ''
  const term = shared.rawSearchTerm
  const token = ++s.searchSeq
  s.pendingOpen = null
  s.pendingDisplay = null
  searchLayout.setAttribute('aria-busy', String(hasQuery))
  if (s.runSearchTimer !== null) {
    window.clearTimeout(s.runSearchTimer)
    s.runSearchTimer = null
  }
  if (s.composing) return
  if (s.completions.update()) {
    results.replaceChildren()
    s.currentHover = null
    searchLayout.setAttribute('aria-busy', 'false')
    clearPreview(s)
    resetProgressBar(s)
    return
  }
  if (!hasQuery || shared.searchMode === 'lexical') {
    void runSearch(s, term, token)
    return
  }
  s.runSearchTimer = window.setTimeout(() => {
    s.runSearchTimer = null
    void runSearch(s, term, token)
  }, 60)
}
