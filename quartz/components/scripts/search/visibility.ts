import { ensureSearchData, updateModeUI } from './engine'
import { clearPreview, setPreviewVisible } from './preview'
import { completeSearchProgress, resetProgressBar, startSearchProgress } from './progress'
import { shared, type SearchSession, type SearchType } from './state'

export function hideSearch(s: SearchSession, focusButton = true) {
  const { image, container, searchBar, searchLayout, results, searchButton } = s
  setPreviewVisible(s, false)
  s.modeRequest++
  s.requestedMode = shared.searchMode
  updateModeUI(s, shared.searchMode)
  image.similarity.cancel()
  image.semantic?.cancel()
  shared.semantic?.cancel()
  s.searchSeq += 1
  s.pressedKeys.clear()
  s.composing = false
  s.pendingNavigation = null
  s.pendingOpen = null
  s.pendingDisplay = null
  if (s.runSearchTimer !== null) {
    window.clearTimeout(s.runSearchTimer)
    s.runSearchTimer = null
  }
  container!.classList.remove('active')
  if (container!.matches(':popover-open')) container!.hidePopover()
  searchBar!.value = ''
  shared.rawSearchTerm = ''
  shared.currentSearchTerm = ''
  s.currentHover = null
  results.replaceChildren()
  clearPreview(s)
  searchLayout!.classList.remove('display-results')
  searchLayout.removeAttribute('aria-busy')
  if (focusButton) {
    const target = s.previousFocus?.isConnected ? s.previousFocus : searchButton
    target?.focus({ preventScroll: true })
  }
  s.previousFocus = null
  resetProgressBar(s)
}

export function showSearch(s: SearchSession, type: SearchType) {
  const { container, searchBar } = s
  if (!container!.matches(':popover-open')) {
    s.previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    container!.showPopover()
  }
  container!.classList.add('active')
  if (!s.data) {
    startSearchProgress(s)
    void ensureSearchData(s).then(
      () => {
        if (container.classList.contains('active')) completeSearchProgress(s)
      },
      error => {
        resetProgressBar(s)
        console.warn('[Search] initialization failed:', error)
      },
    )
  }
  if (type === 'tags') {
    searchBar!.value = '#'
    shared.rawSearchTerm = '#'
  }
  searchBar!.focus()
}
