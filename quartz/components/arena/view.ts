export { mountArena } from './view/lifecycle'
export { closeModal, navigateBlock, showModal } from './view/modal'
export {
  buildSearchIndex,
  clearSearchState,
  closeSearchDropdown,
  focusSearchInput,
  getSearchIndex,
  performSearch,
  renderSearchResults,
  resetActiveResultHighlight,
  setActiveResult,
  setSearchIndex,
  wireSearchResultsInteractions,
} from './view/search'

export function handleCopyButton(button: HTMLElement) {
  const targetUrl = button.getAttribute('data-url')
  if (!targetUrl) {
    return
  }

  navigator.clipboard.writeText(targetUrl).then(
    () => {
      button.classList.add('check')
      setTimeout(() => {
        button.classList.remove('check')
      }, 2000)
    },
    error => console.error(error),
  )
}
