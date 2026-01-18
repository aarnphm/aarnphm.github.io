import type { ArenaEffect } from "./model"
import {
  buildSearchIndex,
  clearSearchState,
  closeModal,
  closeSearchDropdown,
  focusSearchInput,
  handleCopyButton,
  navigateBlock,
  performSearch,
  renderSearchResults,
  resetActiveResultHighlight,
  setActiveResult,
  showModal,
  wireSearchResultsInteractions,
  getSearchIndex,
  setSearchIndex,
} from "./view"

export const runArenaEffect = (effect: ArenaEffect) => {
  switch (effect.type) {
    case "modal.open":
      void showModal(effect.blockId).catch((error) => console.error(error))
      return
    case "modal.close":
      closeModal()
      return
    case "block.navigate":
      void navigateBlock(effect.direction).catch((error) => console.error(error))
      return
    case "copy":
      handleCopyButton(effect.button)
      return
    case "search.focus":
      focusSearchInput(effect.prefill)
      return
    case "search.clear":
      clearSearchState({ blur: effect.blur })
      return
    case "search.close":
      resetActiveResultHighlight()
      closeSearchDropdown()
      return
    case "search.result.activate":
      setActiveResult(effect.index, effect.options)
      return
    case "search.query": {
      if (effect.query.length < 2) {
        resetActiveResultHighlight()
        closeSearchDropdown()
        return
      }
      const results = performSearch(effect.query, getSearchIndex())
      renderSearchResults(results, effect.scope, effect.query)
      wireSearchResultsInteractions()
      resetActiveResultHighlight()
      return
    }
    case "search.index.build":
      buildSearchIndex(effect.scope)
        .then((index) => {
          setSearchIndex(index)
        })
        .catch((error) => {
          console.error("Failed to build search index:", error)
          setSearchIndex([])
        })
      return
    default: {
      const exhaustive: never = effect
      return exhaustive
    }
  }
}
