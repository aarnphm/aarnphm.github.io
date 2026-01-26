import { populateSearchIndex } from "../scripts/search-index"
import { getCommentPageId } from "./identity"
import type { MultiplayerServices } from "./effects"
import type { MultiplayerEvent, MultiplayerModel } from "./state"

type MountDeps = {
  dispatch: (event: MultiplayerEvent) => void
  state: () => MultiplayerModel
  services: MultiplayerServices
}

const parseCommentHash = () => {
  const { hash } = window.location
  if (!hash) return null
  const prefix = "#comment-"
  if (!hash.startsWith(prefix)) return null
  const rawId = hash.slice(prefix.length)
  if (!rawId) return null
  try {
    return decodeURIComponent(rawId)
  } catch {
    return rawId
  }
}

export const mountMultiplayer = ({ dispatch, state, services }: MountDeps) => {
  const cleanups: Array<() => void> = []
  const addCleanup = (cleanup: () => void) => {
    cleanups.push(cleanup)
  }

  const init = async () => {
    dispatch({ type: "nav.enter", pageId: getCommentPageId() })

    const data = await fetchData
    await populateSearchIndex(data)

    dispatch({ type: "nav.ready" })
    dispatch({ type: "ui.hash.changed", commentId: parseCommentHash() })
  }

  void init()

  const mouseUp = (event: MouseEvent) => {
    if (event.button !== 0) return
    if (!event.metaKey && !event.altKey && !event.ctrlKey) return
    if (event.target instanceof Node && event.target.isConnected) {
      const composer = document.body.querySelector(".comment-composer")
      if (composer instanceof HTMLElement && composer.contains(event.target)) {
        return
      }
    }
    services.ui.handleTextSelection()
  }

  let resizeFrame = 0
  const handleResize = () => {
    if (resizeFrame) {
      cancelAnimationFrame(resizeFrame)
    }
    resizeFrame = requestAnimationFrame(() => {
      services.ui.renderAllComments()
      services.ui.refreshActiveModal()
      const selection = state().activeSelection
      if (selection) {
        services.ui.renderSelectionHighlight(selection)
      }
      resizeFrame = 0
    })
  }

  const handleAuthorUpdate = (event: CustomEventMap["commentauthorupdated"]) => {
    const detail = event.detail
    if (!detail?.oldAuthor || !detail?.newAuthor) return
    dispatch({ type: "author.update", oldAuthor: detail.oldAuthor, newAuthor: detail.newAuthor })
  }

  const handleCollapseToggle = () => {
    dispatch({ type: "dom.collapse" })
  }

  const handleHashChange = () => {
    dispatch({ type: "ui.hash.changed", commentId: parseCommentHash() })
  }

  document.addEventListener("mouseup", mouseUp)
  window.addEventListener("resize", handleResize)
  document.addEventListener("collapsibletoggle", handleCollapseToggle)
  document.addEventListener("commentauthorupdated", handleAuthorUpdate)
  window.addEventListener("hashchange", handleHashChange)

  addCleanup(() => {
    services.ui.cleanup()
    services.ws.close()
    document.removeEventListener("mouseup", mouseUp)
    window.removeEventListener("resize", handleResize)
    document.removeEventListener("collapsibletoggle", handleCollapseToggle)
    document.removeEventListener("commentauthorupdated", handleAuthorUpdate)
    window.removeEventListener("hashchange", handleHashChange)
  })

  return () => {
    const pending = cleanups.slice()
    cleanups.length = 0
    for (const cleanup of pending) {
      cleanup()
    }
  }
}
