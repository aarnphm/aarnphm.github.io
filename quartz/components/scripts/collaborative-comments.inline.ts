import { populateSearchIndex } from "./search-index"
import type { Effect, Event } from "../multiplayer"
import {
  createState,
  reduce,
  createCommentsUi,
  restorePendingOps,
  persistPendingOps,
  getCommentPageId,
  createWebSocketManager,
} from "../multiplayer"

let state = createState()
const getState = () => state

const runEffects = (effects: Effect[]) => {
  for (const effect of effects) {
    if (effect.type === "render") {
      ui.renderAllComments()
    } else if (effect.type === "refreshModal") {
      ui.refreshActiveModal()
    } else if (effect.type === "openPendingThread") {
      ui.openPendingCommentThread()
    } else if (effect.type === "persistPendingOps") {
      persistPendingOps(effect.pageId, getState().pendingOps)
    } else if (effect.type === "storage.restore") {
      const restoredOps = restorePendingOps(effect.pageId)
      if (restoredOps.length > 0) {
        dispatch({ type: "storage.pendingOpsRestored", ops: restoredOps })
      }
    } else if (effect.type === "ws.send") {
      ws.send(effect.op)
    } else if (effect.type === "ws.flush") {
      ws.flushPending()
    } else if (effect.type === "ws.connect") {
      ws.connect()
    } else if (effect.type === "selection.highlight") {
      const selection = getState().activeSelection
      if (selection) {
        ui.renderSelectionHighlight(selection)
      }
    } else if (effect.type === "composer.show") {
      const selection = getState().activeSelection
      if (selection) {
        ui.showComposer(selection)
      }
    } else if (effect.type === "composer.hide") {
      ui.hideComposer()
    } else if (effect.type === "popover.hide") {
      ui.hideActionsPopover()
    } else if (effect.type === "modal.close") {
      ui.closeActiveModal()
    }
  }
}

const dispatch = (event: Event) => {
  const result = reduce(state, event)
  state = result.state
  runEffects(result.effects)
}

const ui = createCommentsUi({ getState, dispatch })
const ws = createWebSocketManager({
  getState,
  dispatch,
  getPageId: getCommentPageId,
})

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

document.addEventListener("nav", async () => {
  dispatch({ type: "nav.enter", pageId: getCommentPageId() })

  const data = await fetchData
  await populateSearchIndex(data)

  dispatch({ type: "nav.ready" })
  dispatch({ type: "ui.hash.changed", commentId: parseCommentHash() })

  const mouseUp = (event: MouseEvent) => {
    if (event.button !== 0) return
    if (!event.metaKey && !event.altKey) return
    if (event.target instanceof Node && event.target.isConnected) {
      const composer = document.body.querySelector(".comment-composer")
      if (composer instanceof HTMLElement && composer.contains(event.target)) {
        return
      }
    }
    ui.handleTextSelection()
  }

  let resizeFrame = 0
  const handleResize = () => {
    if (resizeFrame) {
      cancelAnimationFrame(resizeFrame)
    }
    resizeFrame = requestAnimationFrame(() => {
      ui.renderAllComments()
      ui.refreshActiveModal()
      const selection = getState().activeSelection
      if (selection) {
        ui.renderSelectionHighlight(selection)
      }
      resizeFrame = 0
    })
  }

  document.addEventListener("mouseup", mouseUp)
  window.addEventListener("resize", handleResize)

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

  window.addCleanup(() => {
    ui.cleanup()
    ws.close()
    document.removeEventListener("mouseup", mouseUp)
    window.removeEventListener("resize", handleResize)
    document.removeEventListener("collapsibletoggle", handleCollapseToggle)
    document.removeEventListener("commentauthorupdated", handleAuthorUpdate)
    window.removeEventListener("hashchange", handleHashChange)
  })

  document.addEventListener("collapsibletoggle", handleCollapseToggle)
  document.addEventListener("commentauthorupdated", handleAuthorUpdate)
  window.addEventListener("hashchange", handleHashChange)
})
