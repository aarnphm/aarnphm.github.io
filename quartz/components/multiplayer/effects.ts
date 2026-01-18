import type { EffectContext } from "../../functional"
import type { MultiplayerEvent, MultiplayerEffect, MultiplayerModel } from "./state"
import type { createCommentsUi } from "./ui"
import type { createWebSocketManager } from "./ws"
import { persistPendingOps, restorePendingOps } from "./storage"

export type MultiplayerServices = {
  ui: ReturnType<typeof createCommentsUi>
  ws: ReturnType<typeof createWebSocketManager>
}

export const runMultiplayerEffect = (
  effect: MultiplayerEffect,
  ctx: EffectContext<MultiplayerModel, MultiplayerEvent>,
  services: MultiplayerServices,
) => {
  const state = ctx.retrieve()

  switch (effect.type) {
    case "render":
      services.ui.renderAllComments()
      return
    case "refreshModal":
      services.ui.refreshActiveModal()
      return
    case "openPendingThread":
      services.ui.openPendingCommentThread()
      return
    case "persistPendingOps":
      persistPendingOps(effect.pageId, state.pendingOps)
      return
    case "storage.restore": {
      const restoredOps = restorePendingOps(effect.pageId)
      if (restoredOps.length > 0) {
        ctx.dispatch({ type: "storage.pendingOpsRestored", ops: restoredOps })
      }
      return
    }
    case "ws.send":
      services.ws.send(effect.op)
      return
    case "ws.flush":
      services.ws.flushPending()
      return
    case "ws.connect":
      services.ws.connect()
      return
    case "selection.highlight":
      if (state.activeSelection) {
        services.ui.renderSelectionHighlight(state.activeSelection)
      }
      return
    case "composer.show":
      if (state.activeSelection) {
        services.ui.showComposer(state.activeSelection)
      }
      return
    case "composer.hide":
      services.ui.hideComposer()
      return
    case "popover.hide":
      services.ui.hideActionsPopover()
      return
    case "modal.close":
      services.ui.closeActiveModal()
      return
    default: {
      const exhaustive: never = effect
      return exhaustive
    }
  }
}
