import { start } from "../../functional"
import {
  createState,
  reduce,
  runMultiplayerEffect,
  type MultiplayerServices,
  type MultiplayerEffect,
  mountMultiplayer,
  createWebSocketManager,
  getCommentPageId,
  createCommentsUi,
} from "../multiplayer"

document.addEventListener("nav", () => {
  let services: MultiplayerServices | null = null
  const resolveAliases = new Set(["aarnphm", "aarnphm-local"])

  const program = start({
    init: () => ({ model: createState(), effects: [] as MultiplayerEffect[] }),
    reduce,
    effects: (effect, ctx) => {
      if (!services) return
      return runMultiplayerEffect(effect, ctx, services)
    },
    subscriptions: (ctx) => {
      services = {
        ui: createCommentsUi({
          getState: ctx.retrieve,
          dispatch: ctx.dispatch,
          canResolveComment: () => {
            const login = localStorage.getItem("comment-author-github-login")
            if (!login) return false
            return resolveAliases.has(login.toLowerCase())
          },
        }),
        ws: createWebSocketManager({
          getState: ctx.retrieve,
          dispatch: ctx.dispatch,
          getPageId: getCommentPageId,
        }),
      }
      return mountMultiplayer({ dispatch: ctx.dispatch, state: ctx.retrieve, services })
    },
  })
  window.addCleanup(program.stop)
})
