import { start } from '../../functional'
import {
  createCommentsUi,
  createState,
  createWebSocketManager,
  getCommentPageId,
  mountMultiplayer,
  reduce,
  runMultiplayerEffect,
  type MultiplayerEffect,
  type MultiplayerServices,
} from '../multiplayer'

type MultiplayerInit = { model: ReturnType<typeof createState>; effects: MultiplayerEffect[] }

export function mountCollaborativeComments() {
  let services: MultiplayerServices | null = null
  const resolveAliases = new Set(['aarnphm', 'aarnphm-local'])

  const program = start({
    init: (): MultiplayerInit => ({ model: createState(), effects: [] }),
    reduce,
    effects: (effect, ctx) => {
      if (!services) return
      return runMultiplayerEffect(effect, ctx, services)
    },
    subscriptions: ctx => {
      services = {
        ui: createCommentsUi({
          getState: ctx.retrieve,
          dispatch: ctx.dispatch,
          canResolveComment: () => {
            const login = localStorage.getItem('comment-author-github-login')
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
  return program.stop
}
