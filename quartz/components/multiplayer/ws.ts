import type { BroadcastMessage, OperationInput } from "./model"
import type { Event, State } from "./core"

type WebSocketManagerDeps = {
  getState: () => State
  dispatch: (event: Event) => void
  getPageId: () => string
}

export function createWebSocketManager({ getState, dispatch, getPageId }: WebSocketManagerDeps) {
  let ws: WebSocket | null = null

  const send = (op: OperationInput) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "op", op }))
    }
  }

  const flushPending = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    for (const op of getState().pendingOps.values()) {
      ws.send(JSON.stringify({ type: "op", op }))
    }
  }

  const connect = () => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const { hasSnapshot, lastSeq } = getState()
    const pageId = encodeURIComponent(getPageId())
    const sinceParam = hasSnapshot && lastSeq > 0 ? `&since=${lastSeq}` : ""
    const wsUrl = `${protocol}//${window.location.host}/comments/websocket?pageId=${pageId}${sinceParam}`

    ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      flushPending()
    }

    ws.onmessage = (event) => {
      const msg: BroadcastMessage = JSON.parse(event.data)

      if (msg.type === "init") {
        dispatch({ type: "ws.init", comments: msg.comments, latestSeq: msg.latestSeq })
      } else if (msg.type === "delta") {
        dispatch({ type: "ws.delta", ops: msg.ops, latestSeq: msg.latestSeq })
      } else if (msg.type === "op") {
        dispatch({ type: "ws.op", op: msg.op })
      } else if (msg.type === "ack") {
        dispatch({ type: "ws.ack", opId: msg.opId, seq: msg.seq })
      } else if (msg.type === "error") {
        console.error("multiplayer comments error:", msg.message)
      }
    }

    ws.onclose = (event) => {
      console.debug("multiplayer comments disconnected:", {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean,
      })
      setTimeout(connect, 3000)
    }

    ws.onerror = (err) => {
      console.error("multiplayer comments websocket error:", err)
      console.error("websocket state:", ws!.readyState)
      console.error("websocket url:", ws!.url)
    }
  }

  const close = () => {
    if (ws) {
      ws.close()
      ws = null
    }
  }

  return { connect, send, flushPending, close }
}
