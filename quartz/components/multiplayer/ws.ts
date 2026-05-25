import type { BroadcastMessage, OperationInput } from './model'
import type { MultiplayerEvent, MultiplayerModel } from './state'

type WebSocketManagerDeps = {
  getState: () => MultiplayerModel
  dispatch: (event: MultiplayerEvent) => void
  getPageId: () => string
}

export function createWebSocketManager({ getState, dispatch, getPageId }: WebSocketManagerDeps) {
  let ws: WebSocket | null = null
  let reconnectTimer = 0
  let reconnectEnabled = false

  const send = (op: OperationInput) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'op', op }))
    }
  }

  const flushPending = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    for (const op of getState().pendingOps.values()) {
      ws.send(JSON.stringify({ type: 'op', op }))
    }
  }

  const connect = () => {
    reconnectEnabled = true
    if (reconnectTimer) {
      window.clearTimeout(reconnectTimer)
      reconnectTimer = 0
    }
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const { hasSnapshot, lastSeq } = getState()
    const pageId = encodeURIComponent(getPageId())
    const sinceParam = hasSnapshot && lastSeq > 0 ? `&since=${lastSeq}` : ''
    const wsUrl = `${protocol}//${window.location.host}/comments/websocket?pageId=${pageId}${sinceParam}`

    const socket = new WebSocket(wsUrl)
    ws = socket

    socket.onopen = () => {
      flushPending()
    }

    socket.onmessage = event => {
      const msg: BroadcastMessage = JSON.parse(event.data)

      if (msg.type === 'init') {
        dispatch({ type: 'ws.init', comments: msg.comments, latestSeq: msg.latestSeq })
      } else if (msg.type === 'delta') {
        dispatch({ type: 'ws.delta', ops: msg.ops, latestSeq: msg.latestSeq })
      } else if (msg.type === 'op') {
        dispatch({ type: 'ws.op', op: msg.op })
      } else if (msg.type === 'ack') {
        dispatch({ type: 'ws.ack', opId: msg.opId, seq: msg.seq })
      } else if (msg.type === 'error') {
        console.error('multiplayer comments error')
      }
    }

    socket.onclose = event => {
      if (ws === socket) ws = null
      console.debug('multiplayer comments disconnected:', {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean,
      })
      if (!reconnectEnabled) return
      reconnectTimer = window.setTimeout(connect, 3000)
    }

    socket.onerror = err => {
      console.error('multiplayer comments websocket error:', err)
      console.error('websocket state:', socket.readyState)
      console.error('websocket url:', socket.url)
    }
  }

  const close = () => {
    reconnectEnabled = false
    if (reconnectTimer) {
      window.clearTimeout(reconnectTimer)
      reconnectTimer = 0
    }
    const socket = ws
    ws = null
    if (!socket) return
    socket.onopen = null
    socket.onmessage = null
    socket.onclose = null
    socket.onerror = null
    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
      socket.close(1000, 'disabled')
    }
  }

  return { connect, send, flushPending, close }
}
