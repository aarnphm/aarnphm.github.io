import type { MultiplayerComment, OperationInput, OperationRecord } from "./model"

export type State = {
  comments: MultiplayerComment[]
  pendingOps: Map<string, OperationInput>
  lastSeq: number
  hasSnapshot: boolean
  currentPageId: string | null
  activeSelection: Range | null
  activeComposerId: string | null
  activeModalId: string | null
  activeActionsPopoverId: string | null
  pendingHashCommentId: string | null
  bubbleOffsets: Map<string, { x: number; y: number }>
  correctedAnchors: Set<string>
  unreadCommentIds: Set<string>
}

export type Event =
  | { type: "nav.enter"; pageId: string }
  | { type: "nav.ready" }
  | { type: "storage.pendingOpsRestored"; ops: OperationInput[] }
  | { type: "ws.init"; comments: MultiplayerComment[]; latestSeq: number }
  | { type: "ws.delta"; ops: OperationRecord[]; latestSeq: number }
  | { type: "ws.op"; op: OperationRecord }
  | { type: "ws.ack"; opId: string; seq: number }
  | { type: "comment.submit"; op: OperationInput }
  | { type: "author.update"; oldAuthor: string; newAuthor: string }
  | { type: "ui.selection.changed"; range: Range }
  | { type: "ui.selection.cleared" }
  | { type: "ui.hash.changed"; commentId: string | null }
  | { type: "ui.hash.consumed" }
  | { type: "ui.modal.open"; commentId: string }
  | { type: "ui.modal.close" }
  | { type: "ui.popover.open"; commentId: string | null }
  | { type: "ui.popover.close" }
  | { type: "ui.bubble.offsetUpdated"; commentId: string; offset: { x: number; y: number } }
  | { type: "ui.bubbleOffsets.prune"; commentIds: string[] }
  | { type: "ui.correctedAnchor.add"; opId: string }
  | { type: "ui.comment.unread"; commentId: string }
  | { type: "ui.comment.read"; commentId: string }
  | { type: "dom.collapse" }

export type Effect =
  | { type: "render" }
  | { type: "refreshModal" }
  | { type: "openPendingThread" }
  | { type: "persistPendingOps"; pageId: string }
  | { type: "ws.send"; op: OperationInput }
  | { type: "ws.flush" }
  | { type: "ws.connect" }
  | { type: "storage.restore"; pageId: string }
  | { type: "selection.highlight" }
  | { type: "composer.show" }
  | { type: "composer.hide" }
  | { type: "popover.hide" }
  | { type: "modal.close" }

export function createState(): State {
  return {
    comments: [],
    pendingOps: new Map(),
    lastSeq: 0,
    hasSnapshot: false,
    currentPageId: null,
    activeSelection: null,
    activeComposerId: null,
    activeModalId: null,
    activeActionsPopoverId: null,
    pendingHashCommentId: null,
    bubbleOffsets: new Map(),
    correctedAnchors: new Set(),
    unreadCommentIds: new Set(),
  }
}

function upsertComment(
  comments: MultiplayerComment[],
  comment: MultiplayerComment,
): MultiplayerComment[] {
  const idx = comments.findIndex((c) => c.id === comment.id)
  if (idx === -1) return [...comments, comment]
  const next = comments.slice()
  next[idx] = comment
  return next
}

function applyPendingOpsToComments(
  comments: MultiplayerComment[],
  pendingOps: Map<string, OperationInput>,
): MultiplayerComment[] {
  let next = comments
  for (const op of pendingOps.values()) {
    next = upsertComment(next, op.comment)
  }
  return next
}

function removePendingOp(
  pendingOps: Map<string, OperationInput>,
  opId: string,
): { pendingOps: Map<string, OperationInput>; changed: boolean } {
  if (!pendingOps.has(opId)) return { pendingOps, changed: false }
  const next = new Map(pendingOps)
  next.delete(opId)
  return { pendingOps: next, changed: true }
}

export function reduce(state: State, event: Event): { state: State; effects: Effect[] } {
  switch (event.type) {
    case "nav.enter": {
      return {
        state: { ...createState(), currentPageId: event.pageId },
        effects: [{ type: "storage.restore", pageId: event.pageId }],
      }
    }
    case "nav.ready": {
      return {
        state,
        effects: [{ type: "ws.connect" }],
      }
    }
    case "storage.pendingOpsRestored": {
      const pendingOps = new Map(event.ops.map((op) => [op.opId, op]))
      const comments = applyPendingOpsToComments(state.comments, pendingOps)
      return {
        state: { ...state, pendingOps, comments },
        effects: [{ type: "render" }, { type: "refreshModal" }],
      }
    }
    case "ws.init": {
      const comments = applyPendingOpsToComments(event.comments, state.pendingOps)
      return {
        state: {
          ...state,
          comments,
          lastSeq: event.latestSeq,
          hasSnapshot: true,
        },
        effects: [
          { type: "render" },
          { type: "refreshModal" },
          { type: "openPendingThread" },
          { type: "ws.flush" },
        ],
      }
    }
    case "ws.delta": {
      let comments = state.comments
      let pendingOps = state.pendingOps
      let pendingOpsChanged = false
      let lastSeq = state.lastSeq
      for (const op of event.ops) {
        if (op.seq > lastSeq) lastSeq = op.seq
        comments = upsertComment(comments, op.comment)
        const removal = removePendingOp(pendingOps, op.opId)
        pendingOps = removal.pendingOps
        pendingOpsChanged = pendingOpsChanged || removal.changed
      }
      if (event.latestSeq > lastSeq) lastSeq = event.latestSeq
      const effects: Effect[] = [{ type: "render" }, { type: "refreshModal" }, { type: "ws.flush" }]
      if (pendingOpsChanged && state.currentPageId) {
        effects.push({ type: "persistPendingOps", pageId: state.currentPageId })
      }
      return {
        state: {
          ...state,
          comments,
          pendingOps,
          lastSeq,
          hasSnapshot: true,
        },
        effects,
      }
    }
    case "ws.op": {
      let comments = upsertComment(state.comments, event.op.comment)
      const removal = removePendingOp(state.pendingOps, event.op.opId)
      const lastSeq = Math.max(state.lastSeq, event.op.seq)
      const effects: Effect[] = [{ type: "render" }, { type: "refreshModal" }]
      if (removal.changed && state.currentPageId) {
        effects.push({ type: "persistPendingOps", pageId: state.currentPageId })
      }
      return {
        state: {
          ...state,
          comments,
          pendingOps: removal.pendingOps,
          lastSeq,
        },
        effects,
      }
    }
    case "ws.ack": {
      const removal = removePendingOp(state.pendingOps, event.opId)
      const effects: Effect[] = []
      if (removal.changed && state.currentPageId) {
        effects.push({ type: "persistPendingOps", pageId: state.currentPageId })
      }
      return {
        state: {
          ...state,
          pendingOps: removal.pendingOps,
          lastSeq: Math.max(state.lastSeq, event.seq),
        },
        effects,
      }
    }
    case "comment.submit": {
      const comments = upsertComment(state.comments, event.op.comment)
      const pendingOps = new Map(state.pendingOps)
      pendingOps.set(event.op.opId, event.op)
      const effects: Effect[] = [
        { type: "render" },
        { type: "refreshModal" },
        { type: "ws.send", op: event.op },
      ]
      if (state.currentPageId) {
        effects.push({ type: "persistPendingOps", pageId: state.currentPageId })
      }
      return {
        state: {
          ...state,
          comments,
          pendingOps,
        },
        effects,
      }
    }
    case "author.update": {
      let updated = false
      const comments = state.comments.map((comment) => {
        if (comment.author !== event.oldAuthor) return comment
        updated = true
        return { ...comment, author: event.newAuthor }
      })
      return {
        state: updated ? { ...state, comments } : state,
        effects: updated ? [{ type: "render" }, { type: "refreshModal" }] : [],
      }
    }
    case "ui.selection.changed": {
      return {
        state: {
          ...state,
          activeSelection: event.range,
          activeComposerId: "selection",
        },
        effects: [{ type: "selection.highlight" }, { type: "composer.show" }],
      }
    }
    case "ui.selection.cleared": {
      return {
        state: {
          ...state,
          activeSelection: null,
          activeComposerId: null,
        },
        effects: [{ type: "composer.hide" }],
      }
    }
    case "ui.hash.changed": {
      return {
        state: { ...state, pendingHashCommentId: event.commentId },
        effects: event.commentId ? [{ type: "openPendingThread" }] : [],
      }
    }
    case "ui.hash.consumed": {
      return {
        state: { ...state, pendingHashCommentId: null },
        effects: [],
      }
    }
    case "ui.modal.open": {
      return {
        state: { ...state, activeModalId: event.commentId },
        effects: [],
      }
    }
    case "ui.modal.close": {
      return {
        state: { ...state, activeModalId: null },
        effects: [],
      }
    }
    case "ui.popover.open": {
      return {
        state: { ...state, activeActionsPopoverId: event.commentId },
        effects: [],
      }
    }
    case "ui.popover.close": {
      return {
        state: { ...state, activeActionsPopoverId: null },
        effects: [],
      }
    }
    case "ui.bubble.offsetUpdated": {
      const bubbleOffsets = new Map(state.bubbleOffsets)
      bubbleOffsets.set(event.commentId, event.offset)
      return {
        state: { ...state, bubbleOffsets },
        effects: [],
      }
    }
    case "ui.bubbleOffsets.prune": {
      if (event.commentIds.length === 0) {
        return { state, effects: [] }
      }
      const bubbleOffsets = new Map(state.bubbleOffsets)
      for (const commentId of event.commentIds) {
        bubbleOffsets.delete(commentId)
      }
      return {
        state: { ...state, bubbleOffsets },
        effects: [],
      }
    }
    case "ui.correctedAnchor.add": {
      if (state.correctedAnchors.has(event.opId)) {
        return { state, effects: [] }
      }
      const correctedAnchors = new Set(state.correctedAnchors)
      correctedAnchors.add(event.opId)
      return {
        state: { ...state, correctedAnchors },
        effects: [],
      }
    }
    case "ui.comment.unread": {
      if (state.unreadCommentIds.has(event.commentId)) {
        return { state, effects: [] }
      }
      const unreadCommentIds = new Set(state.unreadCommentIds)
      unreadCommentIds.add(event.commentId)
      return {
        state: { ...state, unreadCommentIds },
        effects: [{ type: "render" }],
      }
    }
    case "ui.comment.read": {
      if (!state.unreadCommentIds.has(event.commentId)) {
        return { state, effects: [] }
      }
      const unreadCommentIds = new Set(state.unreadCommentIds)
      unreadCommentIds.delete(event.commentId)
      return {
        state: { ...state, unreadCommentIds },
        effects: [{ type: "render" }],
      }
    }
    case "dom.collapse": {
      return {
        state: {
          ...state,
          activeSelection: null,
          activeComposerId: null,
          activeModalId: null,
          activeActionsPopoverId: null,
        },
        effects: [
          { type: "composer.hide" },
          { type: "popover.hide" },
          { type: "modal.close" },
          { type: "render" },
          { type: "refreshModal" },
          { type: "openPendingThread" },
        ],
      }
    }
  }
}
