import type { MultiplayerComment, OperationInput, OperationRecord } from "./model"
import { Cmd } from "../../functional"

export type MultiplayerModel = {
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

export type MultiplayerEvent =
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

export type MultiplayerEffect =
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

export function createState(): MultiplayerModel {
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

export function reduce(
  model: MultiplayerModel,
  event: MultiplayerEvent,
): { model: MultiplayerModel; effects: Cmd<MultiplayerEffect> } {
  switch (event.type) {
    case "nav.enter": {
      return {
        model: { ...createState(), currentPageId: event.pageId },
        effects: [{ type: "storage.restore", pageId: event.pageId }],
      }
    }
    case "nav.ready": {
      return { model: model, effects: [{ type: "ws.connect" }] }
    }
    case "storage.pendingOpsRestored": {
      const pendingOps = new Map(event.ops.map((op) => [op.opId, op]))
      const comments = applyPendingOpsToComments(model.comments, pendingOps)
      return {
        model: { ...model, pendingOps, comments },
        effects: [{ type: "render" }, { type: "refreshModal" }],
      }
    }
    case "ws.init": {
      const comments = applyPendingOpsToComments(event.comments, model.pendingOps)
      return {
        model: { ...model, comments, lastSeq: event.latestSeq, hasSnapshot: true },
        effects: [
          { type: "render" },
          { type: "refreshModal" },
          { type: "openPendingThread" },
          { type: "ws.flush" },
        ],
      }
    }
    case "ws.delta": {
      let comments = model.comments
      let pendingOps = model.pendingOps
      let pendingOpsChanged = false
      let lastSeq = model.lastSeq
      for (const op of event.ops) {
        if (op.seq > lastSeq) lastSeq = op.seq
        comments = upsertComment(comments, op.comment)
        const removal = removePendingOp(pendingOps, op.opId)
        pendingOps = removal.pendingOps
        pendingOpsChanged = pendingOpsChanged || removal.changed
      }
      if (event.latestSeq > lastSeq) lastSeq = event.latestSeq
      const effects: MultiplayerEffect[] = [
        { type: "render" },
        { type: "refreshModal" },
        { type: "ws.flush" },
      ]
      if (pendingOpsChanged && model.currentPageId) {
        effects.push({ type: "persistPendingOps", pageId: model.currentPageId })
      }
      return { model: { ...model, comments, pendingOps, lastSeq, hasSnapshot: true }, effects }
    }
    case "ws.op": {
      let comments = upsertComment(model.comments, event.op.comment)
      const removal = removePendingOp(model.pendingOps, event.op.opId)
      const lastSeq = Math.max(model.lastSeq, event.op.seq)
      const effects: MultiplayerEffect[] = [{ type: "render" }, { type: "refreshModal" }]
      if (removal.changed && model.currentPageId) {
        effects.push({ type: "persistPendingOps", pageId: model.currentPageId })
      }
      return { model: { ...model, comments, pendingOps: removal.pendingOps, lastSeq }, effects }
    }
    case "ws.ack": {
      const removal = removePendingOp(model.pendingOps, event.opId)
      const effects: MultiplayerEffect[] = []
      if (removal.changed && model.currentPageId) {
        effects.push({ type: "persistPendingOps", pageId: model.currentPageId })
      }
      return {
        model: {
          ...model,
          pendingOps: removal.pendingOps,
          lastSeq: Math.max(model.lastSeq, event.seq),
        },
        effects,
      }
    }
    case "comment.submit": {
      const comments = upsertComment(model.comments, event.op.comment)
      const pendingOps = new Map(model.pendingOps)
      pendingOps.set(event.op.opId, event.op)
      const effects: MultiplayerEffect[] = [
        { type: "render" },
        { type: "refreshModal" },
        { type: "ws.send", op: event.op },
      ]
      if (model.currentPageId) {
        effects.push({ type: "persistPendingOps", pageId: model.currentPageId })
      }
      return { model: { ...model, comments, pendingOps }, effects }
    }
    case "author.update": {
      let updated = false
      const comments = model.comments.map((comment) => {
        if (comment.author !== event.oldAuthor) return comment
        updated = true
        return { ...comment, author: event.newAuthor }
      })
      return {
        model: updated ? { ...model, comments } : model,
        effects: updated ? [{ type: "render" }, { type: "refreshModal" }] : [],
      }
    }
    case "ui.selection.changed": {
      return {
        model: { ...model, activeSelection: event.range, activeComposerId: "selection" },
        effects: [{ type: "selection.highlight" }, { type: "composer.show" }],
      }
    }
    case "ui.selection.cleared": {
      return {
        model: { ...model, activeSelection: null, activeComposerId: null },
        effects: [{ type: "composer.hide" }],
      }
    }
    case "ui.hash.changed": {
      return {
        model: { ...model, pendingHashCommentId: event.commentId },
        effects: event.commentId ? [{ type: "openPendingThread" }] : [],
      }
    }
    case "ui.hash.consumed": {
      return { model: { ...model, pendingHashCommentId: null }, effects: [] }
    }
    case "ui.modal.open": {
      return { model: { ...model, activeModalId: event.commentId }, effects: [] }
    }
    case "ui.modal.close": {
      return { model: { ...model, activeModalId: null }, effects: [] }
    }
    case "ui.popover.open": {
      return { model: { ...model, activeActionsPopoverId: event.commentId }, effects: [] }
    }
    case "ui.popover.close": {
      return { model: { ...model, activeActionsPopoverId: null }, effects: [] }
    }
    case "ui.bubble.offsetUpdated": {
      const bubbleOffsets = new Map(model.bubbleOffsets)
      bubbleOffsets.set(event.commentId, event.offset)
      return { model: { ...model, bubbleOffsets }, effects: [] }
    }
    case "ui.bubbleOffsets.prune": {
      if (event.commentIds.length === 0) {
        return { model: model, effects: [] }
      }
      const bubbleOffsets = new Map(model.bubbleOffsets)
      for (const commentId of event.commentIds) {
        bubbleOffsets.delete(commentId)
      }
      return { model: { ...model, bubbleOffsets }, effects: [] }
    }
    case "ui.correctedAnchor.add": {
      if (model.correctedAnchors.has(event.opId)) {
        return { model: model, effects: [] }
      }
      const correctedAnchors = new Set(model.correctedAnchors)
      correctedAnchors.add(event.opId)
      return { model: { ...model, correctedAnchors }, effects: [] }
    }
    case "ui.comment.unread": {
      if (model.unreadCommentIds.has(event.commentId)) {
        return { model: model, effects: [] }
      }
      const unreadCommentIds = new Set(model.unreadCommentIds)
      unreadCommentIds.add(event.commentId)
      return { model: { ...model, unreadCommentIds }, effects: [{ type: "render" }] }
    }
    case "ui.comment.read": {
      if (!model.unreadCommentIds.has(event.commentId)) {
        return { model: model, effects: [] }
      }
      const unreadCommentIds = new Set(model.unreadCommentIds)
      unreadCommentIds.delete(event.commentId)
      return { model: { ...model, unreadCommentIds }, effects: [{ type: "render" }] }
    }
    case "dom.collapse": {
      return {
        model: {
          ...model,
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
