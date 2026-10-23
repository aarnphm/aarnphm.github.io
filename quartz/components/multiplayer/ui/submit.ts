import type { MultiplayerComment, OperationInput } from '../model'
import type { UiContext } from './context'
import { getAuthor, getCommentPageId } from '../identity'

export const canResolve = (ctx: UiContext, comment: MultiplayerComment) => {
  if (comment.author === getAuthor()) return true
  if (!ctx.canResolveComment) return false
  return ctx.canResolveComment(comment)
}

export const submitComment = (ctx: UiContext, op: OperationInput) => {
  ctx.dispatch({ type: 'comment.submit', op })
}

export const submitNewComment = (ctx: UiContext, comment: MultiplayerComment) => {
  submitComment(ctx, { opId: crypto.randomUUID(), type: 'new', comment })
}

export const ensurePageId = (comment: MultiplayerComment): MultiplayerComment => {
  if (comment.pageId) return comment
  return { ...comment, pageId: getCommentPageId() }
}

export const submitUpdateComment = (ctx: UiContext, comment: MultiplayerComment, opId?: string) => {
  submitComment(ctx, {
    opId: opId ?? crypto.randomUUID(),
    type: 'update',
    comment: ensurePageId(comment),
  })
}

export const submitDeleteComment = (
  ctx: UiContext,
  comment: MultiplayerComment,
  deletedAt: number,
) => {
  const deleted = { ...ensurePageId(comment), deletedAt }
  submitComment(ctx, { opId: crypto.randomUUID(), type: 'delete', comment: deleted })
}

export const submitResolveComment = (
  ctx: UiContext,
  comment: MultiplayerComment,
  resolvedAt: number,
) => {
  const resolved = { ...ensurePageId(comment), resolvedAt }
  submitComment(ctx, { opId: crypto.randomUUID(), type: 'resolve', comment: resolved })
}
