import type { UiDeps } from './ui/context'
import { renderAllComments } from './ui/bubbles'
import {
  handleTextSelection,
  hideComposer,
  renderSelectionHighlight,
  showComposer,
} from './ui/composer'
import { createUiContext } from './ui/context'
import { closeActiveModal, openPendingCommentThread, refreshActiveModal } from './ui/modal'
import { hideActionsPopover } from './ui/popovers'

export function createCommentsUi(deps: UiDeps) {
  const ctx = createUiContext(deps)

  const cleanup = () => {
    window.clearTimeout(ctx.threadOpenTimer)
    ctx.threadOpenTimer = 0
    closeActiveModal(ctx)
    hideComposer(ctx)
    hideActionsPopover(ctx)
    document.querySelectorAll('.comment-highlight-layer').forEach(el => el.remove())
    document.querySelectorAll('.comment-selection-layer').forEach(el => el.remove())
    document
      .querySelectorAll('.comment-highlight')
      .forEach(el => el.replaceWith(...Array.from(el.childNodes)))
    document.querySelectorAll('.comment-bubble').forEach(el => el.remove())
    document.querySelectorAll('.comment-thread-modal').forEach(el => el.remove())
    document.querySelectorAll('.comment-actions-popover').forEach(el => el.remove())
    document.querySelectorAll('.delete-confirmation-modal').forEach(el => el.remove())
  }

  return {
    cleanup,
    closeActiveModal: () => closeActiveModal(ctx),
    handleTextSelection: () => handleTextSelection(ctx),
    hideActionsPopover: () => hideActionsPopover(ctx),
    hideComposer: () => hideComposer(ctx),
    openPendingCommentThread: () => openPendingCommentThread(ctx),
    refreshActiveModal: () => refreshActiveModal(ctx),
    renderAllComments: () => renderAllComments(ctx),
    renderSelectionHighlight: (range: Range) => renderSelectionHighlight(ctx, range),
    showComposer: (range: Range) => showComposer(ctx, range),
  }
}
