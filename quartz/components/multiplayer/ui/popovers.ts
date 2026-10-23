import type { MultiplayerComment } from '../model'
import type { UiContext } from './context'
import { copyToClipboard } from './dom'
import { submitDeleteComment } from './submit'

export const hideActionsPopover = (ctx: UiContext) => {
  ctx.activePopoverCleanup?.()
  ctx.activePopoverCleanup = null
  if (ctx.activeActionsPopover) {
    document.body.removeChild(ctx.activeActionsPopover)
    ctx.activeActionsPopover = null
  }
  ctx.dispatch({ type: 'ui.popover.close' })
}

export const mountActionsPopover = (
  ctx: UiContext,
  popover: HTMLElement,
  trigger: HTMLButtonElement,
) => {
  document.body.appendChild(popover)
  trigger.setAttribute('aria-expanded', 'true')

  const position = () => {
    const anchor = trigger.getBoundingClientRect()
    const rect = popover.getBoundingClientRect()
    const width = document.documentElement.clientWidth
    const height = document.documentElement.clientHeight
    const preferredLeft = anchor.right + 4
    const left =
      preferredLeft + rect.width <= width - 12 ? preferredLeft : anchor.right - rect.width
    popover.style.left = `${Math.max(12, Math.min(left, width - rect.width - 12)) + window.scrollX}px`
    popover.style.top = `${Math.max(12, Math.min(anchor.top, height - rect.height - 12)) + window.scrollY}px`
  }
  position()

  const closeOnClickOutside = (event: MouseEvent) => {
    const target = event.target
    if (target instanceof Node && (popover.contains(target) || trigger.contains(target))) return
    hideActionsPopover(ctx)
  }
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopImmediatePropagation()
      hideActionsPopover(ctx)
      trigger.focus()
    }
  }

  document.addEventListener('mousedown', closeOnClickOutside)
  popover.addEventListener('keydown', onKeyDown)
  window.addEventListener('resize', position)
  popover.querySelector<HTMLButtonElement>('button')?.focus()
  ctx.activePopoverCleanup = () => {
    trigger.setAttribute('aria-expanded', 'false')
    document.removeEventListener('mousedown', closeOnClickOutside)
    popover.removeEventListener('keydown', onKeyDown)
    window.removeEventListener('resize', position)
  }
}

export const showActionsPopover = (
  ctx: UiContext,
  trigger: HTMLButtonElement,
  onEdit: () => void,
  onDelete: () => void,
  commentId: string | null,
  showDelete: boolean = true,
) => {
  hideActionsPopover(ctx)

  const popover = document.createElement('div')
  popover.className = 'comment-actions-popover'
  ctx.activeActionsPopover = popover
  ctx.dispatch({ type: 'ui.popover.open', commentId })

  const editButton = document.createElement('button')
  editButton.className = 'popover-action'
  editButton.innerHTML = `<span class="menu-item">Edit</span>`
  editButton.onclick = () => {
    hideActionsPopover(ctx)
    onEdit()
  }

  popover.appendChild(editButton)

  if (showDelete) {
    const deleteButton = document.createElement('button')
    deleteButton.className = 'popover-action popover-action-danger'
    deleteButton.innerHTML = `<span class="menu-item">Delete comment</span>`
    deleteButton.onclick = () => {
      hideActionsPopover(ctx)
      onDelete()
    }
    popover.appendChild(deleteButton)
  }

  mountActionsPopover(ctx, popover, trigger)
}

export const showThreadActionsPopover = (
  ctx: UiContext,
  comment: MultiplayerComment,
  trigger: HTMLButtonElement,
) => {
  hideActionsPopover(ctx)

  const popover = document.createElement('div')
  popover.className = 'comment-actions-popover'
  ctx.activeActionsPopover = popover
  ctx.dispatch({ type: 'ui.popover.open', commentId: comment.id })

  const markUnreadButton = document.createElement('button')
  const isUnread = ctx.getState().unreadCommentIds.has(comment.id)
  markUnreadButton.className = `popover-action ${isUnread ? 'popover-action-read' : 'popover-action-unread'}`
  markUnreadButton.innerHTML = `<span class="menu-item">${isUnread ? 'Mark as read' : 'Mark as unread'}</span>`
  markUnreadButton.onclick = () => {
    hideActionsPopover(ctx)
    ctx.dispatch({
      type: isUnread ? 'ui.comment.read' : 'ui.comment.unread',
      commentId: comment.id,
    })
  }

  const copyLinkButton = document.createElement('button')
  copyLinkButton.className = 'popover-action'
  copyLinkButton.innerHTML = `<span class="menu-item">Copy link</span>`
  copyLinkButton.onclick = async () => {
    hideActionsPopover(ctx)
    const url = new URL(window.location.href)
    url.hash = `comment-${comment.id}`
    const copied = await copyToClipboard(url.toString())
    if (!copied) {
      console.error('failed to copy link')
    }
  }

  const deleteButton = document.createElement('button')
  deleteButton.className = 'popover-action popover-action-danger'
  deleteButton.innerHTML = `<span class="menu-item">Delete thread...</span>`
  deleteButton.onclick = () => {
    hideActionsPopover(ctx)
    const replies = ctx.getState().comments.filter(c => c.parentId === comment.id && !c.deletedAt)
    showDeleteConfirmation(ctx, comment, deletedAt => {
      submitDeleteComment(ctx, comment, deletedAt)
      for (const reply of replies) {
        submitDeleteComment(ctx, reply, deletedAt)
      }
    })
  }

  popover.appendChild(markUnreadButton)
  popover.appendChild(copyLinkButton)
  popover.appendChild(deleteButton)
  mountActionsPopover(ctx, popover, trigger)
}

export const showDeleteConfirmation = (
  ctx: UiContext,
  comment: MultiplayerComment,
  onConfirm?: (deletedAt: number) => void,
) => {
  const modal = document.createElement('dialog')
  modal.className = 'delete-confirmation-modal'
  modal.setAttribute('aria-label', 'Delete comment')
  const returnFocus = ctx.activeModal?.querySelector<HTMLButtonElement>('.modal-actions-button')

  const message = document.createElement('div')
  message.className = 'delete-confirmation-message'
  message.textContent = 'Delete this comment?'

  const actions = document.createElement('div')
  actions.className = 'delete-confirmation-actions'

  const cancelButton = document.createElement('button')
  cancelButton.className = 'edit-button edit-button-cancel'
  cancelButton.innerHTML = `<span class="button-container"><span class="button-text"><span class="button-content">Cancel</span></span></span>`

  const deleteButton = document.createElement('button')
  deleteButton.className = 'edit-button edit-button-delete'
  deleteButton.innerHTML = `<span class="button-container"><span class="button-text"><span class="button-content">Delete</span></span></span>`

  cancelButton.onclick = () => {
    modal.close()
  }

  deleteButton.onclick = async () => {
    const deletedAt = Date.now()
    modal.close()
    if (onConfirm) {
      onConfirm(deletedAt)
      return
    }
    submitDeleteComment(ctx, comment, deletedAt)
  }

  actions.appendChild(cancelButton)
  actions.appendChild(deleteButton)
  modal.appendChild(message)
  modal.appendChild(actions)
  modal.addEventListener(
    'close',
    () => {
      modal.remove()
      returnFocus?.focus()
    },
    { once: true },
  )
  document.body.appendChild(modal)
  modal.showModal()
  cancelButton.focus()
}
