import type { MarkdownEditor } from '../../scripts/markdown-editor'
import type { MultiplayerComment } from '../model'
import type { UiContext } from './context'
import { getAuthor, getCommentPageId } from '../identity'
import { createMarkdownEditor, keepModalInViewport } from './dom'
import { showThreadActionsPopover } from './popovers'
import { avatarImg, peopleStack, renderThreadContent } from './rows'
import { canResolve, submitNewComment, submitResolveComment } from './submit'

export const closeActiveModal = (ctx: UiContext) => {
  if (!ctx.activeModal) return
  ctx.activeModalCleanup?.()
  ctx.activeModalCleanup = null
  for (const cleanup of ctx.editCleanups) cleanup()
  const activeId = ctx.activeModal.dataset.commentId
  document.body.removeChild(ctx.activeModal)
  ctx.activeModal = null
  ctx.dispatch({ type: 'ui.modal.close' })
  if (activeId) {
    document
      .querySelector<HTMLElement>(`.comment-bubble[data-comment-id="${activeId}"]`)
      ?.classList.remove('modal-active')
  }
}

export const refreshActiveModal = (ctx: UiContext) => {
  if (!ctx.activeModal) return
  keepModalInViewport(ctx.activeModal)
  const commentId = ctx.activeModal.dataset.commentId
  if (!commentId) return
  const comment = ctx.getState().comments.find(c => c.id === commentId)
  if (!comment || comment.deletedAt || comment.resolvedAt) {
    closeActiveModal(ctx)
    return
  }
  const replies = ctx.getState().comments.filter(c => c.parentId === commentId && !c.deletedAt)
  const content = ctx.activeModal.querySelector('.modal-content')
  if (!(content instanceof HTMLElement)) return
  renderThreadContent(ctx, content, comment, replies)
}

export const showCommentThread = (
  ctx: UiContext,
  commentId: string,
  position?: { top: number; left: number },
) => {
  const comment = ctx.getState().comments.find(c => c.id === commentId)
  if (!comment || comment.deletedAt || comment.resolvedAt) return

  if (ctx.activeModal) {
    if (ctx.activeModal.dataset.commentId === commentId) {
      const replies = ctx.getState().comments.filter(c => c.parentId === commentId && !c.deletedAt)
      const content = ctx.activeModal.querySelector('.modal-content')
      if (content instanceof HTMLElement) {
        renderThreadContent(ctx, content, comment, replies)
      }
      ctx.dispatch({ type: 'ui.comment.read', commentId })
      return
    }
    closeActiveModal(ctx)
  }

  const replies = ctx.getState().comments.filter(c => c.parentId === commentId && !c.deletedAt)

  const modal = document.createElement('div')
  modal.className = 'comment-thread-modal'
  modal.dataset.commentId = commentId
  ctx.activeModal = modal
  ctx.dispatch({ type: 'ui.modal.open', commentId })
  ctx.dispatch({ type: 'ui.comment.read', commentId })

  if (position) {
    modal.style.top = `${position.top}px`
    modal.style.left = `${position.left + 50}px`
    modal.style.right = 'auto'
  }

  const header = document.createElement('div')
  header.className = 'modal-header'

  // The thread is named by the passage it hangs on, the way a channel is named.
  const title = document.createElement('div')
  title.className = 'modal-title'
  const excerpt = comment.anchorText.replace(/\s+/g, ' ').trim()
  title.textContent = excerpt || 'comment'
  title.title = excerpt

  const headerActions = document.createElement('div')
  headerActions.className = 'modal-actions'

  const headerMenuButton = document.createElement('button')
  headerMenuButton.className = 'modal-actions-button'
  headerMenuButton.setAttribute('aria-label', 'Thread actions')
  headerMenuButton.setAttribute('aria-expanded', 'false')
  headerMenuButton.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M7.5 12a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m6 0a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m4.5 1.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3" clip-rule="evenodd"></path></svg>`
  headerMenuButton.onclick = (e: MouseEvent) => {
    e.stopPropagation()
    showThreadActionsPopover(ctx, comment, headerMenuButton)
  }

  const closeButton = document.createElement('button')
  closeButton.className = 'modal-close'
  closeButton.setAttribute('aria-label', 'Close comment')
  closeButton.textContent = '×'
  closeButton.onclick = () => {
    closeActiveModal(ctx)
  }

  let isDragging = false
  let dragStartX = 0
  let dragStartY = 0
  let modalStartX = 0
  let modalStartY = 0

  header.onmousedown = (e: MouseEvent) => {
    const target = e.target
    if (target instanceof Node && headerActions.contains(target)) return
    isDragging = true
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
    dragStartX = e.pageX
    dragStartY = e.pageY
    const rect = modal.getBoundingClientRect()
    modalStartX = rect.left + scrollLeft
    modalStartY = rect.top + scrollTop
    modal.style.transform = 'none'
    modal.style.right = 'auto'
    modal.style.top = `${modalStartY}px`
    modal.style.left = `${modalStartX}px`
    e.preventDefault()
  }

  const onMouseMove = (e: MouseEvent) => {
    if (!isDragging) return
    const deltaX = e.pageX - dragStartX
    const deltaY = e.pageY - dragStartY
    modal.style.left = `${modalStartX + deltaX}px`
    modal.style.top = `${modalStartY + deltaY}px`
  }

  const onMouseUp = () => {
    isDragging = false
  }

  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)

  headerActions.appendChild(headerMenuButton)
  if (canResolve(ctx, comment)) {
    const resolveButton = document.createElement('button')
    resolveButton.className = 'modal-actions-button modal-resolve-button'
    resolveButton.setAttribute('aria-label', 'Resolve thread')
    resolveButton.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>`
    resolveButton.onclick = () => {
      submitResolveComment(ctx, comment, Date.now())
      closeActiveModal(ctx)
      document.dispatchEvent(new CustomEvent('toast', { detail: { message: 'Comment resolved' } }))
    }
    headerActions.appendChild(resolveButton)
  }
  headerActions.appendChild(closeButton)

  header.appendChild(title)
  header.appendChild(peopleStack([comment, ...replies].map(c => c.author)))
  header.appendChild(headerActions)

  const content = document.createElement('div')
  content.className = 'modal-content'
  renderThreadContent(ctx, content, comment, replies)

  const replyComposerContainer = document.createElement('div')
  replyComposerContainer.className = 'reply-composer-container'

  const replyAuthorElement = document.createElement('div')
  replyAuthorElement.className = 'reply-author-element'

  const avatar = avatarImg(getAuthor())
  avatar.className = 'avatar'
  replyAuthorElement.appendChild(avatar)

  const inputSectionWrapper = document.createElement('div')
  inputSectionWrapper.className = 'input-section-wrapper composer-empty'

  const editableTypeaheadWrapper = document.createElement('div')
  editableTypeaheadWrapper.className = 'editable-typeahead-wrapper'

  const primitiveWrapper = document.createElement('div')
  primitiveWrapper.className = 'primitive-wrapper'
  primitiveWrapper.style.display = 'block'

  const lexicalWrapper = document.createElement('div')
  lexicalWrapper.className = 'lexical-wrapper'

  const editorMount = document.createElement('div')
  let replyEditor: MarkdownEditor | null = null

  const placeholderWrapper = document.createElement('div')
  placeholderWrapper.setAttribute('aria-hidden', 'true')
  const placeholderText = document.createElement('span')
  placeholderText.className = 'placeholder-text'
  placeholderText.textContent = 'Reply'
  placeholderWrapper.appendChild(placeholderText)

  const actions = document.createElement('div')
  actions.className = 'composer-actions'

  const replyButton = document.createElement('button')
  replyButton.type = 'button'
  replyButton.setAttribute('aria-label', 'Submit')
  replyButton.setAttribute('aria-disabled', 'true')
  replyButton.setAttribute('data-tooltip', 'Submit')
  replyButton.setAttribute('data-tooltip-type', 'text')
  replyButton.tabIndex = 0
  replyButton.className = 'submit-button'
  const buttonIconSpan = document.createElement('span')
  buttonIconSpan.setAttribute('aria-hidden', 'true')
  buttonIconSpan.className = 'button-icon'
  buttonIconSpan.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M12 16a.5.5 0 0 1-.5-.5V8.707l-3.146 3.147a.5.5 0 0 1-.708-.708l4-4a.5.5 0 0 1 .708 0l4 4a.5.5 0 0 1-.708.708L12.5 8.707V15.5a.5.5 0 0 1-.5.5" clip-rule="evenodd"></path></svg>`
  replyButton.appendChild(buttonIconSpan)

  replyButton.onclick = async () => {
    if (replyButton.getAttribute('aria-disabled') === 'true') return

    if (!replyEditor) return

    const content = replyEditor.getValue().trim()
    if (!content) return

    const reply: MultiplayerComment = {
      id: crypto.randomUUID(),
      pageId: getCommentPageId(),
      parentId: comment.id,
      anchorHash: comment.anchorHash,
      anchorStart: comment.anchorStart,
      anchorEnd: comment.anchorEnd,
      anchorText: comment.anchorText,
      content,
      author: getAuthor(),
      createdAt: Date.now(),
      updatedAt: null,
      deletedAt: null,
      resolvedAt: null,
    }

    submitNewComment(ctx, reply)

    replyEditor.setValue('')
    inputSectionWrapper.classList.add('composer-empty')
    replyButton.setAttribute('aria-disabled', 'true')
  }

  lexicalWrapper.appendChild(editorMount)
  lexicalWrapper.appendChild(placeholderWrapper)
  primitiveWrapper.appendChild(lexicalWrapper)
  editableTypeaheadWrapper.appendChild(primitiveWrapper)

  actions.appendChild(replyButton)
  inputSectionWrapper.appendChild(editableTypeaheadWrapper)
  inputSectionWrapper.appendChild(actions)

  replyComposerContainer.appendChild(replyAuthorElement)
  replyComposerContainer.appendChild(inputSectionWrapper)

  modal.appendChild(header)
  modal.appendChild(content)
  modal.appendChild(replyComposerContainer)

  document.body.appendChild(modal)
  keepModalInViewport(modal)
  const onResize = () => keepModalInViewport(modal)
  window.addEventListener('resize', onResize)

  void createMarkdownEditor({
    parent: editorMount,
    ariaLabel: 'Reply to comment',
    onEdited: state => {
      if (state.empty) {
        inputSectionWrapper.classList.add('composer-empty')
        replyButton.setAttribute('aria-disabled', 'true')
      } else {
        inputSectionWrapper.classList.remove('composer-empty')
        replyButton.setAttribute('aria-disabled', 'false')
      }
    },
    onSubmit: () => replyButton.click(),
  })
    .then(editor => {
      if (editorMount.isConnected) {
        replyEditor = editor
      } else {
        editor.destroy()
      }
    })
    .catch(error => {
      console.error('failed to load comment reply editor', error)
    })

  const handleEscape = (e: KeyboardEvent) => {
    if (e.target instanceof Element && e.target.closest('dialog')) return
    if (e.key === 'Escape' && ctx.activeModal === modal) {
      closeActiveModal(ctx)
    }
  }
  document.addEventListener('keydown', handleEscape)
  ctx.activeModalCleanup = () => {
    replyEditor?.destroy()
    replyEditor = null
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
    document.removeEventListener('keydown', handleEscape)
    window.removeEventListener('resize', onResize)
  }
}

export const openPendingCommentThread = (ctx: UiContext) => {
  const targetId = ctx.getState().pendingHashCommentId
  if (!targetId) return
  const comment = ctx
    .getState()
    .comments.find(item => item.id === targetId && !item.deletedAt && !item.resolvedAt)
  if (!comment) return

  const bubble = document.querySelector<HTMLElement>(
    `.comment-bubble[data-comment-id="${targetId}"]`,
  )
  const highlight = document.querySelector<HTMLElement>(
    `.comment-highlight[data-comment-id="${targetId}"]`,
  )
  const target = bubble ?? highlight
  let position: { top: number; left: number } | undefined
  if (target) {
    const rect = target.getBoundingClientRect()
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
    position = { top: rect.top + scrollTop, left: rect.left + scrollLeft }
    target.scrollIntoView({ block: 'center', inline: 'nearest' })
  }

  showCommentThread(ctx, targetId, position)
  ctx.dispatch({ type: 'ui.hash.consumed' })
}
