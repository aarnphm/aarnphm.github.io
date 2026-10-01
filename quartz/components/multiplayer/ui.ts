import type { MarkdownEditor, MarkdownEditorConfig } from '../scripts/markdown-editor'
import type { MultiplayerComment, OperationInput } from './model'
import type { MultiplayerEvent, MultiplayerModel } from './state'
import { renderMarkdown } from '../../util/markdown-renderer'
import { getFullSlug } from '../../util/path'
import {
  computeStructuralAnchor,
  getArticleText,
  getRangeOffsets,
  hashText,
  recoverFromStructuralAnchor,
} from './anchor'
import { getAuthor, getAvatarUrl, getCommentPageId } from './identity'
import { isStructuralAnchor } from './model'

let markdownEditorModule: Promise<typeof import('../scripts/markdown-editor')> | null = null

const notebookCellBoundarySelector = '.notebook-code-cell'

async function createMarkdownEditor(config: MarkdownEditorConfig): Promise<MarkdownEditor> {
  markdownEditorModule ??= import('../scripts/markdown-editor')
  const { MarkdownEditor } = await markdownEditorModule
  return new MarkdownEditor(config)
}

function rangeIntersectsNotebookCellBoundary(range: Range): boolean {
  for (const cell of document.querySelectorAll(notebookCellBoundarySelector)) {
    if (range.intersectsNode(cell)) return true
  }
  return false
}

type UiDeps = {
  getState: () => MultiplayerModel
  dispatch: (event: MultiplayerEvent) => void
  canResolveComment?: (comment: MultiplayerComment) => boolean
}

export function createCommentsUi({ getState, dispatch, canResolveComment }: UiDeps) {
  let activeComposer: HTMLElement | null = null
  let activeModal: HTMLElement | null = null
  let activeActionsPopover: HTMLElement | null = null
  let selectionHighlightLayer: HTMLElement | null = null
  let activeModalCleanup: (() => void) | null = null
  let activeComposerCleanup: (() => void) | null = null
  let activePopoverCleanup: (() => void) | null = null
  let threadOpenTimer = 0
  let composerVersion = 0
  const editCleanups = new Set<() => void>()
  const renderedThreads = new WeakMap<HTMLElement, MultiplayerComment[]>()

  const canResolve = (comment: MultiplayerComment) => {
    if (comment.author === getAuthor()) return true
    if (!canResolveComment) return false
    return canResolveComment(comment)
  }

  const submitComment = (op: OperationInput) => {
    dispatch({ type: 'comment.submit', op })
  }

  const formatRelativeTime = (timestamp: number): string => {
    const now = Date.now()
    const diff = now - timestamp
    const seconds = Math.floor(diff / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)
    const weeks = Math.floor(days / 7)
    const months = Math.floor(days / 30)
    const years = Math.floor(days / 365)

    if (years > 0) return `${years} yr${years > 1 ? 's' : ''} ago`
    if (months > 0) return `${months} mo${months > 1 ? 's' : ''} ago`
    if (weeks > 0) return `${weeks} wk${weeks > 1 ? 's' : ''} ago`
    if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`
    if (hours > 0) return `${hours} hr${hours > 1 ? 's' : ''} ago`
    if (minutes > 0) return `${minutes} min${minutes > 1 ? 's' : ''} ago`
    return 'just now'
  }

  const correctionOpId = (commentId: string, start: number, end: number): string => {
    return `correction:${commentId}:${start}:${end}`
  }

  const clearSelectionHighlight = () => {
    if (selectionHighlightLayer) {
      selectionHighlightLayer.remove()
      selectionHighlightLayer = null
    }
  }

  const hideComposer = () => {
    composerVersion++
    activeComposerCleanup?.()
    activeComposerCleanup = null
    if (activeComposer) {
      document.body.removeChild(activeComposer)
      activeComposer = null
    }
    clearSelectionHighlight()
  }

  const hideActionsPopover = () => {
    activePopoverCleanup?.()
    activePopoverCleanup = null
    if (activeActionsPopover) {
      document.body.removeChild(activeActionsPopover)
      activeActionsPopover = null
    }
    dispatch({ type: 'ui.popover.close' })
  }

  const getTextNodeRects = (range: Range): DOMRect[] => {
    const rects: DOMRect[] = []
    const rootNode =
      range.commonAncestorContainer.nodeType === Node.TEXT_NODE
        ? range.commonAncestorContainer.parentElement
        : range.commonAncestorContainer
    if (!rootNode) return rects
    const walker = document.createTreeWalker(rootNode, NodeFilter.SHOW_TEXT)

    let currentNode: Node | null = null
    while ((currentNode = walker.nextNode())) {
      if (!(currentNode instanceof Text)) continue
      const node = currentNode
      if (!range.intersectsNode(node)) continue

      const nodeRange = document.createRange()
      nodeRange.selectNodeContents(node)

      const startOffset = node === range.startContainer ? range.startOffset : 0
      const endOffset = node === range.endContainer ? range.endOffset : node.length

      if (startOffset >= endOffset) continue

      nodeRange.setStart(node, startOffset)
      nodeRange.setEnd(node, endOffset)

      const nodeRects = nodeRange.getClientRects()
      for (const rect of nodeRects) {
        if (rect.width > 0 && rect.height > 0) {
          rects.push(rect)
        }
      }
    }

    return rects
  }

  const renderSelectionHighlight = (range: Range) => {
    clearSelectionHighlight()
    const rects = getTextNodeRects(range)
    if (rects.length === 0) return
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
    const layer = document.createElement('div')
    layer.className = 'comment-selection-layer'
    for (const rect of rects) {
      const highlight = document.createElement('span')
      highlight.className = 'comment-selection-highlight'
      highlight.style.left = `${rect.left + scrollLeft}px`
      highlight.style.top = `${rect.top + scrollTop}px`
      highlight.style.width = `${rect.width}px`
      highlight.style.height = `${rect.height}px`
      layer.appendChild(highlight)
    }
    document.body.appendChild(layer)
    selectionHighlightLayer = layer
  }

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch (err) {
      console.error('Failed to copy:', err)
      return false
    }
  }

  const closeActiveModal = () => {
    if (!activeModal) return
    activeModalCleanup?.()
    activeModalCleanup = null
    for (const cleanup of editCleanups) cleanup()
    const activeId = activeModal.dataset.commentId
    document.body.removeChild(activeModal)
    activeModal = null
    dispatch({ type: 'ui.modal.close' })
    if (activeId) {
      document
        .querySelector<HTMLElement>(`.comment-bubble[data-comment-id="${activeId}"]`)
        ?.classList.remove('modal-active')
    }
  }

  const keepModalInViewport = (modal: HTMLElement) => {
    const rect = modal.getBoundingClientRect()
    const maxLeft = document.documentElement.clientWidth - rect.width - 12
    const left = Math.max(12, Math.min(rect.left, maxLeft))
    modal.style.left = `${left + window.scrollX}px`
    modal.style.right = 'auto'
  }

  const refreshActiveModal = () => {
    if (!activeModal) return
    keepModalInViewport(activeModal)
    const commentId = activeModal.dataset.commentId
    if (!commentId) return
    const comment = getState().comments.find(c => c.id === commentId)
    if (!comment || comment.deletedAt || comment.resolvedAt) {
      closeActiveModal()
      return
    }
    const replies = getState().comments.filter(c => c.parentId === commentId && !c.deletedAt)
    const content = activeModal.querySelector('.modal-content')
    if (!(content instanceof HTMLElement)) return
    renderThreadContent(content, comment, replies)
  }

  const showActionsPopover = (
    buttonRect: DOMRect,
    onEdit: () => void,
    onDelete: () => void,
    commentId: string | null,
    showDelete: boolean = true,
  ) => {
    hideActionsPopover()

    const popover = document.createElement('div')
    popover.className = 'comment-actions-popover'
    activeActionsPopover = popover
    dispatch({ type: 'ui.popover.open', commentId })

    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop

    popover.style.left = `${buttonRect.right + scrollLeft + 4}px`
    popover.style.top = `${buttonRect.top + scrollTop}px`

    const editButton = document.createElement('button')
    editButton.className = 'popover-action'
    editButton.innerHTML = `<span class="menu-item">Edit</span>`
    editButton.onclick = () => {
      hideActionsPopover()
      onEdit()
    }

    popover.appendChild(editButton)

    if (showDelete) {
      const deleteButton = document.createElement('button')
      deleteButton.className = 'popover-action popover-action-danger'
      deleteButton.innerHTML = `<span class="menu-item">Delete comment</span>`
      deleteButton.onclick = () => {
        hideActionsPopover()
        onDelete()
      }
      popover.appendChild(deleteButton)
    }

    document.body.appendChild(popover)

    const closeOnClickOutside = (e: MouseEvent) => {
      const target = e.target
      if (!(target instanceof Node) || !popover.contains(target)) {
        hideActionsPopover()
        document.removeEventListener('mousedown', closeOnClickOutside)
      }
    }
    const timer = window.setTimeout(
      () => document.addEventListener('mousedown', closeOnClickOutside),
      0,
    )
    activePopoverCleanup = () => {
      window.clearTimeout(timer)
      document.removeEventListener('mousedown', closeOnClickOutside)
    }
  }

  const showThreadActionsPopover = (comment: MultiplayerComment, buttonRect: DOMRect) => {
    hideActionsPopover()

    const popover = document.createElement('div')
    popover.className = 'comment-actions-popover'
    activeActionsPopover = popover
    dispatch({ type: 'ui.popover.open', commentId: comment.id })

    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop

    popover.style.left = `${buttonRect.right + scrollLeft + 4}px`
    popover.style.top = `${buttonRect.top + scrollTop}px`

    const markUnreadButton = document.createElement('button')
    const isUnread = getState().unreadCommentIds.has(comment.id)
    markUnreadButton.className = `popover-action ${isUnread ? 'popover-action-read' : 'popover-action-unread'}`
    markUnreadButton.innerHTML = `<span class="menu-item">${isUnread ? 'Mark as read' : 'Mark as unread'}</span>`
    markUnreadButton.onclick = () => {
      hideActionsPopover()
      dispatch({ type: isUnread ? 'ui.comment.read' : 'ui.comment.unread', commentId: comment.id })
    }

    const copyLinkButton = document.createElement('button')
    copyLinkButton.className = 'popover-action'
    copyLinkButton.innerHTML = `<span class="menu-item">Copy link</span>`
    copyLinkButton.onclick = async () => {
      hideActionsPopover()
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
      hideActionsPopover()
      const replies = getState().comments.filter(c => c.parentId === comment.id && !c.deletedAt)
      showDeleteConfirmation(comment, deletedAt => {
        submitDeleteComment(comment, deletedAt)
        for (const reply of replies) {
          submitDeleteComment(reply, deletedAt)
        }
      })
    }

    popover.appendChild(markUnreadButton)
    popover.appendChild(copyLinkButton)
    popover.appendChild(deleteButton)
    document.body.appendChild(popover)

    const closeOnClickOutside = (e: MouseEvent) => {
      const target = e.target
      if (!(target instanceof Node) || !popover.contains(target)) {
        hideActionsPopover()
        document.removeEventListener('mousedown', closeOnClickOutside)
      }
    }
    const timer = window.setTimeout(
      () => document.addEventListener('mousedown', closeOnClickOutside),
      0,
    )
    activePopoverCleanup = () => {
      window.clearTimeout(timer)
      document.removeEventListener('mousedown', closeOnClickOutside)
    }
  }

  const enterEditMode = async (comment: MultiplayerComment, textElement: HTMLElement) => {
    const wrapper = document.createElement('div')
    wrapper.className = 'comment-edit-wrapper'

    const inputContent = document.createElement('div')
    inputContent.className = 'edit-input-content'

    const editorMount = document.createElement('div')
    editorMount.className = 'edit-input'

    let markdownEditor: MarkdownEditor | null = null

    const exitEditMode = () => {
      editCleanups.delete(exitEditMode)
      textElement.style.display = ''
      if (markdownEditor) {
        markdownEditor.destroy()
        markdownEditor = null
      }
      if (wrapper.parentNode) {
        wrapper.parentNode.removeChild(wrapper)
      }
    }

    const actions = document.createElement('div')
    actions.className = 'edit-actions'

    const cancelButton = document.createElement('button')
    cancelButton.innerHTML = `<span class="button-container"><span class="button-text"><span class="button-content">Cancel</span></span></span>`
    cancelButton.className = 'edit-button edit-button-cancel'
    cancelButton.onclick = () => {
      exitEditMode()
    }

    const saveButton = document.createElement('button')
    saveButton.innerHTML = `<span class="button-container"><span class="button-text"><span class="button-content">Save</span></span></span>`
    saveButton.className = 'edit-button edit-button-save'
    saveButton.onclick = async () => {
      if (!markdownEditor) return
      const newContent = markdownEditor.getValue().trim()
      if (!newContent || newContent === comment.content) {
        exitEditMode()
        return
      }

      const updatedAt = Date.now()
      submitUpdateComment({ ...comment, content: newContent, updatedAt })

      exitEditMode()
    }

    inputContent.appendChild(editorMount)
    actions.appendChild(cancelButton)
    actions.appendChild(saveButton)
    wrapper.appendChild(inputContent)
    wrapper.appendChild(actions)

    textElement.style.display = 'none'
    textElement.parentNode?.insertBefore(wrapper, textElement)
    editCleanups.add(exitEditMode)

    markdownEditor = await createMarkdownEditor({
      parent: editorMount,
      initialContent: comment.content,
      onSubmit: () => saveButton.click(),
      onCancel: exitEditMode,
    })

    if (wrapper.isConnected) {
      markdownEditor.focus()
    } else {
      markdownEditor.destroy()
      markdownEditor = null
    }
  }

  const showDeleteConfirmation = (
    comment: MultiplayerComment,
    onConfirm?: (deletedAt: number) => void,
  ) => {
    const overlay = document.createElement('div')
    overlay.className = 'delete-confirmation-overlay'

    const modal = document.createElement('div')
    modal.className = 'delete-confirmation-modal'

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
      document.body.removeChild(overlay)
    }

    deleteButton.onclick = async () => {
      const deletedAt = Date.now()
      document.body.removeChild(overlay)
      if (onConfirm) {
        onConfirm(deletedAt)
        return
      }
      submitDeleteComment(comment, deletedAt)
    }

    actions.appendChild(cancelButton)
    actions.appendChild(deleteButton)
    modal.appendChild(message)
    modal.appendChild(actions)
    overlay.appendChild(modal)
    document.body.appendChild(overlay)
  }

  const submitNewComment = (comment: MultiplayerComment) => {
    submitComment({ opId: crypto.randomUUID(), type: 'new', comment })
  }

  const ensurePageId = (comment: MultiplayerComment): MultiplayerComment => {
    if (comment.pageId) return comment
    return { ...comment, pageId: getCommentPageId() }
  }

  const submitUpdateComment = (comment: MultiplayerComment, opId?: string) => {
    submitComment({
      opId: opId ?? crypto.randomUUID(),
      type: 'update',
      comment: ensurePageId(comment),
    })
  }

  const submitDeleteComment = (comment: MultiplayerComment, deletedAt: number) => {
    const deleted = { ...ensurePageId(comment), deletedAt }
    submitComment({ opId: crypto.randomUUID(), type: 'delete', comment: deleted })
  }

  const submitResolveComment = (comment: MultiplayerComment, resolvedAt: number) => {
    const resolved = { ...ensurePageId(comment), resolvedAt }
    submitComment({ opId: crypto.randomUUID(), type: 'resolve', comment: resolved })
  }

  const buildThreadItem = (comment: MultiplayerComment) => {
    const item = document.createElement('div')
    item.className = 'reply-item'

    const top = document.createElement('div')
    top.className = 'reply-top'

    const left = document.createElement('div')
    left.className = 'reply-left'

    const avatar = document.createElement('img')
    avatar.className = 'reply-avatar'
    avatar.dataset.ignorePopup = 'true'
    getAvatarUrl(comment.author, 24).then(url => {
      avatar.src = url
    })
    avatar.alt = comment.author

    const author = document.createElement('div')
    author.className = 'reply-author'
    author.textContent = comment.author

    const time = document.createElement('div')
    time.className = 'reply-time'
    time.textContent = formatRelativeTime(comment.createdAt)

    left.appendChild(avatar)
    left.appendChild(author)
    left.appendChild(time)

    const text = document.createElement('div')
    text.className = 'reply-text markdown-content'
    const currentSlug = getFullSlug(window)
    text.innerHTML = renderMarkdown(comment.content, currentSlug)

    const right = document.createElement('div')
    right.className = 'reply-right'

    const actions = document.createElement('button')
    actions.className = 'reply-actions'
    actions.setAttribute('aria-label', 'Comment actions')
    actions.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M7.5 12a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m6 0a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m4.5 1.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3" clip-rule="evenodd"></path></svg>`

    actions.onclick = (e: MouseEvent) => {
      e.stopPropagation()
      const buttonRect = actions.getBoundingClientRect()
      showActionsPopover(
        buttonRect,
        () => {
          void enterEditMode(comment, text).catch(error => {
            console.error('failed to enter comment edit mode', error)
          })
        },
        () => showDeleteConfirmation(comment),
        comment.id,
        comment.parentId !== null,
      )
    }

    right.appendChild(actions)
    top.appendChild(left)
    top.appendChild(right)

    item.appendChild(top)
    item.appendChild(text)

    return item
  }

  const renderThreadContent = (
    content: HTMLElement,
    comment: MultiplayerComment,
    replies: MultiplayerComment[],
  ) => {
    const items = [comment, ...replies]
    const previous = renderedThreads.get(content)
    if (previous?.length === items.length && items.every((item, i) => item === previous[i])) {
      return
    }
    for (const cleanup of editCleanups) cleanup()
    renderedThreads.set(content, items)
    content.replaceChildren()
    content.appendChild(buildThreadItem(comment))
    for (const reply of replies) {
      content.appendChild(buildThreadItem(reply))
    }
  }

  const showCommentThread = (commentId: string, position?: { top: number; left: number }) => {
    const comment = getState().comments.find(c => c.id === commentId)
    if (!comment || comment.deletedAt || comment.resolvedAt) return

    if (activeModal) {
      if (activeModal.dataset.commentId === commentId) {
        const replies = getState().comments.filter(c => c.parentId === commentId && !c.deletedAt)
        const content = activeModal.querySelector('.modal-content')
        if (content instanceof HTMLElement) {
          renderThreadContent(content, comment, replies)
        }
        dispatch({ type: 'ui.comment.read', commentId })
        return
      }
      closeActiveModal()
    }

    const replies = getState().comments.filter(c => c.parentId === commentId && !c.deletedAt)

    const modal = document.createElement('div')
    modal.className = 'comment-thread-modal'
    modal.dataset.commentId = commentId
    activeModal = modal
    dispatch({ type: 'ui.modal.open', commentId })
    dispatch({ type: 'ui.comment.read', commentId })

    if (position) {
      modal.style.top = `${position.top}px`
      modal.style.left = `${position.left + 50}px`
      modal.style.right = 'auto'
    }

    const header = document.createElement('div')
    header.className = 'modal-header'

    const title = document.createElement('div')
    title.className = 'modal-title'
    title.textContent = 'comment'

    const headerActions = document.createElement('div')
    headerActions.className = 'modal-actions'

    const headerMenuButton = document.createElement('button')
    headerMenuButton.className = 'modal-actions-button'
    headerMenuButton.setAttribute('aria-label', 'Thread actions')
    headerMenuButton.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M7.5 12a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m6 0a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m4.5 1.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3" clip-rule="evenodd"></path></svg>`
    headerMenuButton.onclick = (e: MouseEvent) => {
      e.stopPropagation()
      const buttonRect = headerMenuButton.getBoundingClientRect()
      showThreadActionsPopover(comment, buttonRect)
    }

    const closeButton = document.createElement('button')
    closeButton.className = 'modal-close'
    closeButton.textContent = '×'
    closeButton.onclick = () => {
      closeActiveModal()
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
    if (canResolve(comment)) {
      const resolveButton = document.createElement('button')
      resolveButton.className = 'modal-actions-button modal-resolve-button'
      resolveButton.setAttribute('aria-label', 'Resolve thread')
      resolveButton.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg>`
      resolveButton.onclick = () => {
        submitResolveComment(comment, Date.now())
        closeActiveModal()
        document.dispatchEvent(
          new CustomEvent('toast', { detail: { message: 'Comment resolved' } }),
        )
      }
      headerActions.appendChild(resolveButton)
    }
    headerActions.appendChild(closeButton)

    header.appendChild(title)
    header.appendChild(headerActions)

    const content = document.createElement('div')
    content.className = 'modal-content'
    renderThreadContent(content, comment, replies)

    const replyComposerContainer = document.createElement('div')
    replyComposerContainer.className = 'reply-composer-container'

    const replyAuthorElement = document.createElement('div')
    replyAuthorElement.className = 'reply-author-element'

    const avatar = document.createElement('img')
    avatar.className = 'avatar'
    avatar.dataset.ignorePopup = 'true'
    getAvatarUrl(getAuthor(), 24).then(url => {
      avatar.src = url
    })
    avatar.alt = getAuthor()
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
    buttonIconSpan.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="var(--fpl-icon-color, var(--color-icon))" fill-rule="evenodd" d="M12 16a.5.5 0 0 1-.5-.5V8.707l-3.146 3.147a.5.5 0 0 1-.708-.708l4-4a.5.5 0 0 1 .708 0l4 4a.5.5 0 0 1-.708.708L12.5 8.707V15.5a.5.5 0 0 1-.5.5" clip-rule="evenodd"></path></svg>`
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

      submitNewComment(reply)

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

    void createMarkdownEditor({
      parent: editorMount,
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
      if (e.key === 'Escape' && activeModal === modal) {
        closeActiveModal()
      }
    }
    document.addEventListener('keydown', handleEscape)
    activeModalCleanup = () => {
      replyEditor?.destroy()
      replyEditor = null
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
      document.removeEventListener('keydown', handleEscape)
    }
  }

  const showComposer = async (range: Range) => {
    const version = ++composerVersion
    activeComposerCleanup?.()
    activeComposerCleanup = null
    if (activeComposer) {
      document.body.removeChild(activeComposer)
      activeComposer = null
    }
    const article = document.querySelector('article.popover-hint')
    if (!article) return
    const offsets = getRangeOffsets(range, article)
    if (!offsets) return
    const anchorHash = await hashText(offsets.text)
    if (version !== composerVersion || !article.isConnected) return
    const structuralAnchor = computeStructuralAnchor(range, article)

    const composer = document.createElement('div')
    composer.className = 'comment-composer'
    activeComposer = composer

    const rects = Array.from(range.getClientRects()).filter(rect => rect.width && rect.height)
    const rect = rects[0] ?? range.getBoundingClientRect()
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop

    composer.style.left = `${rect.left + scrollLeft}px`
    composer.style.top = `${rect.bottom + scrollTop + 8}px`

    const inputWrapper = document.createElement('div')
    inputWrapper.className = 'composer-input-wrapper composer-empty'

    const inputContainer = document.createElement('div')
    inputContainer.className = 'composer-input'
    inputContainer.setAttribute('role', 'textbox')
    inputContainer.setAttribute('aria-placeholder', 'Commentaire (compatible markdown)')

    const editorMount = document.createElement('div')
    editorMount.className = 'composer-editor-mount'

    const placeholderWrapper = document.createElement('div')
    placeholderWrapper.setAttribute('aria-hidden', 'true')
    const placeholderText = document.createElement('span')
    placeholderText.className = 'placeholder-text'
    placeholderText.textContent = 'Commentaire (compatible markdown)'
    placeholderWrapper.appendChild(placeholderText)

    let editor: MarkdownEditor | null = null
    activeComposerCleanup = () => {
      editor?.destroy()
      editor = null
    }

    const submitButton = document.createElement('button')
    submitButton.className = 'composer-submit'
    submitButton.disabled = true
    submitButton.innerHTML = `<span class="icon"><svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M12 16a.5.5 0 0 1-.5-.5V8.707l-3.146 3.147a.5.5 0 0 1-.708-.708l4-4a.5.5 0 0 1 .708 0l4 4a.5.5 0 0 1-.708.708L12.5 8.707V15.5a.5.5 0 0 1-.5.5" clip-rule="evenodd"></path></svg></span>`

    submitButton.onclick = async () => {
      if (!editor) return

      const content = editor.getValue().trim()
      if (!content) return

      const comment: MultiplayerComment = {
        id: crypto.randomUUID(),
        pageId: getCommentPageId(),
        parentId: null,
        anchorHash,
        anchorStart: offsets.startOffset,
        anchorEnd: offsets.endOffset,
        anchorText: offsets.text,
        content,
        author: getAuthor(),
        createdAt: Date.now(),
        updatedAt: null,
        deletedAt: null,
        resolvedAt: null,
        anchor: structuralAnchor,
        orphaned: null,
        lastRecoveredAt: null,
      }

      submitNewComment(comment)

      hideComposer()
    }

    inputContainer.appendChild(editorMount)
    inputContainer.appendChild(placeholderWrapper)
    inputWrapper.appendChild(inputContainer)
    inputWrapper.appendChild(submitButton)
    composer.appendChild(inputWrapper)
    document.body.appendChild(composer)

    editor = await createMarkdownEditor({
      parent: editorMount,
      onEdited: state => {
        submitButton.disabled = state.empty
        if (state.empty) {
          inputWrapper.classList.add('composer-empty')
        } else {
          inputWrapper.classList.remove('composer-empty')
        }
      },
      onSubmit: () => submitButton.click(),
      onCancel: () => hideComposer(),
    })

    if (composer.isConnected) {
      editor.focus()
    } else {
      editor.destroy()
      editor = null
    }
  }

  const handleTextSelection = () => {
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
      dispatch({ type: 'ui.selection.cleared' })
      return
    }

    const range = selection.getRangeAt(0)
    const article = document.querySelector('article.popover-hint')
    if (!article || !article.contains(range.commonAncestorContainer)) {
      dispatch({ type: 'ui.selection.cleared' })
      return
    }
    if (rangeIntersectsNotebookCellBoundary(range)) {
      dispatch({ type: 'ui.selection.cleared' })
      return
    }
    dispatch({ type: 'ui.selection.changed', range: range.cloneRange() })
  }

  const renderComments = () => {
    const { comments } = getState()
    const hiddenIds = comments
      .filter(comment => comment.deletedAt || comment.resolvedAt)
      .map(comment => comment.id)
    if (hiddenIds.length > 0) {
      dispatch({ type: 'ui.bubbleOffsets.prune', commentIds: hiddenIds })
    }

    document.querySelectorAll('.comment-highlight-layer').forEach(el => el.remove())
    document
      .querySelectorAll('.comment-highlight')
      .forEach(el => el.replaceWith(...Array.from(el.childNodes)))
    document.querySelectorAll('.comment-bubble').forEach(el => el.remove())

    const articleText = getArticleText()
    const article = document.querySelector('article.popover-hint')
    if (!article) return

    const topLevelComments = comments.filter(c => !c.parentId && !c.deletedAt && !c.resolvedAt)
    if (topLevelComments.length === 0) return

    const replyCounts = new Map<string, number>()
    for (const comment of comments) {
      if (comment.parentId && !comment.deletedAt) {
        replyCounts.set(comment.parentId, (replyCounts.get(comment.parentId) ?? 0) + 1)
      }
    }

    const textNodes: Array<{ node: Text; start: number; end: number }> = []
    const walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT)
    let offset = 0
    while (walker.nextNode()) {
      const node = walker.currentNode
      if (!(node instanceof Text)) continue
      if (node.length > 0) textNodes.push({ node, start: offset, end: offset + node.length })
      offset += node.length
    }
    const findTextNode = (position: number, isEnd: boolean) => {
      let low = 0
      let high = textNodes.length
      while (low < high) {
        const mid = (low + high) >>> 1
        const end = textNodes[mid].end
        if (isEnd ? end < position : end <= position) low = mid + 1
        else high = mid
      }
      return textNodes[low]
    }

    // Keep overlays detached while measuring ranges to avoid a layout per comment.
    const overlays = document.createDocumentFragment()
    const highlightLayer = document.createElement('div')
    highlightLayer.className = 'comment-highlight-layer'
    overlays.appendChild(highlightLayer)

    for (const comment of topLevelComments) {
      let startIdx = comment.anchorStart
      let endIdx = comment.anchorEnd

      const textAtOffsets = articleText.substring(startIdx, endIdx)
      const offsetsValid =
        startIdx >= 0 &&
        endIdx <= articleText.length &&
        startIdx < endIdx &&
        textAtOffsets === comment.anchorText

      if (!offsetsValid && comment.anchorText) {
        let recovered = false

        if (isStructuralAnchor(comment.anchor)) {
          const structuralResult = recoverFromStructuralAnchor(
            comment.anchor,
            comment.anchorText,
            article,
          )
          if (structuralResult) {
            startIdx = structuralResult.startIdx
            endIdx = structuralResult.endIdx
            recovered = true
          }
        }

        if (!recovered) {
          const matches: number[] = []
          let searchStart = 0
          while (true) {
            const idx = articleText.indexOf(comment.anchorText, searchStart)
            if (idx === -1) break
            matches.push(idx)
            searchStart = idx + 1
          }

          if (matches.length > 0) {
            const closest = matches.reduce((best, curr) =>
              Math.abs(curr - comment.anchorStart) < Math.abs(best - comment.anchorStart)
                ? curr
                : best,
            )
            startIdx = closest
            endIdx = closest + comment.anchorText.length
            recovered = true
          }
        }

        if (recovered && (startIdx !== comment.anchorStart || endIdx !== comment.anchorEnd)) {
          const opId = correctionOpId(comment.id, startIdx, endIdx)
          if (!getState().correctedAnchors.has(opId)) {
            dispatch({ type: 'ui.correctedAnchor.add', opId })
            submitUpdateComment(
              {
                ...comment,
                anchorStart: startIdx,
                anchorEnd: endIdx,
                lastRecoveredAt: Date.now(),
                updatedAt: Date.now(),
              },
              opId,
            )
          }
        }
      }

      if (startIdx === endIdx || startIdx < 0 || endIdx > articleText.length) {
        if (!comment.orphaned) {
          const opId = `orphan:${comment.id}`
          if (!getState().correctedAnchors.has(opId)) {
            dispatch({ type: 'ui.correctedAnchor.add', opId })
            submitUpdateComment({ ...comment, orphaned: true, updatedAt: Date.now() }, opId)
          }
        }
        continue
      }

      if (comment.orphaned) {
        const opId = `unorphan:${comment.id}`
        if (!getState().correctedAnchors.has(opId)) {
          dispatch({ type: 'ui.correctedAnchor.add', opId })
          submitUpdateComment({ ...comment, orphaned: false, updatedAt: Date.now() }, opId)
        }
      }

      const start = findTextNode(startIdx, false)
      const end = findTextNode(endIdx, true)

      if (start && end) {
        try {
          const range = document.createRange()
          range.setStart(start.node, startIdx - start.start)
          range.setEnd(end.node, endIdx - end.start)

          if (!comment.anchor) {
            const opId = `backfill-anchor:${comment.id}`
            if (!getState().correctedAnchors.has(opId)) {
              dispatch({ type: 'ui.correctedAnchor.add', opId })
              const newAnchor = computeStructuralAnchor(range, article)
              submitUpdateComment({ ...comment, anchor: newAnchor, updatedAt: Date.now() }, opId)
            }
          }

          const rects = getTextNodeRects(range)
          if (rects.length === 0) continue
          const anchorRect = rects[0]
          const scrollTop = window.pageYOffset || document.documentElement.scrollTop
          const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft

          for (const rect of rects) {
            const highlight = document.createElement('span')
            highlight.className = 'comment-highlight'
            highlight.dataset.commentId = comment.id
            highlight.style.left = `${rect.left + scrollLeft}px`
            highlight.style.top = `${rect.top + scrollTop}px`
            highlight.style.width = `${rect.width}px`
            highlight.style.height = `${rect.height}px`
            highlightLayer.appendChild(highlight)
          }

          const bubble = document.createElement('div')
          bubble.className = 'comment-bubble'
          if (getState().unreadCommentIds.has(comment.id)) {
            bubble.classList.add('comment-bubble-unread')
          }
          bubble.dataset.commentId = comment.id
          const baseLeft = anchorRect.right + scrollLeft + 8
          const baseTop = anchorRect.top + scrollTop
          const offset = getState().bubbleOffsets.get(comment.id)
          const initialLeft = baseLeft + (offset?.x ?? 0)
          const initialTop = baseTop + (offset?.y ?? 0)
          bubble.style.top = `${initialTop}px`
          bubble.style.left = `${initialLeft}px`

          const replyTop = document.createElement('div')
          replyTop.className = 'reply-top'

          const replyLeft = document.createElement('div')
          replyLeft.className = 'reply-left'

          const avatar = document.createElement('img')
          avatar.className = 'reply-avatar'
          avatar.dataset.ignorePopup = 'true'
          getAvatarUrl(comment.author, 24).then(url => {
            avatar.src = url
          })
          avatar.alt = comment.author

          const author = document.createElement('span')
          author.className = 'reply-author'
          author.textContent = comment.author

          const time = document.createElement('span')
          time.className = 'reply-time'
          time.textContent = formatRelativeTime(comment.createdAt)

          replyLeft.appendChild(avatar)
          replyLeft.appendChild(author)
          replyLeft.appendChild(time)
          replyTop.appendChild(replyLeft)

          const text = document.createElement('div')
          text.className = 'reply-text markdown-content'
          const currentSlug = getFullSlug(window)
          text.innerHTML = renderMarkdown(comment.content, currentSlug)

          bubble.appendChild(replyTop)
          bubble.appendChild(text)

          const replyCount = replyCounts.get(comment.id) ?? 0
          if (replyCount > 0) {
            const replies = document.createElement('div')
            replies.className = 'preview-replies'
            replies.textContent = `${replyCount} ${replyCount === 1 ? 'reply' : 'replies'}`
            bubble.appendChild(replies)
          }

          bubble.onmouseenter = () => {
            if (activeModal && activeModal.dataset.commentId === comment.id) {
              bubble.classList.add('modal-active')
            }
          }

          bubble.onmouseleave = () => {
            bubble.classList.remove('modal-active')
          }

          bubble.onmousedown = (e: MouseEvent) => {
            if (e.button !== 0) return
            e.preventDefault()
            let isDragging = true
            let dragStartX = e.pageX
            let dragStartY = e.pageY
            let startLeft = parseFloat(bubble.style.left) || initialLeft
            let startTop = parseFloat(bubble.style.top) || initialTop

            const onMouseMove = (moveEvent: MouseEvent) => {
              if (!isDragging) return
              const deltaX = moveEvent.pageX - dragStartX
              const deltaY = moveEvent.pageY - dragStartY
              bubble.style.left = `${startLeft + deltaX}px`
              bubble.style.top = `${startTop + deltaY}px`
            }

            const onMouseUp = () => {
              isDragging = false
              document.removeEventListener('mousemove', onMouseMove)
              document.removeEventListener('mouseup', onMouseUp)
              const currentLeft = parseFloat(bubble.style.left) || startLeft
              const currentTop = parseFloat(bubble.style.top) || startTop
              dispatch({
                type: 'ui.bubble.offsetUpdated',
                commentId: comment.id,
                offset: { x: currentLeft - baseLeft, y: currentTop - baseTop },
              })
            }

            document.addEventListener('mousemove', onMouseMove)
            document.addEventListener('mouseup', onMouseUp)
          }

          bubble.onclick = () => {
            if (activeModal) {
              return
            }
            bubble.classList.add('modal-active')
            const durationRaw = getComputedStyle(bubble)
              .getPropertyValue('--expand-animation-time')
              .trim()
            let delay = 120
            if (durationRaw.endsWith('ms')) {
              const parsed = Number.parseFloat(durationRaw)
              if (!Number.isNaN(parsed)) delay = parsed
            } else if (durationRaw.endsWith('s')) {
              const parsed = Number.parseFloat(durationRaw)
              if (!Number.isNaN(parsed)) delay = parsed * 1000
            }
            window.clearTimeout(threadOpenTimer)
            threadOpenTimer = window.setTimeout(() => {
              threadOpenTimer = 0
              if (!bubble.isConnected) return
              const bubbleRect = bubble.getBoundingClientRect()
              const scrollTop = window.pageYOffset || document.documentElement.scrollTop
              const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
              showCommentThread(comment.id, {
                top: bubbleRect.top + scrollTop,
                left: bubbleRect.left + scrollLeft,
              })
            }, delay)
          }

          overlays.appendChild(bubble)
        } catch (err) {
          console.warn('failed to highlight comment', err)
        }
      }
    }
    document.body.appendChild(overlays)
  }

  let rendering = false
  let renderAgain = false
  const renderAllComments = () => {
    if (rendering) {
      renderAgain = true
      return
    }
    // Anchor recovery dispatches synchronous updates; finish this pass before rerendering.
    do {
      renderAgain = false
      rendering = true
      try {
        renderComments()
      } finally {
        rendering = false
      }
    } while (renderAgain)
  }

  const openPendingCommentThread = () => {
    const targetId = getState().pendingHashCommentId
    if (!targetId) return
    const comment = getState().comments.find(
      item => item.id === targetId && !item.deletedAt && !item.resolvedAt,
    )
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

    showCommentThread(targetId, position)
    dispatch({ type: 'ui.hash.consumed' })
  }

  const cleanup = () => {
    window.clearTimeout(threadOpenTimer)
    threadOpenTimer = 0
    closeActiveModal()
    hideComposer()
    hideActionsPopover()
    document.querySelectorAll('.comment-highlight-layer').forEach(el => el.remove())
    document.querySelectorAll('.comment-selection-layer').forEach(el => el.remove())
    document
      .querySelectorAll('.comment-highlight')
      .forEach(el => el.replaceWith(...Array.from(el.childNodes)))
    document.querySelectorAll('.comment-bubble').forEach(el => el.remove())
    document.querySelectorAll('.comment-thread-modal').forEach(el => el.remove())
    document.querySelectorAll('.comment-actions-popover').forEach(el => el.remove())
    document.querySelectorAll('.delete-confirmation-overlay').forEach(el => el.remove())
    document.querySelectorAll('.delete-confirmation-modal').forEach(el => el.remove())
  }

  return {
    cleanup,
    closeActiveModal,
    handleTextSelection,
    hideActionsPopover,
    hideComposer,
    openPendingCommentThread,
    refreshActiveModal,
    renderAllComments,
    renderSelectionHighlight,
    showComposer,
  }
}
