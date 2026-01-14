import { getFullSlug } from "../../util/path"
import { renderMarkdown } from "../../util/markdown-renderer"
import { MarkdownEditor } from "../scripts/markdown-editor"
import type { Event, State } from "./core"
import type { MultiplayerComment, OperationInput } from "./model"
import {
  computeStructuralAnchor,
  getArticleText,
  getRangeOffsets,
  hashText,
  recoverFromStructuralAnchor,
} from "./anchor"
import { getAuthor, getAvatarUrl, getCommentPageId } from "./identity"

type UiDeps = {
  getState: () => State
  dispatch: (event: Event) => void
}

export function createCommentsUi({ getState, dispatch }: UiDeps) {
  let activeComposer: HTMLElement | null = null
  let activeModal: HTMLElement | null = null
  let activeActionsPopover: HTMLElement | null = null
  let selectionHighlightLayer: HTMLElement | null = null

  const submitComment = (op: OperationInput) => {
    dispatch({ type: "comment.submit", op })
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

    if (years > 0) return `${years} yr${years > 1 ? "s" : ""} ago`
    if (months > 0) return `${months} mo${months > 1 ? "s" : ""} ago`
    if (weeks > 0) return `${weeks} wk${weeks > 1 ? "s" : ""} ago`
    if (days > 0) return `${days} day${days > 1 ? "s" : ""} ago`
    if (hours > 0) return `${hours} hr${hours > 1 ? "s" : ""} ago`
    if (minutes > 0) return `${minutes} min${minutes > 1 ? "s" : ""} ago`
    return "just now"
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
    if (activeComposer) {
      document.body.removeChild(activeComposer)
      activeComposer = null
    }
    clearSelectionHighlight()
  }

  const hideActionsPopover = () => {
    if (activeActionsPopover) {
      document.body.removeChild(activeActionsPopover)
      activeActionsPopover = null
    }
    dispatch({ type: "ui.popover.close" })
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
    const layer = document.createElement("div")
    layer.className = "comment-selection-layer"
    layer.style.width = `${document.documentElement.scrollWidth}px`
    layer.style.height = `${document.documentElement.scrollHeight}px`
    for (const rect of rects) {
      const highlight = document.createElement("span")
      highlight.className = "comment-selection-highlight"
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
      console.error("Failed to copy:", err)
      return false
    }
  }

  const closeActiveModal = () => {
    if (!activeModal) return
    const activeId = activeModal.dataset.commentId
    document.body.removeChild(activeModal)
    activeModal = null
    dispatch({ type: "ui.modal.close" })
    if (activeId) {
      document
        .querySelector<HTMLElement>(`.comment-bubble[data-comment-id=\"${activeId}\"]`)
        ?.classList.remove("modal-active")
    }
  }

  const refreshActiveModal = () => {
    if (!activeModal) return
    const commentId = activeModal.dataset.commentId
    if (!commentId) return
    const comment = getState().comments.find((c) => c.id === commentId)
    if (!comment || comment.deletedAt) {
      closeActiveModal()
      return
    }
    const replies = getState().comments.filter((c) => c.parentId === commentId && !c.deletedAt)
    const content = activeModal.querySelector(".modal-content")
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

    const popover = document.createElement("div")
    popover.className = "comment-actions-popover"
    activeActionsPopover = popover
    dispatch({ type: "ui.popover.open", commentId })

    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop

    popover.style.left = `${buttonRect.right + scrollLeft + 4}px`
    popover.style.top = `${buttonRect.top + scrollTop}px`

    const editButton = document.createElement("button")
    editButton.className = "popover-action"
    editButton.innerHTML = `<span class="menu-item">Edit</span>`
    editButton.onclick = () => {
      hideActionsPopover()
      onEdit()
    }

    popover.appendChild(editButton)

    if (showDelete) {
      const deleteButton = document.createElement("button")
      deleteButton.className = "popover-action popover-action-danger"
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
        document.removeEventListener("mousedown", closeOnClickOutside)
      }
    }
    setTimeout(() => document.addEventListener("mousedown", closeOnClickOutside), 0)
  }

  const showThreadActionsPopover = (comment: MultiplayerComment, buttonRect: DOMRect) => {
    hideActionsPopover()

    const popover = document.createElement("div")
    popover.className = "comment-actions-popover"
    activeActionsPopover = popover
    dispatch({ type: "ui.popover.open", commentId: comment.id })

    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop

    popover.style.left = `${buttonRect.right + scrollLeft + 4}px`
    popover.style.top = `${buttonRect.top + scrollTop}px`

    const markUnreadButton = document.createElement("button")
    markUnreadButton.className = "popover-action"
    markUnreadButton.innerHTML = `<span class="menu-item">Mark as unread</span>`
    markUnreadButton.onclick = () => {
      hideActionsPopover()
    }

    const copyLinkButton = document.createElement("button")
    copyLinkButton.className = "popover-action"
    copyLinkButton.innerHTML = `<span class="menu-item">Copy link</span>`
    copyLinkButton.onclick = async () => {
      hideActionsPopover()
      const url = new URL(window.location.href)
      url.hash = `comment-${comment.id}`
      const copied = await copyToClipboard(url.toString())
      if (!copied) {
        console.error("failed to copy link")
      }
    }

    const deleteButton = document.createElement("button")
    deleteButton.className = "popover-action popover-action-danger"
    deleteButton.innerHTML = `<span class="menu-item">Delete thread...</span>`
    deleteButton.onclick = () => {
      hideActionsPopover()
      const replies = getState().comments.filter((c) => c.parentId === comment.id && !c.deletedAt)
      showDeleteConfirmation(comment, (deletedAt) => {
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
        document.removeEventListener("mousedown", closeOnClickOutside)
      }
    }
    setTimeout(() => document.addEventListener("mousedown", closeOnClickOutside), 0)
  }

  const enterEditMode = (comment: MultiplayerComment, textElement: HTMLElement) => {
    const wrapper = document.createElement("div")
    wrapper.className = "comment-edit-wrapper"

    const inputContent = document.createElement("div")
    inputContent.className = "edit-input-content"

    const editorMount = document.createElement("div")
    editorMount.className = "edit-input"

    let markdownEditor: MarkdownEditor | null

    const exitEditMode = () => {
      textElement.style.display = ""
      if (markdownEditor) {
        markdownEditor.destroy()
        markdownEditor = null
      }
      if (wrapper.parentNode) {
        wrapper.parentNode.removeChild(wrapper)
      }
    }

    const actions = document.createElement("div")
    actions.className = "edit-actions"

    const cancelButton = document.createElement("button")
    cancelButton.innerHTML = `<span class="button-container"><span class="button-text"><span class="button-content">Cancel</span></span></span>`
    cancelButton.className = "edit-button edit-button-cancel"
    cancelButton.onclick = () => {
      exitEditMode()
    }

    const saveButton = document.createElement("button")
    saveButton.innerHTML = `<span class="button-container"><span class="button-text"><span class="button-content">Save</span></span></span>`
    saveButton.className = "edit-button edit-button-save"
    saveButton.onclick = async () => {
      if (!markdownEditor) return
      const newContent = markdownEditor.getValue().trim()
      if (!newContent || newContent === comment.content) {
        exitEditMode()
        return
      }

      const updatedAt = Date.now()
      submitUpdateComment({
        ...comment,
        content: newContent,
        updatedAt,
      })

      exitEditMode()
    }

    markdownEditor = new MarkdownEditor({
      parent: editorMount,
      initialContent: comment.content,
      onSubmit: () => saveButton.click(),
      onCancel: exitEditMode,
    })

    inputContent.appendChild(editorMount)
    actions.appendChild(cancelButton)
    actions.appendChild(saveButton)
    wrapper.appendChild(inputContent)
    wrapper.appendChild(actions)

    textElement.style.display = "none"
    textElement.parentNode?.insertBefore(wrapper, textElement)

    markdownEditor.focus()
  }

  const showDeleteConfirmation = (
    comment: MultiplayerComment,
    onConfirm?: (deletedAt: number) => void,
  ) => {
    const overlay = document.createElement("div")
    overlay.className = "delete-confirmation-overlay"

    const modal = document.createElement("div")
    modal.className = "delete-confirmation-modal"

    const message = document.createElement("div")
    message.className = "delete-confirmation-message"
    message.textContent = "Delete this comment?"

    const actions = document.createElement("div")
    actions.className = "delete-confirmation-actions"

    const cancelButton = document.createElement("button")
    cancelButton.className = "edit-button edit-button-cancel"
    cancelButton.innerHTML = `<span class="button-container"><span class="button-text"><span class="button-content">Cancel</span></span></span>`

    const deleteButton = document.createElement("button")
    deleteButton.className = "edit-button edit-button-delete"
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
    submitComment({ opId: crypto.randomUUID(), type: "new", comment })
  }

  const submitUpdateComment = (comment: MultiplayerComment, opId?: string) => {
    submitComment({ opId: opId ?? crypto.randomUUID(), type: "update", comment })
  }

  const submitDeleteComment = (comment: MultiplayerComment, deletedAt: number) => {
    const deleted = { ...comment, deletedAt }
    submitComment({ opId: crypto.randomUUID(), type: "delete", comment: deleted })
  }

  const buildThreadItem = (comment: MultiplayerComment) => {
    const item = document.createElement("div")
    item.className = "reply-item"

    const top = document.createElement("div")
    top.className = "reply-top"

    const left = document.createElement("div")
    left.className = "reply-left"

    const avatar = document.createElement("img")
    avatar.className = "reply-avatar"
    getAvatarUrl(comment.author, 24).then((url) => {
      avatar.src = url
    })
    avatar.alt = comment.author

    const author = document.createElement("div")
    author.className = "reply-author"
    author.textContent = comment.author

    const time = document.createElement("div")
    time.className = "reply-time"
    time.textContent = formatRelativeTime(comment.createdAt)

    left.appendChild(avatar)
    left.appendChild(author)
    left.appendChild(time)

    const text = document.createElement("div")
    text.className = "reply-text markdown-content"
    const currentSlug = getFullSlug(window)
    text.innerHTML = renderMarkdown(comment.content, currentSlug)

    const right = document.createElement("div")
    right.className = "reply-right"

    const actions = document.createElement("button")
    actions.className = "reply-actions"
    actions.setAttribute("aria-label", "Comment actions")
    actions.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M7.5 12a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m6 0a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m4.5 1.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3" clip-rule="evenodd"></path></svg>`

    actions.onclick = (e: MouseEvent) => {
      e.stopPropagation()
      const buttonRect = actions.getBoundingClientRect()
      showActionsPopover(
        buttonRect,
        () => enterEditMode(comment, text),
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
    content.replaceChildren()
    content.appendChild(buildThreadItem(comment))
    for (const reply of replies) {
      content.appendChild(buildThreadItem(reply))
    }
  }

  const showCommentThread = (commentId: string, position?: { top: number; left: number }) => {
    const comment = getState().comments.find((c) => c.id === commentId)
    if (!comment) return

    if (activeModal) {
      if (activeModal.dataset.commentId === commentId) {
        const replies = getState().comments.filter((c) => c.parentId === commentId && !c.deletedAt)
        const content = activeModal.querySelector(".modal-content")
        if (content instanceof HTMLElement) {
          renderThreadContent(content, comment, replies)
        }
        return
      }
      closeActiveModal()
    }

    const replies = getState().comments.filter((c) => c.parentId === commentId && !c.deletedAt)

    const modal = document.createElement("div")
    modal.className = "comment-thread-modal"
    modal.dataset.commentId = commentId
    activeModal = modal
    dispatch({ type: "ui.modal.open", commentId })

    if (position) {
      modal.style.top = `${position.top}px`
      modal.style.left = `${position.left + 50}px`
      modal.style.right = "auto"
    }

    const header = document.createElement("div")
    header.className = "modal-header"

    const title = document.createElement("div")
    title.className = "modal-title"
    title.textContent = "comment"

    const headerActions = document.createElement("div")
    headerActions.className = "modal-actions"

    const headerMenuButton = document.createElement("button")
    headerMenuButton.className = "modal-actions-button"
    headerMenuButton.setAttribute("aria-label", "Thread actions")
    headerMenuButton.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M7.5 12a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m6 0a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m4.5 1.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3" clip-rule="evenodd"></path></svg>`
    headerMenuButton.onclick = (e: MouseEvent) => {
      e.stopPropagation()
      const buttonRect = headerMenuButton.getBoundingClientRect()
      showThreadActionsPopover(comment, buttonRect)
    }

    const closeButton = document.createElement("button")
    closeButton.className = "modal-close"
    closeButton.textContent = "×"
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
      modal.style.transform = "none"
      modal.style.right = "auto"
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

    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)

    headerActions.appendChild(headerMenuButton)
    headerActions.appendChild(closeButton)

    header.appendChild(title)
    header.appendChild(headerActions)

    const content = document.createElement("div")
    content.className = "modal-content"
    renderThreadContent(content, comment, replies)

    const replyComposerContainer = document.createElement("div")
    replyComposerContainer.className = "reply-composer-container"

    const replyAuthorElement = document.createElement("div")
    replyAuthorElement.className = "reply-author-element"

    const avatar = document.createElement("img")
    avatar.className = "avatar"
    getAvatarUrl(getAuthor(), 24).then((url) => {
      avatar.src = url
    })
    avatar.alt = getAuthor()
    replyAuthorElement.appendChild(avatar)

    const inputSectionWrapper = document.createElement("div")
    inputSectionWrapper.className = "input-section-wrapper composer-empty"

    const editableTypeaheadWrapper = document.createElement("div")
    editableTypeaheadWrapper.className = "editable-typeahead-wrapper"

    const primitiveWrapper = document.createElement("div")
    primitiveWrapper.className = "primitive-wrapper"
    primitiveWrapper.style.display = "block"

    const lexicalWrapper = document.createElement("div")
    lexicalWrapper.className = "lexical-wrapper"

    const editorMount = document.createElement("div")
    let replyEditor: MarkdownEditor

    const placeholderWrapper = document.createElement("div")
    placeholderWrapper.setAttribute("aria-hidden", "true")
    const placeholderText = document.createElement("span")
    placeholderText.className = "placeholder-text"
    placeholderText.textContent = "Reply"
    placeholderWrapper.appendChild(placeholderText)

    const actions = document.createElement("div")
    actions.className = "composer-actions"

    const replyButton = document.createElement("button")
    replyButton.type = "button"
    replyButton.setAttribute("aria-label", "Submit")
    replyButton.setAttribute("aria-disabled", "true")
    replyButton.setAttribute("data-tooltip", "Submit")
    replyButton.setAttribute("data-tooltip-type", "text")
    replyButton.tabIndex = 0
    replyButton.className = "submit-button"
    const buttonIconSpan = document.createElement("span")
    buttonIconSpan.setAttribute("aria-hidden", "true")
    buttonIconSpan.className = "button-icon"
    buttonIconSpan.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="var(--fpl-icon-color, var(--color-icon))" fill-rule="evenodd" d="M12 16a.5.5 0 0 1-.5-.5V8.707l-3.146 3.147a.5.5 0 0 1-.708-.708l4-4a.5.5 0 0 1 .708 0l4 4a.5.5 0 0 1-.708.708L12.5 8.707V15.5a.5.5 0 0 1-.5.5" clip-rule="evenodd"></path></svg>`
    replyButton.appendChild(buttonIconSpan)

    replyButton.onclick = async () => {
      if (replyButton.getAttribute("aria-disabled") === "true") return

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
      }

      submitNewComment(reply)

      replyEditor.setValue("")
      inputSectionWrapper.classList.add("composer-empty")
      replyButton.setAttribute("aria-disabled", "true")
    }

    replyEditor = new MarkdownEditor({
      parent: editorMount,
      onChange: (content) => {
        const isEmpty = content.trim().length === 0
        if (isEmpty) {
          inputSectionWrapper.classList.add("composer-empty")
          replyButton.setAttribute("aria-disabled", "true")
        } else {
          inputSectionWrapper.classList.remove("composer-empty")
          replyButton.setAttribute("aria-disabled", "false")
        }
      },
      onSubmit: () => replyButton.click(),
    })

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

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && activeModal === modal) {
        closeActiveModal()
        document.removeEventListener("keydown", handleEscape)
      }
    }
    document.addEventListener("keydown", handleEscape)
  }

  const showComposer = async (range: Range) => {
    if (activeComposer) {
      document.body.removeChild(activeComposer)
      activeComposer = null
    }
    const article = document.querySelector("article.popover-hint")
    if (!article) return
    const offsets = getRangeOffsets(range, article)
    if (!offsets) return
    const anchorHash = await hashText(offsets.text)
    const structuralAnchor = computeStructuralAnchor(range, article)

    const composer = document.createElement("div")
    composer.className = "comment-composer"
    activeComposer = composer

    const rects = Array.from(range.getClientRects()).filter((rect) => rect.width && rect.height)
    const rect = rects[0] ?? range.getBoundingClientRect()
    const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop

    composer.style.left = `${rect.left + scrollLeft}px`
    composer.style.top = `${rect.bottom + scrollTop + 8}px`

    const inputWrapper = document.createElement("div")
    inputWrapper.className = "composer-input-wrapper composer-empty"

    const inputContainer = document.createElement("div")
    inputContainer.className = "composer-input"
    inputContainer.setAttribute("role", "textbox")
    inputContainer.setAttribute("aria-placeholder", "Add a comment")

    const editorMount = document.createElement("div")
    editorMount.className = "composer-editor-mount"

    const placeholderWrapper = document.createElement("div")
    placeholderWrapper.setAttribute("aria-hidden", "true")
    const placeholderText = document.createElement("span")
    placeholderText.className = "placeholder-text"
    placeholderText.textContent = "Add a comment"
    placeholderWrapper.appendChild(placeholderText)

    let editor: MarkdownEditor

    const submitButton = document.createElement("button")
    submitButton.className = "composer-submit"
    submitButton.disabled = true
    submitButton.innerHTML = `<span class="icon"><svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M12 16a.5.5 0 0 1-.5-.5V8.707l-3.146 3.147a.5.5 0 0 1-.708-.708l4-4a.5.5 0 0 1 .708 0l4 4a.5.5 0 0 1-.708.708L12.5 8.707V15.5a.5.5 0 0 1-.5.5" clip-rule="evenodd"></path></svg></span>`

    submitButton.onclick = async () => {
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
        anchor: structuralAnchor,
        orphaned: null,
        lastRecoveredAt: null,
      }

      submitNewComment(comment)

      hideComposer()
    }

    editor = new MarkdownEditor({
      parent: editorMount,
      onChange: (content) => {
        const trimmed = content.trim()
        submitButton.disabled = trimmed.length === 0
        if (trimmed.length === 0) {
          inputWrapper.classList.add("composer-empty")
        } else {
          inputWrapper.classList.remove("composer-empty")
        }
      },
      onSubmit: () => submitButton.click(),
      onCancel: () => hideComposer(),
    })

    inputContainer.appendChild(editorMount)
    inputContainer.appendChild(placeholderWrapper)
    inputWrapper.appendChild(inputContainer)
    inputWrapper.appendChild(submitButton)
    composer.appendChild(inputWrapper)
    document.body.appendChild(composer)

    editor.focus()
  }

  const handleTextSelection = () => {
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
      dispatch({ type: "ui.selection.cleared" })
      return
    }

    const range = selection.getRangeAt(0)
    const article = document.querySelector("article.popover-hint")
    if (!article || !article.contains(range.commonAncestorContainer)) {
      dispatch({ type: "ui.selection.cleared" })
      return
    }
    dispatch({ type: "ui.selection.changed", range: range.cloneRange() })
  }

  const renderAllComments = () => {
    const { comments } = getState()
    const deletedIds = comments.filter((comment) => comment.deletedAt).map((comment) => comment.id)
    if (deletedIds.length > 0) {
      dispatch({ type: "ui.bubbleOffsets.prune", commentIds: deletedIds })
    }

    document.querySelectorAll(".comment-highlight-layer").forEach((el) => el.remove())
    document
      .querySelectorAll(".comment-highlight")
      .forEach((el) => el.replaceWith(...Array.from(el.childNodes)))
    document.querySelectorAll(".comment-bubble").forEach((el) => el.remove())

    const articleText = getArticleText()
    const article = document.querySelector("article.popover-hint")
    if (!article) return

    const highlightLayer = document.createElement("div")
    highlightLayer.className = "comment-highlight-layer"
    highlightLayer.style.width = `${document.documentElement.scrollWidth}px`
    highlightLayer.style.height = `${document.documentElement.scrollHeight}px`
    document.body.appendChild(highlightLayer)

    const topLevelComments = comments.filter((c) => !c.parentId && !c.deletedAt)

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

        if (comment.anchor) {
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
            dispatch({ type: "ui.correctedAnchor.add", opId })
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
            dispatch({ type: "ui.correctedAnchor.add", opId })
            submitUpdateComment(
              {
                ...comment,
                orphaned: true,
                updatedAt: Date.now(),
              },
              opId,
            )
          }
        }
        continue
      }

      if (comment.orphaned) {
        const opId = `unorphan:${comment.id}`
        if (!getState().correctedAnchors.has(opId)) {
          dispatch({ type: "ui.correctedAnchor.add", opId })
          submitUpdateComment(
            {
              ...comment,
              orphaned: false,
              updatedAt: Date.now(),
            },
            opId,
          )
        }
      }

      const walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT)
      let currentOffset = 0
      let startNode: Text | null = null
      let startNodeOffset = 0
      let endNode: Text | null = null
      let endNodeOffset = 0

      while (walker.nextNode()) {
        const currentNode = walker.currentNode
        if (!(currentNode instanceof Text)) continue
        const textNode = currentNode
        const nodeLength = textNode.length

        if (startNode === null && currentOffset + nodeLength > startIdx) {
          startNode = textNode
          startNodeOffset = startIdx - currentOffset
        }

        if (currentOffset + nodeLength >= endIdx) {
          endNode = textNode
          endNodeOffset = endIdx - currentOffset
          break
        }

        currentOffset += nodeLength
      }

      if (startNode && endNode) {
        try {
          const range = document.createRange()
          range.setStart(startNode, startNodeOffset)
          range.setEnd(endNode, endNodeOffset)

          if (!comment.anchor) {
            const opId = `backfill-anchor:${comment.id}`
            if (!getState().correctedAnchors.has(opId)) {
              dispatch({ type: "ui.correctedAnchor.add", opId })
              const newAnchor = computeStructuralAnchor(range, article)
              submitUpdateComment(
                {
                  ...comment,
                  anchor: newAnchor,
                  updatedAt: Date.now(),
                },
                opId,
              )
            }
          }

          const rects = Array.from(range.getClientRects()).filter(
            (rect) => rect.width && rect.height,
          )
          if (rects.length === 0) continue
          const anchorRect = rects[0]
          const scrollTop = window.pageYOffset || document.documentElement.scrollTop
          const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft

          for (const rect of rects) {
            const highlight = document.createElement("span")
            highlight.className = "comment-highlight"
            highlight.dataset.commentId = comment.id
            highlight.style.left = `${rect.left + scrollLeft}px`
            highlight.style.top = `${rect.top + scrollTop}px`
            highlight.style.width = `${rect.width}px`
            highlight.style.height = `${rect.height}px`
            highlightLayer.appendChild(highlight)
          }

          const bubble = document.createElement("div")
          bubble.className = "comment-bubble"
          bubble.dataset.commentId = comment.id
          const baseLeft = anchorRect.right + scrollLeft + 8
          const baseTop = anchorRect.top + scrollTop
          const offset = getState().bubbleOffsets.get(comment.id)
          const initialLeft = baseLeft + (offset?.x ?? 0)
          const initialTop = baseTop + (offset?.y ?? 0)
          bubble.style.top = `${initialTop}px`
          bubble.style.left = `${initialLeft}px`

          const replyTop = document.createElement("div")
          replyTop.className = "reply-top"

          const replyLeft = document.createElement("div")
          replyLeft.className = "reply-left"

          const avatar = document.createElement("img")
          avatar.className = "reply-avatar"
          getAvatarUrl(comment.author, 24).then((url) => {
            avatar.src = url
          })
          avatar.alt = comment.author

          const author = document.createElement("span")
          author.className = "reply-author"
          author.textContent = comment.author

          const time = document.createElement("span")
          time.className = "reply-time"
          time.textContent = formatRelativeTime(comment.createdAt)

          replyLeft.appendChild(avatar)
          replyLeft.appendChild(author)
          replyLeft.appendChild(time)
          replyTop.appendChild(replyLeft)

          const text = document.createElement("div")
          text.className = "reply-text markdown-content"
          const currentSlug = getFullSlug(window)
          text.innerHTML = renderMarkdown(comment.content, currentSlug)

          bubble.appendChild(replyTop)
          bubble.appendChild(text)

          const replyCount = getState().comments.filter(
            (c) => c.parentId === comment.id && !c.deletedAt,
          ).length
          if (replyCount > 0) {
            const replies = document.createElement("div")
            replies.className = "preview-replies"
            replies.textContent = `${replyCount} ${replyCount === 1 ? "reply" : "replies"}`
            bubble.appendChild(replies)
          }

          bubble.onmouseenter = () => {
            if (activeModal && activeModal.dataset.commentId === comment.id) {
              bubble.classList.add("modal-active")
            }
          }

          bubble.onmouseleave = () => {
            bubble.classList.remove("modal-active")
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
              document.removeEventListener("mousemove", onMouseMove)
              document.removeEventListener("mouseup", onMouseUp)
              const currentLeft = parseFloat(bubble.style.left) || startLeft
              const currentTop = parseFloat(bubble.style.top) || startTop
              dispatch({
                type: "ui.bubble.offsetUpdated",
                commentId: comment.id,
                offset: {
                  x: currentLeft - baseLeft,
                  y: currentTop - baseTop,
                },
              })
            }

            document.addEventListener("mousemove", onMouseMove)
            document.addEventListener("mouseup", onMouseUp)
          }

          bubble.onclick = () => {
            if (activeModal) {
              return
            }
            bubble.classList.add("modal-active")
            const durationRaw = getComputedStyle(bubble)
              .getPropertyValue("--expand-animation-time")
              .trim()
            let delay = 120
            if (durationRaw.endsWith("ms")) {
              const parsed = Number.parseFloat(durationRaw)
              if (!Number.isNaN(parsed)) delay = parsed
            } else if (durationRaw.endsWith("s")) {
              const parsed = Number.parseFloat(durationRaw)
              if (!Number.isNaN(parsed)) delay = parsed * 1000
            }
            window.setTimeout(() => {
              const bubbleRect = bubble.getBoundingClientRect()
              const scrollTop = window.pageYOffset || document.documentElement.scrollTop
              const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
              showCommentThread(comment.id, {
                top: bubbleRect.top + scrollTop,
                left: bubbleRect.left + scrollLeft,
              })
            }, delay)
          }

          document.body.appendChild(bubble)
        } catch (err) {
          console.warn("failed to highlight comment", err)
        }
      }
    }
  }

  const openPendingCommentThread = () => {
    const targetId = getState().pendingHashCommentId
    if (!targetId) return
    const comment = getState().comments.find((item) => item.id === targetId && !item.deletedAt)
    if (!comment) return

    const bubble = document.querySelector<HTMLElement>(
      `.comment-bubble[data-comment-id=\"${targetId}\"]`,
    )
    const highlight = document.querySelector<HTMLElement>(
      `.comment-highlight[data-comment-id=\"${targetId}\"]`,
    )
    const target = bubble ?? highlight
    let position: { top: number; left: number } | undefined
    if (target) {
      const rect = target.getBoundingClientRect()
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop
      const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
      position = { top: rect.top + scrollTop, left: rect.left + scrollLeft }
      target.scrollIntoView({ block: "center", inline: "nearest" })
    }

    showCommentThread(targetId, position)
    dispatch({ type: "ui.hash.consumed" })
  }

  const cleanup = () => {
    closeActiveModal()
    hideComposer()
    hideActionsPopover()
    document.querySelectorAll(".comment-highlight-layer").forEach((el) => el.remove())
    document.querySelectorAll(".comment-selection-layer").forEach((el) => el.remove())
    document
      .querySelectorAll(".comment-highlight")
      .forEach((el) => el.replaceWith(...Array.from(el.childNodes)))
    document.querySelectorAll(".comment-bubble").forEach((el) => el.remove())
    document.querySelectorAll(".comment-thread-modal").forEach((el) => el.remove())
    document.querySelectorAll(".comment-actions-popover").forEach((el) => el.remove())
    document.querySelectorAll(".delete-confirmation-overlay").forEach((el) => el.remove())
    document.querySelectorAll(".delete-confirmation-modal").forEach((el) => el.remove())
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
