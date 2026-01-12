import { getFullSlug } from "../../util/path"
import { MarkdownEditor } from "./markdown-editor"
import { renderMarkdown } from "../../util/markdown-renderer"

type MultiplayerComment = {
  id: string
  pageId: string
  parentId: string | null
  anchorHash: string
  anchorStart: number
  anchorEnd: number
  anchorText: string
  content: string
  author: string
  createdAt: number
  updatedAt: number | null
  deletedAt: number | null
}

type OperationType = "new" | "update" | "delete"

type OperationInput = {
  opId: string
  type: OperationType
  comment: MultiplayerComment
}

type OperationRecord = OperationInput & {
  seq: number
}

// BUG: if sending consecutive messages it seems like it is not being registered to D1
// i.e. some synching problem, which means probably will need to implement WebRTC + CRDT, circa raft.

type BroadcastMessage =
  | {
      type: "init"
      comments: MultiplayerComment[]
      latestSeq: number
    }
  | {
      type: "delta"
      ops: OperationRecord[]
      latestSeq: number
    }
  | {
      type: "op"
      op: OperationRecord
    }
  | {
      type: "ack"
      opId: string
      seq: number
    }
  | {
      type: "error"
      message: string
    }

let ws: WebSocket | null = null
let comments: MultiplayerComment[] = []
let activeSelection: Range | null = null
let bubbleOffsets = new Map<string, { x: number; y: number }>()
let selectionHighlightLayer: HTMLElement | null = null
let pendingHashCommentId: string | null = null
let lastSeq = 0
let hasSnapshot = false
let pendingOps = new Map<string, OperationInput>()
const githubAvatarCache = new Map<string, string>()
const githubAvatarStoragePrefix = "comment-author-github-avatar:"

function getAuthor(): string {
  let author =
    localStorage.getItem("comment-author") ?? localStorage.getItem("comment-author-github-login")
  if (!author) {
    author = `anon-${Math.random().toString(36).slice(2, 8)}`
    localStorage.setItem("comment-author", author)
  }
  return author
}

function getCommentPageId(): string {
  const slug = getFullSlug(window)
  const hostname = window.location.hostname.toLowerCase()
  if (hostname === "stream.aarnphm.xyz") {
    return `stream:${slug}`
  }
  return slug
}

async function getGravatarUrl(identifier: string, size: number = 24): Promise<string> {
  const normalized = identifier.trim().toLowerCase()
  const encoder = new TextEncoder()
  const data = encoder.encode(normalized)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("")
  return `https://gravatar.com/avatar/${hashHex}?s=${size}&d=identicon&r=pg`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

async function getGithubAvatarUrl(login: string): Promise<string | null> {
  const cached = githubAvatarCache.get(login)
  if (cached) return cached
  const storageKey = `${githubAvatarStoragePrefix}${login}`
  try {
    const stored = sessionStorage.getItem(storageKey)
    if (stored) {
      githubAvatarCache.set(login, stored)
      return stored
    }
  } catch {}
  const resp = await fetch(`https://api.github.com/users/${encodeURIComponent(login)}`, {
    headers: { Accept: "application/vnd.github+json" },
  })
  if (!resp.ok) return null
  let data: unknown
  try {
    data = await resp.json()
  } catch {
    return null
  }
  if (!isRecord(data)) return null
  const avatar = data["avatar_url"]
  if (typeof avatar !== "string" || avatar.length === 0) return null
  githubAvatarCache.set(login, avatar)
  try {
    sessionStorage.setItem(storageKey, avatar)
  } catch {}
  return avatar
}

async function getAvatarUrl(author: string, size: number = 24): Promise<string> {
  const login = localStorage.getItem("comment-author-github-login")
  const localAuthor = localStorage.getItem("comment-author")
  if (login && (author === localAuthor || author === login)) {
    const githubUrl = await getGithubAvatarUrl(login)
    if (githubUrl) return githubUrl
  }
  return getGravatarUrl(author, size)
}

function formatRelativeTime(timestamp: number): string {
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

async function hashText(text: string): Promise<string> {
  const encoder = new TextEncoder()
  const data = encoder.encode(text)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("")
}

function getArticleText(): string {
  const article = document.querySelector("article.popover-hint")
  return article?.textContent || ""
}

function parseCommentHash(): string | null {
  const { hash } = window.location
  if (!hash) return null
  const prefix = "#comment-"
  if (!hash.startsWith(prefix)) return null
  const rawId = hash.slice(prefix.length)
  if (!rawId) return null
  try {
    return decodeURIComponent(rawId)
  } catch {
    return rawId
  }
}

function openPendingCommentThread() {
  if (!pendingHashCommentId) return
  const targetId = pendingHashCommentId
  const comment = comments.find((item) => item.id === targetId && !item.deletedAt)
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
    target.scrollIntoView({ block: "center", inline: "nearest" })
  }

  showCommentThread(targetId, position)
  pendingHashCommentId = null
}

function setPendingCommentFromHash() {
  const targetId = parseCommentHash()
  if (!targetId) {
    pendingHashCommentId = null
    return
  }
  pendingHashCommentId = targetId
  openPendingCommentThread()
}

function clearSelectionHighlight() {
  if (selectionHighlightLayer) {
    selectionHighlightLayer.remove()
    selectionHighlightLayer = null
  }
}

function renderSelectionHighlight(range: Range) {
  clearSelectionHighlight()
  const rects = Array.from(range.getClientRects()).filter((rect) => rect.width && rect.height)
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

function getRangeOffsets(range: Range, root: Element) {
  const text = range.toString()
  if (!text) return null
  const startRange = document.createRange()
  startRange.setStart(root, 0)
  startRange.setEnd(range.startContainer, range.startOffset)
  const startOffset = startRange.toString().length
  const endRange = document.createRange()
  endRange.setStart(root, 0)
  endRange.setEnd(range.endContainer, range.endOffset)
  const endOffset = endRange.toString().length
  if (endOffset <= startOffset) return null
  return { text, startOffset, endOffset }
}

async function copyToClipboard(text: string) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      return false
    }
  }

  const textarea = document.createElement("textarea")
  textarea.value = text
  textarea.style.position = "fixed"
  textarea.style.opacity = "0"
  document.body.appendChild(textarea)
  textarea.focus()
  textarea.select()
  let succeeded = false
  try {
    succeeded = document.execCommand("copy")
  } catch {
    succeeded = false
  }
  document.body.removeChild(textarea)
  return succeeded
}

function refreshActiveModal() {
  if (!activeModal) return
  const commentId = activeModal.dataset.commentId
  if (!commentId) return
  const comment = comments.find((c) => c.id === commentId)
  if (!comment) return
  const replies = comments.filter((c) => c.parentId === commentId && !c.deletedAt)
  const content = activeModal.querySelector(".modal-content")
  if (!(content instanceof HTMLElement)) return
  renderThreadContent(content, comment, replies)
}

function upsertComment(comment: MultiplayerComment) {
  const idx = comments.findIndex((c) => c.id === comment.id)
  if (idx === -1) {
    comments.push(comment)
    return
  }
  comments[idx] = comment
}

function applyCommentSilent(comment: MultiplayerComment) {
  upsertComment(comment)
  if (comment.deletedAt) {
    bubbleOffsets.delete(comment.id)
  }
}

function applyComment(comment: MultiplayerComment) {
  applyCommentSilent(comment)
  renderAllComments()
  refreshActiveModal()
}

function updateCommentAuthors(oldAuthor: string, newAuthor: string) {
  let updated = false
  for (const comment of comments) {
    if (comment.author === oldAuthor) {
      comment.author = newAuthor
      updated = true
    }
  }
  if (updated) {
    renderAllComments()
    refreshActiveModal()
  }
}

function applyOperation(op: OperationRecord) {
  if (op.seq > lastSeq) {
    lastSeq = op.seq
  }
  if (pendingOps.has(op.opId)) {
    pendingOps.delete(op.opId)
  }
  applyComment(op.comment)
}

function applyOperations(ops: OperationRecord[]) {
  if (ops.length === 0) return
  for (const op of ops) {
    if (op.seq > lastSeq) {
      lastSeq = op.seq
    }
    if (pendingOps.has(op.opId)) {
      pendingOps.delete(op.opId)
    }
    applyCommentSilent(op.comment)
  }
  renderAllComments()
  refreshActiveModal()
}

function queueOperation(op: OperationInput) {
  pendingOps.set(op.opId, op)
}

function sendOperation(op: OperationInput) {
  queueOperation(op)
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "op", op }))
  }
}

function flushPendingOperations() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return
  for (const op of pendingOps.values()) {
    ws.send(JSON.stringify({ type: "op", op }))
  }
}

function submitNewComment(comment: MultiplayerComment) {
  applyComment(comment)
  sendOperation({ opId: crypto.randomUUID(), type: "new", comment })
}

function submitUpdateComment(comment: MultiplayerComment) {
  applyComment(comment)
  sendOperation({ opId: crypto.randomUUID(), type: "update", comment })
}

function submitDeleteComment(comment: MultiplayerComment, deletedAt: number) {
  const deleted = { ...comment, deletedAt }
  applyComment(deleted)
  sendOperation({ opId: crypto.randomUUID(), type: "delete", comment: deleted })
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
  const pageId = encodeURIComponent(getCommentPageId())
  const sinceParam = hasSnapshot && lastSeq > 0 ? `&since=${lastSeq}` : ""
  const wsUrl = `${protocol}//${window.location.host}/comments/websocket?pageId=${pageId}${sinceParam}`

  ws = new WebSocket(wsUrl)

  ws.onopen = () => {
    flushPendingOperations()
  }

  ws.onmessage = (event) => {
    const msg: BroadcastMessage = JSON.parse(event.data)

    if (msg.type === "init") {
      comments = msg.comments
      lastSeq = msg.latestSeq
      hasSnapshot = true
      for (const op of pendingOps.values()) {
        applyCommentSilent(op.comment)
      }
      renderAllComments()
      refreshActiveModal()
      openPendingCommentThread()
      flushPendingOperations()
    } else if (msg.type === "delta") {
      applyOperations(msg.ops)
      if (msg.latestSeq > lastSeq) {
        lastSeq = msg.latestSeq
      }
      hasSnapshot = true
      flushPendingOperations()
    } else if (msg.type === "op") {
      applyOperation(msg.op)
    } else if (msg.type === "ack") {
      if (msg.seq > lastSeq) {
        lastSeq = msg.seq
      }
      pendingOps.delete(msg.opId)
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
    setTimeout(connectWebSocket, 3000)
  }

  ws.onerror = (err) => {
    console.error("multiplayer comments websocket error:", err)
    console.error("websocket state:", ws!.readyState)
    console.error("websocket url:", ws!.url)
  }
}

let activeComposer: HTMLElement | null = null
let activeModal: HTMLElement | null = null
let activeActionsPopover: HTMLElement | null = null

function hideComposer() {
  if (activeComposer) {
    document.body.removeChild(activeComposer)
    activeComposer = null
  }
  clearSelectionHighlight()
  activeSelection = null
}

function hideActionsPopover() {
  if (activeActionsPopover) {
    document.body.removeChild(activeActionsPopover)
    activeActionsPopover = null
  }
}

function showActionsPopover(
  comment: MultiplayerComment,
  buttonRect: DOMRect,
  onEdit: () => void,
  onDelete: () => void,
  showDelete: boolean = true,
) {
  hideActionsPopover()

  const popover = document.createElement("div")
  popover.className = "comment-actions-popover"
  activeActionsPopover = popover

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
    if (!popover.contains(e.target as Node)) {
      hideActionsPopover()
      document.removeEventListener("mousedown", closeOnClickOutside)
    }
  }
  setTimeout(() => document.addEventListener("mousedown", closeOnClickOutside), 0)
}

function showThreadActionsPopover(comment: MultiplayerComment, buttonRect: DOMRect) {
  hideActionsPopover()

  const popover = document.createElement("div")
  popover.className = "comment-actions-popover"
  activeActionsPopover = popover

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
    const replies = comments.filter((c) => c.parentId === comment.id && !c.deletedAt)
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
    if (!popover.contains(e.target as Node)) {
      hideActionsPopover()
      document.removeEventListener("mousedown", closeOnClickOutside)
    }
  }
  setTimeout(() => document.addEventListener("mousedown", closeOnClickOutside), 0)
}

function enterEditMode(comment: MultiplayerComment, textElement: HTMLElement) {
  const wrapper = document.createElement("div")
  wrapper.className = "comment-edit-wrapper"

  const inputContent = document.createElement("div")
  inputContent.className = "edit-input-content"

  const editorMount = document.createElement("div")
  editorMount.className = "edit-input"

  let markdownEditor: MarkdownEditor | null

  function exitEditMode() {
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

function showDeleteConfirmation(
  comment: MultiplayerComment,
  onConfirm?: (deletedAt: number) => void,
) {
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

function handleTextSelection() {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    hideComposer()
    return
  }

  const range = selection.getRangeAt(0)
  const article = document.querySelector("article.popover-hint")
  if (!article || !article.contains(range.commonAncestorContainer)) {
    hideComposer()
    return
  }
  activeSelection = range.cloneRange()
  renderSelectionHighlight(activeSelection)

  showComposer(activeSelection)
}

async function showComposer(range: Range) {
  if (activeComposer) {
    document.body.removeChild(activeComposer)
    activeComposer = null
  }
  const article = document.querySelector("article.popover-hint")
  if (!article) return
  const offsets = getRangeOffsets(range, article)
  if (!offsets) return
  const anchorHash = await hashText(offsets.text)

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

function renderAllComments() {
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
    if (startIdx >= endIdx || endIdx > articleText.length || startIdx < 0) {
      if (comment.anchorText) {
        const fallbackStart = articleText.indexOf(comment.anchorText)
        if (fallbackStart !== -1) {
          startIdx = fallbackStart
          endIdx = fallbackStart + comment.anchorText.length
        }
      }
    }

    if (startIdx === endIdx) continue
    if (startIdx < 0 || endIdx > articleText.length) continue

    const walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT)
    let currentOffset = 0
    let startNode: Text | null = null
    let startNodeOffset = 0
    let endNode: Text | null = null
    let endNodeOffset = 0

    while (walker.nextNode()) {
      const textNode = walker.currentNode as Text
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

        const rects = Array.from(range.getClientRects()).filter((rect) => rect.width && rect.height)
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
        const offset = bubbleOffsets.get(comment.id)
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
        text.innerHTML = renderMarkdown(comment.content)

        bubble.appendChild(replyTop)
        bubble.appendChild(text)

        const replyCount = comments.filter((c) => c.parentId === comment.id && !c.deletedAt).length
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
            bubbleOffsets.set(comment.id, {
              x: currentLeft - baseLeft,
              y: currentTop - baseTop,
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

function buildThreadItem(comment: MultiplayerComment) {
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
  text.innerHTML = renderMarkdown(comment.content)

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
      comment,
      buttonRect,
      () => enterEditMode(comment, text),
      () => showDeleteConfirmation(comment),
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

function renderThreadContent(
  content: HTMLElement,
  comment: MultiplayerComment,
  replies: MultiplayerComment[],
) {
  content.replaceChildren()
  content.appendChild(buildThreadItem(comment))
  for (const reply of replies) {
    content.appendChild(buildThreadItem(reply))
  }
}

function showCommentThread(commentId: string, position?: { top: number; left: number }) {
  const comment = comments.find((c) => c.id === commentId)
  if (!comment) return

  if (activeModal) {
    if (activeModal.dataset.commentId === commentId) {
      const replies = comments.filter((c) => c.parentId === commentId && !c.deletedAt)
      const content = activeModal.querySelector(".modal-content")
      if (content instanceof HTMLElement) {
        renderThreadContent(content, comment, replies)
      }
      return
    }
    document.body.removeChild(activeModal)
    activeModal = null
  }

  const replies = comments.filter((c) => c.parentId === commentId && !c.deletedAt)

  const modal = document.createElement("div")
  modal.className = "comment-thread-modal"
  modal.dataset.commentId = commentId
  activeModal = modal

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
    if (activeModal) {
      document.body.removeChild(activeModal)
      activeModal = null
    }
  }

  let isDragging = false
  let dragStartX = 0
  let dragStartY = 0
  let modalStartX = 0
  let modalStartY = 0

  header.onmousedown = (e: MouseEvent) => {
    if (headerActions.contains(e.target as Node)) return
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
}

document.addEventListener("nav", () => {
  lastSeq = 0
  hasSnapshot = false
  pendingOps = new Map()
  connectWebSocket()
  setPendingCommentFromHash()

  const mouseUp = (event: MouseEvent) => {
    if (event.button !== 0) return
    if (activeComposer && event.target && activeComposer.contains(event.target as Node)) {
      return
    }
    handleTextSelection()
  }

  let resizeFrame = 0
  const handleResize = () => {
    if (resizeFrame) {
      cancelAnimationFrame(resizeFrame)
    }
    resizeFrame = requestAnimationFrame(() => {
      renderAllComments()
      refreshActiveModal()
      if (activeSelection) {
        renderSelectionHighlight(activeSelection)
      }
      resizeFrame = 0
    })
  }

  document.addEventListener("mouseup", mouseUp)
  window.addEventListener("resize", handleResize)

  const handleAuthorUpdate = (event: CustomEventMap["commentauthorupdated"]) => {
    const detail = event.detail
    if (!detail?.oldAuthor || !detail?.newAuthor) return
    updateCommentAuthors(detail.oldAuthor, detail.newAuthor)
  }

  const handleCollapseToggle = () => {
    hideComposer()
    hideActionsPopover()
    if (activeModal) {
      const activeId = activeModal.dataset.commentId
      document.body.removeChild(activeModal)
      activeModal = null
      if (activeId) {
        document
          .querySelector<HTMLElement>(`.comment-bubble[data-comment-id="${activeId}"]`)
          ?.classList.remove("modal-active")
      }
    }
    requestAnimationFrame(() => {
      renderAllComments()
      refreshActiveModal()
      openPendingCommentThread()
    })
  }

  const handleHashChange = () => {
    setPendingCommentFromHash()
  }

  window.addCleanup(() => {
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

    if (ws) {
      ws.close()
      ws = null
    }
    document.removeEventListener("mouseup", mouseUp)
    window.removeEventListener("resize", handleResize)
    document.removeEventListener("collapsibletoggle", handleCollapseToggle)
    document.removeEventListener("commentauthorupdated", handleAuthorUpdate)
    window.removeEventListener("hashchange", handleHashChange)
  })

  document.addEventListener("collapsibletoggle", handleCollapseToggle)
  document.addEventListener("commentauthorupdated", handleAuthorUpdate)
  window.addEventListener("hashchange", handleHashChange)
})
