import { getFullSlug } from "../../util/path"

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

type BroadcastMessage = {
  type: "init" | "new" | "update" | "delete"
  comment?: MultiplayerComment
  comments?: MultiplayerComment[]
}

let ws: WebSocket | null = null
let comments: MultiplayerComment[] = []
let activeSelection: Range | null = null
let selectionIdleTimer: number | null = null

function getAuthor(): string {
  let author = localStorage.getItem("comment-author")
  if (!author) {
    author = `anon-${Math.random().toString(36).slice(2, 8)}`
    localStorage.setItem("comment-author", author)
  }
  return author
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

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
  const wsUrl = `${protocol}//${window.location.host}/comments/websocket?pageId=${encodeURIComponent(getFullSlug(window))}`

  ws = new WebSocket(wsUrl)

  ws.onmessage = (event) => {
    const msg: BroadcastMessage = JSON.parse(event.data)

    if (msg.type === "init" && msg.comments) {
      comments = msg.comments
      renderAllComments()
    } else if (msg.type === "new" && msg.comment) {
      comments.push(msg.comment)
      renderAllComments()
    } else if (msg.type === "update" && msg.comment) {
      const idx = comments.findIndex((c) => c.id === msg.comment!.id)
      if (idx !== -1) {
        comments[idx] = msg.comment
        renderAllComments()
      }
    } else if (msg.type === "delete" && msg.comment) {
      const idx = comments.findIndex((c) => c.id === msg.comment!.id)
      if (idx !== -1) {
        comments[idx] = msg.comment
        renderAllComments()
      }
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

function hideComposer() {
  if (activeComposer) {
    document.body.removeChild(activeComposer)
    activeComposer = null
  }
  activeSelection = null
}

function scheduleComposer() {
  if (selectionIdleTimer !== null) {
    window.clearTimeout(selectionIdleTimer)
  }
  selectionIdleTimer = window.setTimeout(() => {
    selectionIdleTimer = null
    handleTextSelection()
  }, 120)
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
  hideComposer()
  activeSelection = range.cloneRange()

  setTimeout(() => showComposer(activeSelection!), 10)
}

async function showComposer(range: Range) {
  hideComposer()
  const articleText = getArticleText()
  const selectedText = range.toString()
  const anchorHash = await hashText(selectedText)

  const startOffset = articleText.indexOf(selectedText)
  if (startOffset === -1) {
    return
  }

  const endOffset = startOffset + selectedText.length

  const composer = document.createElement("div")
  composer.className = "comment-composer"
  activeComposer = composer

  const rect = range.getBoundingClientRect()
  const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
  const scrollTop = window.pageYOffset || document.documentElement.scrollTop

  composer.style.left = `${rect.left + scrollLeft}px`
  composer.style.top = `${rect.bottom + scrollTop + 8}px`

  const inputWrapper = document.createElement("div")
  inputWrapper.className = "composer-input-wrapper"

  const input = document.createElement("div")
  input.className = "composer-input"
  input.contentEditable = "true"
  input.setAttribute("role", "textbox")
  input.setAttribute("aria-placeholder", "Add a comment")
  input.dataset.placeholder = "Add a comment"

  const submitButton = document.createElement("button")
  submitButton.className = "composer-submit"
  submitButton.disabled = true
  submitButton.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M12 16a.5.5 0 0 1-.5-.5V8.707l-3.146 3.147a.5.5 0 0 1-.708-.708l4-4a.5.5 0 0 1 .708 0l4 4a.5.5 0 0 1-.708.708L12.5 8.707V15.5a.5.5 0 0 1-.5.5" clip-rule="evenodd"></path></svg>`

  input.addEventListener("input", () => {
    const content = input.textContent?.trim() || ""
    submitButton.disabled = content.length === 0
    if (content.length === 0) {
      input.classList.add("empty")
    } else {
      input.classList.remove("empty")
    }
  })

  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      hideComposer()
    } else if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      submitButton.click()
    }
  })

  submitButton.onclick = async () => {
    const content = input.textContent?.trim()
    if (!content) return

    const comment: MultiplayerComment = {
      id: crypto.randomUUID(),
      pageId: getFullSlug(window),
      parentId: null,
      anchorHash,
      anchorStart: startOffset,
      anchorEnd: endOffset,
      anchorText: selectedText,
      content,
      author: getAuthor(),
      createdAt: Date.now(),
      updatedAt: null,
      deletedAt: null,
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "new", comment }))
    }

    hideComposer()
  }

  inputWrapper.appendChild(input)
  inputWrapper.appendChild(submitButton)
  composer.appendChild(inputWrapper)
  document.body.appendChild(composer)

  input.focus()
}

function renderAllComments() {
  document.querySelectorAll(".comment-highlight").forEach((el) => el.replaceWith(...el.childNodes))
  document.querySelectorAll(".comment-bubble").forEach((el) => el.remove())

  const articleText = getArticleText()
  const article = document.querySelector("article.popover-hint")
  if (!article) return

  const topLevelComments = comments.filter((c) => !c.parentId && !c.deletedAt)

  for (const comment of topLevelComments) {
    const startIdx = comment.anchorStart
    const endIdx = comment.anchorEnd

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

        const span = document.createElement("span")
        span.className = "comment-highlight"
        span.dataset.commentId = comment.id
        range.surroundContents(span)

        const rect = span.getBoundingClientRect()
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop

        const bubble = document.createElement("div")
        bubble.className = "comment-bubble"
        bubble.dataset.commentId = comment.id
        bubble.style.top = `${rect.top + scrollTop}px`

        const avatar = document.createElement("div")
        avatar.className = "bubble-avatar"
        avatar.textContent = comment.author.charAt(0).toUpperCase()
        bubble.appendChild(avatar)

        bubble.onclick = () => {
          showCommentThread(comment.id)
        }

        document.body.appendChild(bubble)
      } catch (err) {
        console.warn("failed to highlight comment", err)
      }
    }
  }
}

function showCommentThread(commentId: string) {
  const comment = comments.find((c) => c.id === commentId)
  if (!comment) return

  const replies = comments.filter((c) => c.parentId === commentId && !c.deletedAt)

  const modal = document.createElement("div")
  modal.className = "comment-thread-modal"

  const header = document.createElement("div")
  header.className = "modal-header"

  const title = document.createElement("div")
  title.className = "modal-title"
  title.textContent = "Comment Thread"

  const closeButton = document.createElement("button")
  closeButton.textContent = "×"
  closeButton.onclick = () => {
    document.body.removeChild(modal)
  }

  header.appendChild(title)
  header.appendChild(closeButton)

  const content = document.createElement("div")
  content.className = "modal-content"

  const commentEl = document.createElement("div")
  commentEl.className = "comment-item"

  const commentText = document.createElement("div")
  commentText.className = "comment-text"
  commentText.textContent = comment.content

  const commentMeta = document.createElement("div")
  commentMeta.className = "comment-meta"
  commentMeta.textContent = `${comment.author} • ${new Date(comment.createdAt).toLocaleString()}`

  commentEl.appendChild(commentText)
  commentEl.appendChild(commentMeta)
  content.appendChild(commentEl)

  for (const reply of replies) {
    const replyEl = document.createElement("div")
    replyEl.className = "reply-item"

    const replyText = document.createElement("div")
    replyText.className = "reply-text"
    replyText.textContent = reply.content

    const replyMeta = document.createElement("div")
    replyMeta.className = "reply-meta"
    replyMeta.textContent = `${reply.author} • ${new Date(reply.createdAt).toLocaleString()}`

    replyEl.appendChild(replyText)
    replyEl.appendChild(replyMeta)
    content.appendChild(replyEl)
  }

  const replyForm = document.createElement("div")
  replyForm.className = "reply-form"

  const replyTextarea = document.createElement("textarea")
  replyTextarea.placeholder = "Add a reply..."

  const replyButton = document.createElement("button")
  replyButton.textContent = "Reply"
  replyButton.onclick = async () => {
    const content = replyTextarea.value.trim()
    if (!content) return

    const reply: MultiplayerComment = {
      id: crypto.randomUUID(),
      pageId: getFullSlug(window),
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

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "new", comment: reply }))
    }

    replyTextarea.value = ""
  }

  replyForm.appendChild(replyTextarea)
  replyForm.appendChild(replyButton)

  modal.appendChild(header)
  modal.appendChild(content)
  modal.appendChild(replyForm)

  document.body.appendChild(modal)
}

document.addEventListener("nav", () => {
  connectWebSocket()

  const mouseUp = (event: MouseEvent) => {
    if (event.button !== 0) return
    if (activeComposer && event.target && activeComposer.contains(event.target as Node)) {
      return
    }
    scheduleComposer()
  }

  document.addEventListener("mouseup", mouseUp)
  const selectionChange = () => {
    if (activeComposer) return
    scheduleComposer()
  }
  document.addEventListener("selectionchange", selectionChange)
  window.addCleanup(() => {
    hideComposer()
    if (selectionIdleTimer !== null) {
      window.clearTimeout(selectionIdleTimer)
      selectionIdleTimer = null
    }
    document
      .querySelectorAll(".comment-highlight")
      .forEach((el) => el.replaceWith(...el.childNodes))
    document.querySelectorAll(".comment-bubble").forEach((el) => el.remove())
    document.querySelectorAll(".comment-thread-modal").forEach((el) => el.remove())

    if (ws) {
      ws.close()
      ws = null
    }
    document.removeEventListener("mouseup", mouseUp)
    document.removeEventListener("selectionchange", selectionChange)
  })
})
