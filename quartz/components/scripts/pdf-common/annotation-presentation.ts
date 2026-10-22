import type { PdfComment, PdfOperationInput, PdfOperationType, PdfState } from './viewer-model'
import {
  filterPdfComments,
  getPdfAuthor,
  isParsedPdfComment,
  isRecord,
  mergePendingPdfComments,
  parsePdfComment,
  parsePdfOperationRecord,
  pdfCommentReplies,
  readCommentRoomEnabled,
  upsertPdfComment,
  visiblePdfTopLevelComments,
} from './annotation-records'
import {
  calculateFitScale,
  invalidatePageMetrics,
  setSlotSize,
  updateAutoPdfEmbedHeight,
  updateControls,
} from './page-layout'
import { enqueuePreloadPages, enqueueVisiblePages } from './page-rendering'
import { createPdfCommentComposer } from './viewer-chrome'

export function closePdfAnnotations(state: PdfState) {
  const annotations = state.annotations
  annotations.reconnectEnabled = false
  if (annotations.reconnectTimer) {
    window.clearTimeout(annotations.reconnectTimer)
    annotations.reconnectTimer = 0
  }
  const websocket = annotations.websocket
  annotations.websocket = null
  if (websocket) {
    websocket.onopen = null
    websocket.onmessage = null
    websocket.onclose = null
    websocket.onerror = null
    if (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING) {
      websocket.close(1000, 'disabled')
    }
  }
}

function flushPdfPendingOps(state: PdfState) {
  const websocket = state.annotations.websocket
  if (!websocket || websocket.readyState !== WebSocket.OPEN) return
  for (const op of state.annotations.pendingOps.values()) {
    websocket.send(JSON.stringify({ type: 'op', op }))
  }
}

export function connectPdfAnnotations(state: PdfState) {
  if (!readCommentRoomEnabled() || state.destroyed) return
  const annotations = state.annotations
  annotations.reconnectEnabled = true
  if (annotations.reconnectTimer) {
    window.clearTimeout(annotations.reconnectTimer)
    annotations.reconnectTimer = 0
  }
  if (
    annotations.websocket &&
    (annotations.websocket.readyState === WebSocket.OPEN ||
      annotations.websocket.readyState === WebSocket.CONNECTING)
  ) {
    return
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const pageId = encodeURIComponent(annotations.pageId)
  const since =
    annotations.hasSnapshot && annotations.lastSeq > 0 ? `&since=${annotations.lastSeq}` : ''
  const websocket = new WebSocket(
    `${protocol}//${window.location.host}/comments/websocket?pageId=${pageId}${since}`,
  )
  annotations.websocket = websocket

  websocket.onopen = () => {
    flushPdfPendingOps(state)
  }

  websocket.onmessage = event => {
    let message: unknown
    try {
      message = JSON.parse(event.data)
    } catch {
      return
    }
    if (!isRecord(message)) return

    const type = message['type']
    if (type === 'init') {
      const rawComments = message['comments']
      const latestSeq = message['latestSeq']
      if (!Array.isArray(rawComments)) return
      if (typeof latestSeq !== 'number' || !Number.isFinite(latestSeq)) return
      const comments = filterPdfComments(
        state,
        rawComments.map(parsePdfComment).filter(isParsedPdfComment),
      )
      annotations.comments = mergePendingPdfComments(state, comments)
      annotations.lastSeq = latestSeq
      annotations.hasSnapshot = true
      renderPdfAnnotations(state)
      return
    }

    if (type === 'delta') {
      const rawOps = message['ops']
      const latestSeq = message['latestSeq']
      if (!Array.isArray(rawOps)) return
      if (typeof latestSeq !== 'number' || !Number.isFinite(latestSeq)) return
      for (const rawOp of rawOps) {
        const op = parsePdfOperationRecord(rawOp)
        if (!op || op.comment.pageId !== annotations.pageId) continue
        annotations.comments = upsertPdfComment(annotations.comments, op.comment)
        annotations.pendingOps.delete(op.opId)
        annotations.lastSeq = Math.max(annotations.lastSeq, op.seq)
      }
      annotations.lastSeq = Math.max(annotations.lastSeq, latestSeq)
      annotations.hasSnapshot = true
      flushPdfPendingOps(state)
      renderPdfAnnotations(state)
      return
    }

    if (type === 'op') {
      const op = parsePdfOperationRecord(message['op'])
      if (!op || op.comment.pageId !== annotations.pageId) return
      annotations.comments = upsertPdfComment(annotations.comments, op.comment)
      annotations.pendingOps.delete(op.opId)
      annotations.lastSeq = Math.max(annotations.lastSeq, op.seq)
      renderPdfAnnotations(state)
      return
    }

    if (type === 'ack') {
      const opId = message['opId']
      const seq = message['seq']
      if (typeof opId !== 'string') return
      if (typeof seq !== 'number' || !Number.isFinite(seq)) return
      annotations.pendingOps.delete(opId)
      annotations.lastSeq = Math.max(annotations.lastSeq, seq)
    }
  }

  websocket.onclose = () => {
    if (annotations.websocket === websocket) annotations.websocket = null
    if (!annotations.reconnectEnabled || state.destroyed || !readCommentRoomEnabled()) return
    annotations.reconnectTimer = window.setTimeout(() => connectPdfAnnotations(state), 3000)
  }

  websocket.onerror = error => {
    console.error('pdf comments websocket error:', error)
  }
}

export function submitPdfOperation(state: PdfState, type: PdfOperationType, comment: PdfComment) {
  const op: PdfOperationInput = { opId: crypto.randomUUID(), type, comment }
  state.annotations.pendingOps.set(op.opId, op)
  state.annotations.comments = upsertPdfComment(state.annotations.comments, comment)
  renderPdfAnnotations(state)
  if (!readCommentRoomEnabled()) return
  connectPdfAnnotations(state)
  flushPdfPendingOps(state)
}

function formatPdfCommentTime(timestamp: number): string {
  const minutes = Math.floor((Date.now() - timestamp) / 60000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  return `${days}d`
}

function scrollToPdfHighlight(state: PdfState, comment: PdfComment) {
  const rect = comment.anchor?.rects[0]
  if (!rect) return
  const slot = state.slots.get(rect.page)
  if (!slot) return
  const top = slot.element.offsetTop + rect.top * slot.element.offsetHeight
  state.scroller.scrollTo({
    top: Math.max(0, top - state.scroller.clientHeight / 3),
    left: state.scroller.scrollLeft,
  })
}

function activatePdfComment(state: PdfState, commentId: string, scrollNote: boolean) {
  state.annotations.activeCommentId = commentId
  renderPdfAnnotations(state)
  if (scrollNote) positionPdfAnnotationCards(state)
}

function buildPdfNoteButton(label: string): HTMLButtonElement {
  const button = document.createElement('button')
  button.type = 'button'
  button.className = 'pdf-annotation-action'
  button.textContent = label
  return button
}

function renderPdfReply(reply: PdfComment): HTMLDivElement {
  const item = document.createElement('div')
  item.className = 'pdf-annotation-reply'

  const meta = document.createElement('div')
  meta.className = 'pdf-annotation-meta'
  meta.textContent = `${reply.author} ${formatPdfCommentTime(reply.createdAt)}`

  const content = document.createElement('div')
  content.className = 'pdf-annotation-content'
  content.textContent = reply.content || reply.anchorText

  item.append(meta, content)
  return item
}

function renderPdfReplyComposer(state: PdfState, parent: PdfComment): HTMLDivElement {
  const composer = createPdfCommentComposer({
    signal: state.events.signal,
    className: 'pdf-annotation-reply-composer',
    placeholder: 'Reply',
    submitLabel: 'Reply',
    onCancel() {
      state.annotations.activeCommentId = null
      renderPdfAnnotations(state)
    },
    onSubmit(content) {
      const reply: PdfComment = {
        ...parent,
        id: crypto.randomUUID(),
        parentId: parent.id,
        content,
        author: getPdfAuthor(),
        createdAt: Date.now(),
        updatedAt: null,
        deletedAt: null,
        resolvedAt: null,
      }
      submitPdfOperation(state, 'new', reply)
    },
  })
  window.requestAnimationFrame(() => composer.focus())
  return composer.element
}

function renderPdfNoteCard(state: PdfState, comment: PdfComment): HTMLDivElement {
  const card = document.createElement('div')
  card.className = 'pdf-annotation-card'
  card.dataset.commentId = comment.id
  if (state.annotations.activeCommentId === comment.id) card.classList.add('is-active')

  const meta = document.createElement('div')
  meta.className = 'pdf-annotation-meta'
  meta.textContent = `${comment.author} ${formatPdfCommentTime(comment.createdAt)}`

  const quote = document.createElement('div')
  quote.className = 'pdf-annotation-quote'
  quote.textContent = comment.anchorText

  const content = document.createElement('div')
  content.className = 'pdf-annotation-content'
  content.textContent = comment.content || 'highlight'

  const actions = document.createElement('div')
  actions.className = 'pdf-annotation-actions'

  const reply = buildPdfNoteButton('Reply')
  reply.addEventListener(
    'click',
    event => {
      event.stopPropagation()
      state.annotations.activeCommentId = comment.id
      renderPdfAnnotations(state)
    },
    { signal: state.events.signal },
  )
  actions.appendChild(reply)

  const author = getPdfAuthor()
  const login = localStorage.getItem('comment-author-github-login')?.toLowerCase()
  if (comment.author === author || login === 'aarnphm') {
    const resolve = buildPdfNoteButton('Resolve')
    resolve.addEventListener(
      'click',
      event => {
        event.stopPropagation()
        submitPdfOperation(state, 'resolve', { ...comment, resolvedAt: Date.now() })
      },
      { signal: state.events.signal },
    )
    const remove = buildPdfNoteButton('Delete')
    remove.addEventListener(
      'click',
      event => {
        event.stopPropagation()
        submitPdfOperation(state, 'delete', { ...comment, deletedAt: Date.now() })
      },
      { signal: state.events.signal },
    )
    actions.append(resolve, remove)
  }

  card.append(meta, quote, content, actions)
  for (const replyComment of pdfCommentReplies(state, comment.id)) {
    card.appendChild(renderPdfReply(replyComment))
  }
  if (state.annotations.activeCommentId === comment.id) {
    card.appendChild(renderPdfReplyComposer(state, comment))
  }

  card.addEventListener(
    'click',
    () => {
      state.annotations.activeCommentId = comment.id
      scrollToPdfHighlight(state, comment)
      renderPdfAnnotations(state)
    },
    { signal: state.events.signal },
  )

  return card
}

function positionPdfAnnotationCards(state: PdfState) {
  const toolbarHeight = state.toolbar.offsetHeight
  const railHeight = state.scroller.clientHeight
  state.root.style.setProperty('--pdf-annotation-rail-top', `${toolbarHeight}px`)
  state.root.style.setProperty('--pdf-annotation-rail-height', `${railHeight}px`)

  let stackHeight = Math.max(
    railHeight,
    state.scroller.scrollHeight,
    state.viewer.scrollHeight,
    state.annotationStack.scrollHeight,
  )
  for (const card of state.annotationStack.querySelectorAll<HTMLElement>('.pdf-annotation-card')) {
    const anchorTop = Number.parseFloat(card.dataset.anchorTop ?? '')
    if (!Number.isFinite(anchorTop)) continue
    card.style.top = `${anchorTop}px`
    stackHeight = Math.max(stackHeight, anchorTop + card.offsetHeight + 8)
  }
  state.annotationStack.style.height = `${stackHeight}px`
  if (Math.abs(state.annotationRail.scrollTop - state.scroller.scrollTop) > 1) {
    state.annotationRail.scrollTop = state.scroller.scrollTop
  }
}

export function syncPdfEmbedViewport(state: PdfState): boolean {
  const heightChanged = updateAutoPdfEmbedHeight(
    state.root,
    state.toolbar,
    state.scroller,
    state.firstViewport,
    state.options,
  )
  const nextWidth = state.scroller.clientWidth
  if (nextWidth === state.layoutWidth && !heightChanged) {
    positionPdfAnnotationCards(state)
    return false
  }
  state.layoutWidth = nextWidth
  invalidatePageMetrics(state)
  if (state.fit === 'custom') {
    positionPdfAnnotationCards(state)
    return false
  }
  rerenderVisiblePages(state)
  return true
}

export function renderPdfAnnotations(state: PdfState) {
  for (const slot of state.slots.values()) {
    slot.highlightLayer.replaceChildren()
  }
  state.annotationStack.replaceChildren()
  state.annotationStack.style.height = ''

  if (!state.options.annotations) {
    state.root.removeAttribute('data-pdf-annotations')
    syncPdfEmbedViewport(state)
    return
  }

  if (!readCommentRoomEnabled()) {
    state.root.removeAttribute('data-pdf-annotations')
    syncPdfEmbedViewport(state)
    return
  }

  const comments = visiblePdfTopLevelComments(state).sort((left, right) => {
    const leftRect = left.anchor?.rects[0]
    const rightRect = right.anchor?.rects[0]
    if (!leftRect || !rightRect) return 0
    const leftSlot = state.slots.get(leftRect.page)
    const rightSlot = state.slots.get(rightRect.page)
    const leftTop =
      (leftSlot?.element.offsetTop ?? 0) + leftRect.top * (leftSlot?.element.offsetHeight ?? 0)
    const rightTop =
      (rightSlot?.element.offsetTop ?? 0) + rightRect.top * (rightSlot?.element.offsetHeight ?? 0)
    return leftTop - rightTop
  })
  if (comments.length === 0) {
    state.root.removeAttribute('data-pdf-annotations')
    syncPdfEmbedViewport(state)
    return
  }

  state.root.dataset.pdfAnnotations = 'true'
  if (syncPdfEmbedViewport(state)) return

  let nextCardTop = 0
  for (const comment of comments) {
    const rects = comment.anchor?.rects ?? []
    const firstRect = rects[0]
    if (!firstRect) continue

    for (const rect of rects) {
      const slot = state.slots.get(rect.page)
      if (!slot) continue
      const span = document.createElement('span')
      span.className = 'pdf-highlight'
      span.dataset.commentId = comment.id
      span.style.left = `${rect.left * 100}%`
      span.style.top = `${rect.top * 100}%`
      span.style.width = `${rect.width * 100}%`
      span.style.height = `${rect.height * 100}%`
      span.addEventListener(
        'click',
        event => {
          event.stopPropagation()
          activatePdfComment(state, comment.id, true)
        },
        { signal: state.events.signal },
      )
      slot.highlightLayer.appendChild(span)
    }

    const card = renderPdfNoteCard(state, comment)
    const slot = state.slots.get(firstRect.page)
    let cardTop = nextCardTop
    if (slot) {
      cardTop = Math.max(
        nextCardTop,
        slot.element.offsetTop + firstRect.top * slot.element.offsetHeight,
      )
    }
    card.dataset.anchorTop = String(Math.max(0, cardTop))
    state.annotationStack.appendChild(card)
    nextCardTop = cardTop + card.offsetHeight + 8
  }
  positionPdfAnnotationCards(state)
}

export function rerenderVisiblePages(state: PdfState) {
  state.renderEpoch += 1
  state.scale = calculateFitScale(
    state.scroller,
    state.firstViewport,
    state.fit,
    state.options.scale,
  )
  for (const slot of state.slots.values()) {
    slot.renderTask?.cancel()
    slot.textRenderTask?.cancel()
    slot.renderTask = undefined
    slot.textRenderTask = undefined
    slot.renderedScale = undefined
    slot.renderingScale = undefined
    slot.textRenderedScale = undefined
    slot.textRenderingScale = undefined
    slot.canvas.width = 0
    slot.canvas.height = 0
    slot.canvas.removeAttribute('style')
    slot.textLayer.replaceChildren()
    setSlotSize(slot, state.firstViewport, state.scale)
  }
  state.renderQueue.clear()
  state.textRenderQueue.clear()
  state.preloadQueue.clear()
  invalidatePageMetrics(state)
  updateControls(state)
  renderPdfAnnotations(state)
  enqueueVisiblePages(state)
  if (state.options.preload) enqueuePreloadPages(state)
}
