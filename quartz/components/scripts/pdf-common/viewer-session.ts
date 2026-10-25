import {
  closePdfAnnotations,
  connectPdfAnnotations,
  renderPdfAnnotations,
  rerenderVisiblePages,
  submitPdfOperation,
} from './annotation-presentation'
import { getPdfAuthor, hashPdfText, readCommentRoomEnabled } from './annotation-records'
import { currentPageFromScroll, updateControls } from './page-layout'
import {
  cancelDistantRenderWork,
  enqueueRenderWindow,
  pruneDistantRenderedPages,
  scrollToPage,
} from './page-rendering'
import { createPdfCommentComposer } from './viewer-chrome'
import {
  clampPdfValue,
  MAX_SCALE,
  MIN_SCALE,
  readFit,
  readInteger,
  type PdfAnchorRect,
  type PdfComment,
  type PdfSelection,
  type PdfState,
} from './viewer-model'

const PDF_COMMENT_ROOM_TOGGLE_EVENT = 'commentsroomtoggle'

export const pdfStates = new WeakMap<HTMLElement, PdfState>()
export const pendingPdfMounts = new WeakMap<HTMLElement, AbortController>()
export let selectedPdfState: PdfState | null = null
export let selectedPdfRoot: HTMLElement | null = null

export function selectPdfRoot(root: HTMLElement) {
  if (selectedPdfRoot && selectedPdfRoot !== root) {
    selectedPdfRoot.removeAttribute('data-pdf-selected')
  }
  selectedPdfRoot = root
  root.dataset.pdfSelected = 'true'
  selectedPdfState = pdfStates.get(root) ?? null
}

export function clearSelectedPdfRoot(root: HTMLElement) {
  if (selectedPdfRoot !== root) return
  selectedPdfRoot = null
  selectedPdfState = null
  root.removeAttribute('data-pdf-selected')
  if (document.activeElement === root) root.blur()
}

export function selectPdfState(state: PdfState) {
  selectPdfRoot(state.root)
  selectedPdfState = state
  state.lastScrollAt = Date.now()
}

function clearSelectedPdfState(state: PdfState) {
  if (selectedPdfState !== state && selectedPdfRoot !== state.root) return
  clearSelectedPdfRoot(state.root)
  state.scroller.blur()
}

function hidePdfComposer(state: PdfState) {
  state.annotations.composer?.remove()
  state.annotations.composer = null
}

function rectIntersection(left: DOMRect, right: DOMRect): DOMRect | null {
  const x1 = Math.max(left.left, right.left)
  const y1 = Math.max(left.top, right.top)
  const x2 = Math.min(left.right, right.right)
  const y2 = Math.min(left.bottom, right.bottom)
  if (x2 <= x1 || y2 <= y1) return null
  return new DOMRect(x1, y1, x2 - x1, y2 - y1)
}

function readPdfSelection(state: PdfState): PdfSelection | null {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null
  if (!selection.anchorNode || !selection.focusNode) return null
  if (!state.viewer.contains(selection.anchorNode) || !state.viewer.contains(selection.focusNode)) {
    return null
  }

  const text = selection.toString().trim()
  if (!text) return null

  const rects: PdfAnchorRect[] = []
  let anchorRect: DOMRect | null = null
  const range = selection.getRangeAt(0)
  for (const rawRect of range.getClientRects()) {
    if (rawRect.width <= 0 || rawRect.height <= 0) continue
    for (const slot of state.slots.values()) {
      const slotRect = slot.element.getBoundingClientRect()
      const intersection = rectIntersection(rawRect, slotRect)
      if (!intersection) continue
      const left = clampPdfValue((intersection.left - slotRect.left) / slotRect.width, 0, 1)
      const top = clampPdfValue((intersection.top - slotRect.top) / slotRect.height, 0, 1)
      const width = clampPdfValue(intersection.width / slotRect.width, 0, 1 - left)
      const height = clampPdfValue(intersection.height / slotRect.height, 0, 1 - top)
      if (width <= 0 || height <= 0) continue
      rects.push({ page: slot.pageNumber, left, top, width, height })
      anchorRect ??= intersection
    }
  }

  if (!anchorRect || rects.length === 0) return null
  return { text, rects, anchorRect }
}

function showPdfComposer(state: PdfState, selection: PdfSelection) {
  hidePdfComposer(state)

  const composer = createPdfCommentComposer({
    signal: state.events.signal,
    className: 'pdf-comment-composer',
    placeholder: 'Commentaire (compatible markdown)',
    submitLabel: 'Submit comment',
    onCancel() {
      hidePdfComposer(state)
    },
    onSubmit(content) {
      return submitPdfSelectionComment(state, selection, content)
    },
  })
  const element = composer.element
  state.annotations.composer = element
  state.scroller.appendChild(element)

  const scrollerRect = state.scroller.getBoundingClientRect()
  element.style.left = `${selection.anchorRect.left - scrollerRect.left + state.scroller.scrollLeft}px`
  element.style.top = `${selection.anchorRect.bottom - scrollerRect.top + state.scroller.scrollTop + 8}px`
  composer.focus()
}

async function submitPdfSelectionComment(
  state: PdfState,
  selection: PdfSelection,
  content: string,
) {
  const pageNumbers = selection.rects.map(rect => rect.page)
  const comment: PdfComment = {
    id: crypto.randomUUID(),
    pageId: state.annotations.pageId,
    parentId: null,
    anchorHash: await hashPdfText(selection.text),
    anchorStart: Math.min(...pageNumbers),
    anchorEnd: Math.max(...pageNumbers),
    anchorText: selection.text,
    content: content.trim(),
    author: getPdfAuthor(),
    createdAt: Date.now(),
    updatedAt: null,
    deletedAt: null,
    resolvedAt: null,
    anchor: { kind: 'pdf', src: state.annotations.src, rects: selection.rects },
    orphaned: null,
    lastRecoveredAt: null,
  }

  submitPdfOperation(state, 'new', comment)
  hidePdfComposer(state)
  window.getSelection()?.removeAllRanges()
  state.annotations.activeCommentId = comment.id
  renderPdfAnnotations(state)
}

export function targetAcceptsTextInput(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target instanceof HTMLInputElement) return true
  if (target instanceof HTMLTextAreaElement) return true
  if (target instanceof HTMLSelectElement) return true
  return target.isContentEditable
}

export function consumePdfNavigationKey(event: KeyboardEvent) {
  event.preventDefault()
  event.stopImmediatePropagation()
}

export function setupPdfEvents(state: PdfState) {
  let syncingAnnotationScroll = false
  const goPrev = () => scrollToPage(state, Math.max(1, state.currentPage - 1))
  const goNext = () => scrollToPage(state, Math.min(state.document.numPages, state.currentPage + 1))
  const setPage = () => scrollToPage(state, readInteger(state.controls.pageInput.value, 1))
  const zoomBy = (factor: number) => {
    state.fit = 'custom'
    state.options.scale = clampPdfValue(state.scale * factor, MIN_SCALE, MAX_SCALE)
    rerenderVisiblePages(state)
  }
  const setFit = () => {
    state.fit = readFit(state.controls.fit.value)
    state.options.scale = undefined
    rerenderVisiblePages(state)
  }
  const onScroll = () => {
    state.lastScrollAt = Date.now()
    if (state.scrollFrame) return
    state.scrollFrame = window.requestAnimationFrame(() => {
      state.scrollFrame = 0
      if (state.root.dataset.pdfAnnotations === 'true' && !syncingAnnotationScroll) {
        syncingAnnotationScroll = true
        state.annotationRail.scrollTop = state.scroller.scrollTop
        window.requestAnimationFrame(() => {
          syncingAnnotationScroll = false
        })
      }
      const page = currentPageFromScroll(state)
      if (page !== state.currentPage) {
        state.currentPage = page
        updateControls(state)
        cancelDistantRenderWork(state, page)
        pruneDistantRenderedPages(state, page)
        enqueueRenderWindow(state, page)
      }
    })
  }
  const onAnnotationRailScroll = () => {
    if (syncingAnnotationScroll) return
    syncingAnnotationScroll = true
    state.scroller.scrollTop = state.annotationRail.scrollTop
    window.requestAnimationFrame(() => {
      syncingAnnotationScroll = false
    })
  }
  const onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return
    selectPdfState(state)
    if (!targetAcceptsTextInput(event.target)) {
      state.scroller.focus({ preventScroll: true })
    }
  }
  const onKeyDown = (event: KeyboardEvent) => {
    if (selectedPdfState !== state) return
    if (targetAcceptsTextInput(event.target)) return
    if (event.metaKey || event.ctrlKey || event.altKey) return

    if (event.key === 'Escape') {
      consumePdfNavigationKey(event)
      clearSelectedPdfState(state)
      return
    }

    if (
      event.key === 'ArrowRight' ||
      event.key === 'ArrowDown' ||
      event.key === 'j' ||
      event.key === 'l' ||
      (event.key === ' ' && !event.shiftKey)
    ) {
      consumePdfNavigationKey(event)
      goNext()
      return
    }

    if (
      event.key === 'ArrowLeft' ||
      event.key === 'ArrowUp' ||
      event.key === 'k' ||
      event.key === 'h' ||
      (event.key === ' ' && event.shiftKey)
    ) {
      consumePdfNavigationKey(event)
      goPrev()
    }
  }
  const onMouseUp = (event: MouseEvent) => {
    if (!state.options.annotations) return
    if (event.button !== 0) return
    if (!readCommentRoomEnabled()) return
    if (!event.metaKey && !event.ctrlKey && !event.altKey) return
    const selection = readPdfSelection(state)
    if (!selection) return
    showPdfComposer(state, selection)
  }
  const onCommentRoomToggle = (event: CustomEventMap['commentsroomtoggle']) => {
    if (!state.options.annotations) return
    const enabled = event.detail.enabled ?? !readCommentRoomEnabled()
    if (enabled) {
      connectPdfAnnotations(state)
      renderPdfAnnotations(state)
      return
    }
    closePdfAnnotations(state)
    hidePdfComposer(state)
    renderPdfAnnotations(state)
  }

  state.controls.prev.addEventListener('click', goPrev, { signal: state.events.signal })
  state.controls.next.addEventListener('click', goNext, { signal: state.events.signal })
  state.controls.pageInput.addEventListener('change', setPage, { signal: state.events.signal })
  state.root
    .querySelector('.pdf-embed-zoom-out')
    ?.addEventListener('click', () => zoomBy(0.9), { signal: state.events.signal })
  state.root
    .querySelector('.pdf-embed-zoom-in')
    ?.addEventListener('click', () => zoomBy(1.1), { signal: state.events.signal })
  state.controls.fit.addEventListener('change', setFit, { signal: state.events.signal })
  state.scroller.addEventListener('scroll', onScroll, {
    passive: true,
    signal: state.events.signal,
  })
  state.annotationRail.addEventListener('scroll', onAnnotationRailScroll, {
    passive: true,
    signal: state.events.signal,
  })
  state.root.addEventListener('pointerdown', onPointerDown, { signal: state.events.signal })
  document.addEventListener('keydown', onKeyDown, { capture: true, signal: state.events.signal })
  state.scroller.addEventListener('mouseup', onMouseUp, { signal: state.events.signal })
  document.addEventListener(PDF_COMMENT_ROOM_TOGGLE_EVENT, onCommentRoomToggle)
  state.cleanup.push(() => {
    document.removeEventListener(PDF_COMMENT_ROOM_TOGGLE_EVENT, onCommentRoomToggle)
    if (state.scrollFrame) {
      window.cancelAnimationFrame(state.scrollFrame)
      state.scrollFrame = 0
    }
  })
}

export function cleanupPdfEmbed(root: HTMLElement) {
  pendingPdfMounts.get(root)?.abort()
  pendingPdfMounts.delete(root)
  root.removeAttribute('data-pdf-status')
  const state = pdfStates.get(root)
  if (!state) return
  state.destroyed = true
  state.events.abort()
  if (selectedPdfState === state || selectedPdfRoot === state.root) {
    selectedPdfState = null
    selectedPdfRoot = null
  }
  state.root.removeAttribute('data-pdf-selected')
  state.observer.disconnect()
  state.resizeObserver.disconnect()
  for (const slot of state.slots.values()) {
    slot.renderTask?.cancel()
    slot.textRenderTask?.cancel()
  }
  state.renderQueue.clear()
  state.textRenderQueue.clear()
  state.preloadQueue.clear()
  if (state.scrollFrame) window.cancelAnimationFrame(state.scrollFrame)
  if (state.resizeFrame) window.cancelAnimationFrame(state.resizeFrame)
  if (state.pruneTimer) window.clearTimeout(state.pruneTimer)
  closePdfAnnotations(state)
  hidePdfComposer(state)
  for (const cleanup of state.cleanup.splice(0)) {
    cleanup()
  }
  state.releaseDocument()
  pdfStates.delete(root)
}
