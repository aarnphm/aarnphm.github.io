import { invalidatePageMetrics, setSlotSize, updateControls } from './page-layout'
import { readNumber, VIEWER_PADDING_PX, type PdfPageSlot, type PdfState } from './viewer-model'

const MAX_RENDER_WINDOW_RADIUS = 4
const PDF_PRELOAD_PAGE_LIMIT = 12

const TEXT_RENDER_IDLE_TIMEOUT_MS = 1000
const TEXT_RENDER_SCROLL_DELAY_MS = 600
const PDF_PRELOAD_SCROLL_DELAY_MS = 700
const PDF_PRUNE_SCROLL_DELAY_MS = 900

function isRenderCancelError(error: unknown): boolean {
  const message = error instanceof Error ? `${error.name} ${error.message}` : String(error)
  return message.includes('RenderingCancelledException') || message.includes('AbortException')
}

export function enqueuePage(state: PdfState, pageNumber: number) {
  if (pageNumber < 1 || pageNumber > state.document.numPages) return
  state.renderQueue.add(pageNumber)
  drainQueue(state)
}

function pdfRenderWindowRadius(state: PdfState): number {
  return Math.max(0, Math.min(state.options.overscan, MAX_RENDER_WINDOW_RADIUS))
}

function shouldPreloadPdfPages(state: PdfState): boolean {
  return state.options.preload && state.document.numPages <= PDF_PRELOAD_PAGE_LIMIT
}

function shouldRenderTextLayer(state: PdfState, pageNumber: number): boolean {
  if (!state.options.textLayer) return false
  if (state.visiblePages.has(pageNumber)) return true
  return Math.abs(pageNumber - state.currentPage) <= pdfRenderWindowRadius(state)
}

function shouldKeepRenderedPage(state: PdfState, pageNumber: number, centerPage: number): boolean {
  if (state.visiblePages.has(pageNumber)) return true
  return Math.abs(pageNumber - centerPage) <= pdfRenderWindowRadius(state)
}

function clearTextLayer(slot: PdfPageSlot) {
  slot.textRenderTask?.cancel()
  slot.textRenderTask = undefined
  slot.textRenderedScale = undefined
  slot.textRenderingScale = undefined
  delete slot.textLayer.dataset.pdfTextReady
  slot.textLayer.replaceChildren()
}

function clearRenderedPage(slot: PdfPageSlot) {
  slot.renderTask?.cancel()
  slot.renderTask = undefined
  slot.renderedScale = undefined
  slot.renderingScale = undefined
  slot.canvas.width = 0
  slot.canvas.height = 0
  slot.canvas.removeAttribute('style')
  clearTextLayer(slot)
}

function pruneDistantRenderedPagesNow(state: PdfState, centerPage: number) {
  if (shouldPreloadPdfPages(state)) return
  cancelDistantRenderWork(state, centerPage)
  for (const slot of state.slots.values()) {
    if (shouldKeepRenderedPage(state, slot.pageNumber, centerPage)) continue
    if (slot.renderedScale !== undefined || slot.textRenderedScale !== undefined) {
      clearRenderedPage(slot)
    }
  }
}

export function cancelDistantRenderWork(state: PdfState, centerPage: number) {
  if (shouldPreloadPdfPages(state)) return
  for (const slot of state.slots.values()) {
    if (shouldKeepRenderedPage(state, slot.pageNumber, centerPage)) continue
    state.renderQueue.delete(slot.pageNumber)
    state.textRenderQueue.delete(slot.pageNumber)
    if (slot.renderingScale !== undefined) {
      slot.renderTask?.cancel()
      slot.renderTask = undefined
      slot.renderingScale = undefined
    }
    if (slot.textRenderingScale !== undefined) {
      slot.textRenderTask?.cancel()
      slot.textRenderTask = undefined
      slot.textRenderingScale = undefined
    }
  }
}

export function pruneDistantRenderedPages(state: PdfState, centerPage: number) {
  if (shouldPreloadPdfPages(state)) return
  if (state.pruneTimer) window.clearTimeout(state.pruneTimer)
  state.pruneTimer = window.setTimeout(() => {
    state.pruneTimer = 0
    pruneDistantRenderedPagesNow(state, centerPage)
  }, PDF_PRUNE_SCROLL_DELAY_MS)
}

export function enqueueRenderWindow(state: PdfState, centerPage: number = state.currentPage) {
  const radius = pdfRenderWindowRadius(state)
  for (let offset = 0; offset <= radius; offset++) {
    const left = centerPage - offset
    const right = centerPage + offset
    if (left >= 1) enqueuePage(state, left)
    if (offset > 0 && right <= state.document.numPages) enqueuePage(state, right)
  }
}

export function enqueueVisiblePages(state: PdfState) {
  for (const pageNumber of state.visiblePages) {
    enqueuePage(state, pageNumber)
  }
  enqueueRenderWindow(state)
}

export function enqueuePreloadPages(state: PdfState) {
  if (!shouldPreloadPdfPages(state)) return
  for (let pageNumber = 1; pageNumber <= state.document.numPages; pageNumber++) {
    const slot = state.slots.get(pageNumber)
    if (!slot) continue
    if (slot.renderedScale === state.scale || slot.renderingScale === state.scale) continue
    state.preloadQueue.add(pageNumber)
  }
  drainPreloadQueue(state)
}

export function scrollToPage(state: PdfState, pageNumber: number) {
  const slot = state.slots.get(pageNumber)
  if (!slot) return
  state.currentPage = pageNumber
  updateControls(state)
  cancelDistantRenderWork(state, pageNumber)
  pruneDistantRenderedPages(state, pageNumber)
  const scrollerRect = state.scroller.getBoundingClientRect()
  const slotRect = slot.element.getBoundingClientRect()
  const scrollPaddingTop = readNumber(
    getComputedStyle(state.scroller).scrollPaddingTop,
    VIEWER_PADDING_PX,
  )
  state.scroller.scrollTo({
    top: Math.max(0, state.scroller.scrollTop + slotRect.top - scrollerRect.top - scrollPaddingTop),
    left: state.scroller.scrollLeft,
  })
  enqueueRenderWindow(state, pageNumber)
}

async function renderPage(state: PdfState, slot: PdfPageSlot, epoch: number) {
  if (state.destroyed) return
  if (slot.renderedScale === state.scale) {
    enqueueTextLayer(state, slot.pageNumber)
    return
  }
  if (slot.renderingScale === state.scale) return

  slot.renderTask?.cancel()
  slot.renderingScale = state.scale
  const page = slot.page ?? (await state.document.getPage(slot.pageNumber))
  slot.page = page
  if (state.destroyed || epoch !== state.renderEpoch) return

  const viewport = page.getViewport({ scale: state.scale })
  const ratio = Math.min(state.options.maxPixelRatio, Math.max(1, window.devicePixelRatio || 1))
  const canvas = slot.canvas
  canvas.width = Math.floor(viewport.width * ratio)
  canvas.height = Math.floor(viewport.height * ratio)
  canvas.style.width = `${Math.ceil(viewport.width)}px`
  canvas.style.height = `${Math.ceil(viewport.height)}px`
  slot.textLayer.style.width = `${Math.ceil(viewport.width)}px`
  slot.textLayer.style.height = `${Math.ceil(viewport.height)}px`
  slot.highlightLayer.style.width = `${Math.ceil(viewport.width)}px`
  slot.highlightLayer.style.height = `${Math.ceil(viewport.height)}px`
  if (setSlotSize(slot, viewport, 1)) invalidatePageMetrics(state)

  const context = canvas.getContext('2d')
  if (!context) return
  const task = page.render({
    canvasContext: context,
    viewport,
    transform: [ratio, 0, 0, ratio, 0, 0],
  })
  slot.renderTask = task
  await task.promise
  if (state.destroyed || epoch !== state.renderEpoch) return
  slot.renderTask = undefined
  slot.renderedScale = state.scale
  slot.renderingScale = undefined
  enqueueTextLayer(state, slot.pageNumber)
}

async function renderTextLayer(state: PdfState, slot: PdfPageSlot, epoch: number) {
  if (state.destroyed) return
  if (slot.textRenderedScale === state.scale || slot.textRenderingScale === state.scale) return
  slot.textRenderTask?.cancel()
  slot.textRenderTask = undefined
  delete slot.textLayer.dataset.pdfTextReady
  slot.textLayer.replaceChildren()
  slot.textRenderingScale = state.scale
  const page = slot.page ?? (await state.document.getPage(slot.pageNumber))
  slot.page = page
  if (state.destroyed || epoch !== state.renderEpoch) return
  const viewport = page.getViewport({ scale: state.scale })
  const textContent = await page.getTextContent()
  if (state.destroyed || epoch !== state.renderEpoch) return
  const textLayer = state.pdfjs.createTextLayer({
    textContentSource: textContent,
    container: slot.textLayer,
    viewport,
  })
  slot.textRenderTask = textLayer
  // pdf.js resets its measuring canvas font between text items, and that canvas sits in
  // <body>, so every font change forces a document style update over whatever is dirty (see
  // pdf-reader/viewer.ts). Render into the detached layer, then reattach it once.
  const parent = slot.textLayer.parentNode
  const anchor = slot.textLayer.nextSibling
  slot.textLayer.remove()
  try {
    await textLayer.render()
  } finally {
    parent?.insertBefore(slot.textLayer, anchor)
  }
  if (state.destroyed || epoch !== state.renderEpoch) return
  slot.textRenderTask = undefined
  slot.textRenderedScale = state.scale
  slot.textRenderingScale = undefined
  slot.textLayer.dataset.pdfTextReady = 'true'
  state.root.dispatchEvent(new CustomEvent('pdf-text-rendered', { detail: slot.pageNumber }))
}

function enqueueTextLayer(state: PdfState, pageNumber: number) {
  const slot = state.slots.get(pageNumber)
  if (!slot) return
  if (!shouldRenderTextLayer(state, pageNumber)) return
  if (slot.textRenderedScale === state.scale || slot.textRenderingScale === state.scale) return
  state.textRenderQueue.add(pageNumber)
  drainTextLayerQueue(state)
}

function schedulePdfIdleWork(callback: () => void) {
  const requestIdle = Reflect.get(window, 'requestIdleCallback')
  if (typeof requestIdle === 'function') {
    Reflect.apply(requestIdle, window, [callback, { timeout: TEXT_RENDER_IDLE_TIMEOUT_MS }])
    return
  }
  window.setTimeout(callback, 32)
}

function takeNextQueuedPage(state: PdfState, queue: Set<number>): number | undefined {
  let bestPage: number | undefined
  let bestDistance = Number.POSITIVE_INFINITY
  for (const pageNumber of queue) {
    const distance = Math.abs(pageNumber - state.currentPage)
    if (distance < bestDistance) {
      bestDistance = distance
      bestPage = pageNumber
    }
  }
  if (bestPage !== undefined) queue.delete(bestPage)
  return bestPage
}

function drainQueue(state: PdfState) {
  if (state.draining || state.destroyed) return
  state.draining = true

  const run = async () => {
    while (!state.destroyed && state.renderQueue.size > 0) {
      const pageNumber = takeNextQueuedPage(state, state.renderQueue)
      if (pageNumber === undefined) break
      const slot = state.slots.get(pageNumber)
      if (slot) {
        try {
          await renderPage(state, slot, state.renderEpoch)
        } catch (error) {
          slot.renderTask = undefined
          slot.renderingScale = undefined
          if (!state.destroyed && !isRenderCancelError(error)) console.error(error)
        }
      }
    }
    state.draining = false
    if (state.preloadQueue.size > 0) drainPreloadQueue(state)
  }

  void run()
}

function drainPreloadQueue(state: PdfState) {
  if (state.preloadDraining || state.destroyed || !shouldPreloadPdfPages(state)) return
  state.preloadDraining = true

  const run = () => {
    void (async () => {
      if (state.destroyed) {
        state.preloadDraining = false
        return
      }
      if (
        state.renderQueue.size > 0 ||
        Date.now() - state.lastScrollAt < PDF_PRELOAD_SCROLL_DELAY_MS
      ) {
        state.preloadDraining = false
        window.setTimeout(() => drainPreloadQueue(state), PDF_PRELOAD_SCROLL_DELAY_MS)
        return
      }

      const pageNumber = takeNextQueuedPage(state, state.preloadQueue)
      const slot = pageNumber === undefined ? undefined : state.slots.get(pageNumber)
      if (slot) {
        try {
          await renderPage(state, slot, state.renderEpoch)
        } catch (error) {
          slot.renderTask = undefined
          slot.renderingScale = undefined
          if (!state.destroyed && !isRenderCancelError(error)) console.error(error)
        }
      }
      state.preloadDraining = false
      if (state.preloadQueue.size > 0) drainPreloadQueue(state)
    })()
  }

  schedulePdfIdleWork(run)
}

function drainTextLayerQueue(state: PdfState) {
  if (state.textDraining || state.destroyed) return
  state.textDraining = true

  const run = () => {
    void (async () => {
      if (state.destroyed) {
        state.textDraining = false
        return
      }
      if (Date.now() - state.lastScrollAt < TEXT_RENDER_SCROLL_DELAY_MS) {
        state.textDraining = false
        window.setTimeout(() => drainTextLayerQueue(state), TEXT_RENDER_SCROLL_DELAY_MS)
        return
      }
      const pageNumber = takeNextQueuedPage(state, state.textRenderQueue)
      const slot = pageNumber === undefined ? undefined : state.slots.get(pageNumber)
      if (slot && shouldRenderTextLayer(state, slot.pageNumber)) {
        try {
          await renderTextLayer(state, slot, state.renderEpoch)
        } catch (error) {
          slot.textRenderTask = undefined
          slot.textRenderingScale = undefined
          if (!state.destroyed && !isRenderCancelError(error)) console.error(error)
        }
      }
      state.textDraining = false
      if (state.textRenderQueue.size > 0) drainTextLayerQueue(state)
    })()
  }

  schedulePdfIdleWork(run)
}
