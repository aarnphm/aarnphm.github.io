import { setPageCount } from './viewer-chrome'
import {
  clampPdfValue,
  MAX_SCALE,
  MIN_SCALE,
  VIEWER_PADDING_PX,
  type PdfFitMode,
  type PdfOptions,
  type PdfPageMetric,
  type PdfPageSlot,
  type PdfState,
  type PdfViewport,
} from './viewer-model'

export function calculateFitScale(
  scroller: HTMLElement,
  viewport: PdfViewport,
  fit: PdfFitMode,
  customScale?: number,
): number {
  if (fit === 'custom' && customScale) return clampPdfValue(customScale, MIN_SCALE, MAX_SCALE)
  if (fit === 'actual') return 1

  const widthScale = (scroller.clientWidth - VIEWER_PADDING_PX * 2) / viewport.width
  if (fit === 'page') {
    const heightScale = (scroller.clientHeight - VIEWER_PADDING_PX * 2) / viewport.height
    return clampPdfValue(Math.min(widthScale, heightScale), MIN_SCALE, MAX_SCALE)
  }
  return clampPdfValue(widthScale, MIN_SCALE, MAX_SCALE)
}

export function updateAutoPdfEmbedHeight(
  root: HTMLElement,
  toolbar: HTMLElement,
  scroller: HTMLElement,
  viewport: PdfViewport,
  options: PdfOptions,
): boolean {
  if (options.height || viewport.width <= 0 || viewport.height <= 0) return false

  const availableWidth = Math.max(1, scroller.clientWidth - VIEWER_PADDING_PX * 2)
  const pageHeight = (availableWidth / viewport.width) * viewport.height
  const toolbarHeight = toolbar.getBoundingClientRect().height
  const totalHeight = Math.ceil(toolbarHeight + pageHeight + VIEWER_PADDING_PX * 2)
  const value = `${totalHeight}px`
  if (root.style.getPropertyValue('--pdf-embed-height') === value) return false
  root.style.setProperty('--pdf-embed-height', value)
  return true
}

export function setSlotSize(slot: PdfPageSlot, viewport: PdfViewport, scale: number): boolean {
  const width = Math.ceil(viewport.width * scale)
  const height = Math.ceil(viewport.height * scale)
  const widthPx = `${width}px`
  const heightPx = `${height}px`
  const changed = slot.element.style.width !== widthPx || slot.element.style.minHeight !== heightPx
  slot.element.style.width = widthPx
  slot.element.style.minHeight = heightPx
  slot.element.style.setProperty(
    '--total-scale-factor',
    String(viewport.scale * viewport.userUnit * scale),
  )
  slot.textLayer.style.width = widthPx
  slot.textLayer.style.height = heightPx
  slot.highlightLayer.style.width = widthPx
  slot.highlightLayer.style.height = heightPx
  return changed
}

export function createPageSlot(
  pageNumber: number,
  viewport: PdfViewport,
  scale: number,
): PdfPageSlot {
  const element = document.createElement('div')
  element.className = 'pdf-embed-page'
  element.dataset.pdfPageNumber = String(pageNumber)

  const label = document.createElement('div')
  label.className = 'pdf-embed-page-label'
  label.textContent = String(pageNumber)

  const canvas = document.createElement('canvas')
  canvas.className = 'pdf-embed-canvas'
  canvas.width = 0
  canvas.height = 0

  const highlightLayer = document.createElement('div')
  highlightLayer.className = 'pdf-embed-highlight-layer'

  const textLayer = document.createElement('div')
  textLayer.className = 'pdf-embed-text-layer textLayer'

  element.append(canvas, highlightLayer, textLayer, label)
  const slot = { pageNumber, element, canvas, textLayer, highlightLayer, label }
  setSlotSize(slot, viewport, scale)
  return slot
}

export function updateControls(state: PdfState) {
  const { controls } = state
  setPageCount(controls, state.document.numPages)
  controls.pageInput.value = String(state.currentPage)
  controls.prev.disabled = state.currentPage <= 1
  controls.next.disabled = state.currentPage >= state.document.numPages
  controls.zoom.textContent = `${Math.round(state.scale * 100)}%`
  if (state.fit !== 'custom') controls.fit.value = state.fit
}

export function invalidatePageMetrics(state: PdfState) {
  state.pageMetricsValid = false
}

function readPageMetrics(state: PdfState): PdfPageMetric[] {
  if (state.pageMetricsValid) return state.pageMetrics
  state.pageMetrics = [...state.slots.values()]
    .sort((left, right) => left.pageNumber - right.pageNumber)
    .map(slot => ({
      pageNumber: slot.pageNumber,
      center: slot.element.offsetTop + slot.element.offsetHeight / 2,
    }))
  state.pageMetricsValid = true
  return state.pageMetrics
}

export function currentPageFromScroll(state: PdfState): number {
  const metrics = readPageMetrics(state)
  if (metrics.length === 0) return state.currentPage
  const center = state.scroller.scrollTop + state.scroller.clientHeight / 2
  let low = 0
  let high = metrics.length - 1
  while (low < high) {
    const mid = Math.floor((low + high) / 2)
    if (metrics[mid].center < center) {
      low = mid + 1
    } else {
      high = mid
    }
  }
  const current = metrics[low]
  const previous = metrics[Math.max(0, low - 1)]
  return Math.abs(previous.center - center) <= Math.abs(current.center - center)
    ? previous.pageNumber
    : current.pageNumber
}
