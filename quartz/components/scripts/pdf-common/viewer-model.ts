const DEFAULT_OVERSCAN = 1

const PDF_INTERSECTION_ROOT_MARGIN = '1200px 0px'
const DEFAULT_MAX_PIXEL_RATIO = 2
export const MIN_SCALE = 0.25
export const MAX_SCALE = 4
export const VIEWER_PADDING_PX = 0

export type PdfFitMode = 'width' | 'page' | 'actual' | 'custom'

export interface PdfViewport {
  width: number
  height: number
  scale: number
  userUnit: number
}

interface PdfRenderTask {
  promise: Promise<void>
  cancel(): void
}

export interface PdfTextRenderTask {
  render(): Promise<void>
  cancel(): void
}

interface PdfTextLayerOptions {
  textContentSource: PdfTextContent
  container: HTMLDivElement
  viewport: PdfViewport
}

interface PdfTextContent {
  items: unknown[]
  styles: Record<string, unknown>
}

interface PdfPage {
  getViewport(options: { scale: number }): PdfViewport
  getTextContent(): Promise<PdfTextContent>
  render(options: {
    canvasContext: CanvasRenderingContext2D
    viewport: PdfViewport
    transform: number[]
  }): PdfRenderTask
}

interface PdfDocument {
  numPages: number
  getPage(pageNumber: number): Promise<PdfPage>
}

interface PdfLoadingTask {
  promise: Promise<PdfDocument>
  destroy(): Promise<void>
}

export interface PdfJsRuntime {
  GlobalWorkerOptions: { workerSrc: string }
  createTextLayer(options: PdfTextLayerOptions): PdfTextRenderTask
  getDocument(options: {
    url: string
    cMapUrl: string
    cMapPacked: boolean
    standardFontDataUrl: string
    wasmUrl: string
  }): PdfLoadingTask
}

export interface PdfOptions {
  src: string
  title: string
  page: number
  fit: PdfFitMode
  scale?: number
  height?: number
  overscan: number
  preload: boolean
  annotations: boolean
  rootMargin: string
  maxPixelRatio: number
  textLayer: boolean
}

export interface PdfDocumentCacheEntry {
  key: string
  loadingTask: PdfLoadingTask
  promise: Promise<PdfDocument>
  refs: number
  destroyTimer: number
  loaded: boolean
}

export interface PdfDocumentHandle {
  loadingTask: PdfLoadingTask
  document: PdfDocument
  release(): void
}

export interface PdfPageSlot {
  pageNumber: number
  element: HTMLDivElement
  canvas: HTMLCanvasElement
  textLayer: HTMLDivElement
  highlightLayer: HTMLDivElement
  label: HTMLDivElement
  page?: PdfPage
  renderTask?: PdfRenderTask
  textRenderTask?: PdfTextRenderTask
  renderedScale?: number
  renderingScale?: number
  textRenderedScale?: number
  textRenderingScale?: number
}

export interface PdfPageMetric {
  pageNumber: number
  center: number
}

export interface PdfControls {
  pageInput: HTMLInputElement
  pageTotal: HTMLSpanElement
  prev: HTMLButtonElement
  next: HTMLButtonElement
  zoom: HTMLSpanElement
  fit: HTMLSelectElement
}

export interface PdfState {
  events: AbortController
  root: HTMLElement
  options: PdfOptions
  pdfjs: PdfJsRuntime
  loadingTask: PdfLoadingTask
  document: PdfDocument
  releaseDocument: () => void
  toolbar: HTMLDivElement
  scroller: HTMLDivElement
  shell: HTMLDivElement
  viewer: HTMLDivElement
  annotationRail: HTMLDivElement
  annotationStack: HTMLDivElement
  controls: PdfControls
  firstViewport: PdfViewport
  slots: Map<number, PdfPageSlot>
  renderQueue: Set<number>
  textRenderQueue: Set<number>
  preloadQueue: Set<number>
  visiblePages: Set<number>
  observer: IntersectionObserver
  resizeObserver: ResizeObserver
  pageMetrics: PdfPageMetric[]
  pageMetricsValid: boolean
  scale: number
  fit: PdfFitMode
  layoutWidth: number
  currentPage: number
  draining: boolean
  textDraining: boolean
  preloadDraining: boolean
  destroyed: boolean
  renderEpoch: number
  scrollFrame: number
  resizeFrame: number
  pruneTimer: number
  lastScrollAt: number
  cleanup: Array<() => void>
  annotations: PdfAnnotationState
}

export interface PdfAnchorRect {
  page: number
  left: number
  top: number
  width: number
  height: number
}

export interface PdfAnchor {
  kind: 'pdf'
  src: string
  rects: PdfAnchorRect[]
}

export interface PdfComment {
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
  resolvedAt: number | null
  anchor?: PdfAnchor | null
  orphaned?: boolean | null
  lastRecoveredAt?: number | null
}

export type PdfOperationType = 'new' | 'update' | 'delete' | 'resolve'

export interface PdfOperationInput {
  opId: string
  type: PdfOperationType
  comment: PdfComment
}

export interface PdfOperationRecord extends PdfOperationInput {
  seq: number
}

export interface PdfAnnotationState {
  pageId: string
  src: string
  comments: PdfComment[]
  pendingOps: Map<string, PdfOperationInput>
  lastSeq: number
  hasSnapshot: boolean
  reconnectEnabled: boolean
  reconnectTimer: number
  websocket: WebSocket | null
  composer: HTMLDivElement | null
  activeCommentId: string | null
}

export interface PdfSelection {
  text: string
  rects: PdfAnchorRect[]
  anchorRect: DOMRect
}

export function readNumber(value: string | undefined, fallback: number): number {
  if (!value) return fallback
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

export function readInteger(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value ?? '', 10)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback
}

function readNonNegativeInteger(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value ?? '', 10)
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback
}

function readBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback
  if (value === 'true' || value === '1' || value === 'on') return true
  if (value === 'false' || value === '0' || value === 'off') return false
  return fallback
}

export function readFit(value: string | undefined): PdfFitMode {
  if (value === 'page' || value === 'actual' || value === 'custom') return value
  return 'width'
}

export function clampPdfValue(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function readPdfOptions(root: HTMLElement): PdfOptions | null {
  const src = root.dataset.pdfSrc
  if (!src) return null

  const scale = root.dataset.pdfScale ? readNumber(root.dataset.pdfScale, 1) : undefined
  return {
    src,
    title: root.dataset.pdfTitle || src.split('/').pop() || 'document.pdf',
    page: readInteger(root.dataset.pdfPage, 1),
    fit: scale ? 'custom' : readFit(root.dataset.pdfFit),
    scale,
    height: root.dataset.pdfHeight ? readInteger(root.dataset.pdfHeight, 0) : undefined,
    overscan: readNonNegativeInteger(root.dataset.pdfOverscan, DEFAULT_OVERSCAN),
    preload: readBoolean(root.dataset.pdfPreload, false),
    annotations: readBoolean(root.dataset.pdfEnableAnnotations, true),
    rootMargin: root.dataset.pdfRootMargin || PDF_INTERSECTION_ROOT_MARGIN,
    maxPixelRatio: clampPdfValue(
      readNumber(root.dataset.pdfMaxPixelRatio, DEFAULT_MAX_PIXEL_RATIO),
      1,
      DEFAULT_MAX_PIXEL_RATIO,
    ),
    textLayer: readBoolean(root.dataset.pdfTextLayer, true),
  }
}

export function normalizePdfSource(src: string): string {
  const url = new URL(src, window.location.href)
  return `${url.pathname}${url.search}`
}
