const PDFJS_SCRIPT_SRC = '/static/pdfjs/pdf.min.mjs'
const PDFJS_WORKER_SRC = '/static/pdfjs/pdf.worker.min.mjs'
const PDF_EMBED_SELECTOR = '.pdf-embed[data-pdf-src]:not([data-pdf-status])'
const PDF_EMBED_CLEANUP_SELECTOR = '.pdf-embed[data-pdf-status]'
const PDF_COMMENT_ROOM_ENABLED_KEY = 'garden:comments-room-enabled:v1'
const PDF_COMMENT_ROOM_TOGGLE_EVENT = 'commentsroomtoggle'
const DEFAULT_OVERSCAN = 1
const MAX_RENDER_WINDOW_RADIUS = 4
const PDF_PRELOAD_PAGE_LIMIT = 12
const PDF_DOCUMENT_CACHE_TTL_MS = 60 * 1000
const PDF_INTERSECTION_ROOT_MARGIN = '1200px 0px'
const DEFAULT_MAX_PIXEL_RATIO = 2
const MIN_SCALE = 0.25
const MAX_SCALE = 4
const VIEWER_PADDING_PX = 0
const TEXT_RENDER_IDLE_TIMEOUT_MS = 1000
const TEXT_RENDER_SCROLL_DELAY_MS = 600
const PDF_PRELOAD_SCROLL_DELAY_MS = 700
const PDF_PRUNE_SCROLL_DELAY_MS = 900
const PDF_ICON_PREV =
  'M8.84182 3.13514C9.04327 3.32401 9.05348 3.64042 8.86462 3.84188L5.43521 7.49991L8.86462 11.1579C9.05348 11.3594 9.04327 11.6758 8.84182 11.8647C8.64036 12.0535 8.32394 12.0433 8.13508 11.8419L4.38508 7.84188C4.20477 7.64955 4.20477 7.35027 4.38508 7.15794L8.13508 3.15794C8.32394 2.95648 8.64036 2.94628 8.84182 3.13514Z'
const PDF_ICON_NEXT =
  'M6.1584 3.13508C6.35985 2.94621 6.67627 2.95642 6.86514 3.15788L10.6151 7.15788C10.7954 7.3502 10.7954 7.64949 10.6151 7.84182L6.86514 11.8418C6.67627 12.0433 6.35985 12.0535 6.1584 11.8646C5.95694 11.6757 5.94673 11.3593 6.1356 11.1579L9.565 7.49985L6.1356 3.84182C5.94673 3.64036 5.95694 3.32394 6.1584 3.13508Z'
const PDF_ICON_MINUS =
  'M3.5 7C3.22386 7 3 7.22386 3 7.5C3 7.77614 3.22386 8 3.5 8H11.5C11.7761 8 12 7.77614 12 7.5C12 7.22386 11.7761 7 11.5 7H3.5Z'
const PDF_ICON_PLUS =
  'M8 2.75C8 2.47386 7.77614 2.25 7.5 2.25C7.22386 2.25 7 2.47386 7 2.75V7H2.75C2.47386 7 2.25 7.22386 2.25 7.5C2.25 7.77614 2.47386 8 2.75 8H7V12.25C7 12.5261 7.22386 12.75 7.5 12.75C7.77614 12.75 8 12.5261 8 12.25V8H12.25C12.5261 8 12.75 7.77614 12.75 7.5C12.75 7.22386 12.5261 7 12.25 7H8V2.75Z'
const PDF_ICON_DOWNLOAD =
  'M7.50005 1.04999C7.74858 1.04999 7.95005 1.25146 7.95005 1.49999V8.41359L10.1819 6.18179C10.3576 6.00605 10.6425 6.00605 10.8182 6.18179C10.994 6.35753 10.994 6.64245 10.8182 6.81819L7.81825 9.81819C7.64251 9.99392 7.35759 9.99392 7.18185 9.81819L4.18185 6.81819C4.00611 6.64245 4.00611 6.35753 4.18185 6.18179C4.35759 6.00605 4.64251 6.00605 4.81825 6.18179L7.05005 8.41359V1.49999C7.05005 1.25146 7.25152 1.04999 7.50005 1.04999ZM2.5 10C2.77614 10 3 10.2239 3 10.5V12C3 12.5539 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2239 12.2239 10 12.5 10C12.7761 10 13 10.2239 13 10.5V12C13 13.1041 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2239 2.22386 10 2.5 10Z'
const PDF_ICON_SUBMIT =
  'M12 16a.5.5 0 0 1-.5-.5V8.707l-3.146 3.147a.5.5 0 0 1-.708-.708l4-4a.5.5 0 0 1 .708 0l4 4a.5.5 0 0 1-.708.708L12.5 8.707V15.5a.5.5 0 0 1-.5.5'

type PdfFitMode = 'width' | 'page' | 'actual' | 'custom'

interface PdfViewport {
  width: number
  height: number
}

interface PdfRenderTask {
  promise: Promise<void>
  cancel(): void
}

interface PdfTextRenderTask {
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
  render(options: { canvasContext: CanvasRenderingContext2D; viewport: PdfViewport }): PdfRenderTask
}

interface PdfDocument {
  numPages: number
  getPage(pageNumber: number): Promise<PdfPage>
  destroy(): Promise<void>
}

interface PdfLoadingTask {
  promise: Promise<PdfDocument>
  destroy(): Promise<void>
}

interface PdfJsRuntime {
  GlobalWorkerOptions: { workerSrc: string }
  createTextLayer(options: PdfTextLayerOptions): PdfTextRenderTask
  getDocument(options: {
    url: string
    cMapUrl: string
    cMapPacked: boolean
    standardFontDataUrl: string
  }): PdfLoadingTask
}

interface QuartzPdfEmbeds {
  mount(root?: ParentNode): void
  cleanup(root?: ParentNode): void
  preload(src?: string): Promise<void>
}

interface PdfOptions {
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

interface PdfDocumentCacheEntry {
  key: string
  loadingTask: PdfLoadingTask
  promise: Promise<PdfDocument>
  refs: number
  destroyTimer: number
  document?: PdfDocument
}

interface PdfDocumentHandle {
  loadingTask: PdfLoadingTask
  document: PdfDocument
  release(): void
}

interface PdfPageSlot {
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

interface PdfPageMetric {
  pageNumber: number
  center: number
}

interface PdfControls {
  pageInput: HTMLInputElement
  pageTotal: HTMLSpanElement
  prev: HTMLButtonElement
  next: HTMLButtonElement
  zoom: HTMLSpanElement
  fit: HTMLSelectElement
}

interface PdfState {
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

interface PdfAnchorRect {
  page: number
  left: number
  top: number
  width: number
  height: number
}

interface PdfAnchor {
  kind: 'pdf'
  src: string
  rects: PdfAnchorRect[]
}

interface PdfComment {
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

type PdfOperationType = 'new' | 'update' | 'delete' | 'resolve'

interface PdfOperationInput {
  opId: string
  type: PdfOperationType
  comment: PdfComment
}

interface PdfOperationRecord extends PdfOperationInput {
  seq: number
}

interface PdfAnnotationState {
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

interface PdfSelection {
  text: string
  rects: PdfAnchorRect[]
  anchorRect: DOMRect
}

interface PdfCommentComposerOptions {
  className: string
  placeholder: string
  submitLabel: string
  onSubmit(content: string): void | Promise<void>
  onCancel(): void
}

interface PdfCommentComposer {
  element: HTMLDivElement
  focus(): void
}

let pdfJsLoad: Promise<PdfJsRuntime> | undefined
const pdfDocumentCache = new Map<string, PdfDocumentCacheEntry>()
const pdfStates = new WeakMap<HTMLElement, PdfState>()
const pendingPdfMounts = new WeakSet<HTMLElement>()
let selectedPdfState: PdfState | null = null
let selectedPdfRoot: HTMLElement | null = null

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readCommentRoomEnabled(storage: Storage = localStorage): boolean {
  return storage.getItem(PDF_COMMENT_ROOM_ENABLED_KEY) === 'true'
}

function ensurePdfJsGlobals() {
  if (Reflect.get(window, 'activeWindow') === window) return
  Object.defineProperty(window, 'activeWindow', {
    value: window,
    configurable: true,
    writable: true,
  })
}

function readPdfTextRenderTask(value: unknown): PdfTextRenderTask | undefined {
  if ((typeof value !== 'object' && typeof value !== 'function') || value === null) {
    return undefined
  }

  const render = Reflect.get(value, 'render')
  const cancel = Reflect.get(value, 'cancel')
  if (typeof render !== 'function') return undefined
  if (typeof cancel !== 'function') return undefined

  return {
    render() {
      return Reflect.apply(render, value, [])
    },
    cancel() {
      Reflect.apply(cancel, value, [])
    },
  }
}

function readPdfJsRuntime(value: unknown): PdfJsRuntime | undefined {
  if ((typeof value !== 'object' && typeof value !== 'function') || value === null) {
    return undefined
  }

  const getDocument = Reflect.get(value, 'getDocument')
  const TextLayer = Reflect.get(value, 'TextLayer')
  const globalWorkerOptions = Reflect.get(value, 'GlobalWorkerOptions')
  if (typeof getDocument !== 'function') return undefined
  if (typeof TextLayer !== 'function') return undefined
  if (
    (typeof globalWorkerOptions !== 'object' && typeof globalWorkerOptions !== 'function') ||
    globalWorkerOptions === null
  ) {
    return undefined
  }

  return {
    createTextLayer(options) {
      const layer = readPdfTextRenderTask(Reflect.construct(TextLayer, [options]))
      if (!layer) throw new Error('PDF.js TextLayer did not initialize')
      return layer
    },
    GlobalWorkerOptions: {
      get workerSrc() {
        const current = Reflect.get(globalWorkerOptions, 'workerSrc')
        return typeof current === 'string' ? current : ''
      },
      set workerSrc(value: string) {
        Reflect.set(globalWorkerOptions, 'workerSrc', value)
      },
    },
    getDocument(options) {
      return Reflect.apply(getDocument, value, [options])
    },
  }
}

function isRenderCancelError(error: unknown): boolean {
  const message = error instanceof Error ? `${error.name} ${error.message}` : String(error)
  return message.includes('RenderingCancelledException') || message.includes('AbortException')
}

function readNumber(value: string | undefined, fallback: number): number {
  if (!value) return fallback
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

function readInteger(value: string | undefined, fallback: number): number {
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

function readFit(value: string | undefined): PdfFitMode {
  if (value === 'page' || value === 'actual' || value === 'custom') return value
  return 'width'
}

function clampPdfValue(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function readPdfOptions(root: HTMLElement): PdfOptions | null {
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

function loadPdfJs(): Promise<PdfJsRuntime> {
  ensurePdfJsGlobals()
  const existing = readPdfJsRuntime(window.pdfjsLib)
  if (existing) return Promise.resolve(existing)
  if (pdfJsLoad) return pdfJsLoad

  pdfJsLoad = import(PDFJS_SCRIPT_SRC).then(module => {
    const runtime = readPdfJsRuntime(module)
    if (!runtime) throw new Error('PDF.js did not initialize')
    window.pdfjsLib = runtime
    return runtime
  })

  return pdfJsLoad
}

function pdfDocumentCacheKey(src: string): string {
  try {
    return normalizePdfSource(src)
  } catch {
    return src
  }
}

function pdfDocumentOptions(src: string) {
  return {
    url: src,
    cMapUrl: '/static/pdfjs/cmaps/',
    cMapPacked: true,
    standardFontDataUrl: '/static/pdfjs/standard_fonts/',
  }
}

function releasePdfDocument(key: string) {
  const entry = pdfDocumentCache.get(key)
  if (!entry) return
  entry.refs = Math.max(0, entry.refs - 1)
  if (entry.refs > 0 || entry.destroyTimer) return
  entry.destroyTimer = window.setTimeout(() => {
    if (entry.refs > 0) {
      entry.destroyTimer = 0
      return
    }

    pdfDocumentCache.delete(key)
    void entry.loadingTask.destroy().catch(() => undefined)
    void entry.document?.destroy().catch(() => undefined)
  }, PDF_DOCUMENT_CACHE_TTL_MS)
}

function loadPdfDocument(pdfjs: PdfJsRuntime, src: string): Promise<PdfDocumentHandle> {
  const key = pdfDocumentCacheKey(src)
  let entry = pdfDocumentCache.get(key)

  if (!entry) {
    const loadingTask = pdfjs.getDocument(pdfDocumentOptions(src))
    let cacheEntry: PdfDocumentCacheEntry
    const promise = loadingTask.promise
      .then(document => {
        cacheEntry.document = document
        return document
      })
      .catch(error => {
        pdfDocumentCache.delete(key)
        throw error
      })
    cacheEntry = { key, loadingTask, promise, refs: 0, destroyTimer: 0 }
    entry = cacheEntry
    pdfDocumentCache.set(key, entry)
  }

  if (entry.destroyTimer) {
    window.clearTimeout(entry.destroyTimer)
    entry.destroyTimer = 0
  }
  entry.refs += 1

  return entry.promise.then(document => {
    let released = false
    return {
      loadingTask: entry.loadingTask,
      document,
      release() {
        if (released) return
        released = true
        releasePdfDocument(key)
      },
    }
  })
}

async function preloadPdfRuntime(src?: string): Promise<void> {
  const pdfjs = await loadPdfJs()
  if (!src) return
  const handle = await loadPdfDocument(pdfjs, src)
  handle.release()
}

function normalizePdfSource(src: string): string {
  const url = new URL(src, window.location.href)
  return `${url.pathname}${url.search}`
}

function pdfCommentPageId(src: string): string {
  return `pdf:${normalizePdfSource(src)}`
}

function getPdfAuthor(): string {
  const login = localStorage.getItem('comment-author-github-login')
  const stored = localStorage.getItem('comment-author')
  if (login && stored !== login) {
    localStorage.setItem('comment-author', login)
    return login
  }
  if (stored) return stored
  if (login) {
    localStorage.setItem('comment-author', login)
    return login
  }
  const author = `anon-${Math.random().toString(36).slice(2, 8)}`
  localStorage.setItem('comment-author', author)
  return author
}

async function hashPdfText(text: string): Promise<string> {
  const data = new TextEncoder().encode(text)
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  return Array.from(new Uint8Array(hashBuffer))
    .map(byte => byte.toString(16).padStart(2, '0'))
    .join('')
}

function readNullableString(value: unknown): string | null | undefined {
  if (value === null || value === undefined) return null
  return typeof value === 'string' ? value : undefined
}

function readNullableNumber(value: unknown): number | null | undefined {
  if (value === null || value === undefined) return null
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function readNullableBoolean(value: unknown): boolean | null | undefined {
  if (value === null || value === undefined) return null
  return typeof value === 'boolean' ? value : undefined
}

function parsePdfAnchorRect(value: unknown): PdfAnchorRect | null {
  if (!isRecord(value)) return null
  const page = value['page']
  const left = value['left']
  const top = value['top']
  const width = value['width']
  const height = value['height']

  if (typeof page !== 'number' || !Number.isInteger(page) || page < 1) return null
  if (typeof left !== 'number' || !Number.isFinite(left)) return null
  if (typeof top !== 'number' || !Number.isFinite(top)) return null
  if (typeof width !== 'number' || !Number.isFinite(width) || width <= 0) return null
  if (typeof height !== 'number' || !Number.isFinite(height) || height <= 0) return null

  return { page, left, top, width, height }
}

function parsePdfAnchor(value: unknown): PdfAnchor | null {
  if (!isRecord(value)) return null
  if (value['kind'] !== 'pdf') return null

  const src = value['src']
  const rects = value['rects']
  if (typeof src !== 'string') return null
  if (!Array.isArray(rects) || rects.length === 0) return null

  const parsedRects: PdfAnchorRect[] = []
  for (const rect of rects) {
    const parsed = parsePdfAnchorRect(rect)
    if (!parsed) return null
    parsedRects.push(parsed)
  }

  return { kind: 'pdf', src, rects: parsedRects }
}

function parsePdfComment(value: unknown): PdfComment | null {
  if (!isRecord(value)) return null

  const id = value['id']
  const pageId = value['pageId']
  const parentId = readNullableString(value['parentId'])
  const anchorHash = value['anchorHash']
  const anchorStart = value['anchorStart']
  const anchorEnd = value['anchorEnd']
  const anchorText = value['anchorText']
  const content = value['content']
  const author = value['author']
  const createdAt = value['createdAt']
  const updatedAt = readNullableNumber(value['updatedAt'])
  const deletedAt = readNullableNumber(value['deletedAt'])
  const resolvedAt = readNullableNumber(value['resolvedAt'])
  const orphaned = readNullableBoolean(value['orphaned'])
  const lastRecoveredAt = readNullableNumber(value['lastRecoveredAt'])
  const anchor = parsePdfAnchor(value['anchor'])

  if (typeof id !== 'string') return null
  if (typeof pageId !== 'string') return null
  if (parentId === undefined) return null
  if (typeof anchorHash !== 'string') return null
  if (typeof anchorStart !== 'number' || !Number.isFinite(anchorStart)) return null
  if (typeof anchorEnd !== 'number' || !Number.isFinite(anchorEnd)) return null
  if (typeof anchorText !== 'string') return null
  if (typeof content !== 'string') return null
  if (typeof author !== 'string') return null
  if (typeof createdAt !== 'number' || !Number.isFinite(createdAt)) return null
  if (updatedAt === undefined) return null
  if (deletedAt === undefined) return null
  if (resolvedAt === undefined) return null
  if (orphaned === undefined) return null
  if (lastRecoveredAt === undefined) return null
  if (!anchor) return null

  return {
    id,
    pageId,
    parentId,
    anchorHash,
    anchorStart,
    anchorEnd,
    anchorText,
    content,
    author,
    createdAt,
    updatedAt,
    deletedAt,
    resolvedAt,
    anchor,
    orphaned,
    lastRecoveredAt,
  }
}

function isParsedPdfComment(value: PdfComment | null): value is PdfComment {
  return value !== null
}

function parsePdfOperationType(value: unknown): PdfOperationType | null {
  if (value === 'new' || value === 'update' || value === 'delete' || value === 'resolve') {
    return value
  }
  return null
}

function parsePdfOperationRecord(value: unknown): PdfOperationRecord | null {
  if (!isRecord(value)) return null
  const seq = value['seq']
  const opId = value['opId']
  const type = parsePdfOperationType(value['type'])
  const comment = parsePdfComment(value['comment'])

  if (typeof seq !== 'number' || !Number.isFinite(seq)) return null
  if (typeof opId !== 'string') return null
  if (!type || !comment) return null

  return { seq, opId, type, comment }
}

function upsertPdfComment(comments: PdfComment[], comment: PdfComment): PdfComment[] {
  const idx = comments.findIndex(item => item.id === comment.id)
  if (idx === -1) return [...comments, comment]
  const next = comments.slice()
  next[idx] = comment
  return next
}

function mergePendingPdfComments(state: PdfState, comments: PdfComment[]): PdfComment[] {
  let next = comments
  for (const op of state.annotations.pendingOps.values()) {
    next = upsertPdfComment(next, op.comment)
  }
  return next
}

function createPdfAnnotationState(options: PdfOptions): PdfAnnotationState {
  const src = normalizePdfSource(options.src)
  return {
    pageId: pdfCommentPageId(options.src),
    src,
    comments: [],
    pendingOps: new Map(),
    lastSeq: 0,
    hasSnapshot: false,
    reconnectEnabled: false,
    reconnectTimer: 0,
    websocket: null,
    composer: null,
    activeCommentId: null,
  }
}

function filterPdfComments(state: PdfState, comments: PdfComment[]): PdfComment[] {
  return comments.filter(comment => {
    if (comment.pageId !== state.annotations.pageId) return false
    if (!comment.anchor || comment.anchor.kind !== 'pdf') return false
    return comment.anchor.src === state.annotations.src
  })
}

function visiblePdfTopLevelComments(state: PdfState): PdfComment[] {
  return state.annotations.comments.filter(comment => {
    if (comment.parentId) return false
    if (comment.deletedAt || comment.resolvedAt) return false
    if (!comment.anchor || comment.anchor.kind !== 'pdf') return false
    return comment.anchor.src === state.annotations.src && comment.anchor.rects.length > 0
  })
}

function pdfCommentReplies(state: PdfState, commentId: string): PdfComment[] {
  return state.annotations.comments.filter(comment => {
    return comment.parentId === commentId && !comment.deletedAt && !comment.resolvedAt
  })
}

function closePdfAnnotations(state: PdfState) {
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

function connectPdfAnnotations(state: PdfState) {
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

function submitPdfOperation(state: PdfState, type: PdfOperationType, comment: PdfComment) {
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

function selectPdfRoot(root: HTMLElement) {
  if (selectedPdfRoot && selectedPdfRoot !== root) {
    selectedPdfRoot.removeAttribute('data-pdf-selected')
  }
  selectedPdfRoot = root
  root.dataset.pdfSelected = 'true'
  selectedPdfState = pdfStates.get(root) ?? null
}

function clearSelectedPdfRoot(root: HTMLElement) {
  if (selectedPdfRoot !== root) return
  selectedPdfRoot = null
  selectedPdfState = null
  root.removeAttribute('data-pdf-selected')
  if (document.activeElement === root) root.blur()
}

function selectPdfState(state: PdfState) {
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

function createPdfCommentSubmitIcon(): SVGSVGElement {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('aria-hidden', 'true')
  svg.setAttribute('focusable', 'false')
  svg.setAttribute('viewBox', '0 0 24 24')
  svg.setAttribute('fill', 'none')
  const pathElement = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  pathElement.setAttribute('d', PDF_ICON_SUBMIT)
  pathElement.setAttribute('fill', 'currentColor')
  pathElement.setAttribute('fill-rule', 'evenodd')
  pathElement.setAttribute('clip-rule', 'evenodd')
  svg.append(pathElement)
  return svg
}

function resizePdfCommentInput(input: HTMLTextAreaElement) {
  input.style.height = 'auto'
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`
}

function createPdfCommentComposer(options: PdfCommentComposerOptions): PdfCommentComposer {
  const element = document.createElement('div')
  element.className = `${options.className} pdf-comment-editor-composer`

  const shell = document.createElement('div')
  shell.className = 'pdf-comment-editor-shell pdf-comment-empty'

  const inputContainer = document.createElement('div')
  inputContainer.className = 'pdf-comment-editor'
  inputContainer.setAttribute('role', 'textbox')
  inputContainer.setAttribute('aria-placeholder', options.placeholder)

  const input = document.createElement('textarea')
  input.className = 'pdf-comment-input'
  input.placeholder = options.placeholder
  input.rows = 1
  input.spellcheck = true
  input.setAttribute('aria-label', options.placeholder)

  const placeholder = document.createElement('div')
  placeholder.className = 'pdf-comment-placeholder'
  placeholder.setAttribute('aria-hidden', 'true')
  const placeholderText = document.createElement('span')
  placeholderText.className = 'pdf-comment-placeholder-text'
  placeholderText.textContent = options.placeholder
  placeholder.appendChild(placeholderText)

  const submit = document.createElement('button')
  submit.type = 'button'
  submit.className = 'pdf-comment-submit'
  submit.ariaLabel = options.submitLabel
  submit.disabled = true
  submit.setAttribute('aria-disabled', 'true')
  const icon = document.createElement('span')
  icon.className = 'pdf-comment-submit-icon'
  icon.appendChild(createPdfCommentSubmitIcon())
  submit.appendChild(icon)

  const sync = () => {
    const empty = input.value.trim().length === 0
    shell.classList.toggle('pdf-comment-empty', empty)
    submit.disabled = empty
    submit.setAttribute('aria-disabled', String(empty))
    resizePdfCommentInput(input)
  }

  let submitting = false
  const save = () => {
    const content = input.value.trim()
    if (submitting || !content) return
    submitting = true
    submit.disabled = true
    submit.setAttribute('aria-disabled', 'true')
    Promise.resolve(options.onSubmit(content))
      .catch(error => {
        console.error('failed to submit pdf comment:', error)
      })
      .finally(() => {
        submitting = false
        if (element.isConnected) sync()
      })
  }

  element.addEventListener('click', event => event.stopPropagation())
  input.addEventListener('input', sync)
  input.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      options.onCancel()
      return
    }
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault()
      event.stopPropagation()
      save()
      return
    }
    event.stopPropagation()
  })
  submit.addEventListener('click', event => {
    event.stopPropagation()
    save()
  })

  inputContainer.append(input, placeholder)
  shell.append(inputContainer, submit)
  element.appendChild(shell)
  sync()

  return {
    element,
    focus() {
      input.focus()
    },
  }
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
  reply.addEventListener('click', event => {
    event.stopPropagation()
    state.annotations.activeCommentId = comment.id
    renderPdfAnnotations(state)
  })
  actions.appendChild(reply)

  const author = getPdfAuthor()
  const login = localStorage.getItem('comment-author-github-login')?.toLowerCase()
  if (comment.author === author || login === 'aarnphm') {
    const resolve = buildPdfNoteButton('Resolve')
    resolve.addEventListener('click', event => {
      event.stopPropagation()
      submitPdfOperation(state, 'resolve', { ...comment, resolvedAt: Date.now() })
    })
    const remove = buildPdfNoteButton('Delete')
    remove.addEventListener('click', event => {
      event.stopPropagation()
      submitPdfOperation(state, 'delete', { ...comment, deletedAt: Date.now() })
    })
    actions.append(resolve, remove)
  }

  card.append(meta, quote, content, actions)
  for (const replyComment of pdfCommentReplies(state, comment.id)) {
    card.appendChild(renderPdfReply(replyComment))
  }
  if (state.annotations.activeCommentId === comment.id) {
    card.appendChild(renderPdfReplyComposer(state, comment))
  }

  card.addEventListener('click', () => {
    state.annotations.activeCommentId = comment.id
    scrollToPdfHighlight(state, comment)
    renderPdfAnnotations(state)
  })

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

function syncPdfEmbedViewport(state: PdfState): boolean {
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

function renderPdfAnnotations(state: PdfState) {
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
      span.addEventListener('click', event => {
        event.stopPropagation()
        activatePdfComment(state, comment.id, true)
      })
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

function createButton(className: string, icon: SVGSVGElement, title: string): HTMLButtonElement {
  const button = document.createElement('button')
  button.className = `pdf-embed-button ${className}`
  button.type = 'button'
  button.title = title
  button.ariaLabel = title
  button.append(icon)
  return button
}

function createPdfIcon(path: string): SVGSVGElement {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('aria-hidden', 'true')
  svg.setAttribute('focusable', 'false')
  svg.setAttribute('viewBox', '0 0 15 15')
  svg.setAttribute('fill', 'none')
  const pathElement = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  pathElement.setAttribute('d', path)
  pathElement.setAttribute('fill', 'currentColor')
  svg.append(pathElement)
  return svg
}

function createDownloadLink(options: PdfOptions): HTMLAnchorElement {
  const link = document.createElement('a')
  link.className = 'pdf-embed-button pdf-embed-download'
  link.href = options.src
  link.download = options.title
  link.title = 'Download PDF'
  link.ariaLabel = 'Download PDF'
  link.append(createPdfIcon(PDF_ICON_DOWNLOAD))
  return link
}

function createControls(options: PdfOptions): { toolbar: HTMLDivElement; controls: PdfControls } {
  const toolbar = document.createElement('div')
  toolbar.className = 'pdf-embed-toolbar'

  const pageGroup = document.createElement('div')
  pageGroup.className = 'pdf-embed-control-group pdf-embed-page-controls'
  const prev = createButton('pdf-embed-prev', createPdfIcon(PDF_ICON_PREV), 'Previous page')
  const pageInput = document.createElement('input')
  pageInput.className = 'pdf-embed-page-input'
  pageInput.type = 'number'
  pageInput.min = '1'
  pageInput.value = String(options.page)
  pageInput.ariaLabel = 'Page'
  const pageTotal = document.createElement('span')
  pageTotal.className = 'pdf-embed-page-total'
  pageTotal.textContent = '/ 0'
  const next = createButton('pdf-embed-next', createPdfIcon(PDF_ICON_NEXT), 'Next page')
  pageGroup.append(prev, pageInput, pageTotal, next)

  const zoomGroup = document.createElement('div')
  zoomGroup.className = 'pdf-embed-control-group pdf-embed-zoom-controls'
  const zoomOut = createButton('pdf-embed-zoom-out', createPdfIcon(PDF_ICON_MINUS), 'Zoom out')
  const zoom = document.createElement('span')
  zoom.className = 'pdf-embed-zoom'
  zoom.textContent = '100%'
  const zoomIn = createButton('pdf-embed-zoom-in', createPdfIcon(PDF_ICON_PLUS), 'Zoom in')
  const download = createDownloadLink(options)
  const fit = document.createElement('select')
  fit.className = 'pdf-embed-fit'
  fit.ariaLabel = 'PDF fit'
  fit.hidden = true
  fit.tabIndex = -1
  fit.setAttribute('aria-hidden', 'true')
  const fitOptions: [PdfFitMode, string][] = [
    ['width', 'Width'],
    ['page', 'Page'],
    ['actual', '100%'],
  ]
  for (const [value, label] of fitOptions) {
    const option = document.createElement('option')
    option.value = value
    option.textContent = label
    fit.appendChild(option)
  }
  fit.value = options.fit === 'custom' ? 'width' : options.fit
  zoomGroup.append(zoomOut, zoom, zoomIn, download, fit)
  toolbar.append(pageGroup, zoomGroup)

  return { toolbar, controls: { pageInput, pageTotal, prev, next, zoom, fit } }
}

function calculateFitScale(
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

function updateAutoPdfEmbedHeight(
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

function setSlotSize(slot: PdfPageSlot, viewport: PdfViewport, scale: number): boolean {
  const width = Math.ceil(viewport.width * scale)
  const height = Math.ceil(viewport.height * scale)
  const widthPx = `${width}px`
  const heightPx = `${height}px`
  const changed = slot.element.style.width !== widthPx || slot.element.style.minHeight !== heightPx
  slot.element.style.width = widthPx
  slot.element.style.minHeight = heightPx
  slot.textLayer.style.width = widthPx
  slot.textLayer.style.height = heightPx
  slot.highlightLayer.style.width = widthPx
  slot.highlightLayer.style.height = heightPx
  return changed
}

function createPageSlot(pageNumber: number, viewport: PdfViewport, scale: number): PdfPageSlot {
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

function updateControls(state: PdfState) {
  const { controls } = state
  controls.pageInput.max = String(state.document.numPages)
  controls.pageInput.value = String(state.currentPage)
  controls.pageTotal.textContent = `/ ${state.document.numPages}`
  controls.prev.disabled = state.currentPage <= 1
  controls.next.disabled = state.currentPage >= state.document.numPages
  controls.zoom.textContent = `${Math.round(state.scale * 100)}%`
  if (state.fit !== 'custom') controls.fit.value = state.fit
}

function invalidatePageMetrics(state: PdfState) {
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

function currentPageFromScroll(state: PdfState): number {
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

function enqueuePage(state: PdfState, pageNumber: number) {
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

function cancelDistantRenderWork(state: PdfState, centerPage: number) {
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

function pruneDistantRenderedPages(state: PdfState, centerPage: number) {
  if (shouldPreloadPdfPages(state)) return
  if (state.pruneTimer) window.clearTimeout(state.pruneTimer)
  state.pruneTimer = window.setTimeout(() => {
    state.pruneTimer = 0
    pruneDistantRenderedPagesNow(state, centerPage)
  }, PDF_PRUNE_SCROLL_DELAY_MS)
}

function enqueueRenderWindow(state: PdfState, centerPage: number = state.currentPage) {
  const radius = pdfRenderWindowRadius(state)
  for (let offset = 0; offset <= radius; offset++) {
    const left = centerPage - offset
    const right = centerPage + offset
    if (left >= 1) enqueuePage(state, left)
    if (offset > 0 && right <= state.document.numPages) enqueuePage(state, right)
  }
}

function enqueueVisiblePages(state: PdfState) {
  for (const pageNumber of state.visiblePages) {
    enqueuePage(state, pageNumber)
  }
  enqueueRenderWindow(state)
}

function enqueuePreloadPages(state: PdfState) {
  if (!shouldPreloadPdfPages(state)) return
  for (let pageNumber = 1; pageNumber <= state.document.numPages; pageNumber++) {
    const slot = state.slots.get(pageNumber)
    if (!slot) continue
    if (slot.renderedScale === state.scale || slot.renderingScale === state.scale) continue
    state.preloadQueue.add(pageNumber)
  }
  drainPreloadQueue(state)
}

function scrollToPage(state: PdfState, pageNumber: number) {
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

function rerenderVisiblePages(state: PdfState) {
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
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  const task = page.render({ canvasContext: context, viewport })
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
  await textLayer.render()
  if (state.destroyed || epoch !== state.renderEpoch) return
  slot.textRenderTask = undefined
  slot.textRenderedScale = state.scale
  slot.textRenderingScale = undefined
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

function schedulePdfStartup(callback: () => void) {
  window.requestAnimationFrame(() => {
    window.setTimeout(callback, 0)
  })
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

function targetAcceptsTextInput(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target instanceof HTMLInputElement) return true
  if (target instanceof HTMLTextAreaElement) return true
  if (target instanceof HTMLSelectElement) return true
  return target.isContentEditable
}

function consumePdfNavigationKey(event: KeyboardEvent) {
  event.preventDefault()
  event.stopImmediatePropagation()
}

function setupPdfEvents(state: PdfState) {
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
      if (!syncingAnnotationScroll) {
        syncingAnnotationScroll = true
        state.annotationRail.scrollTop = state.scroller.scrollTop
        window.requestAnimationFrame(() => {
          syncingAnnotationScroll = false
        })
      }
      positionPdfAnnotationCards(state)
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

  state.controls.prev.addEventListener('click', goPrev)
  state.controls.next.addEventListener('click', goNext)
  state.controls.pageInput.addEventListener('change', setPage)
  state.root.querySelector('.pdf-embed-zoom-out')?.addEventListener('click', () => zoomBy(0.9))
  state.root.querySelector('.pdf-embed-zoom-in')?.addEventListener('click', () => zoomBy(1.1))
  state.controls.fit.addEventListener('change', setFit)
  state.scroller.addEventListener('scroll', onScroll, { passive: true })
  state.annotationRail.addEventListener('scroll', onAnnotationRailScroll, { passive: true })
  state.root.addEventListener('pointerdown', onPointerDown)
  document.addEventListener('keydown', onKeyDown, true)
  state.scroller.addEventListener('mouseup', onMouseUp)
  document.addEventListener(PDF_COMMENT_ROOM_TOGGLE_EVENT, onCommentRoomToggle)
  state.cleanup.push(() => {
    document.removeEventListener('keydown', onKeyDown, true)
    document.removeEventListener(PDF_COMMENT_ROOM_TOGGLE_EVENT, onCommentRoomToggle)
    if (state.scrollFrame) {
      window.cancelAnimationFrame(state.scrollFrame)
      state.scrollFrame = 0
    }
  })
}

function renderPdfError(root: HTMLElement, options: PdfOptions) {
  root.replaceChildren()
  const error = document.createElement('div')
  error.className = 'pdf-embed-error'
  const label = document.createElement('span')
  label.textContent = 'PDF unavailable'
  const link = document.createElement('a')
  link.href = options.src
  link.target = '_blank'
  link.rel = 'noopener noreferrer'
  link.textContent = options.title
  error.append(label, link)
  root.appendChild(error)
}

async function mountPdfEmbed(root: HTMLElement, options: PdfOptions) {
  root.dataset.pdfStatus = 'loading'
  root.replaceChildren()
  if (options.height) root.style.setProperty('--pdf-embed-height', `${options.height}px`)

  const loading = document.createElement('div')
  loading.className = 'pdf-embed-loading'
  loading.textContent = 'Loading PDF'
  root.appendChild(loading)

  let documentHandle: PdfDocumentHandle | undefined
  const pdfjs = await loadPdfJs()
  try {
    if (!root.isConnected || root.dataset.pdfStatus !== 'loading') return
    pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_SRC
    documentHandle = await loadPdfDocument(pdfjs, options.src)
    const { loadingTask, document: pdfDocument } = documentHandle
    if (!root.isConnected || root.dataset.pdfStatus !== 'loading') return
    const firstPage = await pdfDocument.getPage(1)
    if (!root.isConnected || root.dataset.pdfStatus !== 'loading') return
    const firstViewport = firstPage.getViewport({ scale: 1 })

    const { toolbar, controls } = createControls(options)
    controls.pageTotal.textContent = `/ ${pdfDocument.numPages}`

    const scroller = document.createElement('div')
    scroller.className = 'pdf-viewer-container'
    scroller.tabIndex = 0
    const body = document.createElement('div')
    body.className = 'pdf-embed-body'
    const shell = document.createElement('div')
    shell.className = 'pdf-viewer-shell'
    const viewer = document.createElement('div')
    viewer.className = 'pdf-viewer'
    const annotationRail = document.createElement('div')
    annotationRail.className = 'pdf-annotation-rail'
    const annotationStack = document.createElement('div')
    annotationStack.className = 'pdf-annotation-stack'
    annotationRail.appendChild(annotationStack)
    shell.appendChild(viewer)
    scroller.appendChild(shell)
    body.appendChild(scroller)
    root.replaceChildren(toolbar, body, annotationRail)
    updateAutoPdfEmbedHeight(root, toolbar, scroller, firstViewport, options)

    const scale = calculateFitScale(scroller, firstViewport, options.fit, options.scale)
    const observer = new IntersectionObserver(() => undefined)
    const resizeObserver = new ResizeObserver(() => undefined)

    const state: PdfState = {
      root,
      options,
      pdfjs,
      loadingTask,
      document: pdfDocument,
      releaseDocument: documentHandle.release,
      toolbar,
      scroller,
      shell,
      viewer,
      annotationRail,
      annotationStack,
      controls,
      firstViewport,
      slots: new Map(),
      renderQueue: new Set(),
      textRenderQueue: new Set(),
      preloadQueue: new Set(),
      visiblePages: new Set(),
      observer,
      resizeObserver,
      pageMetrics: [],
      pageMetricsValid: false,
      scale,
      fit: options.fit,
      layoutWidth: scroller.clientWidth,
      currentPage: clampPdfValue(options.page, 1, pdfDocument.numPages),
      draining: false,
      textDraining: false,
      preloadDraining: false,
      destroyed: false,
      renderEpoch: 0,
      scrollFrame: 0,
      resizeFrame: 0,
      pruneTimer: 0,
      lastScrollAt: Date.now(),
      cleanup: [],
      annotations: createPdfAnnotationState(options),
    }
    pdfStates.set(root, state)
    state.observer.disconnect()
    state.resizeObserver.disconnect()
    state.observer = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          const pageNumber = readInteger(entry.target.getAttribute('data-pdf-page-number') ?? '', 0)
          if (pageNumber === 0) continue
          if (entry.isIntersecting) {
            state.visiblePages.add(pageNumber)
            enqueuePage(state, pageNumber)
          } else {
            state.visiblePages.delete(pageNumber)
          }
        }
        cancelDistantRenderWork(state, state.currentPage)
      },
      { root: scroller, rootMargin: options.rootMargin },
    )
    state.resizeObserver = new ResizeObserver(() => {
      if (state.resizeFrame) return
      state.resizeFrame = window.requestAnimationFrame(() => {
        state.resizeFrame = 0
        syncPdfEmbedViewport(state)
      })
    })

    for (let pageNumber = 1; pageNumber <= pdfDocument.numPages; pageNumber++) {
      const slot = createPageSlot(pageNumber, firstViewport, scale)
      state.slots.set(pageNumber, slot)
      viewer.appendChild(slot.element)
      state.observer.observe(slot.element)
    }

    state.resizeObserver.observe(root)
    setupPdfEvents(state)
    if (selectedPdfRoot === root) {
      selectPdfState(state)
      state.scroller.focus({ preventScroll: true })
    }
    if (options.annotations && readCommentRoomEnabled()) {
      connectPdfAnnotations(state)
    }
    updateControls(state)
    root.dataset.pdfStatus = 'loaded'
    scrollToPage(state, state.currentPage)
    renderPdfAnnotations(state)
    if (options.preload) enqueuePreloadPages(state)
    documentHandle = undefined
  } finally {
    documentHandle?.release()
  }
}

function cleanupPdfEmbed(root: HTMLElement) {
  pendingPdfMounts.delete(root)
  const state = pdfStates.get(root)
  if (!state) {
    root.removeAttribute('data-pdf-status')
    return
  }
  state.destroyed = true
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

function pdfEmbedRoots(root: ParentNode, selector: string): HTMLElement[] {
  const roots = Array.from(root.querySelectorAll<HTMLElement>(selector))
  if (root instanceof HTMLElement && root.matches(selector)) roots.unshift(root)
  return roots
}

function mountPdfEmbeds(root: ParentNode = document) {
  const roots = pdfEmbedRoots(root, PDF_EMBED_SELECTOR)
  for (const root of roots) {
    if (pendingPdfMounts.has(root)) continue
    root.tabIndex = 0
    pendingPdfMounts.add(root)
    schedulePdfStartup(() => {
      pendingPdfMounts.delete(root)
      if (!root.isConnected || !root.matches(PDF_EMBED_SELECTOR)) return
      const options = readPdfOptions(root)
      if (!options) return
      void mountPdfEmbed(root, options).catch(error => {
        if (!root.isConnected) return
        console.error(error)
        renderPdfError(root, options)
        root.dataset.pdfStatus = 'error'
      })
    })
  }
}

function cleanupPdfEmbeds(root: ParentNode = document) {
  pdfEmbedRoots(root, PDF_EMBED_CLEANUP_SELECTOR).forEach(cleanupPdfEmbed)
}

const pdfEmbeds: QuartzPdfEmbeds = {
  mount: mountPdfEmbeds,
  cleanup: cleanupPdfEmbeds,
  preload: preloadPdfRuntime,
}
window.quartzPdfEmbeds = pdfEmbeds

document.addEventListener('nav', () => {
  const onPdfShellPointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return
    if (targetAcceptsTextInput(event.target)) return
    if (!(event.target instanceof Element)) return
    const root = event.target.closest('.pdf-embed[data-pdf-src]')
    if (!(root instanceof HTMLElement)) return
    selectPdfRoot(root)
    const state = pdfStates.get(root)
    if (state) {
      state.scroller.focus({ preventScroll: true })
      return
    }
    root.focus({ preventScroll: true })
  }
  const onPdfShellKeyDown = (event: KeyboardEvent) => {
    if (event.key !== 'Escape') return
    if (!selectedPdfRoot || selectedPdfState) return
    consumePdfNavigationKey(event)
    clearSelectedPdfRoot(selectedPdfRoot)
  }
  document.addEventListener('pointerdown', onPdfShellPointerDown, true)
  document.addEventListener('keydown', onPdfShellKeyDown, true)
  mountPdfEmbeds(document)
  window.addCleanup(() => {
    document.removeEventListener('pointerdown', onPdfShellPointerDown, true)
    document.removeEventListener('keydown', onPdfShellKeyDown, true)
    cleanupPdfEmbeds(document)
  })
})
