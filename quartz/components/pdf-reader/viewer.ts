import type { ViewportLike } from './geometry'

const PDFJS_SRC = '/static/pdfjs/pdf.min.mjs'
const PDFJS_WORKER_SRC = '/static/pdfjs/pdf.worker.min.mjs'
/** PDF.js user space is 72 dpi; CSS is 96 dpi, so zoom 1 is the printed size. */
const CSS_UNITS = 96 / 72
const MIN_ZOOM = 0.25
const MAX_ZOOM = 4
const MAX_PIXEL_RATIO = 2
const MAX_CANVAS_PIXELS = 16_777_216
const MAX_RENDERED_PAGES = 12
const VIEWPORT_BATCH = 16
const TEXT_SETTLE_MS = 160
const READING_LINE = 0.3

interface PdfViewport extends ViewportLike {
  scale: number
  rotation: number
  userUnit?: number
}

interface PdfRenderTask {
  promise: Promise<void>
  cancel(): void
}

interface PdfTextItem {
  str: string
  transform: number[]
}

interface PdfTextContent {
  items: PdfTextItem[]
  styles: Record<string, unknown>
}

interface PdfAnnotation {
  subtype?: string
  rect?: number[]
  url?: string
  unsafeUrl?: string
  dest?: unknown
}

interface PdfPageProxy {
  getViewport(options: { scale: number }): PdfViewport
  getTextContent(): Promise<PdfTextContent>
  getAnnotations(options: { intent: string }): Promise<PdfAnnotation[]>
  render(options: {
    canvasContext: CanvasRenderingContext2D
    viewport: PdfViewport
    transform: number[]
  }): PdfRenderTask
  cleanup(): void
}

export interface PdfOutlineNode {
  title: string
  dest: unknown
  url?: string | null
  items: PdfOutlineNode[]
}

interface PdfDocumentProxy {
  numPages: number
  getPage(pageNumber: number): Promise<PdfPageProxy>
  getOutline(): Promise<PdfOutlineNode[] | null>
  getDestination(id: string): Promise<unknown[] | null>
  getPageIndex(ref: unknown): Promise<number>
  getPageLabels(): Promise<string[] | null>
  getMetadata(): Promise<{
    info?: Record<string, unknown>
    metadata?: { get(name: string): unknown } | null
  }>
}

interface PdfLoadingTask {
  promise: Promise<PdfDocumentProxy>
  destroy(): Promise<void>
}

interface PdfTextLayer {
  render(): Promise<void>
  cancel(): void
}

interface PdfJsModule {
  getDocument(options: Record<string, unknown>): PdfLoadingTask
  TextLayer: new (options: {
    textContentSource: PdfTextContent
    container: HTMLElement
    viewport: PdfViewport
  }) => PdfTextLayer
  GlobalWorkerOptions: { workerSrc: string }
}

export interface ViewerPage {
  number: number
  element: HTMLDivElement
  canvas: HTMLCanvasElement
  markLayer: HTMLDivElement
  textLayer: HTMLDivElement
  linkLayer: HTMLDivElement
  base: PdfViewport | null
  proxy?: PdfPageProxy
  renderedZoom?: number
  renderingZoom?: number
  renderTask?: PdfRenderTask
  textZoom?: number
  textTask?: PdfTextLayer
  textPromise?: Promise<void>
  linksLoaded: boolean
}

export interface ViewerEvents {
  /** Page sizes or zoom changed; anything positioned against pages must re-measure. */
  onLayout(): void
  onPageChange(page: number): void
  /** A page's own viewport arrived; marks on it can now be painted. */
  onPageViewport(page: number): void
  onTextLayer(page: number): void
}

export interface ResolvedDestination {
  page: number
  /** Fraction of the page height, when the destination names a vertical position. */
  top?: number
}

let pdfJs: Promise<PdfJsModule> | undefined

function loadPdfJs(): Promise<PdfJsModule> {
  if (Reflect.get(window, 'activeWindow') !== window) {
    Object.defineProperty(window, 'activeWindow', {
      value: window,
      configurable: true,
      writable: true,
    })
  }
  pdfJs ??= import(PDFJS_SRC).then((module: PdfJsModule) => {
    module.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_SRC
    return module
  })
  return pdfJs
}

function isCancel(error: unknown): boolean {
  const message = error instanceof Error ? `${error.name} ${error.message}` : String(error)
  return message.includes('RenderingCancelledException') || message.includes('AbortException')
}

function idle(callback: () => void) {
  const requestIdle = Reflect.get(window, 'requestIdleCallback')
  if (typeof requestIdle === 'function') {
    Reflect.apply(requestIdle, window, [callback, { timeout: 800 }])
  } else {
    window.setTimeout(callback, 32)
  }
}

function clampZoom(value: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value))
}

function cleanTitle(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const title = value.replace(/\s+/g, ' ').trim()
  if (title.length < 4 || title.length > 200 || /\.(pdf|docx?|tex|dvi)$/i.test(title)) return null
  if (/^(untitled|microsoft word|title)\b/i.test(title)) return null
  return title
}

function firstPageTitle(content: PdfTextContent, viewport: PdfViewport): string | null {
  const text = content.items
    .filter(item => item.str.trim() && item.transform.length === 6)
    .map(item => {
      const [left, top] = viewport.convertToViewportPoint(item.transform[4], item.transform[5])
      const [endLeft, endTop] = viewport.convertToViewportPoint(
        item.transform[4] + item.transform[0],
        item.transform[5] + item.transform[1],
      )
      return {
        text: item.str.trim(),
        left,
        top,
        size: Math.hypot(item.transform[2], item.transform[3]),
        horizontal: Math.abs(endTop - top) <= Math.abs(endLeft - left) * 0.1,
      }
    })
    .filter(item => item.horizontal && Number.isFinite(item.size) && item.size > 0)
  const sizes = text.map(item => item.size).sort((a, b) => a - b)
  const bodySize = sizes[Math.floor(sizes.length / 2)]
  const heading = text.filter(item => item.top > 0 && item.top < viewport.height / 2)
  const largest = Math.max(0, ...heading.map(item => item.size))
  // Plain body text is too ambiguous; keep the named document when no heading stands out.
  if (!bodySize || largest < bodySize * 1.15) return null
  const prominent = heading
    .filter(item => item.size >= largest * 0.9)
    .sort((a, b) => (Math.abs(a.top - b.top) < largest / 4 ? a.left - b.left : a.top - b.top))
  const first = prominent[0]
  if (!first) return null
  const parts = [first.text]
  let bottom = first.top
  for (const item of prominent.slice(1)) {
    // Wrapped title lines stay close; the gap before authors starts a separate block.
    if (item.top - bottom > largest * 1.6) break
    parts.push(item.text)
    bottom = Math.max(bottom, item.top)
  }
  const title = cleanTitle(parts.join(' '))
  return title &&
    title.split(/\s+/).length > 1 &&
    !/https?:\/\/|^(abstract|contents)\b/i.test(title)
    ? title
    : null
}

export class PdfViewer {
  readonly pages: ViewerPage[] = []
  zoom = 1
  fit: 'width' | 'custom' = 'width'
  currentPage = 1
  pageCount = 0

  private module?: PdfJsModule
  private task?: PdfLoadingTask
  private document?: PdfDocumentProxy
  private renderObserver?: IntersectionObserver
  private textObserver?: IntersectionObserver
  private resizeObserver?: ResizeObserver
  private readonly nearPages = new Set<number>()
  private readonly visiblePages = new Set<number>()
  private readonly renderQueue = new Set<number>()
  private readonly renderedOrder: number[] = []
  private draining = false
  private textTimer = 0
  private scrollFrame = 0
  private destroyed = false
  private lastWidth = 0

  constructor(
    private readonly src: string,
    private readonly scroller: HTMLElement,
    private readonly column: HTMLElement,
    private readonly availableWidth: () => number,
    private readonly events: ViewerEvents,
  ) {}

  async load(): Promise<void> {
    this.module = await loadPdfJs()
    if (this.destroyed) return
    this.task = this.module.getDocument({
      url: this.src,
      cMapUrl: '/static/pdfjs/cmaps/',
      cMapPacked: true,
      standardFontDataUrl: '/static/pdfjs/standard_fonts/',
      wasmUrl: '/static/pdfjs/wasm/',
    })
    this.document = await this.task.promise
    if (this.destroyed) return
    this.pageCount = this.document.numPages

    const first = await this.document.getPage(1)
    const firstBase = first.getViewport({ scale: 1 })
    for (let number = 1; number <= this.pageCount; number++) {
      const page = this.createPage(number)
      if (number === 1) {
        page.proxy = first
        page.base = firstBase
      }
      this.pages.push(page)
      this.column.append(page.element)
    }
    this.zoom = this.fitZoom()
    this.layoutPages()

    this.renderObserver = new IntersectionObserver(entries => this.onRenderIntersect(entries), {
      root: this.scroller,
      rootMargin: '1200px 0px',
    })
    this.textObserver = new IntersectionObserver(entries => this.onTextIntersect(entries), {
      root: this.scroller,
      rootMargin: '240px 0px',
    })
    for (const page of this.pages) {
      this.renderObserver.observe(page.element)
      this.textObserver.observe(page.element)
    }
    this.resizeObserver = new ResizeObserver(() => this.onResize())
    this.resizeObserver.observe(this.scroller)
    this.lastWidth = this.scroller.clientWidth
    this.scroller.addEventListener('scroll', this.onScroll, { passive: true })
    this.events.onPageViewport(1)
    this.fetchViewports(2)
  }

  destroy() {
    this.destroyed = true
    this.renderObserver?.disconnect()
    this.textObserver?.disconnect()
    this.resizeObserver?.disconnect()
    this.scroller.removeEventListener('scroll', this.onScroll)
    window.cancelAnimationFrame(this.scrollFrame)
    window.clearTimeout(this.textTimer)
    for (const page of this.pages) {
      page.renderTask?.cancel()
      page.textTask?.cancel()
    }
    void this.task?.destroy().catch(() => undefined)
  }

  private createPage(number: number): ViewerPage {
    const element = document.createElement('div')
    element.className = 'pdf-reader-page'
    element.dataset.pageNumber = String(number)
    element.setAttribute('role', 'region')
    element.setAttribute('aria-label', `Page ${number}`)

    const canvas = document.createElement('canvas')
    canvas.className = 'pdf-reader-canvas'
    canvas.setAttribute('aria-hidden', 'true')
    const markLayer = document.createElement('div')
    markLayer.className = 'pdf-reader-marks'
    markLayer.setAttribute('aria-hidden', 'true')
    const textLayer = document.createElement('div')
    textLayer.className = 'pdf-reader-text textLayer'
    const linkLayer = document.createElement('div')
    linkLayer.className = 'pdf-reader-links'

    element.append(canvas, markLayer, textLayer, linkLayer)
    return {
      number,
      element,
      canvas,
      markLayer,
      textLayer,
      linkLayer,
      base: null,
      linksLoaded: false,
    }
  }

  private defaultBase(): PdfViewport | null {
    return this.pages[this.currentPage - 1]?.base ?? this.pages[0]?.base ?? null
  }

  private fitZoom(): number {
    const base = this.defaultBase()
    if (!base) return 1
    return clampZoom(Math.max(1, this.availableWidth()) / (base.width * CSS_UNITS))
  }

  private sizePage(page: ViewerPage) {
    const base = page.base ?? this.defaultBase()
    if (!base) return
    const scale = this.zoom * CSS_UNITS
    const width = Math.round(base.width * scale)
    page.element.style.width = `${width}px`
    page.element.style.height = `${Math.round(base.height * scale)}px`
    page.element.toggleAttribute(
      'data-fits-width',
      this.fit === 'width' && width <= Math.round(this.availableWidth()),
    )
    page.element.style.setProperty('--total-scale-factor', String(scale * (base.userUnit ?? 1)))
  }

  /** Keeps the reading position while page boxes change size underneath it. */
  private preservingAnchor(change: () => void) {
    const anchor = this.pages[this.currentPage - 1]
    const before = anchor
      ? (this.scroller.scrollTop - anchor.element.offsetTop) /
        Math.max(1, anchor.element.offsetHeight)
      : 0
    change()
    if (anchor) {
      this.scroller.scrollTop = anchor.element.offsetTop + before * anchor.element.offsetHeight
    }
  }

  private layoutPages() {
    for (const page of this.pages) {
      this.sizePage(page)
      if (page.textZoom !== this.zoom) page.textLayer.hidden = true
    }
    this.events.onLayout()
  }

  setZoom(value: number | 'width') {
    const next = value === 'width' ? this.fitZoom() : clampZoom(value)
    const fit = value === 'width' ? 'width' : 'custom'
    if (Math.abs(next - this.zoom) < 0.001 && this.fit === fit) return
    this.fit = fit
    this.preservingAnchor(() => {
      this.zoom = next
      this.layoutPages()
    })
    this.enqueueNear()
    this.scheduleText()
  }

  zoomBy(factor: number) {
    this.setZoom(Math.round(this.zoom * factor * 100) / 100)
  }

  private onResize() {
    const width = this.scroller.clientWidth
    if (width === this.lastWidth) return
    this.lastWidth = width
    if (this.fit === 'width') {
      const next = this.fitZoom()
      if (Math.abs(next - this.zoom) < 0.001) return
      this.preservingAnchor(() => {
        this.zoom = next
        this.layoutPages()
      })
      this.enqueueNear()
      this.scheduleText()
    }
  }

  private fetchViewports(from: number) {
    if (this.destroyed || !this.document || from > this.pageCount) return
    idle(() => {
      const document = this.document
      if (this.destroyed || !document) return
      const to = Math.min(this.pageCount, from + VIEWPORT_BATCH - 1)
      const pending: Promise<void>[] = []
      for (let number = from; number <= to; number++) {
        const page = this.pages[number - 1]
        if (page.base) continue
        pending.push(
          document.getPage(number).then(proxy => {
            page.proxy = proxy
            page.base = proxy.getViewport({ scale: 1 })
          }),
        )
      }
      void Promise.all(pending)
        .then(() => {
          if (this.destroyed) return
          let resized = false
          this.preservingAnchor(() => {
            for (let number = from; number <= to; number++) {
              const page = this.pages[number - 1]
              const width = page.element.style.width
              const height = page.element.style.height
              this.sizePage(page)
              resized ||= width !== page.element.style.width || height !== page.element.style.height
            }
          })
          if (resized) this.events.onLayout()
          for (let number = from; number <= to; number++) this.events.onPageViewport(number)
          this.fetchViewports(to + 1)
        })
        .catch(error => {
          if (!this.destroyed) console.error(error)
        })
    })
  }

  private async proxyFor(page: ViewerPage): Promise<PdfPageProxy | null> {
    if (page.proxy) return page.proxy
    if (!this.document) return null
    page.proxy = await this.document.getPage(page.number)
    if (!page.base) {
      page.base = page.proxy.getViewport({ scale: 1 })
      this.preservingAnchor(() => this.sizePage(page))
      this.events.onLayout()
      this.events.onPageViewport(page.number)
    }
    return page.proxy
  }

  private onRenderIntersect(entries: IntersectionObserverEntry[]) {
    for (const entry of entries) {
      const number = Number((entry.target as HTMLElement).dataset.pageNumber)
      if (entry.isIntersecting) this.nearPages.add(number)
      else this.nearPages.delete(number)
    }
    this.enqueueNear()
  }

  private onTextIntersect(entries: IntersectionObserverEntry[]) {
    for (const entry of entries) {
      const number = Number((entry.target as HTMLElement).dataset.pageNumber)
      if (entry.isIntersecting) this.visiblePages.add(number)
      else this.visiblePages.delete(number)
    }
    this.scheduleText()
  }

  private enqueueNear() {
    for (const number of this.nearPages) this.renderQueue.add(number)
    void this.drain()
  }

  private async drain() {
    if (this.draining) return
    this.draining = true
    try {
      while (!this.destroyed && this.renderQueue.size > 0) {
        let next = 0
        let distance = Infinity
        for (const number of this.renderQueue) {
          const gap = Math.abs(number - this.currentPage)
          if (gap < distance) {
            distance = gap
            next = number
          }
        }
        this.renderQueue.delete(next)
        if (!this.nearPages.has(next)) continue
        try {
          await this.renderPage(this.pages[next - 1])
        } catch (error) {
          if (!isCancel(error)) console.error(error)
        }
      }
    } finally {
      this.draining = false
    }
  }

  private async renderPage(page: ViewerPage) {
    const zoom = this.zoom
    if (page.renderedZoom === zoom || page.renderingZoom === zoom) return
    page.renderTask?.cancel()
    page.renderingZoom = zoom
    const proxy = await this.proxyFor(page)
    if (!proxy || this.destroyed || this.zoom !== zoom) {
      page.renderingZoom = undefined
      return
    }
    const viewport = proxy.getViewport({ scale: zoom * CSS_UNITS })
    let ratio = Math.min(MAX_PIXEL_RATIO, Math.max(1, window.devicePixelRatio || 1))
    const pixels = viewport.width * viewport.height * ratio * ratio
    if (pixels > MAX_CANVAS_PIXELS) ratio *= Math.sqrt(MAX_CANVAS_PIXELS / pixels)

    // Render offscreen so the stretched previous frame stays up until the new one is ready.
    const canvas = document.createElement('canvas')
    canvas.width = Math.floor(viewport.width * ratio)
    canvas.height = Math.floor(viewport.height * ratio)
    const context = canvas.getContext('2d')
    if (!context) return
    const task = proxy.render({
      canvasContext: context,
      viewport,
      transform: [ratio, 0, 0, ratio, 0, 0],
    })
    page.renderTask = task
    try {
      await task.promise
    } finally {
      if (page.renderTask === task) page.renderTask = undefined
      page.renderingZoom = undefined
    }
    if (this.destroyed || this.zoom !== zoom) return
    canvas.className = page.canvas.className
    canvas.setAttribute('aria-hidden', 'true')
    page.canvas.replaceWith(canvas)
    page.canvas = canvas
    page.renderedZoom = zoom
    this.trackRendered(page.number)
    if (!page.linksLoaded) void this.loadLinks(page, proxy)
  }

  private trackRendered(number: number) {
    const at = this.renderedOrder.indexOf(number)
    if (at >= 0) this.renderedOrder.splice(at, 1)
    this.renderedOrder.push(number)
    while (this.renderedOrder.length > MAX_RENDERED_PAGES) {
      const evicted = this.renderedOrder.findIndex(candidate => !this.nearPages.has(candidate))
      if (evicted < 0) break
      const [number] = this.renderedOrder.splice(evicted, 1)
      const page = this.pages[number - 1]
      page.canvas.width = 0
      page.canvas.height = 0
      page.renderedZoom = undefined
    }
  }

  private scheduleText() {
    window.clearTimeout(this.textTimer)
    this.textTimer = window.setTimeout(() => {
      for (const number of this.visiblePages) void this.ensureTextLayer(number)
    }, TEXT_SETTLE_MS)
  }

  /** Renders the page's text layer at the current zoom; re-anchoring calls it for offscreen pages. */
  async ensureTextLayer(number: number): Promise<HTMLElement | null> {
    const page = this.pages[number - 1]
    if (!page || !this.module) return null
    const zoom = this.zoom
    if (page.textZoom === zoom && !page.textLayer.hidden) return page.textLayer
    if (page.textPromise) {
      await page.textPromise
      return page.textZoom === this.zoom ? page.textLayer : this.ensureTextLayer(number)
    }
    const module = this.module
    page.textPromise = (async () => {
      const proxy = await this.proxyFor(page)
      if (!proxy || this.destroyed) return
      const content = await proxy.getTextContent()
      if (this.destroyed || this.zoom !== zoom) return
      page.textTask?.cancel()
      page.textLayer.replaceChildren()
      const layer = new module.TextLayer({
        textContentSource: content,
        container: page.textLayer,
        viewport: proxy.getViewport({ scale: zoom * CSS_UNITS }),
      })
      page.textTask = layer
      await layer.render()
      if (this.destroyed || this.zoom !== zoom) return
      // PDF.js's viewer keeps selections from jumping across gaps with this sentinel.
      const end = document.createElement('div')
      end.className = 'endOfContent'
      page.textLayer.append(end)
      page.textTask = undefined
      page.textZoom = zoom
      page.textLayer.hidden = false
      this.events.onTextLayer(number)
    })()
      .catch(error => {
        if (!isCancel(error)) console.error(error)
      })
      .finally(() => {
        page.textPromise = undefined
      })
    await page.textPromise
    if (page.textZoom === zoom) return page.textLayer
    return this.zoom === zoom ? null : this.ensureTextLayer(number)
  }

  private async loadLinks(page: ViewerPage, proxy: PdfPageProxy) {
    page.linksLoaded = true
    const base = page.base
    if (!base) return
    const annotations = await proxy.getAnnotations({ intent: 'display' }).catch(() => [])
    if (this.destroyed) return
    const links: HTMLAnchorElement[] = []
    for (const annotation of annotations) {
      if (annotation.subtype !== 'Link' || !annotation.rect || annotation.rect.length !== 4)
        continue
      const [x0, y0, x1, y1] = annotation.rect
      const [ax, ay] = base.convertToViewportPoint(x0, y0)
      const [bx, by] = base.convertToViewportPoint(x1, y1)
      const left = Math.min(ax, bx) / base.width
      const top = Math.min(ay, by) / base.height
      const width = Math.abs(bx - ax) / base.width
      const height = Math.abs(by - ay) / base.height
      if (width <= 0 || height <= 0) continue

      const link = document.createElement('a')
      link.style.left = `${left * 100}%`
      link.style.top = `${top * 100}%`
      link.style.width = `${width * 100}%`
      link.style.height = `${height * 100}%`
      const url = annotation.url ?? annotation.unsafeUrl
      if (url && /^https?:/i.test(url)) {
        link.href = url
        link.target = '_blank'
        link.rel = 'noopener noreferrer'
        link.title = url
        link.setAttribute('aria-label', url)
      } else if (annotation.dest) {
        const dest = annotation.dest
        link.href = '#'
        link.dataset.routerIgnore = ''
        link.setAttribute('aria-label', 'Go to linked location in this document')
        link.addEventListener('click', event => {
          event.preventDefault()
          void this.resolveDestination(dest).then(target => {
            if (target) this.scrollToPage(target.page, target.top)
          })
        })
      } else {
        continue
      }
      links.push(link)
    }
    page.linkLayer.replaceChildren(...links)
  }

  async resolveDestination(dest: unknown): Promise<ResolvedDestination | null> {
    const document = this.document
    if (!document) return null
    const explicit = typeof dest === 'string' ? await document.getDestination(dest) : dest
    if (!Array.isArray(explicit) || explicit.length === 0) return null
    const ref = explicit[0]
    const index = typeof ref === 'number' ? ref : await document.getPageIndex(ref).catch(() => -1)
    if (!Number.isInteger(index) || index < 0 || index >= this.pageCount) return null
    const page = this.pages[index]
    await this.proxyFor(page)
    const mode = (explicit[1] as { name?: string } | undefined)?.name
    const y =
      mode === 'XYZ' ? explicit[3] : mode === 'FitH' || mode === 'FitBH' ? explicit[2] : undefined
    if (typeof y !== 'number' || !page.base) return { page: index + 1 }
    const [, top] = page.base.convertToViewportPoint(0, y)
    return { page: index + 1, top: Math.min(1, Math.max(0, top / page.base.height)) }
  }

  /** Scrolls so `top` (a fraction of the page) sits at the reading line. */
  scrollToPage(number: number, top = 0, behavior: ScrollBehavior = 'auto') {
    const page = this.pages[Math.min(this.pageCount, Math.max(1, number)) - 1]
    if (!page) return
    const offset = top > 0 ? this.scroller.clientHeight * READING_LINE : 12
    this.scroller.scrollTo({
      top: Math.max(0, page.element.offsetTop + top * page.element.offsetHeight - offset),
      behavior,
    })
    this.setCurrentPage(page.number)
  }

  private setCurrentPage(number: number) {
    if (number === this.currentPage) return
    this.currentPage = number
    this.events.onPageChange(number)
  }

  private readonly onScroll = () => {
    if (this.scrollFrame) return
    this.scrollFrame = window.requestAnimationFrame(() => {
      this.scrollFrame = 0
      const line = this.scroller.scrollTop + this.scroller.clientHeight * READING_LINE
      let low = 0
      let high = this.pages.length - 1
      while (low < high) {
        const mid = Math.ceil((low + high) / 2)
        if (this.pages[mid].element.offsetTop <= line) low = mid
        else high = mid - 1
      }
      if (this.pages[low]) this.setCurrentPage(this.pages[low].number)
      this.enqueueNear()
      this.scheduleText()
    })
  }

  async outline(): Promise<PdfOutlineNode[]> {
    return (await this.document?.getOutline().catch(() => null)) ?? []
  }

  async pageLabels(): Promise<string[] | null> {
    const labels = await this.document?.getPageLabels().catch(() => null)
    if (!labels || labels.length !== this.pageCount) return null
    // Documents often label pages 1..n anyway; only distinct labels earn screen space.
    return labels.some((label, index) => label !== String(index + 1)) ? labels : null
  }

  async documentTitle(): Promise<string | null> {
    const metadata = await this.document?.getMetadata().catch(() => null)
    const title =
      cleanTitle(metadata?.info?.Title) ?? cleanTitle(metadata?.metadata?.get('dc:title'))
    if (title) return title
    const first = this.pages[0]
    const content = await first?.proxy?.getTextContent().catch(() => null)
    return content && first.base ? firstPageTitle(content, first.base) : null
  }
}
