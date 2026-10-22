import { arenaPdfViewerSource } from '../../../util/arena-embed'

interface PdfEmbedOptions {
  preload?: boolean
  annotations?: boolean
  rootMargin?: string
  overscan?: number
  maxPixelRatio?: number
  textLayer?: boolean
}

const PDF_LOAD_TIMEOUT = 8 * 1000
const PDF_IDLE_DOCUMENT_PRELOAD_LIMIT = 1
export const PDF_PREVIEW_OPTIONS: PdfEmbedOptions = {
  preload: false,
  annotations: false,
  rootMargin: '0px',
  overscan: 0,
  maxPixelRatio: 2,
  textLayer: false,
}
export const PDF_SIDE_PANEL_OPTIONS = PDF_PREVIEW_OPTIONS

let pdfRuntimePreloadScheduled = false

export function cleanArxivIdentifier(identifier: string): string {
  return identifier.replace(/^(arxiv:)?(pdf\/)?/i, '').replace(/\.pdf$/i, '')
}

export function arxivPdfUrl(identifier: string): URL {
  const path = cleanArxivIdentifier(identifier).split('/').map(encodeURIComponent).join('/')
  return new URL(`https://arxiv.org/pdf/${path}`)
}

export function isPdfUrl(url: URL): boolean {
  return url.pathname.toLowerCase().endsWith('.pdf')
}

function pdfRuntimeSource(rawSrc: string): string {
  try {
    const url = new URL(rawSrc, window.location.href)
    if (url.protocol === 'http:' || url.protocol === 'https:') {
      if (url.origin !== window.location.origin) return arenaPdfViewerSource(url.toString())
      return `${url.pathname}${url.search}`
    }
  } catch {
    return rawSrc
  }

  return rawSrc
}

export function pdfTitleFromUrl(rawSrc: string): string {
  try {
    const url = new URL(rawSrc, window.location.href)
    const filename = url.pathname.split('/').filter(Boolean).pop()
    return filename ? decodeURIComponent(filename) : 'document.pdf'
  } catch {
    return 'document.pdf'
  }
}

export function createPdfEmbed(
  src: string,
  title = pdfTitleFromUrl(src),
  options: PdfEmbedOptions = {},
): HTMLDivElement {
  const pdf = document.createElement('div')
  pdf.classList.add('internal-embed', 'pdf-embed', 'pdf-embed-preview')
  pdf.dataset.pdfSrc = pdfRuntimeSource(src)
  pdf.dataset.pdfTitle = title
  pdf.dataset.pdfFit = 'width'
  if (options.preload !== undefined) pdf.dataset.pdfPreload = String(options.preload)
  if (options.annotations !== undefined) {
    pdf.dataset.pdfEnableAnnotations = String(options.annotations)
  }
  if (options.rootMargin !== undefined) pdf.dataset.pdfRootMargin = options.rootMargin
  if (options.overscan !== undefined) pdf.dataset.pdfOverscan = String(options.overscan)
  if (options.maxPixelRatio !== undefined) {
    pdf.dataset.pdfMaxPixelRatio = String(options.maxPixelRatio)
  }
  if (options.textLayer !== undefined) pdf.dataset.pdfTextLayer = String(options.textLayer)
  pdf.tabIndex = 0

  const loading = document.createElement('span')
  loading.className = 'pdf-embed-loading'
  loading.textContent = 'Loading PDF'
  pdf.appendChild(loading)

  return pdf
}

type IdleCallback = (callback: () => void, options?: { timeout: number }) => number

function isIdleCallback(value: unknown): value is IdleCallback {
  return typeof value === 'function'
}

function schedulePdfRuntimePreload(sources: string[]) {
  if (pdfRuntimePreloadScheduled) return
  pdfRuntimePreloadScheduled = true
  const preloadSources = [...new Set(sources)].slice(0, PDF_IDLE_DOCUMENT_PRELOAD_LIMIT)

  const preload = () => {
    const runtime = window.quartzPdfEmbeds
    if (!runtime) {
      pdfRuntimePreloadScheduled = false
      return
    }

    void runtime
      .preload()
      .then(() => {
        for (const source of preloadSources) {
          void runtime.preload(source).catch(console.error)
        }
      })
      .catch(console.error)
  }

  const requestIdle: unknown = Reflect.get(window, 'requestIdleCallback')
  if (isIdleCallback(requestIdle)) {
    requestIdle(preload, { timeout: 1500 })
    return
  }

  window.setTimeout(preload, 250)
}

function pdfPreviewSource(link: HTMLAnchorElement): string | null {
  if (link.dataset.arxivId) return pdfRuntimeSource(arxivPdfUrl(link.dataset.arxivId).toString())

  try {
    const targetUrl = new URL(link.href)
    targetUrl.hash = ''
    targetUrl.search = ''
    return isPdfUrl(targetUrl) ? pdfRuntimeSource(targetUrl.toString()) : null
  } catch {
    return null
  }
}

export function mountPdfPreview(
  pdf: HTMLElement,
  signal: AbortSignal,
  onReady: () => void,
  onFailed: () => void,
) {
  const runtime = window.quartzPdfEmbeds
  if (!runtime) {
    onFailed()
    return
  }
  const pdfRuntime = runtime

  let settled = false
  const timeoutId = window.setTimeout(() => ready(), PDF_LOAD_TIMEOUT)
  const observer = new MutationObserver(() => {
    if (pdf.dataset.pdfStatus === 'loaded') ready()
    if (pdf.dataset.pdfStatus === 'error') failed()
  })

  function cleanup() {
    window.clearTimeout(timeoutId)
    signal.removeEventListener('abort', aborted)
    observer.disconnect()
  }

  function finish(cb: () => void) {
    if (settled) return
    settled = true
    cleanup()
    cb()
  }

  function ready() {
    finish(onReady)
  }

  function failed() {
    finish(onFailed)
  }

  function aborted() {
    pdfRuntime.cleanup(pdf)
    pdf.remove()
    finish(onFailed)
  }

  observer.observe(pdf, { attributes: true, attributeFilter: ['data-pdf-status'] })
  signal.addEventListener('abort', aborted, { once: true })
  pdfRuntime.mount(pdf)

  if (signal.aborted) aborted()
}

export function mountPdfPreviews(root: ParentNode) {
  window.quartzPdfEmbeds?.mount(root)
}

export async function handlePdfContent(
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
) {
  const src = response.url || targetUrl.toString()
  const pdf = createPdfEmbed(src, pdfTitleFromUrl(targetUrl.toString()), PDF_PREVIEW_OPTIONS)
  popoverInner.appendChild(pdf)
}

export function preloadPopoverLinks(container: Document | HTMLElement = document): void {
  const links = ([...container.getElementsByClassName('internal')] as HTMLAnchorElement[]).filter(
    link => !link.closest('#stacked-notes-container'),
  )

  const pdfSources = links.flatMap(link => {
    const source = pdfPreviewSource(link)
    return source ? [source] : []
  })
  if (pdfSources.length > 0) schedulePdfRuntimePreload(pdfSources)
}
