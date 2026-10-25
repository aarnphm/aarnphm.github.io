import { createPdfEmbed, mountPdfPreview } from '../popover/pdf-preview'
import { highlightPdfPreview } from './pdf-highlights'

const MAX_PREVIEWS = 3
const MAX_CANVAS_BYTES = 96 * 1024 * 1024

interface PreviewEntry {
  inner: HTMLElement
  pdf: HTMLElement
  scrollTop: number
  scrollLeft: number
  observer: MutationObserver
}

class PdfPreviewCache {
  private entries = new Map<string, PreviewEntry>()
  private active: PreviewEntry | undefined

  private destroy(key: string, entry: PreviewEntry) {
    entry.observer.disconnect()
    window.quartzPdfEmbeds?.cleanup(entry.pdf)
    for (const canvas of entry.pdf.querySelectorAll('canvas')) {
      canvas.width = 0
      canvas.height = 0
    }
    entry.inner.remove()
    this.entries.delete(key)
  }

  private trim() {
    let bytes = 0
    for (const entry of this.entries.values()) {
      for (const canvas of entry.pdf.querySelectorAll('canvas')) {
        bytes += canvas.width * canvas.height * 4
      }
    }
    for (const [key, entry] of this.entries) {
      if (this.entries.size <= MAX_PREVIEWS && bytes <= MAX_CANVAS_BYTES) break
      // The visible viewer keeps its normal resolution even when it exceeds the cache budget.
      if (entry === this.active) continue
      for (const canvas of entry.pdf.querySelectorAll('canvas')) {
        bytes -= canvas.width * canvas.height * 4
      }
      this.destroy(key, entry)
    }
  }

  park() {
    const entry = this.active
    if (!entry) return
    const scroller = entry.pdf.querySelector('.pdf-viewer-container')
    entry.scrollTop = scroller?.scrollTop ?? 0
    entry.scrollLeft = scroller?.scrollLeft ?? 0
    entry.inner.remove()
    this.active = undefined
    for (const [key, cached] of this.entries) {
      if (cached.pdf.dataset.pdfStatus !== 'loaded') this.destroy(key, cached)
    }
    this.trim()
  }

  dispose() {
    this.active = undefined
    for (const [key, entry] of this.entries) this.destroy(key, entry)
  }

  async show(
    preview: HTMLElement,
    url: URL,
    title: string,
    signal: AbortSignal,
    page = 1,
    query = '',
  ) {
    this.park()
    const key = `${url.origin}${url.pathname}${url.search}#${page}`
    let entry = this.entries.get(key)
    const reused = entry?.pdf.dataset.pdfStatus === 'loaded'
    if (!entry) {
      const inner = document.createElement('div')
      inner.className = 'preview-inner search-pdf-preview'
      const pdf = createPdfEmbed(url.toString(), title, {
        preload: false,
        annotations: false,
        rootMargin: '0px',
        overscan: 1,
        maxPixelRatio: 2,
        textLayer: true,
      })
      pdf.dataset.pdfPage = String(page)
      inner.append(pdf)
      const observer = new MutationObserver(() => this.trim())
      observer.observe(pdf, {
        attributes: true,
        attributeFilter: ['width', 'height'],
        subtree: true,
      })
      entry = { inner, pdf, scrollTop: 0, scrollLeft: 0, observer }
    }
    this.entries.delete(key)
    this.entries.set(key, entry)
    this.active = entry
    preview.replaceChildren(entry.inner)
    const scroller = entry.pdf.querySelector('.pdf-viewer-container')
    if (scroller) {
      scroller.scrollTop = entry.scrollTop
      scroller.scrollLeft = entry.scrollLeft
    }
    highlightPdfPreview(entry.pdf, url, page, query, signal)
    this.trim()
    if (reused) return
    const pdf = entry.pdf
    await new Promise<void>((resolve, reject) => {
      mountPdfPreview(
        pdf,
        signal,
        resolve,
        () => {
          if (signal.aborted) resolve()
          else reject(new Error('PDF preview could not load.'))
        },
        true,
      )
    })
  }
}

const caches = new WeakMap<HTMLElement, PdfPreviewCache>()

export function parkPdfPreview(preview: HTMLElement) {
  caches.get(preview)?.park()
}

export function disposePdfPreviews(preview: HTMLElement) {
  caches.get(preview)?.dispose()
  caches.delete(preview)
}

export function renderPdfPreview(
  preview: HTMLElement,
  url: URL,
  title: string,
  signal: AbortSignal,
  page?: number,
  query?: string,
) {
  let cache = caches.get(preview)
  if (!cache) {
    cache = new PdfPreviewCache()
    caches.set(preview, cache)
  }
  return cache.show(preview, url, title, signal, page, query)
}
