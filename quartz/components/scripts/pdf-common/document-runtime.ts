import {
  normalizePdfSource,
  type PdfDocumentCacheEntry,
  type PdfDocumentHandle,
  type PdfJsRuntime,
  type PdfTextRenderTask,
} from './viewer-model'

const PDFJS_SCRIPT_SRC = '/static/pdfjs/pdf.min.mjs'
const PDFJS_WORKER_SRC = '/static/pdfjs/pdf.worker.min.mjs'

const PDF_DOCUMENT_CACHE_TTL_MS = 60 * 1000
const MAX_IDLE_DOCUMENTS = 2

let pdfJsLoad: Promise<PdfJsRuntime> | undefined
const pdfDocumentCache = new Map<string, PdfDocumentCacheEntry>()

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

function configurePdfJsRuntime(pdfjs: PdfJsRuntime): PdfJsRuntime {
  pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_SRC
  return pdfjs
}

export function loadPdfJs(): Promise<PdfJsRuntime> {
  ensurePdfJsGlobals()
  const existing = readPdfJsRuntime(window.pdfjsLib)
  if (existing) return Promise.resolve(configurePdfJsRuntime(existing))
  if (pdfJsLoad) return pdfJsLoad

  pdfJsLoad = import(PDFJS_SCRIPT_SRC).then(module => {
    const runtime = readPdfJsRuntime(module)
    if (!runtime) throw new Error('PDF.js did not initialize')
    const configured = configurePdfJsRuntime(runtime)
    window.pdfjsLib = configured
    return configured
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
    wasmUrl: '/static/pdfjs/wasm/',
  }
}

function destroyIdleDocument(key: string, entry: PdfDocumentCacheEntry) {
  if (entry.refs > 0 || pdfDocumentCache.get(key) !== entry) return
  window.clearTimeout(entry.destroyTimer)
  pdfDocumentCache.delete(key)
  void entry.loadingTask.destroy().catch(() => undefined)
}

function releasePdfDocument(key: string) {
  const entry = pdfDocumentCache.get(key)
  if (!entry) return
  entry.refs = Math.max(0, entry.refs - 1)
  if (entry.refs > 0 || entry.destroyTimer) return
  pdfDocumentCache.delete(key)
  pdfDocumentCache.set(key, entry)
  entry.destroyTimer = window.setTimeout(
    () => destroyIdleDocument(key, entry),
    PDF_DOCUMENT_CACHE_TTL_MS,
  )
  const idle = [...pdfDocumentCache].filter(([, document]) => document.refs === 0)
  for (const [idleKey, document] of idle.slice(0, -MAX_IDLE_DOCUMENTS)) {
    destroyIdleDocument(idleKey, document)
  }
}

export function loadPdfDocument(pdfjs: PdfJsRuntime, src: string): Promise<PdfDocumentHandle> {
  const key = pdfDocumentCacheKey(src)
  let entry = pdfDocumentCache.get(key)

  if (!entry) {
    const loadingTask = pdfjs.getDocument(pdfDocumentOptions(src))
    const promise = loadingTask.promise.catch(error => {
      pdfDocumentCache.delete(key)
      throw error
    })
    entry = { key, loadingTask, promise, refs: 0, destroyTimer: 0 }
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

export async function preloadPdfRuntime(src?: string): Promise<void> {
  const pdfjs = await loadPdfJs()
  if (!src) return
  const handle = await loadPdfDocument(pdfjs, src)
  handle.release()
}
