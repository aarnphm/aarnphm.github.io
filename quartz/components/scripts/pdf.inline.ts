import {
  connectPdfAnnotations,
  renderPdfAnnotations,
  syncPdfEmbedViewport,
} from './pdf-common/annotation-presentation'
import { createPdfAnnotationState, readCommentRoomEnabled } from './pdf-common/annotation-records'
import { loadPdfDocument, loadPdfJs, preloadPdfRuntime } from './pdf-common/document-runtime'
import {
  calculateFitScale,
  createPageSlot,
  updateAutoPdfEmbedHeight,
  updateControls,
} from './pdf-common/page-layout'
import {
  cancelDistantRenderWork,
  enqueuePage,
  enqueuePreloadPages,
  scrollToPage,
} from './pdf-common/page-rendering'
import { createControls, resetControls, setPageCount } from './pdf-common/viewer-chrome'
import {
  clampPdfValue,
  readInteger,
  readPdfOptions,
  type PdfControls,
  type PdfDocumentHandle,
  type PdfOptions,
  type PdfState,
} from './pdf-common/viewer-model'
import {
  cleanupPdfEmbed,
  clearSelectedPdfRoot,
  consumePdfNavigationKey,
  pdfStates,
  pendingPdfMounts,
  selectedPdfRoot,
  selectedPdfState,
  selectPdfRoot,
  selectPdfState,
  setupPdfEvents,
  targetAcceptsTextInput,
} from './pdf-common/viewer-session'
import { setStatusText } from './status-flap'

const PDF_EMBED_SELECTOR = '.pdf-embed[data-pdf-src]:not([data-pdf-status])'
const PDF_EMBED_CLEANUP_SELECTOR = '.pdf-embed[data-pdf-src], .pdf-embed[data-pdf-status]'

interface QuartzPdfEmbeds {
  mount(root?: ParentNode): void
  cleanup(root?: ParentNode): void
  preload(src?: string): Promise<void>
}

interface PdfChrome {
  toolbar: HTMLDivElement
  controls: PdfControls
  body: HTMLDivElement
  scroller: HTMLDivElement
  shell: HTMLDivElement
  viewer: HTMLDivElement
  annotationRail: HTMLDivElement
  annotationStack: HTMLDivElement
}

// A cleaned embed remounted with a new source keeps its chrome, so only the pages reload.
const pdfChromes = new WeakMap<HTMLElement, PdfChrome>()

function createPdfChrome(options: PdfOptions): PdfChrome {
  const { toolbar, controls } = createControls(options)
  const status = document.createElement('span')
  status.className = 'pdf-embed-loading'
  status.setAttribute('role', 'status')
  toolbar.append(status)
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
  return { toolbar, controls, body, scroller, shell, viewer, annotationRail, annotationStack }
}

function reusePdfChrome(root: HTMLElement, options: PdfOptions): PdfChrome | null {
  const chrome = pdfChromes.get(root)
  if (!chrome || chrome.toolbar.parentElement !== root || chrome.body.parentElement !== root) {
    return null
  }
  // The previous pages stay until the next document's slots replace them in one swap.
  if (chrome.scroller.parentElement !== chrome.body) chrome.body.replaceChildren(chrome.scroller)
  resetControls(chrome.toolbar, chrome.controls, options)
  chrome.annotationStack.replaceChildren()
  return chrome
}

function schedulePdfStartup(callback: () => void) {
  window.requestAnimationFrame(() => {
    window.setTimeout(callback, 0)
  })
}

function renderPdfError(root: HTMLElement, options: PdfOptions) {
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
  // The toolbar stays, so a failed document keeps the frame of a loaded one.
  const chrome = pdfChromes.get(root)
  if (chrome && chrome.toolbar.parentElement === root && chrome.body.parentElement === root) {
    const status = chrome.toolbar.querySelector<HTMLElement>('.pdf-embed-loading')
    if (status) setStatusText(status, '')
    chrome.body.replaceChildren(error)
    return
  }
  root.replaceChildren(error)
}

async function mountPdfEmbed(root: HTMLElement, options: PdfOptions, events: AbortController) {
  root.dataset.pdfStatus = 'loading'
  let chrome = reusePdfChrome(root, options)
  if (!chrome) {
    chrome = createPdfChrome(options)
    root.replaceChildren(chrome.toolbar, chrome.body, chrome.annotationRail)
    pdfChromes.set(root, chrome)
    resetControls(chrome.toolbar, chrome.controls, options)
  }
  const loading = chrome.toolbar.querySelector<HTMLElement>('.pdf-embed-loading')
  if (loading) setStatusText(loading, 'loading PDF…')
  if (options.height) root.style.setProperty('--pdf-embed-height', `${options.height}px`)

  let documentHandle: PdfDocumentHandle | undefined
  try {
    const pdfjs = await loadPdfJs()
    if (events.signal.aborted || !root.isConnected || root.dataset.pdfStatus !== 'loading') return
    documentHandle = await loadPdfDocument(pdfjs, options.src, events.signal)
    const { loadingTask, document: pdfDocument } = documentHandle
    if (events.signal.aborted || !root.isConnected || root.dataset.pdfStatus !== 'loading') return
    const firstPage = await pdfDocument.getPage(1)
    if (events.signal.aborted || !root.isConnected || root.dataset.pdfStatus !== 'loading') return
    const firstViewport = firstPage.getViewport({ scale: 1 })

    chrome.viewer.replaceChildren()
    const { toolbar, controls, scroller, shell, viewer, annotationRail, annotationStack } = chrome
    setPageCount(controls, pdfDocument.numPages)
    updateAutoPdfEmbedHeight(root, toolbar, scroller, firstViewport, options)

    const scale = calculateFitScale(scroller, firstViewport, options.fit, options.scale)
    const observer = new IntersectionObserver(() => undefined)
    const resizeObserver = new ResizeObserver(() => undefined)

    const state: PdfState = {
      events,
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
    const status = toolbar.querySelector<HTMLElement>('.pdf-embed-loading')
    if (status) setStatusText(status, '')
    scrollToPage(state, state.currentPage)
    renderPdfAnnotations(state)
    if (options.preload) enqueuePreloadPages(state)
    documentHandle = undefined
  } finally {
    documentHandle?.release()
  }
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
    const events = new AbortController()
    pendingPdfMounts.set(root, events)
    schedulePdfStartup(() => {
      if (events.signal.aborted) return
      if (!root.isConnected || !root.matches(PDF_EMBED_SELECTOR)) {
        pendingPdfMounts.delete(root)
        return
      }
      const options = readPdfOptions(root)
      if (!options) {
        pendingPdfMounts.delete(root)
        return
      }
      void mountPdfEmbed(root, options, events)
        .catch(error => {
          if (events.signal.aborted || !root.isConnected) return
          console.error(error)
          renderPdfError(root, options)
          root.dataset.pdfStatus = 'error'
        })
        .finally(() => {
          if (pendingPdfMounts.get(root) === events) pendingPdfMounts.delete(root)
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
