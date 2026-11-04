import { useEffect, useRef, useState } from 'preact/hooks'
import type { MarkBook } from '../book'
import { parsePdfFragment, type PdfMark } from '../../../util/pdf-marks'
import { fetchMarks } from '../marks'
import { loadPosition, savePosition } from '../store'
import { PdfViewer } from '../viewer'
import { flattenOutline, describeLoadFailure } from './utils'

interface ReaderRef<T> {
  current: T | null
}

interface ReaderLifecycleOptions {
  pdfUrl: string
  slug: string
  docId: string
  signal: AbortSignal
  book: MarkBook
  rootRef: ReaderRef<HTMLDivElement>
  scrollerRef: ReaderRef<HTMLDivElement>
  columnRef: ReaderRef<HTMLDivElement>
}

export function useReaderLifecycle({
  pdfUrl,
  slug,
  docId,
  signal,
  book,
  rootRef,
  scrollerRef,
  columnRef,
}: ReaderLifecycleOptions) {
  const viewerRef = useRef<PdfViewer | null>(null)
  const staleRef = useRef<PdfMark[]>([])
  const [, setBookVersion] = useState(0)
  const [phase, setPhase] = useState<'loading' | 'ready' | 'error'>('loading')
  const [failure, setFailure] = useState('')
  const [marksState, setMarksState] = useState<'loading' | 'ready' | 'error'>('loading')
  const [page, setPage] = useState(1)
  const [pageCount, setPageCount] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [fit, setFit] = useState<'width' | 'custom'>('width')
  const [layoutTick, setLayoutTick] = useState(0)
  const [viewportTick, setViewportTick] = useState(0)
  const [outline, setOutline] = useState<ReturnType<typeof flattenOutline>>([])
  const [labels, setLabels] = useState<string[] | null>(null)
  const [extractedTitle, setExtractedTitle] = useState<string | null>(null)
  const [pageInput, setPageInput] = useState('1')

  useEffect(() => book.subscribe(() => setBookVersion(book.version)), [book])

  useEffect(() => {
    const scroller = scrollerRef.current
    const column = columnRef.current
    const root = rootRef.current
    if (!scroller || !column || !root) return
    let layoutFrame = 0
    let viewportFrame = 0
    const viewer = new PdfViewer(
      pdfUrl,
      scroller,
      column,
      () => {
        const style = getComputedStyle(column)
        const padding =
          Number.parseFloat(style.paddingInlineStart) + Number.parseFloat(style.paddingInlineEnd)
        const margin = root.querySelector<HTMLElement>('.pdf-reader-side')
        const side = margin && getComputedStyle(margin).display !== 'none' ? margin.offsetWidth : 0
        return scroller.clientWidth - side - padding
      },
      {
        onLayout() {
          window.cancelAnimationFrame(layoutFrame)
          layoutFrame = window.requestAnimationFrame(() => {
            setLayoutTick(tick => tick + 1)
            setZoom(viewer.zoom)
            setFit(viewer.fit)
          })
        },
        onPageChange: setPage,
        onPageViewport() {
          if (viewportFrame) return
          viewportFrame = window.requestAnimationFrame(() => {
            viewportFrame = 0
            setViewportTick(tick => tick + 1)
          })
        },
        onTextLayer() {},
      },
    )
    viewerRef.current = viewer
    viewer
      .load()
      .then(async () => {
        if (signal.aborted) return
        setPageCount(viewer.pageCount)
        setZoom(viewer.zoom)
        setPhase('ready')
        const { page: hashPage } = parsePdfFragment(location.hash)
        if (hashPage) {
          viewer.scrollToPage(hashPage)
        } else if (!parsePdfFragment(location.hash).markId) {
          const position = await loadPosition(docId).catch(() => null)
          if (position && !signal.aborted) viewer.scrollToPage(position.page, position.top)
        }
        void viewer.outline().then(nodes => !signal.aborted && setOutline(flattenOutline(nodes)))
        void viewer.pageLabels().then(value => !signal.aborted && setLabels(value))
        void viewer.documentTitle().then(value => !signal.aborted && setExtractedTitle(value))
      })
      .catch(async error => {
        if (signal.aborted) return
        console.error(error)
        setPhase('error')
        const message = error instanceof Error ? error.message : String(error)
        setFailure(message)
        const described = await describeLoadFailure(pdfUrl, message)
        if (!signal.aborted) setFailure(described)
      })
    return () => {
      window.cancelAnimationFrame(layoutFrame)
      window.cancelAnimationFrame(viewportFrame)
      viewer.destroy()
      viewerRef.current = null
    }
  }, [pdfUrl])

  useEffect(() => {
    fetchMarks(slug, signal)
      .then(async response => {
        if (signal.aborted) return
        if (response.canWrite) staleRef.current = response.stale
        await book.load(response)
        setMarksState('ready')
      })
      .catch(error => {
        if (signal.aborted) return
        console.error(error)
        setMarksState('error')
      })
    const online = () => book.retry()
    window.addEventListener('online', online)
    return () => {
      window.removeEventListener('online', online)
      book.flush()
    }
  }, [book])

  useEffect(() => {
    if (phase !== 'ready') return
    const timer = window.setTimeout(() => {
      const viewer = viewerRef.current
      const scroller = scrollerRef.current
      const element = viewer?.pages[page - 1]?.element
      if (!viewer || !scroller || !element) return
      const top = (scroller.scrollTop - element.offsetTop) / Math.max(1, element.offsetHeight)
      void savePosition(docId, { page, top: Math.min(1, Math.max(0, top)) }).catch(() => undefined)
    }, 800)
    return () => window.clearTimeout(timer)
  }, [page, phase, docId])

  useEffect(() => setPageInput(labels?.[page - 1] ?? String(page)), [page, labels])

  return {
    viewerRef,
    staleRef,
    phase,
    failure,
    marksState,
    page,
    pageCount,
    zoom,
    fit,
    layoutTick,
    viewportTick,
    outline,
    labels,
    extractedTitle,
    pageInput,
    setPageInput,
  }
}
