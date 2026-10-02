import { render } from 'preact'
import { useCallback, useEffect, useMemo, useRef, useState } from 'preact/hooks'
import {
  parsePdfDocumentRecord,
  parsePdfFragment,
  pdfReaderPath,
  type PdfCitation,
  type PdfDocumentRecord,
  type PdfMark,
  type PdfMarkKind,
  type PdfMarkTarget,
  type PdfQuote,
} from '../../util/pdf-marks'
import { rangesForQuotes } from '../arena-feed/quote-highlights'
import { MarkBook, type MarkEntry } from './book'
import {
  boxQuad,
  boxRect,
  hitTest,
  selectionOnPage,
  targetBoxes,
  type PageBox,
  type PageSelection,
} from './geometry'
import { fetchMarks } from './marks'
import {
  CitationCard,
  IndexPanel,
  kindMeta,
  Margin,
  MarkCard,
  notePath,
  SelectionBubble,
  Strip,
  type MarginItem,
  type OutlineEntry,
  type StripTick,
} from './parts'
import { loadPosition, savePosition } from './store'
import { PdfViewer, type PdfOutlineNode, type ViewerPage } from './viewer'

interface DocumentData {
  slug: string
  readPath: string
  document: PdfDocumentRecord
}

interface IndexDocument {
  slug: string
  readPath: string
  title: string
  citations: number
}

type ReaderData =
  | ({ mode: 'document' } & DocumentData)
  | { mode: 'index'; documents: IndexDocument[] }

const COMPACT_WIDTH = 720

function readData(): ReaderData | null {
  try {
    const value: unknown = JSON.parse(document.getElementById('pdf-reader-data')?.textContent ?? '')
    if (typeof value !== 'object' || value === null) return null
    const record = value as Record<string, unknown>
    if (record.mode === 'index' && Array.isArray(record.documents)) {
      return {
        mode: 'index',
        documents: record.documents.filter(
          (doc): doc is IndexDocument =>
            typeof doc === 'object' &&
            doc !== null &&
            typeof doc.slug === 'string' &&
            typeof doc.readPath === 'string' &&
            typeof doc.title === 'string' &&
            typeof doc.citations === 'number',
        ),
      }
    }
    const parsed = parsePdfDocumentRecord(record.document)
    if (record.mode === 'document' && typeof record.slug === 'string' && parsed) {
      return {
        mode: 'document',
        slug: record.slug,
        readPath: pdfReaderPath(record.slug),
        document: parsed,
      }
    }
  } catch {
    /* fall through to the missing-data message */
  }
  return null
}

export function mountPdfReader(root: HTMLElement, signal: AbortSignal): () => void {
  const data = readData()
  root.replaceChildren()
  if (!data) {
    root.append('This reader page is missing its document. Reload to retry.')
    return () => undefined
  }
  render(
    data.mode === 'index' ? (
      <ReaderIndex documents={data.documents} />
    ) : (
      <DocumentReader data={data} signal={signal} />
    ),
    root,
  )
  return () => render(null, root)
}

function ReaderIndex({ documents }: { documents: IndexDocument[] }) {
  const [filter, setFilter] = useState('')
  const query = filter.trim().toLowerCase()
  const visible = documents.filter(
    doc =>
      !query || doc.title.toLowerCase().includes(query) || doc.slug.toLowerCase().includes(query),
  )
  return (
    <div class="pdf-reader-library">
      <header>
        <h1>reader</h1>
        <p>Open any PDFs to read with annotations.</p>
        <input
          type="search"
          value={filter}
          placeholder="filter by title or path"
          aria-label="Filter documents"
          onInput={event => setFilter(event.currentTarget.value)}
        />
      </header>
      {visible.length === 0 ? (
        <p class="pdf-reader-empty">No documents match.</p>
      ) : (
        <ol class="pdf-reader-library-list">
          {visible.map(doc => (
            <li key={doc.slug}>
              <a href={encodeURI(doc.readPath)}>
                <span class="pdf-reader-library-title">{doc.title}</span>
                <span class="pdf-reader-library-meta">
                  {doc.slug} · cited by {doc.citations}
                </span>
              </a>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function toast(
  message: string,
  action?: { label: string; onClick: () => void },
  durationMs = 2400,
) {
  document.dispatchEvent(new CustomEvent('toast', { detail: { message, action, durationMs } }))
}

async function copyText(text: string, message: string) {
  try {
    await navigator.clipboard.writeText(text)
    toast(message)
    return true
  } catch {
    toast('Copy failed. The clipboard is unavailable here.')
    return false
  }
}

function flattenOutline(
  nodes: PdfOutlineNode[],
  depth = 0,
  out: OutlineEntry[] = [],
): OutlineEntry[] {
  for (const node of nodes) {
    out.push({
      title: node.title.replace(/\s+/g, ' ').trim(),
      dest: node.dest,
      url: node.url ?? null,
      depth,
    })
    if (node.items?.length && out.length < 800) flattenOutline(node.items, depth + 1, out)
  }
  return out
}

function isEditable(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target.isContentEditable)
  )
}

interface ActiveSelection extends PageSelection {
  range: Range
}

function DocumentReader({ data, signal }: { data: DocumentData; signal: AbortSignal }) {
  const { slug, readPath, document: record } = data
  const pdfUrl = `/${encodeURI(slug)}`
  const rootRef = useRef<HTMLDivElement>(null)
  const scrollerRef = useRef<HTMLDivElement>(null)
  const columnRef = useRef<HTMLDivElement>(null)
  const bracketRef = useRef<HTMLDivElement>(null)
  const sheetRef = useRef<HTMLDialogElement>(null)
  const keysRef = useRef<HTMLDialogElement>(null)
  const viewerRef = useRef<PdfViewer | null>(null)
  const staleRef = useRef<PdfMark[]>([])
  const book = useMemo(() => new MarkBook(slug, record.doc), [slug, record.doc])

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
  const [view, setView] = useState<'margin' | 'index'>('margin')
  const [compact, setCompact] = useState(false)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [focusId, setFocusId] = useState<string | null>(null)
  const [selection, setSelection] = useState<ActiveSelection | null>(null)
  const [tool, setTool] = useState<'select' | 'region'>('select')
  const [outline, setOutline] = useState<OutlineEntry[]>([])
  const [labels, setLabels] = useState<string[] | null>(null)
  const [metaTitle, setMetaTitle] = useState<string | null>(null)
  const [orphans, setOrphans] = useState<PdfMark[]>([])
  const [pageInput, setPageInput] = useState('1')

  useEffect(() => book.subscribe(() => setBookVersion(book.version)), [book])

  const label = useCallback((number: number) => labels?.[number - 1] ?? String(number), [labels])
  const title = metaTitle ?? record.title

  // Viewer lifecycle.
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
          const position = await loadPosition(record.doc).catch(() => null)
          if (position && !signal.aborted) viewer.scrollToPage(position.page, position.top)
        }
        void viewer.outline().then(nodes => !signal.aborted && setOutline(flattenOutline(nodes)))
        void viewer.pageLabels().then(value => !signal.aborted && setLabels(value))
        void viewer.metadataTitle().then(value => !signal.aborted && setMetaTitle(value))
      })
      .catch(error => {
        if (signal.aborted) return
        console.error(error)
        setPhase('error')
        setFailure(error instanceof Error ? error.message : String(error))
      })
    return () => {
      window.cancelAnimationFrame(layoutFrame)
      window.cancelAnimationFrame(viewportFrame)
      viewer.destroy()
      viewerRef.current = null
    }
  }, [pdfUrl])

  // Marks.
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

  // Phone layout: the margin becomes a sheet below 720 px of reader width.
  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    const observer = new ResizeObserver(() => setCompact(root.clientWidth < COMPACT_WIDTH))
    observer.observe(root)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    viewerRef.current?.setZoom(viewerRef.current.fit === 'width' ? 'width' : viewerRef.current.zoom)
  }, [compact])

  useEffect(() => setPageInput(label(page)), [page, label])

  // Reading position, saved once scrolling settles.
  useEffect(() => {
    if (phase !== 'ready') return
    const timer = window.setTimeout(() => {
      const viewer = viewerRef.current
      const scroller = scrollerRef.current
      const element = viewer?.pages[page - 1]?.element
      if (!viewer || !scroller || !element) return
      const top = (scroller.scrollTop - element.offsetTop) / Math.max(1, element.offsetHeight)
      void savePosition(record.doc, { page, top: Math.min(1, Math.max(0, top)) }).catch(
        () => undefined,
      )
    }, 800)
    return () => window.clearTimeout(timer)
  }, [page, phase, record.doc])

  const boxesById = useMemo(() => {
    const result = new Map<string, { page: number; boxes: PageBox[] }>()
    const viewer = viewerRef.current
    if (!viewer) return result
    for (const { mark } of book.entries.values()) {
      const base = viewer.pages[mark.target.page - 1]?.base
      if (base)
        result.set(mark.id, { page: mark.target.page, boxes: targetBoxes(mark.target, base) })
    }
    return result
  }, [book.version, viewportTick, phase])

  const firstTop = useCallback(
    (mark: PdfMark) => boxesById.get(mark.id)?.boxes[0]?.top ?? 0,
    [boxesById],
  )
  const entries = useMemo(() => book.sorted(firstTop), [book.version, firstTop])

  // Paint marks into each page's mark layer; boxes are page fractions, so zoom needs no repaint.
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return
    const byPage = new Map<number, HTMLElement[]>()
    for (const entry of entries) {
      const placed = boxesById.get(entry.mark.id)
      if (!placed) continue
      const nodes = byPage.get(placed.page) ?? []
      placed.boxes.forEach((box, index) => {
        const node = document.createElement('div')
        node.className = 'pdf-reader-mark'
        node.dataset.kind = entry.mark.kind
        node.dataset.markId = entry.mark.id
        if (index === 0) node.dataset.lead = ''
        if (entry.mark.id === activeId) node.dataset.active = ''
        if (entry.mark.target.type === 'region') node.dataset.region = ''
        node.style.left = `${box.left * 100}%`
        node.style.top = `${box.top * 100}%`
        node.style.width = `${box.width * 100}%`
        node.style.height = `${box.height * 100}%`
        nodes.push(node)
      })
      byPage.set(placed.page, nodes)
    }
    for (const viewerPage of viewer.pages) {
      const nodes = byPage.get(viewerPage.number)
      if (nodes || viewerPage.markLayer.childElementCount > 0)
        viewerPage.markLayer.replaceChildren(...(nodes ?? []))
    }
  }, [entries, boxesById, activeId])

  const pageOffset = useCallback(
    (number: number, top = 0) => {
      const element = viewerRef.current?.pages[number - 1]?.element
      return element ? element.offsetTop + top * element.offsetHeight : 0
    },
    [layoutTick, viewportTick],
  )

  const scrollToMark = useCallback(
    (id: string, behavior: ScrollBehavior = 'smooth') => {
      const entry = book.entries.get(id)
      const viewer = viewerRef.current
      if (!entry || !viewer) return
      viewer.scrollToPage(entry.mark.target.page, boxesById.get(id)?.boxes[0]?.top ?? 0, behavior)
    },
    [book, boxesById],
  )

  const activate = useCallback(
    (id: string | null, options: { scroll?: boolean; focus?: boolean } = {}) => {
      setActiveId(id)
      setFocusId(options.focus && id ? id : null)
      if (!id) return
      if (options.scroll) scrollToMark(id)
      if (compact && !sheetRef.current?.open) sheetRef.current?.showModal()
      history.replaceState(history.state, '', `${location.pathname}${location.search}#^${id}`)
    },
    [scrollToMark, compact],
  )

  // `#^id` waits for marks and the target page's viewport; `#page=N` is handled on load.
  const pendingHash = useRef(parsePdfFragment(location.hash).markId ?? null)
  useEffect(() => {
    const id = pendingHash.current
    if (!id || marksState !== 'ready' || phase !== 'ready' || !boxesById.has(id)) return
    pendingHash.current = null
    activate(id)
    scrollToMark(id, 'auto')
  }, [marksState, phase, boxesById, activate, scrollToMark])

  useEffect(() => {
    const onHash = () => {
      const fragment = parsePdfFragment(location.hash)
      if (fragment.page) viewerRef.current?.scrollToPage(fragment.page)
      if (fragment.markId && book.entries.has(fragment.markId)) {
        setActiveId(fragment.markId)
        scrollToMark(fragment.markId)
      }
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [book, scrollToMark])

  // Owner only: carry marks from an older copy of this PDF onto the current one by their quotes.
  useEffect(() => {
    const viewer = viewerRef.current
    if (phase !== 'ready' || marksState !== 'ready' || !viewer || staleRef.current.length === 0)
      return
    const stale = staleRef.current
    staleRef.current = []
    void (async () => {
      const lost: PdfMark[] = []
      for (const mark of stale) {
        if (signal.aborted) return
        const target = await reanchor(viewer, mark.target)
        if (target) book.adopt(mark, target)
        else lost.push(mark)
      }
      if (!signal.aborted) setOrphans(lost)
      if (stale.length > lost.length) {
        toast(`Moved ${stale.length - lost.length} marks onto the updated PDF.`)
      }
    })()
  }, [phase, marksState, book])

  // Text selection → bubble.
  useEffect(() => {
    let timer = 0
    const onChange = () => {
      window.clearTimeout(timer)
      timer = window.setTimeout(() => {
        const current = document.getSelection()
        const column = columnRef.current
        if (!current || current.isCollapsed || current.rangeCount === 0 || !column) {
          setSelection(null)
          return
        }
        const range = current.getRangeAt(0)
        if (!column.contains(range.commonAncestorContainer)) {
          setSelection(null)
          return
        }
        const resolved = selectionOnPage(range)
        setSelection(resolved ? { ...resolved, range: range.cloneRange() } : null)
      }, 90)
    }
    const release = () => {
      for (const layer of columnRef.current?.querySelectorAll('.pdf-reader-text.selecting') ?? []) {
        layer.classList.remove('selecting')
      }
    }
    document.addEventListener('selectionchange', onChange)
    window.addEventListener('pointerup', release)
    return () => {
      window.clearTimeout(timer)
      document.removeEventListener('selectionchange', onChange)
      window.removeEventListener('pointerup', release)
    }
  }, [])

  const quoteText = useCallback(
    (quote: string | null, page: number, id?: string) => {
      const anchor = id ? `#^${id}` : `#page=${page}`
      const link = `[[${slug}${anchor}|${title}, p. ${label(page)}]]`
      if (!quote) return link
      return `${quote
        .split('\n')
        .map(line => `> ${line}`)
        .join('\n')}\n> ${link}`
    },
    [slug, title, label],
  )

  const markFromSelection = useCallback(
    (kind: PdfMarkKind, focus: boolean) => {
      const viewer = viewerRef.current
      if (!selection || !viewer || !book.canWrite) return
      const base = viewer.pages[selection.page - 1]?.base
      if (!base) return
      const target: PdfMarkTarget = {
        type: 'text',
        page: selection.page,
        quads: selection.boxes.map(box => boxQuad(box, base)),
        quote: selection.quote,
      }
      const entry = book.create(kind, target)
      document.getSelection()?.removeAllRanges()
      setSelection(null)
      activate(entry.mark.id, { focus })
      if (selection.clipped)
        toast(`Marks stay on one page, so this one covers p. ${label(selection.page)}.`)
    },
    [selection, book, activate, label],
  )

  const onBubble = useCallback(
    (action: PdfMarkKind | 'note' | 'quote') => {
      if (action === 'quote') {
        if (!selection) return
        void copyText(
          quoteText(selection.quote.exact, selection.page),
          'Quote copied as a garden link.',
        )
        return
      }
      if (action === 'note') markFromSelection('mark', true)
      else markFromSelection(action, action !== 'mark')
    },
    [selection, quoteText, markFromSelection],
  )

  const pageNote = useCallback(() => {
    if (!book.canWrite) return
    const entry = book.create('mark', { type: 'page', page })
    activate(entry.mark.id, { focus: true })
  }, [book, page, activate])

  const removeMark = useCallback(
    (id: string) => {
      book.remove(id)
      setActiveId(current => (current === id ? null : current))
      toast('Mark deleted.', { label: 'undo', onClick: () => book.undo() }, 6000)
    },
    [book],
  )

  const step = useCallback(
    (direction: 1 | -1) => {
      if (entries.length === 0) return
      const index = entries.findIndex(entry => entry.mark.id === activeId)
      let next: MarkEntry | undefined
      if (index >= 0) next = entries[index + direction]
      else if (direction > 0) next = entries.find(entry => entry.mark.target.page >= page)
      else next = [...entries].reverse().find(entry => entry.mark.target.page <= page)
      if (next) activate(next.mark.id, { scroll: true })
    },
    [entries, activeId, page, activate],
  )

  // Reader keymap. Capture phase, but only for keys the reader owns and never inside fields.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (isEditable(event.target) || event.altKey) return
      if (document.querySelector('dialog[open]:not(.pdf-reader-sheet)')) return
      const viewer = viewerRef.current
      const mod = event.metaKey || event.ctrlKey
      let handled = true
      if (mod && event.key.toLowerCase() === 'z' && !event.shiftKey && book.canUndo) {
        const restored = book.undo()
        if (restored) activate(restored.mark.id)
      } else if (mod) {
        handled = false
      } else if (event.key === 'Escape') {
        if (tool === 'region') setTool('select')
        else if (selection) {
          document.getSelection()?.removeAllRanges()
          setSelection(null)
        } else if (activeId) activate(null)
        else handled = false
      } else if (event.key === '?') {
        keysRef.current?.showModal()
      } else if (event.key === '[' || event.key === ']') {
        step(event.key === ']' ? 1 : -1)
      } else if (event.key === '+' || event.key === '=') {
        viewer?.zoomBy(1.1)
      } else if (event.key === '-') {
        viewer?.zoomBy(1 / 1.1)
      } else if (event.key === '0') {
        viewer?.setZoom('width')
      } else if (event.key === 'c' && (selection || activeId)) {
        if (selection) onBubble('quote')
        else if (activeId) void copyMarkQuote(activeId)
      } else if (!book.canWrite) {
        handled = false
      } else if (event.key === 'a' || event.key === 'q' || event.key === 'x') {
        const kind: PdfMarkKind =
          event.key === 'a' ? 'mark' : event.key === 'q' ? 'question' : 'contra'
        if (selection) markFromSelection(kind, kind !== 'mark')
        else if (activeId) book.update(activeId, { kind }, true)
        else handled = false
      } else if (event.key === 'n') {
        if (selection) markFromSelection('mark', true)
        else pageNote()
      } else if (event.key === 'r') {
        setTool(current => (current === 'region' ? 'select' : 'region'))
      } else if ((event.key === 'Delete' || event.key === 'Backspace') && activeId) {
        removeMark(activeId)
      } else {
        handled = false
      }
      if (handled) {
        event.preventDefault()
        event.stopImmediatePropagation()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  })

  const copyMarkQuote = async (id: string): Promise<boolean> => {
    const entry = book.entries.get(id)
    if (!entry) return false
    const quote = entry.mark.target.type === 'text' ? entry.mark.target.quote.exact : null
    return copyText(quoteText(quote, entry.mark.target.page, id), 'Quote copied as a garden link.')
  }

  // Pointer: activate painted marks on click, start text selection, or drag a region.
  const onPointerDown = (event: PointerEvent) => {
    const target = event.target as Element
    const pageElement = target.closest<HTMLElement>('[data-page-number]')
    if (!pageElement) return
    if (tool !== 'region') {
      pageElement.querySelector('.pdf-reader-text')?.classList.add('selecting')
      return
    }
    const viewer = viewerRef.current
    const number = Number(pageElement.dataset.pageNumber)
    const base = viewer?.pages[number - 1]?.base
    if (!base || event.button !== 0) return
    event.preventDefault()
    const rect = pageElement.getBoundingClientRect()
    const start = {
      x: (event.clientX - rect.left) / rect.width,
      y: (event.clientY - rect.top) / rect.height,
    }
    const draft = document.createElement('div')
    draft.className = 'pdf-reader-region-draft'
    pageElement.append(draft)
    let box: PageBox = { left: start.x, top: start.y, width: 0, height: 0 }
    const move = (moveEvent: PointerEvent) => {
      const x = Math.min(1, Math.max(0, (moveEvent.clientX - rect.left) / rect.width))
      const y = Math.min(1, Math.max(0, (moveEvent.clientY - rect.top) / rect.height))
      box = {
        left: Math.min(start.x, x),
        top: Math.min(start.y, y),
        width: Math.abs(x - start.x),
        height: Math.abs(y - start.y),
      }
      Object.assign(draft.style, {
        left: `${box.left * 100}%`,
        top: `${box.top * 100}%`,
        width: `${box.width * 100}%`,
        height: `${box.height * 100}%`,
      })
    }
    const finish = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', finish)
      window.removeEventListener('pointercancel', finish)
      draft.remove()
      setTool('select')
      if (box.width < 0.01 || box.height < 0.005) return
      const entry = book.create('mark', { type: 'region', page: number, rect: boxRect(box, base) })
      activate(entry.mark.id, { focus: true })
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', finish)
    window.addEventListener('pointercancel', finish)
  }

  const onPointerUp = (event: PointerEvent) => {
    if (tool === 'region' || event.button !== 0) return
    const current = document.getSelection()
    if (current && !current.isCollapsed) return
    const target = event.target as Element
    if (target.closest('a')) return
    const pageElement = target.closest<HTMLElement>('[data-page-number]')
    if (!pageElement) return
    const number = Number(pageElement.dataset.pageNumber)
    const rect = pageElement.getBoundingClientRect()
    const x = (event.clientX - rect.left) / rect.width
    const y = (event.clientY - rect.top) / rect.height
    const onPage = new Map<string, PageBox[]>()
    for (const [id, placed] of boxesById) if (placed.page === number) onPage.set(id, placed.boxes)
    const hits = hitTest(onPage, x, y)
    if (hits.length > 0) activate(hits[hits.length - 1])
    else if (activeId) activate(null)
  }

  // Overview strip and its viewport bracket.
  const columnHeight = columnRef.current?.scrollHeight ?? 0
  const ticks = useMemo<StripTick[]>(() => {
    if (!columnHeight) return []
    const result: StripTick[] = entries.map(entry => ({
      key: entry.mark.id,
      kind: entry.mark.kind,
      top:
        pageOffset(entry.mark.target.page, boxesById.get(entry.mark.id)?.boxes[0]?.top ?? 0) /
        columnHeight,
    }))
    record.citedBy.forEach((citation, index) => {
      if (citation.page) {
        result.push({
          key: `cite-${index}`,
          kind: 'cite',
          top: pageOffset(citation.page) / columnHeight,
        })
      }
    })
    return result
  }, [entries, boxesById, pageOffset, columnHeight, record.citedBy])

  useEffect(() => {
    const scroller = scrollerRef.current
    const bracket = bracketRef.current
    if (!scroller || !bracket) return
    let frame = 0
    const update = () => {
      frame = 0
      const height = scroller.scrollHeight || 1
      bracket.style.top = `${(scroller.scrollTop / height) * 100}%`
      bracket.style.height = `${Math.max(1, (scroller.clientHeight / height) * 100)}%`
    }
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(update)
    }
    update()
    scroller.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      scroller.removeEventListener('scroll', onScroll)
      window.cancelAnimationFrame(frame)
    }
  }, [layoutTick, phase])

  // Trackpad pinch and ctrl+wheel zoom the document instead of the page.
  useEffect(() => {
    const scroller = scrollerRef.current
    if (!scroller) return
    let factor = 1
    let frame = 0
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey) return
      event.preventDefault()
      factor *= Math.exp(-event.deltaY * 0.01)
      if (!frame) {
        frame = window.requestAnimationFrame(() => {
          frame = 0
          viewerRef.current?.zoomBy(factor)
          factor = 1
        })
      }
    }
    scroller.addEventListener('wheel', onWheel, { passive: false })
    return () => {
      scroller.removeEventListener('wheel', onWheel)
      window.cancelAnimationFrame(frame)
    }
  }, [])

  const cardFor = (entry: MarkEntry) => (
    <MarkCard
      key={entry.mark.id}
      entry={entry}
      active={entry.mark.id === activeId}
      canWrite={book.canWrite}
      pageLabel={label(entry.mark.target.page)}
      autoFocus={entry.mark.id === focusId}
      onActivate={() => activate(entry.mark.id, { scroll: true })}
      onBody={body => book.update(entry.mark.id, { body })}
      onKind={kind => book.update(entry.mark.id, { kind }, true)}
      onVisibility={visibility => book.update(entry.mark.id, { visibility }, true)}
      onDelete={() => removeMark(entry.mark.id)}
      onResolve={choice => book.resolve(entry.mark.id, choice)}
      onCopyLink={() =>
        copyText(`${location.origin}${encodeURI(readPath)}#^${entry.mark.id}`, 'Link copied.')
      }
      onCopyQuote={() => copyMarkQuote(entry.mark.id)}
    />
  )

  // Cards render fresh each pass, so this is plain computation rather than a memo.
  const marginItems: MarginItem[] = []
  if (phase === 'ready') {
    const items: MarginItem[] = entries.map(entry => ({
      key: entry.mark.id,
      top: pageOffset(entry.mark.target.page, boxesById.get(entry.mark.id)?.boxes[0]?.top ?? 0),
      node: cardFor(entry),
    }))
    record.citedBy.forEach((citation: PdfCitation, index) => {
      const markTop =
        citation.markId && book.entries.has(citation.markId)
          ? pageOffset(
              book.entries.get(citation.markId)!.mark.target.page,
              boxesById.get(citation.markId)?.boxes[0]?.top ?? 0,
            )
          : null
      items.push({
        key: `cite-${index}`,
        top: markTop ?? (citation.page ? pageOffset(citation.page) : 0),
        node: (
          <CitationCard
            citation={citation}
            pageLabel={citation.page ? label(citation.page) : undefined}
            onJump={
              citation.page
                ? () => viewerRef.current?.scrollToPage(citation.page!, 0, 'smooth')
                : undefined
            }
          />
        ),
      })
    })
    marginItems.push(...items)
  }

  const indexPanel = (
    <IndexPanel
      entries={entries}
      citations={record.citedBy}
      outline={outline}
      orphans={orphans}
      canWrite={book.canWrite}
      label={label}
      activeId={activeId}
      renderCard={cardFor}
      onActivate={id => activate(id, { scroll: true })}
      onOutline={entry => {
        if (entry.url) {
          window.open(entry.url, '_blank', 'noopener,noreferrer')
          return
        }
        void viewerRef.current?.resolveDestination(entry.dest).then(found => {
          if (found) viewerRef.current?.scrollToPage(found.page, found.top, 'smooth')
          if (compact) sheetRef.current?.close()
        })
      }}
      onJumpPage={number => viewerRef.current?.scrollToPage(number, 0, 'smooth')}
      onDropOrphan={mark => {
        setOrphans(current => current.filter(item => item.id !== mark.id))
        book.entries.set(mark.id, { mark, revision: mark.revision, status: 'saved' })
        removeMark(mark.id)
      }}
    />
  )

  const markCount = book.entries.size
  const showSide = !compact
  const viewer = viewerRef.current

  return (
    <div
      ref={rootRef}
      class="pdf-reader-app"
      data-phase={phase}
      data-tool={tool}
      data-view={view}
      data-compact={compact}
    >
      <header class="pdf-reader-bar">
        <div class="pdf-reader-title">
          <a href="/read" class="pdf-reader-crumb" aria-label="All documents">
            reader
          </a>
          <span aria-hidden="true">/</span>
          <h1 title={slug}>{title}</h1>
        </div>
        <p class="pdf-reader-meta">
          {pageCount > 0 && <span>{pageCount} pp</span>}
          <span class="pdf-reader-meta-marks">
            {markCount} {markCount === 1 ? 'mark' : 'marks'}
          </span>
          {record.citedBy.length > 0 && <span>cited by {record.citedBy.length}</span>}
          <a
            class="pdf-reader-download"
            href={`${pdfUrl}?raw=1`}
            download={slug.split('/').pop()}
            aria-label="Download PDF"
            title="Download PDF"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
              focusable="false"
            >
              <path d="M12 3v12m-5-5 5 5 5-5M5 15v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5" />
            </svg>
          </a>
        </p>
        <div class="pdf-reader-controls" role="toolbar" aria-label="Reader controls">
          <form
            class="pdf-reader-page-form"
            onSubmit={event => {
              event.preventDefault()
              const wanted = pageInput.trim()
              const byLabel = labels?.indexOf(wanted) ?? -1
              const number = byLabel >= 0 ? byLabel + 1 : Number.parseInt(wanted, 10)
              if (Number.isInteger(number)) viewer?.scrollToPage(number)
              scrollerRef.current?.focus({ preventScroll: true })
            }}
          >
            <label for="pdf-reader-page">p.</label>
            <input
              id="pdf-reader-page"
              value={pageInput}
              inputMode="numeric"
              size={Math.max(2, String(pageCount).length)}
              aria-label={`Page, of ${pageCount}`}
              onInput={event => setPageInput(event.currentTarget.value)}
              onBlur={() => setPageInput(label(page))}
            />
            <span>/ {pageCount || '…'}</span>
          </form>
          <div class="pdf-reader-row" role="group" aria-label="Zoom">
            <button
              type="button"
              aria-label="Zoom out"
              title="zoom out (−)"
              onClick={() => viewer?.zoomBy(1 / 1.1)}
            >
              −
            </button>
            <output class="pdf-reader-zoom" aria-live="polite">
              {Math.round(zoom * 100)}%
            </output>
            <button
              type="button"
              aria-label="Zoom in"
              title="zoom in (+)"
              onClick={() => viewer?.zoomBy(1.1)}
            >
              +
            </button>
            <button
              type="button"
              aria-pressed={fit === 'width'}
              title="fit width (0)"
              onClick={() => viewer?.setZoom('width')}
            >
              fit
            </button>
          </div>
          {showSide ? (
            <div class="pdf-reader-segmented" role="group" aria-label="Side panel">
              <button
                type="button"
                aria-pressed={view === 'margin'}
                onClick={() => setView('margin')}
              >
                margin
              </button>
              <button
                type="button"
                aria-pressed={view === 'index'}
                onClick={() => setView('index')}
              >
                index
              </button>
            </div>
          ) : (
            <button
              type="button"
              aria-haspopup="dialog"
              onClick={() => sheetRef.current?.showModal()}
            >
              marks ({markCount})
            </button>
          )}
          {book.canWrite && (
            <button
              type="button"
              aria-pressed={tool === 'region'}
              title="mark a region: drag a box on the page (r)"
              onClick={() => setTool(tool === 'region' ? 'select' : 'region')}
            >
              region
            </button>
          )}
          <button
            type="button"
            aria-label="Keyboard shortcuts"
            title="shortcuts (?)"
            onClick={() => keysRef.current?.showModal()}
          >
            ?
          </button>
        </div>
      </header>

      <div class="pdf-reader-body">
        <div
          ref={scrollerRef}
          class="pdf-reader-scroller"
          tabIndex={0}
          aria-label={`${title}, ${pageCount} pages`}
          onPointerDown={onPointerDown}
          onPointerUp={onPointerUp}
        >
          <div class="pdf-reader-spread">
            <div ref={columnRef} class="pdf-reader-column">
              {phase === 'loading' && (
                <p class="pdf-reader-status" role="status">
                  Loading {record.title}…
                </p>
              )}
              {phase === 'error' && (
                <div class="pdf-reader-status" role="alert">
                  <p>This PDF failed to load{failure ? `: ${failure}` : '.'}</p>
                  <p>
                    <a href={`${pdfUrl}?raw=1`}>Open the file directly</a> or reload to retry.
                  </p>
                </div>
              )}
            </div>
            {showSide && (
              <aside class="pdf-reader-side" aria-label="Notes">
                <Margin
                  items={marginItems}
                  activeKey={activeId}
                  layoutKey={`${marginItems.map(item => item.key).join()}|${layoutTick}|${viewportTick}`}
                  hidden={view !== 'margin'}
                />
                {marksState === 'error' && (
                  <p class="pdf-reader-empty" role="status">
                    Marks are unavailable right now.
                  </p>
                )}
              </aside>
            )}
          </div>
        </div>
        {showSide && view === 'index' && <div class="pdf-reader-index-overlay">{indexPanel}</div>}
        {showSide && (
          <Strip
            ticks={ticks}
            bracket={bracketRef}
            onScrub={fraction => {
              const scroller = scrollerRef.current
              if (scroller)
                scroller.scrollTop = fraction * scroller.scrollHeight - scroller.clientHeight / 2
            }}
          />
        )}
      </div>

      {selection && tool === 'select' && (
        <SelectionBubble range={selection.range} canWrite={book.canWrite} onAction={onBubble} />
      )}

      <dialog ref={sheetRef} class="pdf-reader-sheet" aria-label="Marks and contents">
        <header class="pdf-reader-row">
          <h2>{title}</h2>
          <button type="button" onClick={() => sheetRef.current?.close()}>
            close
          </button>
        </header>
        {compact && indexPanel}
      </dialog>

      <dialog ref={keysRef} class="pdf-reader-keys" aria-labelledby="pdf-reader-keys-title">
        <h2 id="pdf-reader-keys-title">shortcuts</h2>
        <dl>
          {book.canWrite && (
            <>
              <dt>a · q · x</dt>
              <dd>
                mark the selection as {kindMeta.mark.label}, {kindMeta.question.label} or{' '}
                {kindMeta.contra.label}; on an active mark, change its kind
              </dd>
              <dt>n</dt>
              <dd>mark with a note, or a note on this page</dd>
              <dt>r</dt>
              <dd>drag a box around a figure or equation</dd>
              <dt>delete</dt>
              <dd>delete the active mark (⌘Z restores it)</dd>
            </>
          )}
          <dt>c</dt>
          <dd>copy the selection or active mark as a garden quote</dd>
          <dt>[ · ]</dt>
          <dd>previous or next mark</dd>
          <dt>+ · − · 0</dt>
          <dd>zoom in, out, fit width</dd>
          <dt>esc</dt>
          <dd>clear the selection, tool or active mark</dd>
        </dl>
        {!book.canWrite && (
          <p>
            Marks here are aarnphm's. <a href={book.signInUrl}>Owner sign-in</a>
          </p>
        )}
        <form method="dialog">
          <button type="submit">close</button>
        </form>
      </dialog>
    </div>
  )
}

/** Finds a stale mark's quote near its old page and rebuilds its quads from the new text layer. */
async function reanchor(viewer: PdfViewer, target: PdfMarkTarget): Promise<PdfMarkTarget | null> {
  if (target.type !== 'text') return target.page <= viewer.pageCount ? target : null
  const quote: PdfQuote = target.quote
  for (const offset of [0, -1, 1, -2, 2]) {
    const number = target.page + offset
    if (number < 1 || number > viewer.pageCount) continue
    const layer = await viewer.ensureTextLayer(number)
    const page: ViewerPage | undefined = viewer.pages[number - 1]
    if (!layer || !page?.base) continue
    const [range] = rangesForQuotes(layer, [quote])
    if (!range) continue
    const found = selectionOnPage(range)
    if (!found || found.page !== number) continue
    return {
      type: 'text',
      page: number,
      quads: found.boxes.map(box => boxQuad(box, page.base!)),
      quote: found.quote,
    }
  }
  return null
}

export { notePath }
