import { render } from 'preact'
import { useCallback, useEffect, useMemo, useRef, useState } from 'preact/hooks'
import { parsePdfFragment, type PdfCitation, type PdfMark } from '../../util/pdf-marks'
import { readData, type DocumentData } from './app/data'
import { ReaderHeader } from './app/header'
import { ReaderIndex } from './app/index'
import { useReaderKeyboard } from './app/keyboard'
import { useReaderLifecycle } from './app/lifecycle'
import { useMarkActions } from './app/mark-actions'
import { createPointerHandlers } from './app/pointer'
import { useReaderSelection, type ActiveSelection } from './app/selection'
import { ReaderShortcuts } from './app/shortcuts'
import { reanchor, toast } from './app/utils'
import { MarkBook, type MarkEntry } from './book'
import { targetBoxes, type PageBox } from './geometry'
import {
  CitationCard,
  IndexPanel,
  Margin,
  MarkCard,
  SelectionBubble,
  Strip,
  type MarginItem,
  type StripTick,
} from './parts'

const COMPACT_WIDTH = 720

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

function DocumentReader({ data, signal }: { data: DocumentData; signal: AbortSignal }) {
  const { slug, readPath, document: record, chrome } = data
  const pdfUrl = `/${encodeURI(slug)}`
  const embedded = document.body.dataset.pdfEmbedded === 'course'
  const rootRef = useRef<HTMLDivElement>(null)
  const scrollerRef = useRef<HTMLDivElement>(null)
  const columnRef = useRef<HTMLDivElement>(null)
  const bracketRef = useRef<HTMLDivElement>(null)
  const sheetRef = useRef<HTMLDialogElement>(null)
  const keysRef = useRef<HTMLDialogElement>(null)
  const book = useMemo(() => new MarkBook(slug, record.doc), [slug, record.doc])

  const [view, setView] = useState<'margin' | 'index'>('margin')
  const [compact, setCompact] = useState(embedded)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [focusId, setFocusId] = useState<string | null>(null)
  const [selection, setSelection] = useState<ActiveSelection | null>(null)
  const [tool, setTool] = useState<'select' | 'region'>('select')
  const [orphans, setOrphans] = useState<PdfMark[]>([])
  const reader = useReaderLifecycle({
    pdfUrl,
    slug,
    docId: record.doc,
    signal,
    book,
    rootRef,
    scrollerRef,
    columnRef,
  })
  const {
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
  } = reader
  const label = useCallback((number: number) => labels?.[number - 1] ?? String(number), [labels])
  const title =
    record.titleSource === 'frontmatter' ? record.title : (extractedTitle ?? record.title)

  useEffect(() => {
    if (!signal.aborted) document.title = title
  }, [title, signal])

  // Embedded readers keep marks in a sheet so resizing gives the width to the PDF.
  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    const observer = new ResizeObserver(() =>
      setCompact(embedded || root.clientWidth < COMPACT_WIDTH),
    )
    observer.observe(root)
    return () => observer.disconnect()
  }, [embedded])

  useEffect(() => {
    viewerRef.current?.setZoom(viewerRef.current.fit === 'width' ? 'width' : viewerRef.current.zoom)
  }, [compact])

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

  useReaderSelection(columnRef, setSelection)

  const { onBubble, markFromSelection, pageNote, removeMark, step, copyMarkQuote, copyLink } =
    useMarkActions({
      slug,
      title,
      label,
      viewerRef,
      book,
      selection,
      setSelection,
      activate,
      page,
      entries,
      activeId,
      setActiveId,
      readPath,
    })

  useReaderKeyboard({
    activeId,
    compact,
    label,
    page,
    tool,
    selection,
    book,
    viewerRef,
    keysRef,
    activate,
    step,
    onBubble,
    copyMarkQuote,
    markFromSelection,
    pageNote,
    setTool,
    setSelection,
    removeMark,
  })

  useEffect(() => {
    if (phase !== 'ready') return
    const viewer = viewerRef.current
    if (!viewer) return
    const needed = new Set<number>()
    for (const entry of entries) needed.add(entry.mark.target.page)
    for (const citation of record.citedBy) {
      if (citation.page) needed.add(citation.page)
    }
    if (needed.size > 0) viewer.ensurePageViewports(needed)
  }, [phase, entries, record.citedBy])

  const { onPointerDown, onPointerUp } = createPointerHandlers({
    tool,
    viewerRef,
    book,
    activate,
    setTool,
    boxesById,
    activeId,
  })

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
      onCopyLink={() => copyLink(entry.mark.id)}
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
      data-keyboard-scroll-scope
      data-phase={phase}
      data-tool={tool}
      data-view={view}
      data-compact={compact}
    >
      <ReaderHeader
        data={data}
        title={title}
        pdfUrl={pdfUrl}
        pageCount={pageCount}
        markCount={markCount}
        labels={labels}
        page={page}
        pageInput={pageInput}
        setPageInput={setPageInput}
        label={label}
        zoom={zoom}
        fit={fit}
        compact={compact}
        view={view}
        setView={setView}
        tool={tool}
        setTool={setTool}
        canWrite={book.canWrite}
        viewer={viewer}
        onFocusReader={() => scrollerRef.current?.focus({ preventScroll: true })}
        onOpenMarks={() => sheetRef.current?.showModal()}
        onOpenKeys={() => keysRef.current?.showModal()}
      />

      <div class="pdf-reader-body">
        <div
          ref={scrollerRef}
          class="pdf-reader-scroller"
          data-keyboard-scroll
          tabIndex={0}
          aria-label={`${title}, ${pageCount} pages`}
          onPointerDown={onPointerDown}
          onPointerUp={onPointerUp}
        >
          <div class="pdf-reader-spread">
            <div ref={columnRef} class="pdf-reader-column">
              {phase === 'loading' && !embedded && (
                <>
                  <p class="pdf-reader-status" role="status">
                    Loading {record.title}…
                  </p>
                  <div class="pdf-reader-skeleton" aria-hidden="true" />
                  <div class="pdf-reader-skeleton" aria-hidden="true" />
                </>
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
                  layoutKey={`${marginItems.map(item => `${item.key}:${Math.round(item.top)}`).join()}|${layoutTick}|${viewportTick}`}
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

      {embedded && phase === 'loading' && (
        <p class="pdf-reader-status pdf-reader-loading" role="status">
          Loading the reader…
        </p>
      )}

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

      <ReaderShortcuts
        dialogRef={keysRef}
        canWrite={book.canWrite}
        signInUrl={book.signInUrl}
        chrome={chrome}
      />
    </div>
  )
}

export { notePath } from './parts'
