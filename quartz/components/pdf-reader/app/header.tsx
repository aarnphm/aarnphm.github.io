import type { PdfViewer } from '../viewer'
import type { DocumentData } from './data'
import { ArxivLink } from '../arxiv-link'

interface ReaderHeaderProps {
  data: DocumentData
  title: string
  pdfUrl: string
  pageCount: number
  markCount: number
  labels: string[] | null
  page: number
  pageInput: string
  setPageInput: (value: string) => void
  label: (page: number) => string
  zoom: number
  fit: 'width' | 'custom'
  compact: boolean
  view: 'margin' | 'index'
  setView: (value: 'margin' | 'index') => void
  tool: 'select' | 'region'
  setTool: (value: 'select' | 'region') => void
  canWrite: boolean
  viewer: PdfViewer | null
  onFocusReader: () => void
  onOpenMarks: () => void
  onOpenKeys: () => void
}

export function ReaderHeader({
  data,
  title,
  pdfUrl,
  pageCount,
  markCount,
  labels,
  page,
  pageInput,
  setPageInput,
  label,
  zoom,
  fit,
  compact,
  view,
  setView,
  tool,
  setTool,
  canWrite,
  viewer,
  onFocusReader,
  onOpenMarks,
  onOpenKeys,
}: ReaderHeaderProps) {
  const { slug, chrome } = data
  // Reserve the longest page number or label while paging.
  const fieldChars = Math.min(
    6,
    Math.max(String(pageCount).length, ...(labels ?? []).map(l => l.length)),
  )
  const showSide = !compact
  return (
    <header class="pdf-reader-bar">
      {chrome !== 'course' && (
        <div class="pdf-reader-title">
          <a href="/read" class="pdf-reader-crumb" aria-label="All documents">
            reader
          </a>
          <span aria-hidden="true">/</span>
          <h1 title={slug}>{title}</h1>
        </div>
      )}
      <p class="pdf-reader-meta">
        {pageCount > 0 && <span>{pageCount} pp</span>}
        <span class="pdf-reader-meta-marks">
          {markCount} {markCount === 1 ? 'mark' : 'marks'}
        </span>
        {data.document.citedBy.length > 0 && <span>cited by {data.document.citedBy.length}</span>}
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
        <ArxivLink slug={slug} />
      </p>
      <div class="pdf-reader-controls" role="toolbar" aria-label="Reader controls">
        <form
          class="pdf-reader-page-form"
          role="group"
          aria-label="Page"
          onSubmit={event => {
            event.preventDefault()
            const wanted = pageInput.trim()
            const byLabel = labels?.indexOf(wanted) ?? -1
            const number = byLabel >= 0 ? byLabel + 1 : Number.parseInt(wanted, 10)
            if (Number.isInteger(number)) viewer?.scrollToPage(number)
            onFocusReader()
          }}
        >
          <button
            type="button"
            aria-label="Previous page"
            title="previous page"
            disabled={page <= 1}
            onClick={() => viewer?.scrollToPage(page - 1)}
          >
            ‹
          </button>
          <span class="pdf-reader-page-cell">
            <label for="pdf-reader-page">p.</label>
            <input
              id="pdf-reader-page"
              value={pageInput}
              inputMode="numeric"
              style={`--pdf-instrument-digits: ${fieldChars}`}
              aria-label={`Page, of ${pageCount}`}
              onInput={event => setPageInput(event.currentTarget.value)}
              onBlur={() => setPageInput(label(page))}
            />
            <span>/ {pageCount || '…'}</span>
          </span>
          <button
            type="button"
            aria-label="Next page"
            title="next page"
            disabled={page >= pageCount}
            onClick={() => viewer?.scrollToPage(page + 1)}
          >
            ›
          </button>
        </form>
        <div class="pdf-reader-zoom-group" role="group" aria-label="Zoom">
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
            <button type="button" aria-pressed={view === 'index'} onClick={() => setView('index')}>
              index
            </button>
          </div>
        ) : (
          <button type="button" aria-haspopup="dialog" onClick={() => onOpenMarks()}>
            marks ({markCount})
          </button>
        )}
        {canWrite && (
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
          onClick={() => onOpenKeys()}
        >
          ?
        </button>
      </div>
    </header>
  )
}
