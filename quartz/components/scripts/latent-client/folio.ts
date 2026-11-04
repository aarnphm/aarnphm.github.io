import type { Runtime } from './runtime'
import { createPdfEmbed } from '../popover/pdf-preview'
import { chunkUnderLine, keepLine } from './reader'

/**
 * A pdf passage's page, opened under it in the site's pdf embed (pdf.inline.ts, the viewer note
 * pages and the arXiv previews use). The embed is fitted to the reader's column, so it is one
 * page tall and the passage's box shows without scrolling inside it; the box is marked on the
 * page as the reader marks its focus, with a rule at its left, over a wash of the pdf colour.
 *
 * One folio is open at a time: pdf.js holds a document per embed, and a second click elsewhere
 * is a new reading. Nothing loads before a click. pdf.js itself is fetched by the runtime on
 * first mount, and the file is the site's own copy (`/thoughts/pdfs/…`).
 */

const TOGGLE = 'button.lat-folio-toggle[data-folio]'
// an A4/letter page under a toolbar, held until the runtime measures the real page
const PAGE_ASPECT = 1.294
const TOOLBAR = 44

interface Folio {
  article: HTMLElement
  toggle: HTMLButtonElement
  folio: HTMLElement
  pdf: HTMLElement
  watch: MutationObserver
  frame: number
}

function parseBox(raw: string | undefined): [number, number, number, number] | null {
  const box = (raw ?? '').split(',').map(Number)
  if (box.length !== 4 || !box.every(n => Number.isFinite(n) && n >= 0 && n <= 1000)) return null
  const [x0, y0, x1, y1] = box
  return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : null
}

/** Put the passage's box on its page slot; the slot outlives re-renders, the mark may not. */
function markPage(pdf: HTMLElement, page: number, box: [number, number, number, number]): void {
  const slot = pdf.querySelector<HTMLElement>(`.pdf-embed-page[data-pdf-page-number="${page}"]`)
  if (!slot || slot.querySelector(':scope > .lat-folio-mark')) return
  // a little air around the OCR box, so the mark does not clip the glyphs it holds
  const pad = 6
  const [x0, y0, x1, y1] = box
  const left = Math.max(0, x0 - pad)
  const top = Math.max(0, y0 - pad)
  const mark = document.createElement('span')
  mark.className = 'lat-folio-mark'
  mark.setAttribute('aria-hidden', 'true')
  mark.style.left = `${left / 10}%`
  mark.style.top = `${top / 10}%`
  mark.style.width = `${(Math.min(1000, x1 + pad) - left) / 10}%`
  mark.style.height = `${(Math.min(1000, y1 + pad) - top) / 10}%`
  // under the text layer, so the passage stays selectable through its mark
  slot.insertBefore(mark, slot.querySelector(':scope > .pdf-embed-text-layer'))
}

export function bindFolios(rt: Runtime): void {
  let open: Folio | null = null

  const release = (folio: Folio) => {
    folio.watch.disconnect()
    cancelAnimationFrame(folio.frame)
    window.quartzPdfEmbeds?.cleanup(folio.folio)
    folio.toggle.setAttribute('aria-expanded', 'false')
    delete folio.article.dataset.folio
  }

  // `still` is the passage that must not move: the one whose glyph was clicked. Holding the
  // passage under the reading line instead threw a clicked passage above it off the screen as
  // its page grew under it.
  const close = (still: HTMLElement | null) => {
    const folio = open
    if (!folio) return
    open = null
    release(folio)
    if (folio.folio.isConnected) keepLine(still ?? chunkUnderLine(rt), () => folio.folio.remove())
  }

  const openFolio = (toggle: HTMLButtonElement) => {
    const article = toggle.closest<HTMLElement>('.lat-chunk[data-row]')
    const runtime = window.quartzPdfEmbeds
    const src = toggle.dataset.folio
    const page = Number(toggle.dataset.folioPage)
    if (!article || !runtime || !src || !Number.isInteger(page) || page < 1) {
      // no runtime on this page: the locus link is the way to the page
      const read = article?.querySelector<HTMLAnchorElement>('.lat-locus a[href]')
      if (read) window.location.assign(read.href)
      return
    }
    const box = parseBox(article.dataset.box)
    const title =
      article.closest('.lat-doc')?.querySelector('.lat-doc-title')?.textContent?.trim() ||
      src.split('/').pop() ||
      'document.pdf'

    const folio = document.createElement('div')
    folio.className = 'lat-folio'
    folio.setAttribute('role', 'region')
    folio.setAttribute('aria-label', `${title}, page ${page}`)
    const pdf = createPdfEmbed(src, title, {
      preload: false,
      annotations: false,
      overscan: 1,
      maxPixelRatio: 2,
      textLayer: true,
    })
    pdf.dataset.pdfPage = String(page)
    folio.append(pdf)

    keepLine(article, () => {
      article.append(folio)
      // hold a page's height now, so the page arriving grows nothing under the reader
      const width = folio.clientWidth
      if (width > 0)
        pdf.style.setProperty(
          '--pdf-embed-height',
          `${Math.round(width * PAGE_ASPECT + TOOLBAR)}px`,
        )
    })

    const state: Folio = {
      article,
      toggle,
      folio,
      pdf,
      watch: new MutationObserver(() => {
        if (state.frame || !box) return
        state.frame = requestAnimationFrame(() => {
          state.frame = 0
          if (pdf.dataset.pdfStatus === 'loaded') markPage(pdf, page, box)
        })
      }),
      frame: 0,
    }
    state.watch.observe(pdf, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['data-pdf-status'],
    })
    open = state
    toggle.setAttribute('aria-expanded', 'true')
    article.dataset.folio = ''
    runtime.mount(folio)
  }

  rt.reader.addEventListener(
    'click',
    event => {
      const toggle = (event.target as Element | null)?.closest?.<HTMLButtonElement>(TOGGLE)
      if (!toggle) return
      event.preventDefault()
      const same = open?.toggle === toggle
      close(toggle.closest<HTMLElement>('.lat-chunk[data-row]'))
      if (!same) openFolio(toggle)
    },
    { signal: rt.signal },
  )

  // the reader drops documents far from the open one and replaces its contents on a jump; a
  // folio that left with them gives its pdf.js document back
  const gone = new MutationObserver(() => {
    if (open && !open.folio.isConnected) {
      const folio = open
      open = null
      release(folio)
    }
  })
  gone.observe(rt.reader, { childList: true })
  rt.signal.addEventListener(
    'abort',
    () => {
      gone.disconnect()
      if (open) release(open)
      open = null
    },
    { once: true },
  )
}
