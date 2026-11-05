import type { Runtime } from './runtime'
import { createPdfEmbed } from '../popover/pdf-preview'

/**
 * A pdf passage's page as a popover: the site's pdf embed (pdf.inline.ts, the viewer note pages
 * and the arXiv previews use) in the site's popover frame (popover.scss), opened on the page the
 * passage came from with its box marked as the reader marks focus: a rule at its left, over a
 * wash of the pdf colour.
 *
 * Desktop: hovering a pdf passage's locus (its `p. N ¶M` and page glyph) opens it, leaving both
 * closes it, and a click on the glyph pins it until a click elsewhere or Escape. Touch: a tap on
 * the glyph opens it pinned. It sits beside the passage, never over it, or under the locus when
 * the column has no room beside it.
 *
 * There is one popover. Nothing loads before the first open; pdf.js comes with the runtime's
 * first mount, and its document cache keeps the file across pages and reopenings.
 */

const LOCUS = '.lat-locus'
const TOGGLE = 'button.lat-folio-toggle[data-folio]'
const SHOW_DELAY = 140
const HIDE_DELAY = 220
const GAP = 10
const EDGE = 8
const MIN_WIDTH = 320
const MAX_WIDTH = 600
// letter/A4 page plus the embed toolbar, so one page shows whole at the chosen width
const PAGE_ASPECT = 1.294
const TOOLBAR = 46

type Box = [number, number, number, number]

function parseBox(raw: string | undefined): Box | null {
  const box = (raw ?? '').split(',').map(Number)
  if (box.length !== 4 || !box.every(n => Number.isFinite(n) && n >= 0 && n <= 1000)) return null
  const [x0, y0, x1, y1] = box
  return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : null
}

/** Put the passage's box on its page slot, under the text layer so the text stays selectable. */
function markPage(pdf: HTMLElement, page: number, box: Box): HTMLElement | null {
  const slot = pdf.querySelector<HTMLElement>(`.pdf-embed-page[data-pdf-page-number="${page}"]`)
  if (!slot) return null
  const have = slot.querySelector<HTMLElement>(':scope > .lat-folio-mark')
  if (have) return have
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
  slot.insertBefore(mark, slot.querySelector(':scope > .pdf-embed-text-layer'))
  return mark
}

/** Scroll the viewer so the mark sits a third of the way down, when it is not already in view. */
function revealMark(pdf: HTMLElement, mark: HTMLElement): void {
  const scroller = pdf.querySelector<HTMLElement>('.pdf-viewer-container')
  if (!scroller) return
  const view = scroller.getBoundingClientRect()
  const at = mark.getBoundingClientRect()
  if (at.top >= view.top + 8 && at.bottom <= view.bottom - 8) return
  scroller.scrollTop += at.top - view.top - scroller.clientHeight / 3
}

interface Placement {
  side: 'right' | 'left' | 'bottom' | 'top'
  x: number
  y: number
  width: number
  height: number
  arrow: number
}

/**
 * Beside the passage (right, else left) with the arrow at the glyph; on a narrow screen, under
 * the locus (else over it) across the width.
 */
function place(article: DOMRect, anchor: DOMRect): Placement {
  const vw = document.documentElement.clientWidth
  const vh = window.innerHeight
  const right = vw - article.right - GAP - EDGE
  const left = article.left - GAP - EDGE
  const pageHeight = (w: number) => Math.round((w - 16) * PAGE_ASPECT + TOOLBAR)
  if (Math.max(right, left) >= MIN_WIDTH) {
    const side = right >= MIN_WIDTH ? 'right' : 'left'
    const width = Math.min(MAX_WIDTH, side === 'right' ? right : left)
    const height = Math.min(vh - 2 * EDGE, pageHeight(width))
    const mid = anchor.top + anchor.height / 2
    const y = Math.max(EDGE, Math.min(vh - EDGE - height, mid - height * 0.3))
    const x = side === 'right' ? article.right + GAP : article.left - GAP - width
    return { side, x, y, width, height, arrow: mid - y }
  }
  const width = Math.min(MAX_WIDTH, vw - 2 * EDGE)
  const below = vh - anchor.bottom - GAP - EDGE
  const above = anchor.top - GAP - EDGE
  const side = below >= above ? 'bottom' : 'top'
  const height = Math.max(160, Math.min(pageHeight(width), side === 'bottom' ? below : above))
  const x = Math.max(EDGE, Math.min(vw - EDGE - width, anchor.left - 24))
  const y = side === 'bottom' ? anchor.bottom + GAP : anchor.top - GAP - height
  return { side, x, y, width, height, arrow: anchor.left + anchor.width / 2 - x }
}

export function bindFolios(rt: Runtime): void {
  const hover = window.matchMedia('(hover: hover) and (pointer: fine)')
  let popover: HTMLElement | null = null
  let inner: HTMLDivElement | null = null
  let pdf: HTMLElement | null = null
  let watch: MutationObserver | null = null
  // what the popover shows now: the glyph it hangs from
  let current: HTMLButtonElement | null = null
  let pinned = false
  let showTimer = 0
  let hideTimer = 0
  let releaseTimer = 0
  let markFrame = 0
  let moveFrame = 0

  const frameFor = (): { popover: HTMLElement; inner: HTMLDivElement } => {
    if (popover?.isConnected && inner) return { popover, inner }
    const el = document.createElement('div')
    el.className = 'popover lat-folio-popover'
    const arrow = document.createElement('div')
    arrow.className = 'popover-arrow'
    const panel = document.createElement('div')
    panel.className = 'popover-inner'
    panel.dataset.contentType = 'application/pdf'
    el.append(arrow, panel)
    el.addEventListener('pointerenter', () => window.clearTimeout(hideTimer))
    el.addEventListener('pointerleave', event => {
      if (!pinned && event.pointerType === 'mouse') scheduleHide()
    })
    document.body.append(el)
    popover = el
    inner = panel
    pdf = null
    return { popover: el, inner: panel }
  }

  const releasePdf = () => {
    watch?.disconnect()
    watch = null
    cancelAnimationFrame(markFrame)
    markFrame = 0
    if (pdf) window.quartzPdfEmbeds?.cleanup(pdf)
    pdf?.remove()
    pdf = null
  }

  const position = () => {
    if (!popover || !current) return
    const article = current.closest<HTMLElement>('.lat-chunk')
    const locus = current.closest<HTMLElement>(LOCUS) ?? current
    if (!article?.isConnected) return hide()
    const anchor = current.getBoundingClientRect()
    // the glyph scrolled out of the window: the popover has nothing to hang from
    if (anchor.bottom < 0 || anchor.top > window.innerHeight) return hide()
    const at = place(article.getBoundingClientRect(), locus.getBoundingClientRect())
    popover.dataset.placement = at.side
    popover.style.transform = `translate(${Math.round(at.x)}px, ${Math.round(at.y)}px)`
    // the popover's padding is its arrow gutter; the inner panel takes the rest
    const padding = parseFloat(getComputedStyle(popover).paddingLeft) || 0
    popover.style.setProperty('--lat-folio-w', `${Math.round(at.width - 2 * padding)}px`)
    popover.style.setProperty('--lat-folio-h', `${Math.round(at.height - 2 * padding)}px`)
    const arrow = popover.querySelector<HTMLElement>(':scope > .popover-arrow')
    if (arrow) {
      arrow.style.cssText = ''
      const inset = 'var(--popover-arrow-inset)'
      if (at.side === 'right' || at.side === 'left') {
        arrow.style.top = `calc(${Math.round(at.arrow)}px - var(--popover-arrow-size) / 2)`
        arrow.style[at.side === 'right' ? 'left' : 'right'] = inset
      } else {
        arrow.style.left = `calc(${Math.round(at.arrow)}px - var(--popover-arrow-size) / 2)`
        arrow.style[at.side === 'bottom' ? 'top' : 'bottom'] = inset
      }
    }
  }

  const load = (src: string, page: number, box: Box | null, title: string) => {
    const { inner: panel } = frameFor()
    if (pdf?.isConnected && pdf.dataset.latSrc === src && pdf.dataset.pdfPage === `${page}`) return
    releasePdf()
    const embed = createPdfEmbed(src, title, {
      preload: false,
      annotations: false,
      overscan: 1,
      maxPixelRatio: 2,
      textLayer: true,
    })
    embed.dataset.pdfPage = String(page)
    embed.dataset.latSrc = src
    panel.replaceChildren(embed)
    pdf = embed
    let revealed = false
    const observer = new MutationObserver(() => {
      if (markFrame || !box) return
      markFrame = requestAnimationFrame(() => {
        markFrame = 0
        if (embed.dataset.pdfStatus !== 'loaded') return
        const mark = markPage(embed, page, box)
        if (mark && !revealed) {
          revealed = true
          revealMark(embed, mark)
        }
      })
    })
    observer.observe(embed, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['data-pdf-status'],
    })
    watch = observer
    window.quartzPdfEmbeds?.mount(panel)
  }

  const show = (toggle: HTMLButtonElement, pin: boolean) => {
    window.clearTimeout(showTimer)
    window.clearTimeout(hideTimer)
    window.clearTimeout(releaseTimer)
    const src = toggle.dataset.folio
    const page = Number(toggle.dataset.folioPage)
    const article = toggle.closest<HTMLElement>('.lat-chunk[data-row]')
    if (!src || !article || !Number.isInteger(page) || page < 1) return
    if (!window.quartzPdfEmbeds) {
      // no pdf runtime on this page: the locus link is the way to the page
      if (pin) article.querySelector<HTMLAnchorElement>(`${LOCUS} a[href]`)?.click()
      return
    }
    const title =
      article.closest('.lat-doc')?.querySelector('.lat-doc-title')?.textContent?.trim() ||
      src.split('/').pop() ||
      'document.pdf'
    if (current && current !== toggle) current.setAttribute('aria-expanded', 'false')
    current = toggle
    pinned = pin
    toggle.setAttribute('aria-expanded', 'true')
    load(src, page, parseBox(article.dataset.box), title)
    const { popover: el } = frameFor()
    el.setAttribute('role', 'dialog')
    el.setAttribute('aria-label', `${title}, page ${page}`)
    el.classList.toggle('lat-folio-pinned', pin)
    position()
    // one popover on the page at a time: the site's step aside. This one keeps its own open
    // class, since the site clears `active-popover` from every popover whenever a link is left
    for (const other of document.querySelectorAll('.popover.active-popover'))
      other.classList.remove('active-popover')
    el.classList.add('lat-folio-open')
  }

  function hide() {
    window.clearTimeout(showTimer)
    window.clearTimeout(hideTimer)
    current?.setAttribute('aria-expanded', 'false')
    current = null
    pinned = false
    popover?.classList.remove('lat-folio-open', 'lat-folio-pinned')
    // the fade runs first; the document stays in the runtime's cache for a reopening
    window.clearTimeout(releaseTimer)
    releaseTimer = window.setTimeout(() => {
      if (!current) releasePdf()
    }, 400)
  }

  function scheduleHide() {
    window.clearTimeout(showTimer)
    window.clearTimeout(hideTimer)
    hideTimer = window.setTimeout(() => {
      if (!pinned) hide()
    }, HIDE_DELAY)
  }

  const locusToggle = (target: EventTarget | null) =>
    (target as Element | null)?.closest?.(LOCUS)?.querySelector<HTMLButtonElement>(TOGGLE) ?? null

  // ---- desktop: hover the locus ------------------------------------------------------------
  rt.reader.addEventListener(
    'pointerover',
    event => {
      if (event.pointerType !== 'mouse' || !hover.matches) return
      const toggle = locusToggle(event.target)
      if (!toggle) return
      window.clearTimeout(hideTimer)
      if (pinned || current === toggle) return
      window.clearTimeout(showTimer)
      showTimer = window.setTimeout(() => show(toggle, false), SHOW_DELAY)
    },
    { signal: rt.signal },
  )
  rt.reader.addEventListener(
    'pointerout',
    event => {
      if (event.pointerType !== 'mouse') return
      const locus = (event.target as Element | null)?.closest?.(LOCUS)
      if (!locus || !locusToggle(locus)) return
      const to = event.relatedTarget as Node | null
      if (to && (locus.contains(to) || popover?.contains(to))) return
      window.clearTimeout(showTimer)
      if (!pinned) scheduleHide()
    },
    { signal: rt.signal },
  )
  // keyboard: the glyph's focus opens it, its blur closes it unless it was pinned
  rt.reader.addEventListener(
    'focusin',
    event => {
      const toggle = (event.target as Element | null)?.closest?.<HTMLButtonElement>(TOGGLE)
      if (toggle?.matches(':focus-visible') && !pinned) show(toggle, false)
    },
    { signal: rt.signal },
  )
  rt.reader.addEventListener(
    'focusout',
    event => {
      const toggle = (event.target as Element | null)?.closest?.(TOGGLE)
      const to = event.relatedTarget as Node | null
      if (toggle && !pinned && !(to && popover?.contains(to))) scheduleHide()
    },
    { signal: rt.signal },
  )

  // ---- click: the glyph pins the popover (the only way in on touch) -------------------------
  rt.reader.addEventListener(
    'click',
    event => {
      const toggle = (event.target as Element | null)?.closest?.<HTMLButtonElement>(TOGGLE)
      if (!toggle) return
      event.preventDefault()
      if (current === toggle && pinned) hide()
      else show(toggle, true)
    },
    { signal: rt.signal },
  )
  document.addEventListener(
    'pointerdown',
    event => {
      if (!current) return
      const target = event.target as Node | null
      if (target && (popover?.contains(target) || current.contains(target))) return
      hide()
    },
    { capture: true, signal: rt.signal },
  )
  document.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Escape' || !current) return
      const toggle = current
      const inside = popover?.contains(document.activeElement) ?? false
      hide()
      if (inside) toggle.focus({ preventScroll: true })
    },
    { signal: rt.signal },
  )

  // the popover is fixed; the passage scrolls under it
  const follow = () => {
    if (!current || moveFrame) return
    moveFrame = requestAnimationFrame(() => {
      moveFrame = 0
      position()
    })
  }
  window.addEventListener('scroll', follow, { passive: true, signal: rt.signal })
  window.addEventListener('resize', follow, { passive: true, signal: rt.signal })

  rt.signal.addEventListener(
    'abort',
    () => {
      window.clearTimeout(showTimer)
      window.clearTimeout(hideTimer)
      window.clearTimeout(releaseTimer)
      cancelAnimationFrame(moveFrame)
      releasePdf()
      popover?.remove()
      popover = null
      current = null
    },
    { once: true },
  )
}
