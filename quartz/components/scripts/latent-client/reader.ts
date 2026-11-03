import { toHtml } from 'hast-util-to-html'
import type { LatentDoc, LatentText } from '../../../util/latent/types'
import { chunkArticles, docHead, moreSentinel, readerDocument } from '../../../util/latent/markup'
import { docOfRow, partOf, textOf, type Runtime } from './runtime'
import { renderLinks } from './search'

/**
 * Hydration for `layout: latent`. The build rendered the first screen; this makes the reader a
 * spine: scrolling moves a reading line through the open document, the chunk under it is the
 * focus, and the rail, links and probe follow. Past a document's end the next one in time is
 * mounted and the rail collapses the old one. Each focus change fetches one kNN block and the
 * preview blocks it names; a document shift fetches that document's text.
 */

const READING_LINE = 0.38

function stubText(doc: LatentDoc): LatentText {
  return {
    d: doc.d,
    id: doc.id,
    store: doc.s,
    title: doc.t,
    date: doc.date,
    prec: doc.prec,
    total: doc.n,
    part: 0,
    chunks: [],
  }
}

/** HTML of a document rendered around one part. */
async function documentHtml(rt: Runtime, d: number, part: number, focus: number): Promise<string> {
  const head = await textOf(rt, d)
  const chunks = part === 0 ? head.chunks : await partOf(rt, d, part)
  const offset = part * head.part
  return toHtml(readerDocument({ ...head, chunks }, rt.docs[d], focus, { offset }) as never)
}

// The chunk under the reading line and its offset in the document. Anything above it that
// grows or shrinks (a document mounted or dropped, a part filled, math typeset, its stylesheet
// or a font arriving) moves that offset; hold() scrolls by the same amount before the frame is
// painted, so the passage under the line never moves. Scrolling leaves the offset alone, so a
// scroll between the anchor and the check is not undone. The browser's own scroll anchoring is
// off for this layout (latent.scss), so there is one correction, not two.
const steady: { el: HTMLElement | null; y: number } = { el: null, y: 0 }

function anchorTo(el: HTMLElement | null): void {
  steady.el = el
  steady.y = el ? el.getBoundingClientRect().top + window.scrollY : 0
}

function hold(): void {
  const el = steady.el
  if (!el || !el.isConnected) return
  const y = el.getBoundingClientRect().top + window.scrollY
  const moved = y - steady.y
  steady.y = y
  if (Math.abs(moved) > 0.5) window.scrollBy({ top: moved, behavior: 'instant' })
}

/** Keep the text under the reading line still when something above it grows or shrinks. */
export function keepLine(anchor: HTMLElement | null, mutate: () => void): void {
  if (anchor?.isConnected && anchor !== steady.el) anchorTo(anchor)
  const el = steady.el
  // read the anchor's place on screen before the change: dropping documents above can shorten
  // the page so much that the browser clamps the scroll position while laying the change out,
  // and a correction by the offset alone would then carry the clamp (tens of pixels) as a jump
  const screen = el?.isConnected ? el.getBoundingClientRect().top : null
  mutate()
  if (!el || screen === null || !el.isConnected) return hold()
  const top = el.getBoundingClientRect().top
  if (Math.abs(top - screen) > 0.5)
    window.scrollTo({ top: window.scrollY + top - screen, behavior: 'instant' })
  steady.y = el.getBoundingClientRect().top + window.scrollY
}

/** Fill a part sentinel with its part, moving the sentinel on to the next one. */
async function fillMore(rt: Runtime, more: HTMLElement): Promise<void> {
  if (more.dataset.loading !== undefined) return
  more.dataset.loading = ''
  const d = Number(more.dataset.doc)
  const part = Number(more.dataset.part)
  const side = more.dataset.side as 'before' | 'after'
  const [head, chunks] = await Promise.all([textOf(rt, d), partOf(rt, d, part)])
  if (rt.signal.aborted || !more.isConnected) return
  const docEl = more.closest<HTMLElement>('.lat-doc')
  if (!docEl) return
  const html = chunkArticles(head, rt.docs[d], chunks, rt.focus)
    .map(el => toHtml(el as never))
    .join('')
  const anchor = chunkUnderLine(rt)
  keepLine(anchor, () => {
    more.insertAdjacentHTML(side === 'before' ? 'afterend' : 'beforebegin', html)
    const lo = side === 'before' ? part : Number(docEl.dataset.lo)
    const hi = side === 'after' ? part : Number(docEl.dataset.hi)
    docEl.dataset.lo = String(lo)
    docEl.dataset.hi = String(hi)
    const remaining =
      side === 'before' ? lo * head.part : head.total - Math.min(head.total, (hi + 1) * head.part)
    if (remaining <= 0) more.remove()
    else {
      more.outerHTML = toHtml(
        moreSentinel(head, side, remaining, side === 'before' ? lo - 1 : hi + 1) as never,
      )
    }
  })
}

function readingLineY(): number {
  return window.innerHeight * READING_LINE
}

/** Which chunk sits under the reading line right now. */
export function chunkUnderLine(rt: Runtime): HTMLElement | null {
  const y = readingLineY()
  let best: HTMLElement | null = null
  let bestDist = Infinity
  // a row joined to the one before it (`chunkArticles`) has no height of its own
  for (const chunk of rt.reader.querySelectorAll<HTMLElement>(
    '.lat-chunk[data-row]:not([data-joined])',
  )) {
    const rect = chunk.getBoundingClientRect()
    if (rect.bottom < y - 1 || rect.top > y + 1) {
      const dist = rect.top > y ? rect.top - y : y - rect.bottom
      if (dist < bestDist && rect.bottom > 0 && rect.top < window.innerHeight) {
        bestDist = dist
        best = chunk
      }
      continue
    }
    return chunk
  }
  return best
}

/** A row's article; a row joined to an earlier one (one equation over several rows) is that one. */
function rowChunk(rt: Runtime, row: number): HTMLElement | null {
  const chunk = rt.reader.querySelector<HTMLElement>(`.lat-chunk[data-row="${row}"]`)
  const first = chunk?.dataset.joined
  return (
    (first && rt.reader.querySelector<HTMLElement>(`.lat-chunk[data-row="${first}"]`)) || chunk
  )
}

function setFocus(rt: Runtime, row: number, options: { push?: boolean } = {}): void {
  if (rt.focus === row) return
  rt.focus = row
  rt.open = docOfRow(rt, row).d
  rt.root.dataset.latentFocus = String(row)
  rt.root.dataset.latentDoc = String(rt.open)
  for (const chunk of rt.reader.querySelectorAll<HTMLElement>('.lat-chunk[data-focus]'))
    delete chunk.dataset.focus
  const chunk = rowChunk(rt, row)
  if (chunk) chunk.dataset.focus = ''
  if (options.push !== false) {
    const url = new URL(window.location.href)
    url.searchParams.set('focus', String(row))
    history.replaceState(history.state, '', url)
  }
  void renderLinks(rt, row).catch(error => {
    if (!rt.signal.aborted) console.warn('[latent]', error)
  })
}

/** The reader's frozen head (see `docHead`), or null on a page built before it. */
function readerHead(rt: Runtime): HTMLElement | null {
  return rt.reader.querySelector<HTMLElement>(':scope > .lat-reader-head')
}

/** Retitle the frozen head for a document; it never moves, only its text changes. */
function setReaderHead(rt: Runtime, d = rt.open): void {
  const doc = rt.docs[d]
  if (!doc) return
  const html = toHtml(docHead(doc, true) as never)
  const head = readerHead(rt)
  if (head) head.outerHTML = html
  else rt.reader.insertAdjacentHTML('afterbegin', html)
  readerHead(rt)?.setAttribute('data-doc', String(d))
}

/**
 * The document the frozen head names: the last one whose own head has reached it. The reading
 * line sits at 38%, so it enters the next document while the tail of the one before still fills
 * the top band; titling by the reading line put the next title over the previous text.
 */
function docAtHead(rt: Runtime, frozen: HTMLElement): number {
  const line = frozen.getBoundingClientRect().top + 1
  let at = rt.open
  for (const head of rt.reader.querySelectorAll<HTMLElement>(
    ':scope > .lat-doc > .lat-doc-head',
  )) {
    if (head.getBoundingClientRect().top > line) break
    at = Number(head.parentElement!.dataset.doc)
  }
  return at
}

function syncReaderHead(rt: Runtime): void {
  const frozen = readerHead(rt)
  if (!frozen) return
  const d = docAtHead(rt, frozen)
  if (frozen.dataset.doc !== String(d)) setReaderHead(rt, d)
}

/** Put a document before the current reader contents (after the frozen head). */
function prependDocument(rt: Runtime, html: string): void {
  const head = readerHead(rt)
  if (head) head.insertAdjacentHTML('afterend', html)
  else rt.reader.insertAdjacentHTML('afterbegin', html)
}

/** Mount a document into the reader (replacing its stub), keeping one stub either side. */
function mountDocument(
  rt: Runtime,
  d: number,
  part: number | 'last' = 0,
): Promise<HTMLElement | null> {
  const existing = rt.reader.querySelector<HTMLElement>(
    `.lat-doc[data-doc="${d}"]:not(.lat-doc-stub)`,
  )
  if (existing) return Promise.resolve(existing)
  const pending = rt.mounting.get(d)
  if (pending) return pending

  const mounting = (async (): Promise<HTMLElement | null> => {
    const head = await textOf(rt, d)
    const index =
      part === 'last' ? Math.max(0, Math.ceil(head.total / Math.max(1, head.part)) - 1) : part
    const html = await documentHtml(rt, d, index, rt.focus)
    if (rt.signal.aborted) return null
    // Pruning can remove this stub while its data is in flight. Only replace the original slot;
    // appending a late result at the reader's end creates duplicate or out-of-order documents.
    const stub = rt.reader.querySelector<HTMLElement>(`.lat-doc-stub[data-doc="${d}"]`)
    if (!stub) return null
    const anchor = chunkUnderLine(rt)
    // the previous document mounted above must not move the text under the reading line
    keepLine(anchor, () => {
      stub.outerHTML = html
      // a document mounted above gets the stub above it in the same step: a short document
      // replacing a tall stub would otherwise shrink the page above the reading line by more
      // than the scroll offset near the top, the correction would clamp at 0, and the reading
      // line would slide onto the document before, which then mounts in turn
      if (d > 0 && d < rt.open && !rt.reader.querySelector(`.lat-doc[data-doc="${d - 1}"]`))
        prependDocument(
          rt,
          toHtml(
            readerDocument(stubText(rt.docs[d - 1]), rt.docs[d - 1], -1, { stub: true }) as never,
          ),
        )
    })
    return rt.reader.querySelector<HTMLElement>(`.lat-doc[data-doc="${d}"]:not(.lat-doc-stub)`)
  })().finally(() => {
    rt.mounting.delete(d)
  })
  rt.mounting.set(d, mounting)
  return mounting
}

function ensureStubs(rt: Runtime): void {
  const mounted = [...rt.reader.querySelectorAll<HTMLElement>('.lat-doc:not(.lat-doc-stub)')].map(
    el => Number(el.dataset.doc),
  )
  if (mounted.length === 0) return
  const lo = Math.min(...mounted)
  const hi = Math.max(...mounted)
  const stubHtml = (d: number) =>
    toHtml(readerDocument(stubText(rt.docs[d]), rt.docs[d], -1, { stub: true }) as never)
  // a stub added or a document dropped above the reading line must not move the text under it
  keepLine(chunkUnderLine(rt), () => {
    if (lo > 0 && !rt.reader.querySelector(`.lat-doc[data-doc="${lo - 1}"]`))
      prependDocument(rt, stubHtml(lo - 1))
    if (hi + 1 < rt.docs.length && !rt.reader.querySelector(`.lat-doc[data-doc="${hi + 1}"]`))
      rt.reader.insertAdjacentHTML('beforeend', stubHtml(hi + 1))
    // unmount documents two away from the open one once they are well off screen: short
    // documents (an image, a two-row base) fit several to a viewport, and pruning by count
    // alone left the reader without page above it at the top of the scroll range
    for (const el of rt.reader.querySelectorAll<HTMLElement>('.lat-doc')) {
      const d = Number(el.dataset.doc)
      if (Math.abs(d - rt.open) <= 2) continue
      const rect = el.getBoundingClientRect()
      if (rect.bottom < -window.innerHeight || rect.top > 2 * window.innerHeight) el.remove()
    }
  })
}

/** Open a row's document in the reader and put that row under the reading line. */
export async function goTo(rt: Runtime, row: number): Promise<void> {
  const doc = docOfRow(rt, row)
  let chunk = rowChunk(rt, row)
  if (!chunk) {
    const head = await textOf(rt, doc.d)
    if (rt.signal.aborted) return
    const part = Math.floor((row - doc.r0) / Math.max(1, head.part))
    const html = await documentHtml(rt, doc.d, part, row)
    if (rt.signal.aborted) return
    const prev = doc.d > 0 ? rt.docs[doc.d - 1] : null
    const next = doc.d + 1 < rt.docs.length ? rt.docs[doc.d + 1] : null
    const stub = (d: LatentDoc) =>
      toHtml(readerDocument(stubText(d), d, -1, { stub: true }) as never)
    const frozen = readerHead(rt)
    rt.reader.innerHTML = [prev ? stub(prev) : '', html, next ? stub(next) : ''].join('')
    if (frozen) rt.reader.prepend(frozen)
    chunk = rowChunk(rt, row)
  }
  if (!chunk) return
  const top = chunk.getBoundingClientRect().top + window.scrollY - readingLineY() + 8
  window.scrollTo({ top, behavior: 'instant' })
  anchorTo(chunk)
  setFocus(rt, row)
  // a scroll clamped at 0 fires no scroll event; let the reader look for stubs to mount anyway
  window.dispatchEvent(new Event('scroll'))
}

export function bindReader(rt: Runtime): void {
  // a page built before the frozen head existed gets one now
  if (!readerHead(rt)) setReaderHead(rt)
  let frame = 0
  const update = () => {
    frame = 0
    const chunk = chunkUnderLine(rt)
    if (!chunk) {
      syncReaderHead(rt)
      return
    }
    if (chunk !== steady.el) anchorTo(chunk)
    const row = Number(chunk.dataset.row)
    if (row !== rt.focus) {
      const open = rt.open
      setFocus(rt, row)
      // crossing into another document moves the one-stub-either-side window with it; without
      // this the reader runs out of page above a document mounted from its stub
      if (rt.open !== open) ensureStubs(rt)
    }
    syncReaderHead(rt)
    const near = (el: HTMLElement) => {
      const rect = el.getBoundingClientRect()
      return rect.top < window.innerHeight * 1.5 && rect.bottom > -window.innerHeight * 0.5
    }
    // approach the rest of the open document: fill the part sentinel within one viewport
    for (const more of rt.reader.querySelectorAll<HTMLElement>('.lat-more')) {
      if (near(more)) void fillMore(rt, more)
    }
    // approach the neighbours: mount the stub that is within one viewport
    for (const stub of rt.reader.querySelectorAll<HTMLElement>('.lat-doc-stub')) {
      if (near(stub)) {
        const d = Number(stub.dataset.doc)
        void mountDocument(rt, d, d < rt.open ? 'last' : 0).then(() => ensureStubs(rt))
      }
    }
  }
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(update)
  }
  window.addEventListener('scroll', schedule, { passive: true, signal: rt.signal })
  // at the top of the page a scroll up fires no scroll event, so the stub above the opening
  // document would never mount; look once on arrival
  schedule()
  // every change of the reader's height, whatever made it, is held against the anchor
  const sizes = new ResizeObserver(() => hold())
  sizes.observe(rt.reader)
  rt.signal.addEventListener('abort', () => {
    sizes.disconnect()
    anchorTo(null)
  })
  window.addEventListener('resize', schedule, { passive: true, signal: rt.signal })
  rt.signal.addEventListener('abort', () => cancelAnimationFrame(frame))
}
