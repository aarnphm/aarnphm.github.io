import { toHtml } from 'hast-util-to-html'
import { groupDocumentLinks, documentLink, linksPane, type LinkItem } from '../../../util/latent/document-links'
import type { LatentPreview } from '../../../util/latent/types'
import type { Runtime } from './runtime'

const windows = new WeakMap<Runtime, () => void>()
const WINDOW_ROWS = 28
const OVERSCAN = 6

/** Rows have one fixed line; only the visible document window remains in the DOM. */
export function renderDocumentLinks(rt: Runtime, focus: LatentPreview, items: LinkItem[], label = 'links'): void {
  windows.get(rt)?.()
  const body = rt.links.querySelector<HTMLElement>('.lat-links-body') ?? rt.links
  const pane = linksPane(focus, items, rt.kinds, label)
  body.innerHTML = toHtml(pane.children as never, { allowDangerousHtml: false })
  rt.links.scrollTop = 0
  const groups = groupDocumentLinks(items)
  const cleanups: (() => void)[] = []
  for (const list of body.querySelectorAll<HTMLOListElement>('.lat-document-list')) {
    const own = groups.filter(group => group.doc.s === list.dataset.kind)
    if (own.length <= WINDOW_ROWS) continue
    const sample = list.firstElementChild
    if (!(sample instanceof HTMLElement)) continue
    let rowHeight = sample.getBoundingClientRect().height
    if (!rowHeight) continue
    let start = -1
    const before = document.createElement('li')
    const after = document.createElement('li')
    for (const spacer of [before, after]) {
      spacer.className = 'lat-list-spacer'
      spacer.setAttribute('aria-hidden', 'true')
    }
    const paint = () => {
      const top = rt.links.getBoundingClientRect().top - list.getBoundingClientRect().top
      const next = Math.min(Math.max(0, own.length - WINDOW_ROWS), Math.max(0, Math.floor(top / rowHeight) - OVERSCAN))
      if (next === start) return
      // A tabbed chunk remains mounted until focus leaves its document.
      if (list.contains(document.activeElement) && start >= 0) return
      start = next
      const end = Math.min(own.length, start + WINDOW_ROWS)
      before.style.height = `${start * rowHeight}px`
      after.style.height = `${(own.length - end) * rowHeight}px`
      list.innerHTML = own.slice(start, end).map(group => toHtml(documentLink(group, focus.d) as never)).join('')
      list.prepend(before)
      list.append(after)
      list.start = start + 1
      rt.root.dispatchEvent(new CustomEvent('latent:links-window'))
    }
    let frame = 0
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(() => { frame = 0; paint() })
    }
    const resize = new ResizeObserver(() => {
      const mounted = list.querySelector<HTMLElement>('.lat-link-document')
      rowHeight = mounted?.getBoundingClientRect().height || rowHeight
      start = -1
      schedule()
    })
    resize.observe(rt.links)
    rt.links.addEventListener('scroll', schedule, { passive: true, signal: rt.signal })
    list.addEventListener('focusout', schedule, { signal: rt.signal })
    cleanups.push(() => {
      resize.disconnect()
      cancelAnimationFrame(frame)
      rt.links.removeEventListener('scroll', schedule)
      list.removeEventListener('focusout', schedule)
    })
    paint()
  }
  const cleanup = () => cleanups.forEach(stop => stop())
  windows.set(rt, cleanup)
  rt.signal.addEventListener('abort', cleanup, { once: true })
}
