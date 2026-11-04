import { toHtml } from 'hast-util-to-html'
import type { Runtime } from './runtime'
import {
  groupDocumentLinks,
  documentLink,
  linksPane,
  type LinkItem,
} from '../../../util/latent/document-links'
import { inScope, type LatentPreview } from '../../../util/latent/types'

interface ListWindow {
  cleanup: () => void
  rows: number[]
  active: number
  reveal: Map<number, () => HTMLElement | null>
}

const windows = new WeakMap<Runtime, ListWindow>()
const OVERSCAN = 6

/** Metadata sufficient for a grouped list; text sidecars wait for a chunk preview. */
export function linkItems(rt: Runtime, rows: [number, number][]): LinkItem[] {
  return rows.flatMap(([row, score]) => {
    const doc = rt.docByRow[row]
    if (!doc) return []
    return [
      {
        doc,
        score,
        inScope: inScope(doc.id, rt.scope),
        preview: { r: row, d: doc.d, s: doc.s, t: doc.t, c: row - doc.r0, p: null, l: '', x: '' },
      },
    ]
  })
}

export function stepDocumentLinks(rt: Runtime, step: number): void {
  const state = windows.get(rt)
  if (!state?.rows.length) return
  state.active = Math.max(0, Math.min(state.rows.length - 1, state.active + step))
  const row = state.reveal.get(state.rows[state.active])?.()
  for (const item of rt.links.querySelectorAll('[data-keyboard-active]'))
    item.removeAttribute('data-keyboard-active')
  row?.setAttribute('data-keyboard-active', '')
  row?.scrollIntoView({ block: 'nearest' })
}

/** Rows have one fixed line; only the visible document window remains in the DOM. */
export function renderDocumentLinks(
  rt: Runtime,
  focus: LatentPreview,
  items: LinkItem[],
  label = 'links',
): void {
  rt.root.dispatchEvent(new CustomEvent('latent:links-render'))
  windows.get(rt)?.cleanup()
  const body = rt.links.querySelector<HTMLElement>('.lat-links-body') ?? rt.links
  const pane = linksPane(focus, items, rt.kinds, label)
  body.innerHTML = toHtml(pane.children as never, { allowDangerousHtml: false })
  rt.links.scrollTop = 0
  const groups = groupDocumentLinks(items)
  const cleanups: (() => void)[] = []
  const state: ListWindow = { cleanup, rows: [], active: -1, reveal: new Map() }
  windows.set(rt, state)
  function cleanup() {
    cleanups.forEach(stop => stop())
    rt.signal.removeEventListener('abort', cleanup)
  }
  for (const list of body.querySelectorAll<HTMLOListElement>('.lat-document-list')) {
    const own = groups.filter(group => group.doc.s === list.dataset.kind)
    state.rows.push(...own.map(group => group.doc.d))
    const find = (doc: number) =>
      list.querySelector<HTMLElement>(`.lat-link-document[data-doc="${doc}"]`)
    const sample = list.firstElementChild
    if (!(sample instanceof HTMLElement)) continue
    let rowHeight = sample.getBoundingClientRect().height
    let viewportHeight = rt.links.clientHeight
    const windowRows = () => Math.ceil(viewportHeight / rowHeight) + 2 * OVERSCAN
    if (!rowHeight) {
      for (const group of own) state.reveal.set(group.doc.d, () => find(group.doc.d))
      continue
    }
    let start = -1
    const before = document.createElement('li')
    const after = document.createElement('li')
    for (const spacer of [before, after]) {
      spacer.className = 'lat-list-spacer'
      spacer.setAttribute('aria-hidden', 'true')
    }
    const paint = (at?: number) => {
      const top = rt.links.getBoundingClientRect().top - list.getBoundingClientRect().top
      const next = Math.min(
        Math.max(0, own.length - windowRows()),
        Math.max(0, at ?? Math.floor(top / rowHeight) - OVERSCAN),
      )
      if (next === start) return
      if (at === undefined && list.contains(document.activeElement) && start >= 0) return
      start = next
      const end = Math.min(own.length, start + windowRows())
      before.style.height = `${start * rowHeight}px`
      after.style.height = `${(own.length - end) * rowHeight}px`
      list.innerHTML = own
        .slice(start, end)
        .map((group, index) => {
          const row = documentLink(group, focus.d)
          row.properties.ariaSetSize = own.length
          row.properties.ariaPosInSet = start + index + 1
          return toHtml(row as never)
        })
        .join('')
      list.prepend(before)
      list.append(after)
      list.start = start + 1
    }
    own.forEach((group, index) =>
      state.reveal.set(group.doc.d, () => {
        if (!find(group.doc.d)) paint(Math.max(0, index - OVERSCAN))
        return find(group.doc.d)
      }),
    )
    const onTab = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || !(event.target instanceof HTMLElement)) return
      const row = event.target.closest<HTMLElement>('.lat-link-document')
      if (!row) return
      const index = own.findIndex(group => group.doc.d === Number(row.dataset.doc))
      const targets = [...row.querySelectorAll<HTMLElement>('a')]
      const edge = event.shiftKey ? targets[0] : targets[targets.length - 1]
      const neighbour = event.shiftKey ? row.previousElementSibling : row.nextElementSibling
      if (event.target !== edge || !neighbour?.classList.contains('lat-list-spacer')) return
      const next = index + (event.shiftKey ? -1 : 1)
      if (next < 0 || next >= own.length) return
      event.preventDefault()
      const revealed = state.reveal.get(own[next].doc.d)?.()
      const links = revealed?.querySelectorAll<HTMLAnchorElement>('a')
      const link = event.shiftKey ? links?.[links.length - 1] : links?.[0]
      link?.focus()
    }
    let frame = 0
    const schedule = () => {
      if (!frame)
        frame = requestAnimationFrame(() => {
          frame = 0
          paint()
        })
    }
    const resize = new ResizeObserver(() => {
      const mounted = list.querySelector<HTMLElement>('.lat-link-document')
      const height = mounted?.getBoundingClientRect().height
      const viewport = rt.links.clientHeight
      if (!height || (height === rowHeight && viewport === viewportHeight)) return
      rowHeight = height
      viewportHeight = viewport
      start = -1
      schedule()
    })
    resize.observe(rt.links)
    rt.links.addEventListener('scroll', schedule, { passive: true, signal: rt.signal })
    list.addEventListener('focusout', schedule, { signal: rt.signal })
    list.addEventListener('keydown', onTab, { signal: rt.signal })
    cleanups.push(() => {
      resize.disconnect()
      cancelAnimationFrame(frame)
      rt.links.removeEventListener('scroll', schedule)
      list.removeEventListener('focusout', schedule)
      list.removeEventListener('keydown', onTab)
    })
    paint()
  }
  rt.signal.addEventListener('abort', cleanup, { once: true })
}
