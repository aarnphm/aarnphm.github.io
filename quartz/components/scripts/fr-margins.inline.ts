// Margin column of /fr pages, Locher style: numbered `§ label (n)` entries (review of the section's
// cards through /api/flashcards, the reader's notes, the lexicon, citing lessons) beside the block
// that carries the same number. The markup comes from renderPage/fr-margins.ts. Under the width
// that leaves a margin, each group folds under its paragraph and the label toggles it.

import { mountClamps, type ClampFit } from './fr-margins-clamp'
import { layout, mountNotes, mountReview } from './fr-margins-review'

// Room for a 15.5rem column and its narrowest 3rem gutter left of the text (fr-margins.scss).
const MARGIN_COLUMN = 248 + 48 + 16
const GROUP_GAP = 12
// a leader stops this short of the number, or of the line a sidenote marks
const STOP = 6
// stacked elbows take their own trunk columns through the gutter, this far apart
const STEP = 4
// where a section's rail stands between the gutter and the number, the leader breaks at the rail
// and a dash this long resumes before the number, this short of it (Locher's broken leaders)
const TICK = 6
const TICK_STOP = 4

const SVG = 'http://www.w3.org/2000/svg'

/** the middle of an element's first line of text, in the viewport */
function firstLine(el: HTMLElement): number {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, node =>
    node.textContent?.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP,
  )
  const text = walker.nextNode()
  if (text) {
    const range = document.createRange()
    range.selectNodeContents(text)
    const rect = range.getClientRects()[0]
    if (rect) return rect.top + rect.height / 2
  }
  const r = el.getBoundingClientRect()
  return r.top + r.height / 2
}

function mountLayout(article: HTMLElement, signal: AbortSignal) {
  const groups = Array.from(article.querySelectorAll<HTMLElement>('.fr-group'))
  if (groups.length === 0) return () => {}
  // a group in a table cell would be clipped by the table's scroller (margin) or squeezed into the
  // cell (inline), so it leaves the cell: beside the article in the margin, after the table when folded
  const anchors = new Map(groups.map(g => [g, g.parentElement as HTMLElement]))
  const scroller = (g: HTMLElement) =>
    anchors.get(g)!.closest<HTMLElement>('.table-container, table')
  const place = (group: HTMLElement, mode: string) => {
    const host = scroller(group)
    if (!host) return
    group.toggleAttribute('data-out', true)
    if (mode === 'margin') {
      if (group.parentElement !== article) article.append(group)
    } else {
      // groups of one table stack in document order after it, not each straight behind the host
      const behind = tail.get(host) ?? host
      if (behind.nextElementSibling !== group) behind.after(group)
      tail.set(host, group)
    }
  }
  const tail = new Map<Element, Element>()
  const leaders = document.createElementNS(SVG, 'svg')
  leaders.classList.add('fr-leaders')
  leaders.setAttribute('aria-hidden', 'true')
  article.append(leaders)
  // a group's entries each take a leader to the block's number; a sidenote takes one
  const leaderOf = new Map<HTMLElement, SVGPathElement[]>()
  // a sidenote keeps its box; neighbours take different tones (fr-margins.scss), so two boxes
  // stacked in the margin read apart
  const notes = Array.from(article.querySelectorAll<HTMLElement>('.sidenote-content'))
  const labelOf = (content: HTMLElement) =>
    content.previousElementSibling?.querySelector<HTMLElement>('.sidenote-label') ?? null
  notes.forEach((content, i) => (content.dataset.frTone = String((i % 3) + 1)))

  // folded, an open entry's body leaves the parenthesis before the text for a fold under the block
  const entries = Array.from(article.querySelectorAll<HTMLElement>('.fr-entry'))
  const bodies = new Map(
    entries.map(e => [e, e.querySelector<HTMLElement>(':scope > .fr-entry-body')!]),
  )
  const settle = (entry: HTMLElement) => {
    const body = bodies.get(entry)!
    const group = entry.parentElement as HTMLElement
    const anchor = anchors.get(group)!
    const out =
      article.dataset.frMode === 'inline' &&
      entry.hasAttribute('data-open') &&
      group.parentElement === anchor
    if (out) {
      let fold = anchor.querySelector<HTMLElement>(':scope > .fr-fold')
      if (!fold) {
        fold = document.createElement('span')
        fold.className = 'fr-fold'
        anchor.append(fold)
      }
      if (body.parentElement !== fold) fold.append(body)
    } else {
      if (body.parentElement !== entry) entry.append(body)
      const fold = anchor.querySelector(':scope > .fr-fold')
      if (fold && fold.childElementCount === 0) fold.remove()
    }
  }

  for (const label of article.querySelectorAll<HTMLButtonElement>('.fr-entry-label')) {
    label.addEventListener(
      'click',
      () => {
        if (article.dataset.frMode !== 'inline') return
        const entry = label.closest<HTMLElement>('.fr-entry')
        if (!entry) return
        const open = entry.toggleAttribute('data-open')
        label.setAttribute('aria-expanded', String(open))
        settle(entry)
      },
      { signal },
    )
  }

  // the number is the cross-reference; pointing at either side lights the other
  const siblings = new Map<HTMLElement, HTMLElement[]>()
  for (const group of groups) {
    const anchor = anchors.get(group)!
    siblings.set(anchor, [...(siblings.get(anchor) ?? []), group])
  }
  for (const [anchor, mine] of siblings) {
    const light = (on: boolean) => {
      anchor.classList.toggle('fr-hot', on)
      for (const group of mine) {
        group.classList.toggle('fr-hot', on)
        for (const path of leaderOf.get(group) ?? []) path.classList.toggle('fr-hot', on)
      }
    }
    for (const node of [anchor, ...mine]) {
      node.addEventListener('pointerenter', () => light(true), { signal })
      node.addEventListener('pointerleave', () => light(false), { signal })
    }
  }
  for (const content of notes) {
    const light = (on: boolean) => {
      for (const path of leaderOf.get(content) ?? []) path.classList.toggle('fr-hot', on)
    }
    for (const node of [content, labelOf(content)]) {
      node?.addEventListener('pointerenter', () => light(true), { signal })
      node?.addEventListener('pointerleave', () => light(false), { signal })
    }
  }

  // the article's own blocks: one that grows or shrinks (a figure leaving its intrinsic size, a
  // fold, an image) moves every number below it, often without resizing the article, which the
  // page grid may stretch. The last one's bottom is where the text ends
  const blocks = Array.from(article.children).filter(
    (child): child is HTMLElement =>
      child instanceof HTMLElement && !child.matches('.fr-group, .fr-leaders'),
  )
  const clamps = mountClamps(article, () => queue(), signal)

  // a citation's reference sits in a margin bibliographie, which stands in for its popover; pointing
  // at the citation opens a cut entry until the pointer leaves both, an open entry stays as it is
  for (const cite of article.querySelectorAll<HTMLAnchorElement>('a[data-bib]')) {
    const id = decodeURIComponent(new URL(cite.href).hash.slice(1))
    const entry = id
      ? article.querySelector<HTMLElement>(`.fr-gloss#${CSS.escape(id)}`)?.closest<HTMLElement>(
          '.fr-entry',
        )
      : null
    if (!entry) continue
    let held = false
    const release = (event: PointerEvent) => {
      const to = event.relatedTarget as Node | null
      if (!held || (to && (cite.contains(to) || entry.contains(to)))) return
      held = false
      clamps.hold(entry, false)
    }
    cite.addEventListener(
      'pointerenter',
      () => {
        if (held || article.dataset.frMode !== 'margin') return
        held = clamps.hold(entry, true)
      },
      { signal },
    )
    cite.addEventListener('pointerleave', release, { signal })
    entry.addEventListener('pointerleave', release, { signal })
  }
  const measured = new WeakMap<Element, { width: number; height: number }>()
  const measureBox = (element: Element) => {
    const rect = element.getBoundingClientRect()
    measured.set(element, { width: rect.width, height: rect.height })
    return rect
  }
  const paint = () => {
    const box = measureBox(article)
    const mode =
      article.closest('.preview-container') || box.left < MARGIN_COLUMN ? 'inline' : 'margin'
    article.dataset.frMode = mode
    for (const label of article.querySelectorAll<HTMLButtonElement>('.fr-entry-label')) {
      if (mode === 'margin') {
        label.removeAttribute('aria-expanded')
        label.tabIndex = -1
      } else {
        label.setAttribute(
          'aria-expanded',
          String(label.closest('.fr-entry')?.hasAttribute('data-open') ?? false),
        )
        label.removeAttribute('tabindex')
      }
    }
    tail.clear()
    for (const group of groups) place(group, mode)
    for (const entry of entries) settle(entry)
    leaders.replaceChildren()
    leaderOf.clear()
    clamps.reset()
    article.style.removeProperty('margin-block-end')
    if (mode === 'inline') {
      for (const group of groups) group.style.removeProperty('top')
      leaders.removeAttribute('viewBox')
      return
    }
    leaders.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`)
    leaders.setAttribute('width', String(box.width))
    leaders.setAttribute('height', String(box.height))

    // Apply the base positions together, then measure once for the placement calculation.
    const hoisted = (g: HTMLElement) => g.parentElement === article
    const mid = (r: DOMRect) => r.top + r.height / 2
    const groupBases = groups.map(group => {
      const anchor = anchors.get(group)!
      const rect = measureBox(anchor)
      const visible = anchor.checkVisibility({ visibilityProperty: true })
      return { group, anchor, rect, visible, top: hoisted(group) ? rect.top - box.top : 0 }
    })
    for (const { group, rect, top, visible } of groupBases) {
      group.style.top = `${top}px`
      if (visible)
        group.style.setProperty('--fr-dx', hoisted(group) ? '0px' : `${rect.left - box.left}px`)
    }
    for (const content of notes) content.style.removeProperty('translate')
    const fits = clamps.measure(
      groupBases.filter(({ visible }) => visible).map(({ group }) => group),
    )
    interface Item {
      el: HTMLElement
      side: 'left' | 'right'
      top: number
      height: number
      left: number
      right: number
      end: { x: number; y: number }
      /** Centres relative to the final top, updated by the clamp's flex-column model. */
      heads: number[]
      tick?: [number, number]
      move: (by: number) => void
      fit?: (room: number) => ClampFit
    }
    const items: Item[] = []
    for (const { group, anchor, top, visible } of groupBases) {
      const mark = anchor.querySelector<HTMLElement>(':scope > .fr-n')
      const label = group.querySelector<HTMLElement>('.fr-entry-label')
      if (!mark || !label || !visible) continue
      const m = mark.getBoundingClientRect()
      const rect = group.getBoundingClientRect()
      const align = mid(m) - mid(label.getBoundingClientRect())
      let markX = Math.round(m.left - box.left)
      const li = anchor.closest<HTMLElement>('li')
      if (li && getComputedStyle(li).listStylePosition === 'outside') {
        const list = li.parentElement!.getBoundingClientRect()
        markX = Math.round(
          Math.min(
            list.left,
            li.getBoundingClientRect().left - 1.5 * parseFloat(getComputedStyle(li).fontSize),
          ) - box.left,
        )
      }
      items.push({
        el: group,
        side: 'left',
        top: rect.top + align,
        height: rect.height,
        left: rect.left,
        right: rect.right,
        heads: [],
        end: { x: -STOP, y: mid(m) - box.top },
        tick:
          markX > STOP + TICK + TICK_STOP
            ? [markX - TICK_STOP - TICK, markX - TICK_STOP]
            : undefined,
        move: by => {
          group.style.top = `${top + align + by}px`
        },
        fit: fits.get(group),
      })
    }
    for (const content of notes) {
      const label = labelOf(content)
      const side = content.classList.contains('sidenote-right') ? 'right' : 'left'
      if (!label || !content.classList.contains(`sidenote-${side}`)) continue
      if (!label.checkVisibility({ visibilityProperty: true })) continue
      const l = mid(label.getBoundingClientRect())
      const head = firstLine(content)
      const dy = l - head
      const rect = content.getBoundingClientRect()
      items.push({
        el: content,
        side,
        top: rect.top + dy,
        height: rect.height,
        left: rect.left,
        right: rect.right,
        heads: [head - rect.top],
        end: { x: side === 'left' ? -STOP : box.width + STOP, y: l - box.top },
        move: by => {
          content.style.translate = `0 ${dy + by}px`
        },
      })
    }
    items.sort((x, y) => x.top - y.top)
    // the column stops where the text does, not where the grid stretched the article to
    const end = blocks.reduce((low, block) => {
      const rect = block.getBoundingClientRect()
      return rect.height > 0 ? Math.max(low, rect.bottom) : low
    }, -Infinity)
    const limit = Number.isFinite(end) ? Math.min(end, box.bottom) : box.bottom
    // top-down: each entry stands at its number, or below the one before it
    const floors = { left: -Infinity, right: -Infinity }
    const placed: number[] = []
    for (const [i, item] of items.entries()) {
      const floor = floors[item.side]
      const at = Math.max(item.top, floor + GROUP_GAP)
      // a group ends before the next entry on its side would stand; the last one may also take
      // the room above it, since the tail of the column is stacked from the text's end upward
      const next = items.slice(i + 1).find(other => other.side === item.side)
      const fit = item.fit?.(
        next ? next.top - GROUP_GAP - at : limit - Math.max(floor + GROUP_GAP, box.top),
      )
      if (fit) {
        item.height = fit.height
        item.heads = fit.heads
      }
      placed[i] = at
      floors[item.side] = at + item.height
    }
    // bottom-up: an entry that would run past the text's end rises until it ends there, and lifts
    // the ones above it in turn, so a body opened near the end grows upward and the page keeps its
    // height. A column taller than the article keeps the top-down stack
    for (const side of ['left', 'right'] as const) {
      const lifted = [...placed]
      let ceiling = limit
      let fits = true
      for (let i = items.length - 1; i >= 0; i--) {
        if (items[i].side !== side) continue
        if (lifted[i] + items[i].height <= ceiling) break
        lifted[i] = ceiling - items[i].height
        if (lifted[i] < box.top) {
          fits = false
          break
        }
        ceiling = lifted[i] - GROUP_GAP
      }
      if (!fits) continue
      floors[side] = -Infinity
      for (const [i, item] of items.entries())
        if (item.side === side) {
          placed[i] = lifted[i]
          floors[side] = Math.max(floors[side], lifted[i] + item.height)
        }
    }
    // entries that point at the same line share one leader: a trunk through the gutter that each
    // head's arm meets, one stem from the trunk to the target, one dash before the number. The
    // stroke is translucent, so a stretch drawn twice would read darker than its neighbours
    interface Target {
      side: 'left' | 'right'
      /** where the stem ends and the y of what it points at, in the article's frame */
      x1: number
      y1: number
      tick?: [number, number]
      /** each head's y and the entry's near edge, in the article's frame */
      heads: { item: Item; x0: number; y0: number }[]
      lo: number
      hi: number
      level: number
    }
    const targets = new Map<string, Target>()
    for (const [i, item] of items.entries()) {
      const placedTop = placed[i]
      item.move(placedTop - item.top)
      const y1 = Math.round(item.end.y) + 0.5
      const key = `${item.side} ${y1}`
      let target = targets.get(key)
      if (!target) {
        target = { side: item.side, x1: item.end.x, y1, heads: [], lo: y1, hi: y1, level: 0 }
        targets.set(key, target)
      }
      target.tick ??= item.tick
      // every entry of a group points at the same number: the first level with it, the rest as
      // elbows that meet its line
      for (const head of item.heads) {
        const y0 = Math.round(placedTop + head - box.top) + 0.5
        target.heads.push({
          item,
          y0,
          x0: (item.side === 'left' ? item.right : item.left) - box.left,
        })
        target.lo = Math.min(target.lo, y0)
        target.hi = Math.max(target.hi, y0)
      }
    }
    // a column too tall to lift may still run past the article; the article grows to hold it, so
    // the column never reaches the footer
    const overrun = Math.max(floors.left, floors.right) + GROUP_GAP - box.bottom
    if (overrun > 0) article.style.setProperty('margin-block-end', `${Math.ceil(overrun)}px`)
    clamps.refocus()
    // one unbroken hairline from the entry to what it points at: straight when the entry is
    // level with it, an elbow through the gutter when the stack pushed the entry down. A trunk
    // spans its target and its heads; a later trunk on the same side whose span overlaps takes the
    // next column inward, so neither one's arms cross the other's trunk
    const depth = { left: 0, right: 0 }
    const trunks: Target[] = []
    for (const target of targets.values()) {
      if (target.lo === target.hi) continue
      for (const prior of trunks)
        if (prior.side === target.side && prior.lo < target.hi && target.lo < prior.hi)
          target.level = Math.max(target.level, prior.level + 1)
      depth[target.side] = Math.max(depth[target.side], target.level)
      trunks.push(target)
    }
    const paths = document.createDocumentFragment()
    for (const target of targets.values()) {
      const { side, x1, y1, tick, heads, level } = target
      const dir = side === 'left' ? 1 : -1
      // the trunk stands in the gutter, never past the entries' edge
      const gutter = Math.min(...heads.map(h => Math.abs(h.x0 - x1)))
      const turn = x1 - dir * Math.min(STEP * (1 + depth[side] - level), gutter - STEP)
      const straight = heads.find(h => h.y0 === y1)
      // the farthest head on each side of the line carries the trunk; the stem joins the first
      // trunk unless a straight leader already runs along it
      const below = heads.filter(h => h.y0 > y1).sort((p, q) => q.y0 - p.y0)[0]
      const above = heads.filter(h => h.y0 < y1).sort((p, q) => p.y0 - q.y0)[0]
      let stem = !straight
      let dash = tick ? `M${tick[0]} ${y1}H${tick[1]}` : ''
      for (const head of heads) {
        const { item, x0, y0 } = head
        let d =
          head === straight
            ? `M${x0} ${y0}H${x1}`
            : head === below || head === above
              ? `M${x0} ${y0}H${turn}V${y1}${stem ? `H${x1}` : ''}`
              : `M${x0} ${y0}H${turn}`
        if (head === below || head === above) stem = false
        d += dash
        dash = ''
        const path = document.createElementNS(SVG, 'path')
        path.setAttribute('d', d)
        paths.append(path)
        leaderOf.set(item.el, [...(leaderOf.get(item.el) ?? []), path])
      }
    }
    leaders.replaceChildren(paths)
  }

  let frame = 0
  const queue = () => {
    cancelAnimationFrame(frame)
    frame = requestAnimationFrame(paint)
  }
  layout.relayout = queue
  // a block's height when the observer last saw it; its first notice only sets the baseline, since
  // the first paint is already queued
  const heights = new Map<Element, number>()
  const observer = new ResizeObserver(changes => {
    // Initial and fonts-ready paints can already include the dimensions delivered by the observer.
    // Skip only those notifications; explicit content/control relayouts still always paint.
    let moved = false
    for (const change of changes) {
      if (!heights.has(change.target)) continue
      const before = heights.get(change.target)!
      const now = change.borderBoxSize[0]?.blockSize ?? change.contentRect.height
      heights.set(change.target, now)
      if (!Number.isNaN(before) && Math.abs(now - before) > 0.01) moved = true
    }
    if (
      moved ||
      changes.some(change => {
        if (heights.has(change.target)) return false
        if (article.dataset.frMode !== 'margin') return true
        const size = change.borderBoxSize[0]
        const previous = measured.get(change.target)
        return (
          !size ||
          !previous ||
          Math.abs(size.inlineSize - previous.width) > 0.01 ||
          Math.abs(size.blockSize - previous.height) > 0.01
        )
      })
    )
      queue()
  })
  observer.observe(article)
  for (const anchor of article.querySelectorAll('.fr-anchor')) observer.observe(anchor)
  for (const block of blocks) {
    if (block.matches('.fr-anchor')) continue
    heights.set(block, NaN)
    observer.observe(block)
  }
  // the sidenote script hangs or folds its notes after this one runs, and again on resize
  const hung = new MutationObserver(records => {
    if (records.some(r => (r.target as Element).classList.contains('sidenote-content'))) queue()
  })
  hung.observe(article, { attributes: true, attributeFilter: ['class'], subtree: true })
  signal.addEventListener(
    'abort',
    () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      hung.disconnect()
      leaders.remove()
      layout.relayout = () => {}
    },
    { once: true },
  )
  void document.fonts?.ready.then(queue)
  queue()
}

function mountRoot(root: HTMLElement, signal: AbortSignal) {
  if (!root.querySelector('.fr-group')) return
  mountLayout(root, signal)
  root.querySelectorAll<HTMLElement>('[data-fr-notes]').forEach(el => mountNotes(el, signal))
  root.querySelectorAll<HTMLElement>('[data-fr-rev]').forEach(el => void mountReview(el, signal))
}

document.addEventListener('nav', () => {
  const controller = new AbortController()
  window.addCleanup(() => controller.abort())
  const article = document.querySelector<HTMLElement>('article')
  if (article) mountRoot(article, controller.signal)

  for (const preview of document.querySelectorAll<HTMLElement>('.preview-container')) {
    let previewController: AbortController | null = null
    const syncPreview = () => {
      previewController?.abort()
      previewController = null
      const inner = preview.querySelector<HTMLElement>(':scope > .preview-inner')
      if (!inner?.querySelector('.fr-group')) return
      previewController = new AbortController()
      mountRoot(inner, previewController.signal)
    }
    const observer = new MutationObserver(syncPreview)
    observer.observe(preview, { childList: true })
    controller.signal.addEventListener(
      'abort',
      () => {
        observer.disconnect()
        previewController?.abort()
      },
      { once: true },
    )
    syncPreview()
  }
})
