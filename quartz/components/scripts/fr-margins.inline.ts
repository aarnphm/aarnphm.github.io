// Margin column of /fr pages, Locher style: numbered `§ label (n)` entries (review of the section's
// cards through /api/flashcards, the reader's notes, the lexicon, citing lessons) beside the block
// that carries the same number. The markup comes from renderPage/fr-margins.ts. Under the width
// that leaves a margin, each group folds under its paragraph and the label toggles it.

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

  const paint = () => {
    const box = article.getBoundingClientRect()
    const mode = box.left >= MARGIN_COLUMN ? 'margin' : 'inline'
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
    if (mode === 'inline') {
      for (const group of groups) group.style.removeProperty('top')
      leaders.removeAttribute('viewBox')
      return
    }
    leaders.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`)
    leaders.setAttribute('width', String(box.width))
    leaders.setAttribute('height', String(box.height))

    // groups hang left of their block, the label level with its number, pushed down where two
    // overlap
    const hoisted = (g: HTMLElement) => g.parentElement === article
    for (const group of groups) {
      const a = anchors.get(group)!.getBoundingClientRect()
      group.style.top = hoisted(group) ? `${a.top - box.top}px` : '0px'
    }
    // sidenotes the page already hung in either margin stack with the groups, so nothing overlaps
    interface Item {
      el: HTMLElement
      side: 'left' | 'right'
      top: number
      /** where the leader ends, in the article's frame, and the y of what it points at */
      end: { x: number; y: number }
      /** the y of each leader's head on the entry: its labels, or a sidenote's first line */
      heads: () => number[]
      /** the dash before a number that sits past the text column's edge, x0 to x1 */
      tick?: [number, number]
      move: (by: number) => void
    }
    const items: Item[] = []
    const mid = (r: DOMRect) => r.top + r.height / 2
    for (const group of groups) {
      const anchor = anchors.get(group)!
      const mark = anchor.querySelector<HTMLElement>(':scope > .fr-n')
      const labels = [...group.querySelectorAll<HTMLElement>('.fr-entry-label')]
      const label = labels[0]
      // a folded callout keeps its blocks, hidden and clipped; their groups take no leader
      if (!mark || !label || !anchor.checkVisibility({ visibilityProperty: true })) continue
      const a = anchor.getBoundingClientRect()
      group.style.setProperty('--fr-dx', hoisted(group) ? '0px' : `${a.left - box.left}px`)
      const m = mark.getBoundingClientRect()
      group.style.top = `${parseFloat(group.style.top) + mid(m) - mid(label.getBoundingClientRect())}px`
      // the leader stays in the gutter and stops at the text column's edge, short of a list item's
      // marker or a cell's rule; a number set in from the edge (past a section's rail) gets its dash
      const markX = Math.round(m.left - box.left)
      items.push({
        el: group,
        side: 'left',
        top: group.getBoundingClientRect().top,
        end: { x: -STOP, y: mid(m) - box.top },
        tick:
          markX > STOP + TICK + TICK_STOP
            ? [markX - TICK_STOP - TICK, markX - TICK_STOP]
            : undefined,
        heads: () => labels.map(l => mid(l.getBoundingClientRect())),
        move: by => (group.style.top = `${parseFloat(group.style.top) + by}px`),
      })
    }
    // a sidenote's first line sits level with its label, as an entry's does with its number; its
    // leader points at the label's line from the text column's edge
    for (const content of notes) {
      content.style.removeProperty('translate')
      const label = labelOf(content)
      const side = content.classList.contains('sidenote-right') ? 'right' : 'left'
      if (!label || !content.classList.contains(`sidenote-${side}`)) continue
      if (!label.checkVisibility({ visibilityProperty: true })) continue
      const l = mid(label.getBoundingClientRect())
      let dy = l - firstLine(content)
      content.style.translate = `0 ${dy}px`
      items.push({
        el: content,
        side,
        top: content.getBoundingClientRect().top,
        end: { x: side === 'left' ? -STOP : box.width + STOP, y: l - box.top },
        heads: () => [firstLine(content)],
        move: by => content.style.setProperty('translate', `0 ${(dy += by)}px`),
      })
    }
    items.sort((x, y) => x.top - y.top)
    const floors = { left: -Infinity, right: -Infinity }
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
    for (const item of items) {
      const shift = Math.max(0, floors[item.side] + GROUP_GAP - item.top)
      if (shift > 0) item.move(shift)
      const placed = item.el.getBoundingClientRect()
      floors[item.side] = placed.bottom
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
      for (const head of item.heads()) {
        const y0 = Math.round(head - box.top) + 0.5
        target.heads.push({
          item,
          y0,
          x0: (item.side === 'left' ? placed.right : placed.left) - box.left,
        })
        target.lo = Math.min(target.lo, y0)
        target.hi = Math.max(target.hi, y0)
      }
    }
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
        leaders.append(path)
        leaderOf.set(item.el, [...(leaderOf.get(item.el) ?? []), path])
      }
    }
  }

  let frame = 0
  const queue = () => {
    cancelAnimationFrame(frame)
    frame = requestAnimationFrame(paint)
  }
  layout.relayout = queue
  const observer = new ResizeObserver(queue)
  observer.observe(article)
  for (const anchor of article.querySelectorAll('.fr-anchor')) observer.observe(anchor)
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

document.addEventListener('nav', () => {
  const article = document.querySelector<HTMLElement>('article')
  if (!article?.querySelector('.fr-group')) return
  const controller = new AbortController()
  window.addCleanup(() => controller.abort())
  mountLayout(article, controller.signal)
  article
    .querySelectorAll<HTMLElement>('[data-fr-notes]')
    .forEach(root => mountNotes(root, controller.signal))
  article
    .querySelectorAll<HTMLElement>('[data-fr-rev]')
    .forEach(root => void mountReview(root, controller.signal))
})
