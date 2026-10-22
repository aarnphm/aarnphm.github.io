// Margin column of /fr pages, Locher style: numbered `§ label (n)` entries (review of the section's
// cards through /api/flashcards, the reader's notes, the lexicon, citing lessons) beside the block
// that carries the same number. The markup comes from renderPage/fr-margins.ts. Under the width
// that leaves a margin, each group folds under its paragraph and the label toggles it.

interface Face {
  front: string
  back: string
  note: string
}

interface CardState {
  cardId: string
  due: number
}

const GRADES = [
  ['encore', 1],
  ['difficile', 2],
  ['bien', 3],
  ['facile', 4],
] as const

const el = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] => {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

const button = (label: string, className = 'fr-btn') => {
  const node = el('button', className, label)
  node.type = 'button'
  return node
}

// Margins change height when a card flips or a note is added; the stack below follows.
let relayout: () => void = () => {}

const decks = new Map<string, Promise<Map<string, Face>>>()

const loadDeck = (deck: string, signal: AbortSignal): Promise<Map<string, Face>> => {
  const cached = decks.get(deck)
  if (cached) return cached
  const pending = fetch(`/${deck}`, { headers: { Accept: 'text/html' }, signal })
    .then(res => (res.ok ? res.text() : Promise.reject(new Error(String(res.status)))))
    .then(html => {
      const doc = new DOMParser().parseFromString(html, 'text/html')
      const faces = new Map<string, Face>()
      for (const card of doc.querySelectorAll<HTMLElement>('.flashcard[data-card-id]')) {
        const note = card.querySelector('.flashcard-note')
        note?.remove()
        faces.set(card.dataset.cardId!, {
          front: card.querySelector('[data-face="front"]')?.innerHTML ?? '',
          back: card.querySelector('[data-face="back"]')?.innerHTML ?? '',
          note: note?.innerHTML ?? '',
        })
      }
      return faces
    })
  pending.catch(() => decks.delete(deck))
  decks.set(deck, pending)
  return pending
}

async function mountReview(root: HTMLElement, signal: AbortSignal) {
  const deck = root.dataset.deck
  const ids = (root.dataset.ids ?? '').split(',').filter(Boolean)
  const count = root.querySelector<HTMLElement>('[data-fr-count]')
  const box = root.querySelector<HTMLElement>('[data-fr-card]')
  if (!deck || ids.length === 0 || !count || !box) return

  const [faces, state] = await Promise.all([
    loadDeck(deck, signal),
    fetch(`/api/flashcards/state?deck=${encodeURIComponent(deck)}`, { signal })
      .then(res => (res.ok ? res.json() : { login: null, states: [] }))
      .catch(() => ({ login: null, states: [] })) as Promise<{
      login: string | null
      states: CardState[]
    }>,
  ]).catch(() => [undefined, undefined] as const)
  if (!faces || !state || signal.aborted) return

  let login = state.login
  const dueAt = new Map(state.states.map(s => [s.cardId, s.due]))
  const now = Date.now()
  // a card with no row is new, so it is due
  const queue = ids.filter(id => faces.has(id) && (dueAt.get(id) ?? 0) <= now)
  const total = queue.length
  let revealed = false

  if (!login) {
    const link = el('a', 'fr-meta', 'connexion')
    link.href = `/comments/github/login?returnTo=${encodeURIComponent(location.pathname)}`
    root.querySelector('.fr-label')?.append(' · ', link)
  }

  const paint = () => {
    box.replaceChildren()
    const id = queue[0]
    if (id === undefined) {
      count.textContent = total === 0 ? 'à jour' : `${total}/${total}`
      box.hidden = true
      relayout()
      return
    }
    const face = faces.get(id)!
    count.textContent = `${total - queue.length + 1}/${total}`
    const front = el('div', 'fr-q')
    front.innerHTML = face.front
    box.append(front)
    const bar = el('div', 'fr-bar')
    const show = button('montrer')
    bar.append(show)
    const flip = () => {
      revealed = true
      const back = el('div', 'fr-a')
      back.innerHTML = face.back
      if (face.note) {
        const meta = el('div', 'fr-meta')
        meta.innerHTML = face.note
        // the card cites this page; that link points at where the reader already is
        for (const a of meta.querySelectorAll<HTMLAnchorElement>('a[data-slug]'))
          if (a.dataset.slug === document.body.dataset.slug) a.remove()
        back.append(meta)
      }
      for (const play of back.querySelectorAll<HTMLButtonElement>('.speech-play'))
        play.disabled = false
      box.insertBefore(back, bar)
      bar.replaceChildren(
        ...GRADES.map(([label, grade]) => {
          const b = button(label)
          b.dataset.grade = String(grade)
          return b
        }),
      )
      relayout()
    }
    show.addEventListener('click', flip, { signal })
    bar.addEventListener(
      'click',
      event => {
        const target = (event.target as Element).closest<HTMLButtonElement>('button[data-grade]')
        if (!target || !revealed) return
        void grade(id, Number(target.dataset.grade))
      },
      { signal },
    )
    box.append(bar)
    box.hidden = false
    revealed = false
    relayout()
  }

  const grade = async (id: string, value: number) => {
    queue.shift()
    if (value === 1) queue.push(id)
    paint()
    if (!login) return
    try {
      const res = await fetch('/api/flashcards/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cardId: id, deckSlug: deck, grade: value }),
        signal,
      })
      if (res.status === 401) login = null
    } catch {
      /* the schedule stays as it was; the next grade retries the session */
    }
  }

  paint()
}

type NotesModule = typeof import('./fr-notes.client')
let notesModule: Promise<NotesModule> | undefined

function mountNotes(root: HTMLElement, signal: AbortSignal) {
  notesModule ??= import(new URL('fr-notes.js', import.meta.url).href)
  void notesModule
    .then(({ mountFrNotes }) => {
      if (signal.aborted || !root.isConnected) return
      mountFrNotes(root, document.body.dataset.slug ?? location.pathname.slice(1), signal, () =>
        relayout(),
      )
    })
    .catch(error => {
      notesModule = undefined
      if (!signal.aborted) console.error(error)
    })
}

// Room for a 15.5rem column and its 3.5rem gutter beside the text (fr-margins.scss).
const GUTTER = 56
const MARGIN_COLUMN = 248 + GUTTER + 16
const GROUP_GAP = 12
// a leader ends in a gap and a short dash before its number, as in Locher's margins
const DASH = 8
const GAP = 6
// stacked elbows take their own trunk columns through the gutter, this far apart
const STEP = 4

const SVG = 'http://www.w3.org/2000/svg'

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
  const leaderOf = new Map<HTMLElement, SVGPathElement>()

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
        leaderOf.get(group)?.classList.toggle('fr-hot', on)
      }
    }
    for (const node of [anchor, ...mine]) {
      node.addEventListener('pointerenter', () => light(true), { signal })
      node.addEventListener('pointerleave', () => light(false), { signal })
    }
  }

  const paint = () => {
    const box = article.getBoundingClientRect()
    const room = Math.min(box.left, document.documentElement.clientWidth - box.right)
    const mode = room >= MARGIN_COLUMN ? 'margin' : 'inline'
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

    // groups hang beside their block, the label level with its number, pushed down where two overlap
    const hoisted = (g: HTMLElement) => g.parentElement === article
    for (const group of groups) {
      const a = anchors.get(group)!.getBoundingClientRect()
      group.style.top = hoisted(group) ? `${a.top - box.top}px` : '0px'
    }
    // sidenotes the page already hung in the margins stack with the groups, so nothing overlaps
    interface Item {
      el: HTMLElement
      side: 'left' | 'right'
      top: number
      /** where the leader ends and the y of the number it points at; sidenotes have none */
      end?: { x: number; y: number }
      move: (by: number) => void
    }
    const items: Item[] = []
    for (const group of groups) {
      const anchor = anchors.get(group)!
      const mark = anchor.querySelector<HTMLElement>(':scope > .fr-n')
      const label = group.querySelector<HTMLElement>('.fr-entry-label')
      if (!mark || !label) continue
      const a = anchor.getBoundingClientRect()
      const side = group.dataset.side === 'right' ? 'right' : 'left'
      const inset = side === 'right' ? box.right - a.right : a.left - box.left
      group.style.setProperty('--fr-dx', hoisted(group) ? '0px' : `${inset}px`)
      const m = mark.getBoundingClientRect()
      const l = label.getBoundingClientRect()
      group.style.top = `${parseFloat(group.style.top) + m.top + m.height / 2 - (l.top + l.height / 2)}px`
      // the leader stays in the gutter: it stops just before the block, or at the text column's
      // edge for a block inside a table
      const edge = hoisted(group)
        ? side === 'right'
          ? box.right
          : box.left
        : side === 'right'
          ? a.right
          : a.left
      const x = side === 'right' ? edge + GAP : edge - GAP
      items.push({
        el: group,
        side,
        top: group.getBoundingClientRect().top,
        end: { x: x - box.left, y: m.top + m.height / 2 - box.top },
        move: by => (group.style.top = `${parseFloat(group.style.top) + by}px`),
      })
    }
    for (const content of document.querySelectorAll<HTMLElement>(
      '.sidenote-content.sidenote-left, .sidenote-content.sidenote-right',
    )) {
      if (!article.contains(content)) continue
      content.style.removeProperty('translate')
      const label = content.previousElementSibling?.querySelector<HTMLElement>('.sidenote-label')
      if (!label) continue
      const side = content.classList.contains('sidenote-right') ? 'right' : 'left'
      items.push({
        el: content,
        side,
        top: content.getBoundingClientRect().top,
        move: by => content.style.setProperty('translate', `0 ${by}px`),
      })
    }
    items.sort((x, y) => x.top - y.top)
    const floors = { left: -Infinity, right: -Infinity }
    interface Lead {
      item: Item
      /** the label's y, the number's y and the entry's near edge, in the article's frame */
      y0: number
      y1: number
      x0: number
      level: number
    }
    const leads: Lead[] = []
    for (const item of items) {
      const shift = Math.max(0, floors[item.side] + GROUP_GAP - item.top)
      if (shift > 0) item.move(shift)
      const placed = item.el.getBoundingClientRect()
      floors[item.side] = placed.bottom
      if (!item.end) continue
      const label = item.el.querySelector<HTMLElement>('.fr-entry-label')!.getBoundingClientRect()
      leads.push({
        item,
        y0: Math.round(label.top + label.height / 2 - box.top) + 0.5,
        y1: Math.round(item.end.y) + 0.5,
        x0: (item.side === 'right' ? placed.left : placed.right) - box.left,
        level: 0,
      })
    }
    // one hairline from the label to the number: straight when the entry is level with it, an
    // elbow through the gutter when the stack pushed the entry down. An elbow's trunk spans its
    // number and its label; a later elbow whose span overlaps takes the next column inward, so
    // neither one's arms cross the other's trunk
    const depth = { left: 0, right: 0 }
    for (const lead of leads) {
      if (lead.y0 === lead.y1) continue
      for (const prior of leads) {
        if (prior === lead) break
        if (prior.item.side !== lead.item.side || prior.y0 === prior.y1) continue
        if (prior.y1 < lead.y0 && lead.y1 < prior.y0)
          lead.level = Math.max(lead.level, prior.level + 1)
      }
      depth[lead.item.side] = Math.max(depth[lead.item.side], lead.level)
    }
    for (const { item, y0, y1, x0, level } of leads) {
      const dir = item.side === 'right' ? -1 : 1
      const x1 = item.end!.x
      const reach = Math.min(GAP + DASH + STEP * (1 + depth[item.side] - level), GUTTER - STEP)
      const turn = item.side === 'right' ? box.width + reach : -reach
      const d =
        y0 === y1
          ? `M${x0} ${y0}H${x1 - dir * (DASH + GAP)}M${x1 - dir * DASH} ${y1}H${x1}`
          : `M${x0} ${y0}H${turn}V${y1}H${x1 - dir * (DASH + GAP)}M${x1 - dir * DASH} ${y1}H${x1}`
      const path = document.createElementNS(SVG, 'path')
      path.setAttribute('d', d)
      leaders.append(path)
      leaderOf.set(item.el, path)
    }
  }

  let frame = 0
  const queue = () => {
    cancelAnimationFrame(frame)
    frame = requestAnimationFrame(paint)
  }
  relayout = queue
  const observer = new ResizeObserver(queue)
  observer.observe(article)
  for (const anchor of article.querySelectorAll('.fr-anchor')) observer.observe(anchor)
  // the sidenote script hangs or folds its notes after this one runs, and again on resize
  const notes = new MutationObserver(records => {
    if (records.some(r => (r.target as Element).classList.contains('sidenote-content'))) queue()
  })
  notes.observe(article, { attributes: true, attributeFilter: ['class'], subtree: true })
  signal.addEventListener(
    'abort',
    () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      notes.disconnect()
      leaders.remove()
      relayout = () => {}
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
