// Margin column of /fr pages, Locher style: numbered `§ label (n)` entries (review of the section's
// cards through /api/flashcards, the reader's notes, the lexicon, citing lessons) joined to their
// paragraph by a leader line and a tick. The markup comes from renderPage/fr-margins.ts. Under the
// width that leaves a margin, each group folds under its paragraph and the label toggles it.

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

// Margins change height when a card flips or a note is added; the leaders follow.
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

// Room for a 15.5rem column and its gutter left of the text, as the sidenotes measure it.
const MARGIN_COLUMN = 248 + 40 + 16
const GROUP_GAP = 12
const TICK = 5

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

  for (const label of article.querySelectorAll<HTMLButtonElement>('.fr-entry-label')) {
    label.addEventListener(
      'click',
      () => {
        if (article.dataset.frMode !== 'inline') return
        const entry = label.closest<HTMLElement>('.fr-entry')
        if (!entry) return
        const open = entry.toggleAttribute('data-open')
        label.setAttribute('aria-expanded', String(open))
      },
      { signal },
    )
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
    leaders.replaceChildren()
    if (mode === 'inline') {
      for (const group of groups) group.style.removeProperty('top')
      leaders.removeAttribute('viewBox')
      return
    }
    leaders.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`)
    leaders.setAttribute('width', String(box.width))
    leaders.setAttribute('height', String(box.height))

    // groups hang beside their paragraph, pushed down where two would overlap
    const hoisted = (g: HTMLElement) => g.parentElement === article
    for (const group of groups) {
      const a = anchors.get(group)!.getBoundingClientRect()
      group.style.top = hoisted(group) ? `${a.top - box.top}px` : '0px'
    }
    const floors = { left: -Infinity, right: -Infinity }
    const d: string[] = []
    for (const group of groups) {
      const anchor = anchors.get(group)!
      const mark = anchor.querySelector<HTMLElement>(':scope > .fr-n')
      const label = group.querySelector<HTMLElement>('.fr-entry-label')
      if (!mark || !label) continue
      const a = anchor.getBoundingClientRect()
      const side = group.dataset.side === 'right' ? 'right' : 'left'
      const inset = side === 'right' ? box.right - a.right : a.left - box.left
      group.style.setProperty('--fr-dx', hoisted(group) ? '0px' : `${inset}px`)
      const m0 = mark.getBoundingClientRect()
      const l0 = label.getBoundingClientRect()
      // the label sits level with its number, so the line is one horizontal stroke
      const level = m0.top + m0.height / 2 - (l0.top + l0.height / 2)
      group.style.top = `${parseFloat(group.style.top) + level}px`
      const natural = group.getBoundingClientRect()
      const shift = Math.max(0, floors[side] + GROUP_GAP - natural.top)
      if (shift > 0) group.style.top = `${parseFloat(group.style.top) + shift}px`
      const placed = group.getBoundingClientRect()
      floors[side] = placed.bottom
      const l = label.getBoundingClientRect()
      const gy = l.top + l.height / 2 - box.top
      const gx = (side === 'right' ? placed.left : placed.right) - box.left
      const tx = (side === 'right' ? a.right + 10 : a.left - 10) - box.left
      // a group pushed down by a neighbour keeps its line straight at the label's height
      d.push(`M${gx} ${gy}H${tx}`, `M${tx} ${gy - TICK}V${gy + TICK}`)
    }
    const path = document.createElementNS(SVG, 'path')
    path.setAttribute('d', d.join(''))
    leaders.append(path)
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
  signal.addEventListener(
    'abort',
    () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
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
