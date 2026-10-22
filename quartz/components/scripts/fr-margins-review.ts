// Review and notes entries of the /fr margin (fr-margins.inline.ts): the section's cards through
// /api/flashcards, and the reader's notes through fr-notes.client.ts.

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

// Margins change height when a card flips or a note is added; the stack below follows
// (fr-margins.inline.ts sets the hook while its layout is mounted).
export const layout = { relayout: () => {} }

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

export async function mountReview(root: HTMLElement, signal: AbortSignal) {
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
      layout.relayout()
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
      layout.relayout()
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
    layout.relayout()
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

export function mountNotes(root: HTMLElement, signal: AbortSignal) {
  notesModule ??= import(new URL('fr-notes.js', import.meta.url).href)
  void notesModule
    .then(({ mountFrNotes }) => {
      if (signal.aborted || !root.isConnected) return
      mountFrNotes(root, document.body.dataset.slug ?? location.pathname.slice(1), signal, () =>
        layout.relayout(),
      )
    })
    .catch(error => {
      notesModule = undefined
      if (!signal.aborted) console.error(error)
    })
}
