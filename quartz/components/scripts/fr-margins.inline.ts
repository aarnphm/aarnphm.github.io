// Left margin of /fr pages: review of the section's cards (FSRS through /api/flashcards) and the
// reader's private notes. The markup comes from renderPage/fr-margins.ts.

interface Face {
  front: string
  back: string
  note: string
}

interface CardState {
  cardId: string
  due: number
}

interface Note {
  id: string
  text: string
  at: number
}

const GRADES = [
  ['encore', 1],
  ['difficile', 2],
  ['bien', 3],
  ['facile', 4],
] as const

const MONTHS = [
  'janv.',
  'févr.',
  'mars',
  'avr.',
  'mai',
  'juin',
  'juil.',
  'août',
  'sept.',
  'oct.',
  'nov.',
  'déc.',
]

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

// margins change height when a card flips or a note is added; the sidenote manager re-lays out on resize
const relayout = () => window.dispatchEvent(new Event('resize'))

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

const notesKey = (section: string) =>
  `fr-notes:v1:${document.body.dataset.slug ?? location.pathname}:${section}`

const readNotes = (key: string): Note[] => {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(key) ?? '[]')
    return Array.isArray(value) ? (value as Note[]) : []
  } catch {
    return []
  }
}

const writeNotes = (key: string, notes: Note[]) => {
  try {
    if (notes.length === 0) localStorage.removeItem(key)
    else localStorage.setItem(key, JSON.stringify(notes))
  } catch {
    /* private mode: the note lives until the page closes */
  }
}

const dayLabel = (at: number) => {
  const d = new Date(at)
  return `${d.getDate()} ${MONTHS[d.getMonth()]}`
}

function mountNotes(root: HTMLElement, signal: AbortSignal) {
  const section = root.dataset.section ?? ''
  const key = notesKey(section)
  const list = root.querySelector<HTMLElement>('[data-fr-note-list]')
  const add = root.querySelector<HTMLButtonElement>('[data-fr-act="add"]')
  if (!list || !add) return
  let notes = readNotes(key)
  let editor: HTMLElement | undefined

  const paint = () => {
    list.replaceChildren(
      ...notes.map(note => {
        const row = el('div', 'fr-note')
        row.append(el('span', undefined, note.text))
        const meta = el('span', 'fr-meta', `${dayLabel(note.at)} · privé`)
        const card = button('carte', 'fr-btn fr-link')
        card.addEventListener(
          'click',
          () => {
            const slug = document.body.dataset.slug ?? location.pathname.slice(1)
            const source = `Q: ${note.text}\nA: \nN: [[${slug}#${section}]]\n`
            void navigator.clipboard?.writeText(source).then(() => {
              card.textContent = 'copié'
              setTimeout(() => (card.textContent = 'carte'), 1500)
            })
          },
          { signal },
        )
        const drop = button('retirer', 'fr-btn fr-link')
        drop.addEventListener(
          'click',
          () => {
            notes = notes.filter(n => n.id !== note.id)
            writeNotes(key, notes)
            paint()
          },
          { signal },
        )
        meta.append(' ', card, ' ', drop)
        row.append(meta)
        return row
      }),
    )
    relayout()
  }

  const close = () => {
    editor?.remove()
    editor = undefined
    add.hidden = false
    relayout()
  }

  add.addEventListener(
    'click',
    () => {
      const area = el('textarea', 'fr-area')
      area.rows = 3
      area.setAttribute('aria-label', 'nouvelle note')
      const save = button('ok')
      const commit = () => {
        const text = area.value.trim()
        if (text) {
          notes = [...notes, { id: crypto.randomUUID(), text, at: Date.now() }]
          writeNotes(key, notes)
          paint()
        }
        close()
      }
      save.addEventListener('click', commit, { signal })
      area.addEventListener(
        'keydown',
        event => {
          if (event.key === 'Escape') close()
          else if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) commit()
        },
        { signal },
      )
      editor = el('div', 'fr-editor')
      editor.append(area, save)
      add.hidden = true
      add.after(editor)
      area.focus()
      relayout()
    },
    { signal },
  )

  paint()
}

document.addEventListener('nav', () => {
  const reviews = document.querySelectorAll<HTMLElement>('article [data-fr-rev]')
  const notes = document.querySelectorAll<HTMLElement>('article [data-fr-notes]')
  if (reviews.length === 0 && notes.length === 0) return
  const controller = new AbortController()
  window.addCleanup(() => controller.abort())
  notes.forEach(root => mountNotes(root, controller.signal))
  reviews.forEach(root => void mountReview(root, controller.signal))
})
