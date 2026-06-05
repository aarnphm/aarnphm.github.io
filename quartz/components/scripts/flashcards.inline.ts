interface CardState {
  cardId: string
  due: number
}

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.flashcards-root')
  const cards = Array.from(document.querySelectorAll<HTMLElement>('.flashcard[data-card-id]'))
  if (!root || cards.length === 0) return

  const reveal = (card: HTMLElement, show: boolean) => {
    const front = card.querySelector<HTMLElement>('[data-face="front"]')
    const back = card.querySelector<HTMLElement>('[data-face="back"]')
    if (!front || !back) return
    card.classList.toggle('is-flipped', show)
    back.toggleAttribute('hidden', !show)
    front.toggleAttribute('hidden', show)
    card.setAttribute('aria-pressed', String(show))
  }

  const cleanups: Array<() => void> = []
  let onFlip: (() => void) | null = null

  for (const card of cards) {
    if (!card.querySelector('[data-face="front"]') || !card.querySelector('[data-face="back"]'))
      continue
    const toggle = () => {
      reveal(card, !card.classList.contains('is-flipped'))
      onFlip?.()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        toggle()
      }
    }
    card.setAttribute('tabindex', '0')
    card.setAttribute('role', 'button')
    card.setAttribute('aria-pressed', 'false')
    card.setAttribute('aria-label', 'reveal answer')
    card.addEventListener('click', toggle)
    card.addEventListener('keydown', onKey)
    cleanups.push(() => {
      card.removeEventListener('click', toggle)
      card.removeEventListener('keydown', onKey)
    })
  }

  const drill = root.querySelector<HTMLElement>('.flashcards-drill')
  const deckSlug = drill?.dataset.deck
  let login: string | null = null
  try {
    login = localStorage.getItem('comment-author-github-login')
  } catch {}

  if (drill && deckSlug && login) {
    const toggleBtn = drill.querySelector<HTMLButtonElement>('.flashcards-drill-toggle')
    const status = drill.querySelector<HTMLElement>('.flashcards-drill-status')
    const grade = drill.querySelector<HTMLElement>('.flashcards-grade')
    drill.hidden = false

    let queue: HTMLElement[] = []
    let active: HTMLElement | null = null
    let drilling = false

    const setStatus = (text: string) => {
      if (status) status.textContent = text
    }
    const syncGrade = () => {
      const flipped = !!active && active.classList.contains('is-flipped')
      grade?.toggleAttribute('hidden', !(drilling && flipped))
    }

    const stop = () => {
      drilling = false
      active?.classList.remove('is-active')
      active = null
      queue = []
      root.removeAttribute('data-drilling')
      grade?.toggleAttribute('hidden', true)
      if (toggleBtn) toggleBtn.textContent = 'start drill'
    }

    const showCard = () => {
      active?.classList.remove('is-active')
      active = queue[0] ?? null
      if (!active) {
        setStatus('all caught up')
        stop()
        return
      }
      active.classList.add('is-active')
      reveal(active, false)
      syncGrade()
      setStatus(`${queue.length} due`)
    }

    const submit = async (g: number) => {
      const card = active
      if (!card) return
      const cardId = card.dataset.cardId!
      const group = card.dataset.group
      queue.shift()
      if (group) queue = queue.filter(other => other.dataset.group !== group)
      try {
        await fetch('/api/flashcards/review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cardId, deckSlug, grade: g, login }),
        })
      } catch {}
      showCard()
    }

    const start = async () => {
      setStatus('loading…')
      let states: CardState[] = []
      try {
        const res = await fetch(
          `/api/flashcards/state?deck=${encodeURIComponent(deckSlug)}&login=${encodeURIComponent(login)}`,
        )
        if (res.ok) {
          const data = (await res.json()) as { states?: CardState[] }
          states = data.states ?? []
        }
      } catch {}
      const dueByCard = new Map(states.map(s => [s.cardId, s.due]))
      const now = Date.now()
      queue = cards
        .filter(card => (dueByCard.get(card.dataset.cardId!) ?? 0) <= now)
        .sort(
          (a, b) =>
            (dueByCard.get(a.dataset.cardId!) ?? 0) - (dueByCard.get(b.dataset.cardId!) ?? 0),
        )
      drilling = true
      root.setAttribute('data-drilling', '')
      if (toggleBtn) toggleBtn.textContent = 'stop drill'
      showCard()
    }

    onFlip = syncGrade

    const onToggle = () => {
      if (drilling) stop()
      else void start()
    }
    const onGrade = (e: MouseEvent) => {
      const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('[data-grade]')
      if (!btn) return
      void submit(Number(btn.dataset.grade))
    }
    const onDigit = (e: KeyboardEvent) => {
      if (!drilling || !active?.classList.contains('is-flipped')) return
      if (['1', '2', '3', '4'].includes(e.key)) {
        e.preventDefault()
        void submit(Number(e.key))
      }
    }

    toggleBtn?.addEventListener('click', onToggle)
    grade?.addEventListener('click', onGrade)
    document.addEventListener('keydown', onDigit)

    cleanups.push(() => {
      toggleBtn?.removeEventListener('click', onToggle)
      grade?.removeEventListener('click', onGrade)
      document.removeEventListener('keydown', onDigit)
      stop()
    })
  }

  window.addCleanup(() => {
    for (const fn of cleanups) fn()
  })
})
