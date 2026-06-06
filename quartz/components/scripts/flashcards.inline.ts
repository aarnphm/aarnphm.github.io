interface CardState {
  cardId: string
  due: number
}

interface LogEntry {
  el: HTMLElement
  cardId: string
  grade: number
  requeued: boolean
}

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.flashcards-root')
  if (!root) return
  const cards = Array.from(root.querySelectorAll<HTMLElement>('.flashcard[data-card-id]'))
  if (cards.length === 0) return

  const deckSlug = root.dataset.deck ?? ''
  const cardBox = root.querySelector<HTMLElement>('.flashcards-card')
  const cardBody = root.querySelector<HTMLElement>('.flashcards-card-body')
  const progressEl = root.querySelector<HTMLElement>('.flashcards-progress')
  const progressFill = root.querySelector<HTMLElement>('.flashcards-progress-fill')
  const statusEl = root.querySelector<HTMLElement>('.flashcards-status')
  const finishedEl = root.querySelector<HTMLElement>('.flashcards-finished')
  const summaryEl = root.querySelector<HTMLElement>('.flashcards-summary')
  const undoBtn = root.querySelector<HTMLButtonElement>('.fc-undo')
  const revealBtn = root.querySelector<HTMLButtonElement>('.fc-reveal')
  const grades = root.querySelector<HTMLElement>('.fc-grades')
  const endBtn = root.querySelector<HTMLButtonElement>('.fc-end')

  let login: string | null = null
  try {
    login = localStorage.getItem('comment-author-github-login')
  } catch {}

  let queue: HTMLElement[] = []
  let total = 0
  let active: HTMLElement | null = null
  let revealed = false
  let finished = false
  let persistError = false
  let startedAt = 0
  const log: LogEntry[] = []

  const shuffle = (arr: HTMLElement[]) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[arr[i], arr[j]] = [arr[j], arr[i]]
    }
  }

  const setBack = (el: HTMLElement, show: boolean) => {
    el.querySelector<HTMLElement>('[data-face="back"]')?.toggleAttribute('hidden', !show)
  }

  const setReveal = (show: boolean) => {
    revealed = show
    revealBtn?.toggleAttribute('hidden', show)
    grades?.toggleAttribute('hidden', !show)
    if (active) {
      setBack(active, show)
      active.setAttribute('aria-expanded', String(show))
    }
  }

  const updateProgress = () => {
    const done = total - queue.length
    const pct = total > 0 ? (done / total) * 100 : 100
    if (progressFill) progressFill.style.width = `${pct}%`
    progressEl?.setAttribute('aria-valuenow', String(done))
    if (statusEl) statusEl.textContent = `${done} / ${total}`
    if (undoBtn) undoBtn.disabled = log.length === 0
  }

  const summaryNumber = (value: number) => {
    const span = document.createElement('span')
    span.className = 'flashcards-summary-number'
    span.textContent = String(value)
    return span
  }

  const reviewedCount = () => new Set(log.map(entry => entry.cardId)).size

  const elapsedSeconds = () => {
    const start = startedAt === 0 ? Date.now() : startedAt
    return Math.round((Date.now() - start) / 1000)
  }

  const setSummary = () => {
    if (!summaryEl) return
    if (total === 0) {
      summaryEl.textContent = 'nothing due right now.'
      return
    }
    const reviewed = reviewedCount()
    const secs = elapsedSeconds()
    summaryEl.replaceChildren(
      document.createTextNode('reviewed '),
      summaryNumber(reviewed),
      document.createTextNode(` ${reviewed === 1 ? 'card' : 'cards'} in `),
      summaryNumber(secs),
      document.createTextNode('s.'),
    )
    if (persistError) summaryEl.append(document.createTextNode(' reviews did not save.'))
  }

  const show = () => {
    for (const el of cards) el.classList.remove('is-active')
    active = queue[0] ?? null
    if (!active) {
      finish()
      return
    }
    setReveal(false)
    void active.offsetWidth
    active.classList.add('is-active')
    if (cardBody) cardBody.scrollTop = 0
    updateProgress()
  }

  const submit = async (grade: number) => {
    if (!revealed || !active || finished) return
    const el = queue.shift()
    if (!el) return
    const cardId = el.dataset.cardId
    if (!cardId) {
      queue.unshift(el)
      return
    }
    const requeued = grade <= 2
    if (requeued) queue.push(el)
    log.push({ el, cardId, grade, requeued })
    if (login && deckSlug) {
      try {
        const res = await fetch('/api/flashcards/review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cardId, deckSlug, grade, login }),
        })
        if (!res.ok) persistError = true
      } catch {
        persistError = true
      }
    }
    if (queue.length === 0) {
      finish()
      return
    }
    show()
  }

  const undo = () => {
    const last = log.pop()
    if (!last) return
    if (last.requeued) {
      const idx = queue.lastIndexOf(last.el)
      if (idx !== -1) queue.splice(idx, 1)
    }
    queue.unshift(last.el)
    if (finished) {
      finished = false
      root.removeAttribute('data-state')
      finishedEl?.toggleAttribute('hidden', true)
    }
    show()
  }

  const finish = () => {
    finished = true
    setReveal(false)
    root.setAttribute('data-state', 'finished')
    setSummary()
    root.toggleAttribute('data-persist-error', persistError)
    finishedEl?.toggleAttribute('hidden', false)
    updateProgress()
  }

  const onCardClick = (event: MouseEvent) => {
    if (event.target instanceof Element && event.target.closest('a')) return
    if (!finished && !revealed) setReveal(true)
  }
  const onReveal = () => {
    if (!finished && !revealed) setReveal(true)
  }
  const onGrade = (event: MouseEvent) => {
    const btn = (event.target as HTMLElement).closest<HTMLButtonElement>('.fc-grade')
    if (!btn) return
    void submit(Number(btn.dataset.grade))
  }
  const onEnd = () => finish()
  const onKey = (event: KeyboardEvent) => {
    const target = event.target
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      (target instanceof HTMLElement && target.isContentEditable)
    ) {
      return
    }
    if (event.shiftKey || event.ctrlKey || event.altKey || event.metaKey) return
    if (event.key === ' ') {
      if (finished) return
      event.preventDefault()
      if (!revealed) setReveal(true)
    } else if (event.key === 'u') {
      event.preventDefault()
      undo()
    } else if (event.key === '1' || event.key === '2' || event.key === '3' || event.key === '4') {
      if (!revealed) return
      event.preventDefault()
      void submit(Number(event.key))
    } else if (event.key === 'e') {
      if (finished) return
      event.preventDefault()
      finish()
    }
  }

  const setHints = (show: boolean) => {
    root.classList.toggle('fc-hints', show)
  }
  const onModDown = (event: KeyboardEvent) => {
    if (event.key === 'Meta' || event.key === 'Control') setHints(true)
  }
  const onModUp = (event: KeyboardEvent) => {
    if (event.key === 'Meta' || event.key === 'Control') setHints(false)
  }
  const onBlur = () => setHints(false)

  cardBox?.addEventListener('click', onCardClick)
  revealBtn?.addEventListener('click', onReveal)
  grades?.addEventListener('click', onGrade)
  undoBtn?.addEventListener('click', undo)
  endBtn?.addEventListener('click', onEnd)
  window.addEventListener('keydown', onKey)
  window.addEventListener('keydown', onModDown)
  window.addEventListener('keyup', onModUp)
  window.addEventListener('blur', onBlur)

  const start = async () => {
    let order = cards.slice()
    if (login && deckSlug) {
      try {
        const res = await fetch(
          `/api/flashcards/state?deck=${encodeURIComponent(deckSlug)}&login=${encodeURIComponent(login)}`,
        )
        if (res.ok) {
          const data = (await res.json()) as { states?: CardState[] }
          const dueByCard = new Map((data.states ?? []).map(s => [s.cardId, s.due]))
          const now = Date.now()
          order = cards.filter(el => {
            const cardId = el.dataset.cardId
            return cardId !== undefined && (dueByCard.get(cardId) ?? 0) <= now
          })
        }
      } catch {}
    }
    shuffle(order)
    const limit = Number(new URLSearchParams(window.location.search).get('n'))
    if (Number.isInteger(limit) && limit > 0) order = order.slice(0, limit)
    queue = order
    total = queue.length
    startedAt = Date.now()
    if (total === 0) {
      finish()
      return
    }
    show()
  }
  void start()

  window.addCleanup(() => {
    cardBox?.removeEventListener('click', onCardClick)
    revealBtn?.removeEventListener('click', onReveal)
    grades?.removeEventListener('click', onGrade)
    undoBtn?.removeEventListener('click', undo)
    endBtn?.removeEventListener('click', onEnd)
    window.removeEventListener('keydown', onKey)
    window.removeEventListener('keydown', onModDown)
    window.removeEventListener('keyup', onModUp)
    window.removeEventListener('blur', onBlur)
  })
})
