import { currentNavSignal } from './nav-lifecycle'

interface CardState {
  cardId: string
  due: number
}

interface LogEntry {
  el: HTMLElement
  cardId: string
  grade: number
  requeued: boolean
  /** siblings of a cloze group pulled from the queue when this card was graded */
  buried: HTMLElement[]
  reviewId?: number
}

let activeSignal: AbortSignal | undefined

document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  if (activeSignal === signal) return

  const root = document.querySelector<HTMLElement>('.flashcards-root')
  if (!root) return
  const cards = Array.from(root.querySelectorAll<HTMLElement>('.flashcard[data-card-id]'))
  if (cards.length === 0) return

  activeSignal = signal
  signal.addEventListener(
    'abort',
    () => {
      if (activeSignal === signal) activeSignal = undefined
    },
    { once: true },
  )

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
  const syncEl = root.querySelector<HTMLAnchorElement>('.flashcards-sync')

  // The Worker resolves the owner's GitHub session; without one the drill is stateless.
  let login: string | null = null

  const setSync = (sync: 'off' | 'on' | 'error', detail: string) => {
    if (!syncEl) return
    syncEl.hidden = false
    syncEl.dataset.sync = sync
    syncEl.title = detail
    if (sync === 'off') {
      const returnTo = `${window.location.pathname}${window.location.search}`
      syncEl.href = `/comments/github/login?returnTo=${encodeURIComponent(returnTo)}`
      syncEl.textContent = 'not saved'
    } else {
      syncEl.removeAttribute('href')
      syncEl.textContent = sync === 'on' ? `@${login}` : 'sync failed'
    }
  }

  let queue: HTMLElement[] = []
  let total = 0
  let active: HTMLElement | null = null
  let revealed = false
  let finished = false
  let persistError = false
  let submitting = false
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
    if (!revealed || !active || finished || submitting) return
    const el = active
    const cardId = el.dataset.cardId
    if (!cardId) return

    submitting = true
    setReveal(false)
    queue.shift()
    const requeued = grade <= 2
    if (requeued) queue.push(el)
    // cloze siblings show the answer just graded, so they wait for the next session
    const group = el.dataset.group
    const buried = group ? queue.filter(other => other !== el && other.dataset.group === group) : []
    if (buried.length > 0) {
      queue = queue.filter(other => !buried.includes(other))
      total -= buried.length
    }
    const entry: LogEntry = { el, cardId, grade, requeued, buried }
    log.push(entry)
    if (login && deckSlug) {
      try {
        const res = await fetch('/api/flashcards/review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cardId, deckSlug, grade }),
          signal,
        })
        if (res.status === 401) login = null
        if (!res.ok) persistError = true
        else {
          const data = (await res.json()) as { reviewId?: number }
          if (typeof data.reviewId === 'number') entry.reviewId = data.reviewId
        }
      } catch {
        if (!signal.aborted) persistError = true
      }
      if (!login) setSync('off', 'session expired; later reviews are not saved')
      else if (persistError) setSync('error', 'a review failed to save; its schedule is unchanged')
    }
    if (signal.aborted) return
    submitting = false
    if (queue.length === 0) {
      finish()
      return
    }
    show()
  }

  const undo = () => {
    if (submitting) return
    const last = log.pop()
    if (!last) return
    if (last.requeued) {
      const idx = queue.lastIndexOf(last.el)
      if (idx !== -1) queue.splice(idx, 1)
    }
    queue.unshift(last.el)
    if (last.buried.length > 0) {
      queue.push(...last.buried)
      total += last.buried.length
    }
    if (login && last.reviewId !== undefined) {
      const reviewId = last.reviewId
      void fetch('/api/flashcards/undo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviewId }),
        signal,
      })
        .then(res => {
          if (!res.ok) setSync('error', 'undo did not save; the review stands on the server')
        })
        .catch(() => {
          if (!signal.aborted)
            setSync('error', 'undo did not save; the review stands on the server')
        })
    }
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
    if (
      event.target instanceof Element &&
      event.target.closest('a, button, input, select, textarea, summary')
    )
      return
    if (!finished && !revealed && !submitting) setReveal(true)
  }
  const onReveal = () => {
    if (!finished && !revealed && !submitting) setReveal(true)
  }
  const onGrade = (event: MouseEvent) => {
    if (!(event.target instanceof Element)) return
    const btn = event.target.closest<HTMLButtonElement>('.fc-grade')
    if (!btn) return
    void submit(Number(btn.dataset.grade))
  }
  const onEnd = () => {
    if (!submitting) finish()
  }
  const onKey = (event: KeyboardEvent) => {
    const target = event.target
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      (target instanceof Element && target.closest('button, a, summary')) ||
      (target instanceof HTMLElement && target.isContentEditable)
    ) {
      return
    }
    if (submitting) return
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

  cardBox?.addEventListener('click', onCardClick, { signal })
  revealBtn?.addEventListener('click', onReveal, { signal })
  grades?.addEventListener('click', onGrade, { signal })
  undoBtn?.addEventListener('click', undo, { signal })
  endBtn?.addEventListener('click', onEnd, { signal })
  window.addEventListener('keydown', onKey, { signal })
  window.addEventListener('keydown', onModDown, { signal })
  window.addEventListener('keyup', onModUp, { signal })
  window.addEventListener('blur', onBlur, { signal })

  const start = async () => {
    let order = cards.slice()
    if (deckSlug) {
      try {
        const res = await fetch(`/api/flashcards/state?deck=${encodeURIComponent(deckSlug)}`, {
          signal,
        })
        if (!res.ok) throw new Error(`state ${res.status}`)
        const data = (await res.json()) as { login?: string | null; states?: CardState[] }
        login = typeof data.login === 'string' ? data.login : null
        if (login) {
          const dueByCard = new Map((data.states ?? []).map(s => [s.cardId, s.due]))
          const now = Date.now()
          order = cards.filter(el => {
            const cardId = el.dataset.cardId
            return cardId !== undefined && (dueByCard.get(cardId) ?? 0) <= now
          })
          setSync('on', `reviews save to ${login}`)
        } else {
          setSync(
            'off',
            'progress is not saved; only the site owner can sign in to schedule reviews',
          )
        }
      } catch {
        if (!signal.aborted) setSync('error', 'could not load progress; showing every card')
      }
    }
    if (signal.aborted) return
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
})
