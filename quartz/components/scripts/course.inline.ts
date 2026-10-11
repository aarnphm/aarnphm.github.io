import { currentNavSignal } from './nav-lifecycle'

/** Clipboard write with the legacy fallback for panes that deny the async clipboard. */
async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
    return
  } catch {
    // fall through
  }
  const area = document.createElement('textarea')
  area.value = text
  area.setAttribute('readonly', '')
  area.style.position = 'fixed'
  area.style.opacity = '0'
  document.body.appendChild(area)
  area.select()
  const ok = document.execCommand('copy')
  area.remove()
  if (!ok) throw new Error('copy rejected')
}

interface CardState {
  cardId: string
  due: number
  lastReviewedAt: number
  r: number | null
}

interface StateResponse {
  login?: string | null
  states?: CardState[]
}

interface CountsResponse {
  canWrite?: boolean
  counts?: Record<string, { mark?: number; question?: number; contra?: number }>
}

const age = (ms: number): string => {
  const minutes = Math.round(ms / 60_000)
  if (minutes < 60) return `${Math.max(minutes, 0)}m`
  const hours = Math.round(minutes / 60)
  if (hours < 48) return `${hours}h`
  return `${Math.round(hours / 24)}d`
}

// Fills the owner-only cells (due, last, mean recall) from one prefix query per page. Signed
// out, every cell keeps its "–" and the action turns into the sign-in link.
document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  const roots = Array.from(document.querySelectorAll<HTMLElement>('[data-course-prefix]'))
  if (roots.length === 0) return

  const loginHref = () =>
    `/comments/github/login?returnTo=${encodeURIComponent(`${location.pathname}${location.search}`)}`

  const fill = (root: HTMLElement, rows: Map<string, CardState>, login: string | null) => {
    const now = Date.now()
    for (const el of root.querySelectorAll<HTMLElement>('[data-cards]')) {
      const ids = (el.dataset.cards ?? '').split(/\s+/).filter(Boolean)
      if (ids.length === 0) continue
      if (!login) {
        el.textContent = '–'
        continue
      }
      let due = 0
      let seen = 0
      let last = 0
      let rSum = 0
      let rCount = 0
      for (const id of ids) {
        const row = rows.get(id)
        if (!row) continue
        seen++
        if (row.due <= now) due++
        if (row.lastReviewedAt > last) last = row.lastReviewedAt
        if (typeof row.r === 'number') {
          rSum += row.r
          rCount++
        }
      }
      const fresh = ids.length - seen
      const dueEl = el.matches('[data-due]') ? el : el.querySelector<HTMLElement>('[data-due]')
      if (dueEl) {
        dueEl.replaceChildren(document.createTextNode(String(due)))
        if (fresh > 0) {
          const badge = document.createElement('span')
          badge.className = 'course-new'
          badge.textContent = `+${fresh}`
          badge.title = `${fresh} new ${fresh === 1 ? 'card' : 'cards'}`
          dueEl.append(badge)
        }
        dueEl.title = `${due} due · ${seen} seen of ${ids.length}`
      }
      const lastEl = root.querySelector<HTMLElement>(
        `[data-last][data-for="${el.dataset.unit ?? ''}"]`,
      )
      if (lastEl) {
        if (last > 0) {
          lastEl.textContent = age(now - last)
          lastEl.title = new Date(last).toISOString()
        } else {
          lastEl.textContent = '–'
          lastEl.removeAttribute('title')
        }
      }
      const rEl = root.querySelector<HTMLElement>(
        `[data-retention][data-for="${el.dataset.unit ?? ''}"]`,
      )
      if (rEl) {
        rEl.textContent = rCount > 0 ? `${Math.round((rSum / rCount) * 100)}%` : '–'
        rEl.title =
          rCount > 0 ? `mean predicted recall over ${rCount} seen cards (fsrs estimate)` : ''
      }
    }
    for (const action of root.querySelectorAll<HTMLElement>('.course-action[data-course-action]')) {
      if (login) continue
      const link = document.createElement('a')
      link.className = action.className
      link.setAttribute('data-router-ignore', '')
      link.href = loginHref()
      link.textContent = 'sign in to schedule'
      action.replaceWith(link)
    }
  }

  const byPrefix = new Map<
    string,
    Promise<{ login: string | null; rows: Map<string, CardState> }>
  >()
  const load = (prefix: string) => {
    let pending = byPrefix.get(prefix)
    if (!pending) {
      pending = fetch(`/api/flashcards/state?prefix=${encodeURIComponent(prefix)}`, {
        signal,
      }).then(async res => {
        if (!res.ok) throw new Error(`state ${res.status}`)
        const data = (await res.json()) as StateResponse
        const login = typeof data.login === 'string' ? data.login : null
        return { login, rows: new Map((data.states ?? []).map(s => [s.cardId, s])) }
      })
      byPrefix.set(prefix, pending)
    }
    return pending
  }

  for (const root of roots) {
    const prefix = root.dataset.coursePrefix ?? ''
    void load(prefix)
      .then(({ login, rows }) => {
        if (!signal.aborted) fill(root, rows, login)
      })
      .catch(() => {
        if (signal.aborted) return
        for (const el of root.querySelectorAll<HTMLElement>('[data-due]')) {
          el.title = 'could not load review state'
        }
      })
  }
})

// Reader marks per PDF (`7 · 2?`: marks, then open questions) from one counts query per course
// home. Visitors get public counts; the cell keeps its "–" when the API is away (R6).
document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  const roots = Array.from(document.querySelectorAll<HTMLElement>('[data-marks-prefix]'))
  if (roots.length === 0) return
  for (const root of roots) {
    const prefix = root.dataset.marksPrefix ?? ''
    const cells = Array.from(root.querySelectorAll<HTMLElement>('[data-marks-src]'))
    if (!prefix || cells.length === 0) continue
    fetch(`/api/pdf/marks/counts?prefix=${encodeURIComponent(prefix)}`, { signal })
      .then(async res => {
        if (!res.ok) throw new Error(`counts ${res.status}`)
        return (await res.json()) as CountsResponse
      })
      .then(data => {
        if (signal.aborted) return
        for (const cell of cells) {
          const src = cell.dataset.marksSrc ?? ''
          const counts = data.counts?.[src]
          const marks = (counts?.mark ?? 0) + (counts?.contra ?? 0)
          const questions = counts?.question ?? 0
          if (marks === 0 && questions === 0) continue
          cell.replaceChildren(
            document.createTextNode(questions > 0 ? `${marks} · ${questions}?` : String(marks)),
          )
          cell.title = `${marks} ${marks === 1 ? 'mark' : 'marks'} · ${questions} open ${questions === 1 ? 'question' : 'questions'}`
        }
      })
      .catch(() => {
        if (signal.aborted) return
        for (const cell of cells) cell.title = 'could not load mark counts'
      })
  }
})

// LaTeX notes: `fold proofs` collapses every foldable proof callout at once, and the `.course-latex`
// panel fetches the served `.tex` on first open and shows the section's lines.
document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  const fold = document.querySelector<HTMLButtonElement>('[data-course-fold-proofs]')
  if (fold) {
    const apply = (collapsed: boolean) => {
      const proofs = document.querySelectorAll<HTMLElement>(
        "article.course-notes blockquote.callout.is-collapsible[data-callout='proof']",
      )
      for (const proof of proofs) {
        proof.classList.toggle('is-collapsed', collapsed)
        const content = proof.querySelector<HTMLElement>('.callout-content')
        if (content) content.style.gridTemplateRows = collapsed ? '0fr' : '1fr'
      }
    }
    fold.addEventListener(
      'click',
      () => {
        const next = fold.getAttribute('aria-pressed') !== 'true'
        fold.setAttribute('aria-pressed', String(next))
        apply(next)
      },
      { signal },
    )
  }

  const panel = document.querySelector<HTMLDetailsElement>(
    'details.course-latex[data-course-latex]',
  )
  if (!panel) return
  const code = panel.querySelector<HTMLElement>('pre code')
  const status = panel.querySelector<HTMLElement>('[role="status"]')
  const copy = panel.querySelector<HTMLButtonElement>('[data-course-latex-copy]')
  const src = panel.dataset.courseLatex ?? ''
  const [startRaw, endRaw] = (panel.dataset.courseLatexLines ?? '').split('-')
  const start = Number(startRaw)
  const end = Number(endRaw)
  if (!code || !src || !Number.isFinite(start) || !Number.isFinite(end)) return
  let loaded = false
  let text = ''
  const say = (message: string) => {
    if (status) status.textContent = message
  }
  const load = async () => {
    if (loaded) return
    loaded = true
    say('loading…')
    try {
      const res = await fetch(src, { signal })
      if (!res.ok) throw new Error(`${res.status}`)
      const lines = (await res.text()).split('\n')
      text = lines.slice(start - 1, end).join('\n')
      code.textContent = text
      say(`${end - start + 1} lines of ${src.split('/').pop() ?? 'source'}`)
    } catch (error) {
      if (signal.aborted) return
      loaded = false
      say(`could not load the source (${error instanceof Error ? error.message : 'error'})`)
    }
  }
  panel.addEventListener(
    'toggle',
    () => {
      if (panel.open) void load()
    },
    { signal },
  )
  if (panel.open) void load()
  copy?.addEventListener(
    'click',
    async () => {
      if (!text) await load()
      try {
        await copyText(text)
        copy.textContent = 'copied'
        window.setTimeout(() => {
          if (!signal.aborted) copy.textContent = 'copy'
        }, 1500)
      } catch {
        copy.textContent = 'copy failed'
      }
    },
    { signal },
  )
})

const today = (): string => {
  const now = new Date()
  const pad = (value: number) => String(value).padStart(2, '0')
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
}

// Dev only: the read/problems toggles post to the study endpoint in quartz/scripts/dev.ts, which
// edits `lectures/index.md` and leaves the change for a commit. The watcher rebuilds the pages.
document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  for (const root of document.querySelectorAll<HTMLElement>('.course-study[data-study-endpoint]')) {
    const status = root.querySelector<HTMLElement>('.course-study-status')
    const say = (message: string) => {
      if (status) status.textContent = message
    }
    for (const button of root.querySelectorAll<HTMLButtonElement>('[data-study-key]')) {
      button.addEventListener(
        'click',
        async () => {
          if (button.getAttribute('aria-busy') === 'true') return
          const key = button.dataset.studyKey
          const date = button.getAttribute('aria-pressed') === 'true' ? null : today()
          button.setAttribute('aria-busy', 'true')
          try {
            const res = await fetch(`${root.dataset.studyEndpoint}/study`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                dir: root.dataset.studyDir,
                n: Number(root.dataset.studyN),
                key,
                date,
              }),
              signal,
            })
            const body = (await res.json().catch(() => ({}))) as { error?: string; path?: string }
            if (!res.ok) throw new Error(body.error ?? `study ${res.status}`)
            button.setAttribute('aria-pressed', date ? 'true' : 'false')
            if (date) button.dataset.studyDate = date
            else delete button.dataset.studyDate
            button.textContent = date ? `${key} ${date.slice(5)}` : (key ?? '')
            button.title = date ? `${key} ${date} · click to clear` : `mark ${key} today`
            say(`${body.path ?? 'lectures/index.md'} (uncommitted)`)
          } catch (error) {
            if (signal.aborted) return
            say(`not saved: ${error instanceof Error ? error.message : 'error'}`)
          } finally {
            button.removeAttribute('aria-busy')
          }
        },
        { signal },
      )
    }
  }
})
