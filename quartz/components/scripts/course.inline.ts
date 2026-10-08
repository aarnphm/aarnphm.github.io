import { currentNavSignal } from './nav-lifecycle'

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
