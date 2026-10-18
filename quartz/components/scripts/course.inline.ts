import { invalidateCanonicalCache } from '../../util/fetch-canonical'
import { parsePdfMark, type PdfMark } from '../../util/pdf-marks'
import { loadNotes, type LectureNote } from '../course-workspace/store'
import { promoteDrafts, type PromoteSource } from './course-promote'
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

const COURSE_API_FRESH_MS = 10_000

interface CachedStateEntry {
  login: string | null
  rows: Map<string, CardState>
  fetchedAt: number
}

interface CachedCountsEntry {
  data: CountsResponse
  fetchedAt: number
}

const stateCache = new Map<string, CachedStateEntry>()
const countsCache = new Map<string, CachedCountsEntry>()

function findCachedState(prefix: string): CachedStateEntry | undefined {
  const exact = stateCache.get(prefix)
  if (exact) return exact
  for (const [cachedPrefix, entry] of stateCache) {
    if (prefix.startsWith(cachedPrefix)) return entry
  }
  return undefined
}

function findCachedCounts(prefix: string): CachedCountsEntry | undefined {
  const exact = countsCache.get(prefix)
  if (exact) return exact
  for (const [cachedPrefix, entry] of countsCache) {
    if (prefix.startsWith(cachedPrefix)) return entry
  }
  return undefined
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
    const lastByUnit = new Map<string, HTMLElement>()
    for (const el of root.querySelectorAll<HTMLElement>('[data-last][data-for]')) {
      const key = el.dataset.for
      if (key) lastByUnit.set(key, el)
    }
    const retentionByUnit = new Map<string, HTMLElement>()
    for (const el of root.querySelectorAll<HTMLElement>('[data-retention][data-for]')) {
      const key = el.dataset.for
      if (key) retentionByUnit.set(key, el)
    }
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
      const unitKey = el.dataset.unit ?? ''
      const lastEl = lastByUnit.get(unitKey)
      if (lastEl) {
        if (last > 0) {
          lastEl.textContent = age(now - last)
          lastEl.title = new Date(last).toISOString()
        } else {
          lastEl.textContent = '–'
          lastEl.removeAttribute('title')
        }
      }
      const rEl = retentionByUnit.get(unitKey)
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
        const rows = new Map((data.states ?? []).map(s => [s.cardId, s]))
        stateCache.set(prefix, { login, rows, fetchedAt: Date.now() })
        return { login, rows }
      })
      byPrefix.set(prefix, pending)
    }
    return pending
  }

  for (const root of roots) {
    const prefix = root.dataset.coursePrefix ?? ''
    const cached = findCachedState(prefix)
    if (cached) {
      fill(root, cached.rows, cached.login)
      if (Date.now() - cached.fetchedAt < COURSE_API_FRESH_MS) continue
    }
    void load(prefix)
      .then(({ login, rows }) => {
        if (!signal.aborted) fill(root, rows, login)
      })
      .catch(() => {
        if (signal.aborted || cached) return
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

  const fillCounts = (cells: HTMLElement[], data: CountsResponse) => {
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
  }

  for (const root of roots) {
    const prefix = root.dataset.marksPrefix ?? ''
    const cells = Array.from(root.querySelectorAll<HTMLElement>('[data-marks-src]'))
    if (!prefix || cells.length === 0) continue
    const cached = findCachedCounts(prefix)
    if (cached) {
      fillCounts(cells, cached.data)
      if (Date.now() - cached.fetchedAt < COURSE_API_FRESH_MS) continue
    }
    fetch(`/api/pdf/marks/counts?prefix=${encodeURIComponent(prefix)}`, { signal })
      .then(async res => {
        if (!res.ok) throw new Error(`counts ${res.status}`)
        return (await res.json()) as CountsResponse
      })
      .then(data => {
        countsCache.set(prefix, { data, fetchedAt: Date.now() })
        if (!signal.aborted) fillCounts(cells, data)
      })
      .catch(() => {
        if (signal.aborted || cached) return
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
    // success is announced only; the toggle label already shows the date
    const say = (message: string, visible = false) => {
      if (!status) return
      status.textContent = message
      status.classList.toggle('visually-hidden', !visible)
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
            const body = (await res.json().catch(() => ({}))) as { error?: string }
            if (!res.ok) throw new Error(body.error ?? `study ${res.status}`)
            button.setAttribute('aria-pressed', date ? 'true' : 'false')
            if (date) button.dataset.studyDate = date
            else delete button.dataset.studyDate
            button.title = date ? `${key} ${date} · click to clear` : `mark ${key} today`
            say(date ? `${key} marked` : `${key} cleared`)
          } catch (error) {
            if (signal.aborted) return
            button.title = `not saved: ${error instanceof Error ? error.message : 'error'}`
            say('not saved', true)
          } finally {
            button.removeAttribute('aria-busy')
          }
        },
        { signal },
      )
    }
  }
})

async function marksFor(src: string, signal: AbortSignal): Promise<PdfMark[] | null> {
  const res = await fetch(`/api/pdf/marks?src=${encodeURIComponent(src)}`, {
    credentials: 'same-origin',
    cache: 'no-store',
    signal,
  }).catch(() => null)
  if (!res?.ok) return null
  const body = (await res.json().catch(() => null)) as { marks?: unknown } | null
  if (!Array.isArray(body?.marks)) return null
  return body.marks.map(parsePdfMark).filter((mark): mark is PdfMark => mark !== null)
}

/** Notes the signed-out workspace keeps in this browser, as passage marks. */
function browserMarks(slug: string): PdfMark[] {
  let notes: LectureNote[]
  try {
    notes = loadNotes(slug)
  } catch {
    return []
  }
  return notes
    .filter(note => note.deletedAt === null)
    .map(note => ({
      id: note.id,
      doc: '',
      src: slug,
      kind: note.kind,
      target: {
        type: 'passage',
        page: 0,
        quote: note.quote,
        heading: note.heading,
        ...(note.quoteText ? { text: note.quoteText } : {}),
      },
      body: note.body,
      visibility: 'private',
      revision: note.revision,
      createdAt: note.createdAt,
      updatedAt: note.createdAt,
    }))
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`

// `start notes` on the pdf strip of a unit without notes: the pdf's marks, the passage marks on
// the unit and on the rendered notes page, and this browser's workspace drafts become a
// `lectures/<nn>.md` skeleton and a `.fc` draft. Dev writes both through the study endpoint;
// the built site copies them.
document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  for (const details of document.querySelectorAll<HTMLDetailsElement>('details.course-promote')) {
    const body = details.querySelector<HTMLElement>('.course-promote-body')
    if (!body) continue
    let loaded = false
    details.addEventListener(
      'toggle',
      async () => {
        if (!details.open || loaded) return
        loaded = true
        const data = details.dataset
        const status = body.querySelector<HTMLElement>('.course-promote-status')
        const say = (message: string) => {
          if (status) status.textContent = message
        }
        const source: PromoteSource = {
          dir: data.promoteDir ?? '',
          n: Number(data.promoteN),
          unitKind: data.promoteKind ?? 'lecture',
          title: data.promoteTitle ?? '',
          resource: data.promoteResource ?? '',
          pdf: data.promotePdf ?? '',
          unit: data.promoteUnit ?? '',
          ...(data.promoteText ? { text: data.promoteText } : {}),
          date: today(),
        }
        const pages = [source.unit, ...(source.text ? [source.text] : [])]
        const [pdfMarks, ...pageMarks] = await Promise.all([
          marksFor(source.pdf, signal),
          ...pages.map(slug => marksFor(slug, signal)),
        ])
        if (signal.aborted) return
        const drafts = pages.flatMap(browserMarks)
        const result = await promoteDrafts(source, pdfMarks ?? [], [
          ...pageMarks.flatMap(marks => marks ?? []),
          ...drafts,
        ])
        const file = `lectures/${String(source.n).padStart(2, '0')}`
        status?.remove()

        const lede = document.createElement('p')
        lede.className = 'course-promote-lede'
        const path = document.createElement('code')
        path.textContent = `${file}.md`
        lede.append(path, ' does not exist yet.')
        const hint = document.createElement('p')
        hint.className = 'course-promote-hint'
        hint.textContent =
          pdfMarks === null
            ? 'Sign in to read your marks. The note starts as an empty skeleton.'
            : result.marks === 0
              ? 'No marks yet. Marks on the pdf and the notes page fill it in page order; you can also start from the empty skeleton.'
              : `Your marks become the first draft, in page order${result.fc ? '; questions become cards' : ''}.`
        body.append(lede, hint)

        if (result.marks > 0) {
          const counts = document.createElement('dl')
          counts.className = 'course-promote-counts'
          const row = (label: string, value: string) => {
            const term = document.createElement('dt')
            term.textContent = label
            const detail = document.createElement('dd')
            detail.textContent = value
            counts.append(term, detail)
          }
          const saved = Math.max(0, result.passages - drafts.length)
          row('pdf', plural(result.pdf, 'mark'))
          if (saved > 0) row('passages', plural(saved, 'mark'))
          if (drafts.length > 0) row('this browser', plural(drafts.length, 'draft'))
          if (result.questions > 0) row('cards', plural(result.questions, 'question'))
          body.append(counts)
        }

        const actions = document.createElement('div')
        actions.className = 'course-promote-actions'
        const button = (label: string, run: () => Promise<string>) => {
          const element = document.createElement('button')
          element.type = 'button'
          element.className = 'course-desk-toggle'
          element.textContent = label
          element.addEventListener(
            'click',
            async () => {
              if (element.getAttribute('aria-busy') === 'true') return
              element.setAttribute('aria-busy', 'true')
              try {
                say(await run())
              } catch (error) {
                if (!signal.aborted)
                  say(`failed: ${error instanceof Error ? error.message : 'error'}`)
              } finally {
                element.removeAttribute('aria-busy')
              }
            },
            { signal },
          )
          actions.append(element)
        }
        const endpoint = data.promoteEndpoint
        if (endpoint)
          button(`write ${file}.md${result.fc ? ' + .fc' : ''}`, async () => {
            const res = await fetch(`${endpoint}/promote`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ dir: source.dir, n: source.n, md: result.md, fc: result.fc }),
              signal,
            })
            const written = (await res.json().catch(() => ({}))) as {
              error?: string
              written?: string[]
            }
            if (!res.ok) throw new Error(written.error ?? `promote ${res.status}`)
            return `wrote ${(written.written ?? []).join(', ')} (uncommitted)`
          })
        button(result.marks === 0 ? 'copy skeleton' : 'copy note', async () => {
          await copyText(result.md)
          return `copied ${file}.md`
        })
        if (result.fc)
          button('copy cards', async () => {
            await copyText(result.fc)
            return `copied ${file}.fc`
          })
        const announce = document.createElement('p')
        announce.className = 'course-promote-status'
        announce.setAttribute('role', 'status')
        announce.setAttribute('aria-live', 'polite')
        const preview = (label: string, text: string) => {
          const fold = document.createElement('details')
          fold.className = 'course-promote-preview'
          const summary = document.createElement('summary')
          summary.textContent = `preview ${label}`
          const pre = document.createElement('pre')
          pre.setAttribute('aria-label', label)
          pre.textContent = text
          fold.append(summary, pre)
          return fold
        }
        body.append(actions, announce, preview(`${file}.md`, result.md))
        if (result.fc) body.append(preview(`${file}.fc`, result.fc))
      },
      { signal },
    )
  }
})

async function postPset(endpoint: string, input: Record<string, unknown>, signal: AbortSignal) {
  const res = await fetch(`${endpoint}/pset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
    signal,
  })
  const body = (await res.json().catch(() => ({}))) as {
    error?: string
    path?: string
    record?: { state: string; date?: string; reveal?: string } | null
  }
  if (!res.ok) throw new Error(body.error ?? `pset ${res.status}`)
  return body
}

// Dev only: the desk's state buttons and `reveal` write `psets/<nn>.md` through the study
// endpoint; `open desk` on a problem set strip creates that file.
document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  for (const desk of document.querySelectorAll<HTMLElement>('.course-desk[data-desk-endpoint]')) {
    const endpoint = desk.dataset.deskEndpoint!
    const status = desk.querySelector<HTMLElement>('.course-study-status')
    const say = (message: string) => {
      if (status) status.textContent = message
    }
    const base = { dir: desk.dataset.deskDir, n: Number(desk.dataset.deskN) }
    for (const row of desk.querySelectorAll<HTMLElement>('[data-desk-problem]')) {
      const problem = Number(row.dataset.deskProblem)
      const logCell = row.querySelector<HTMLElement>('[data-desk-log]')
      const show = (record: { state: string; date?: string; reveal?: string } | null) => {
        for (const button of row.querySelectorAll<HTMLButtonElement>('[data-desk-state]'))
          button.setAttribute(
            'aria-pressed',
            button.dataset.deskState === (record?.state ?? 'open') ? 'true' : 'false',
          )
        if (!logCell) return
        const parts: string[] = []
        if (record && record.state !== 'open')
          parts.push(`${record.state}${record.date ? ` ${record.date}` : ''}`)
        if (record?.reveal) parts.push(`revealed ${record.reveal}`)
        logCell.textContent = parts.join(' · ') || '–'
      }
      const run = async (button: HTMLButtonElement, input: Record<string, unknown>) => {
        if (button.getAttribute('aria-busy') === 'true') return false
        button.setAttribute('aria-busy', 'true')
        try {
          const body = await postPset(endpoint, { ...base, problem, ...input }, signal)
          show(body.record ?? null)
          say('saved')
          return true
        } catch (error) {
          if (!signal.aborted) say(`not saved: ${error instanceof Error ? error.message : 'error'}`)
          return false
        } finally {
          button.removeAttribute('aria-busy')
        }
      }
      for (const button of row.querySelectorAll<HTMLButtonElement>('[data-desk-state]'))
        button.addEventListener(
          'click',
          () => void run(button, { state: button.dataset.deskState, date: today() }),
          { signal },
        )
      const reveal = row.querySelector<HTMLButtonElement>('[data-desk-reveal]')
      reveal?.addEventListener(
        'click',
        async () => {
          if (await run(reveal, { reveal: today() }))
            window.location.assign(reveal.dataset.deskReveal!)
        },
        { signal },
      )
    }
  }
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-desk-open]')) {
    button.addEventListener(
      'click',
      async () => {
        if (button.getAttribute('aria-busy') === 'true') return
        button.setAttribute('aria-busy', 'true')
        try {
          await postPset(
            button.dataset.deskOpen!,
            { dir: button.dataset.deskDir, n: Number(button.dataset.deskN) },
            signal,
          )
          // the watcher rebuilds before the new page exists; wait for it, then open the desk
          button.textContent = 'opening…'
          const href = new URL(button.dataset.deskHref!, location.href)
          for (let tries = 0; tries < 180; tries++) {
            const res = await fetch(href, { method: 'HEAD', cache: 'no-store', signal })
            if (res.ok) break
            await new Promise(resolve => setTimeout(resolve, Math.min(350, 120 + tries * 30)))
          }
          if (signal.aborted) return
          invalidateCanonicalCache(href)
          if (window.spaNavigate) window.spaNavigate(href)
          else location.assign(href)
        } catch (error) {
          if (signal.aborted) return
          button.textContent = 'open desk'
          button.title = `not created: ${error instanceof Error ? error.message : 'error'}`
        } finally {
          button.removeAttribute('aria-busy')
        }
      },
      { signal },
    )
  }
})

// a note's statement link lands on its row; SPA navigation leaves `:target` unset, so mark it
document.addEventListener('nav', () => {
  const table = document.querySelector('.course-statements [data-course-rows]')
  if (!table) return
  const mark = () => {
    table.querySelector('[data-arrived]')?.removeAttribute('data-arrived')
    const id = decodeURIComponent(location.hash.slice(1))
    if (id) table.querySelector(`tr[id="${CSS.escape(id)}"]`)?.setAttribute('data-arrived', '')
  }
  mark()
  window.addEventListener('hashchange', mark)
  window.addCleanup(() => window.removeEventListener('hashchange', mark))
})
