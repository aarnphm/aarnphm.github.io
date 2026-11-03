import { rootNavSignal } from './root-lifecycle'

const hydrated = new WeakMap<HTMLElement, AbortSignal>()

function hydrateArticle(article: HTMLElement) {
  if (article.closest('[data-layout]')?.getAttribute('data-layout') !== 'newspaper') return
  const signal = rootNavSignal(article)
  if (hydrated.get(article) === signal) return
  hydrated.set(article, signal)
  const chapters = [
    ...article.querySelectorAll<HTMLElement>(
      "section.collapsible-header[data-level='2'] > .mag-running-head",
    ),
  ].map(head => ({ head, heading: head.parentElement?.querySelector('h2') ?? null }))
  let frame = 0
  const updateHeads = () => {
    frame = 0
    const states = chapters.map(({ head, heading }) => {
      const rect = head.getBoundingClientRect()
      return {
        head,
        running: heading !== null && heading.getBoundingClientRect().bottom < rect.bottom,
      }
    })
    for (const { head, running } of states) {
      head.toggleAttribute('data-running', running)
    }
  }
  const scheduleHeads = () => {
    if (!frame) frame = requestAnimationFrame(updateHeads)
  }
  scheduleHeads()
  void document.fonts?.ready.then(() => {
    if (!signal.aborted) scheduleHeads()
  })
  const scroller = article.closest('.stacked-note') ?? window
  scroller.addEventListener('scroll', scheduleHeads, { passive: true, signal })
  window.addEventListener('resize', scheduleHeads, { passive: true, signal: signal })
  signal.addEventListener('abort', () => cancelAnimationFrame(frame), { once: true })

  // Below the rail width each note is folded under its paragraph (marginalia.scss); tapping the
  // mark (sidenote word or footnote square) opens or closes it. With the rail, a note already sits
  // beside its mark, so the mark keeps its usual behaviour.
  for (const mark of article.querySelectorAll<HTMLElement>('[data-note]')) {
    const note = article.querySelector<HTMLElement>(`[id="${CSS.escape(mark.dataset.note ?? '')}"]`)
    if (!note || note.classList.contains('profile-note-boxed')) continue
    const link = mark instanceof HTMLAnchorElement
    mark.setAttribute('aria-controls', note.id)
    mark.setAttribute('aria-expanded', 'false')
    if (!link) {
      mark.setAttribute('role', 'button')
      mark.tabIndex = 0
    }
    const toggle = (event: Event) => {
      if (getComputedStyle(note).position === 'absolute') return
      event.preventDefault()
      event.stopImmediatePropagation()
      const open = !note.hasAttribute('data-open')
      note.toggleAttribute('data-open', open)
      mark.setAttribute('aria-expanded', String(open))
    }
    mark.addEventListener('click', toggle, { capture: true, signal })
    if (!link)
      mark.addEventListener(
        'keydown',
        event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            toggle(event)
          }
        },
        { signal },
      )
  }
}

function hydrate(root: Document | HTMLElement) {
  if (root instanceof HTMLElement && root.matches('article')) hydrateArticle(root)
  root.querySelectorAll<HTMLElement>('article').forEach(hydrateArticle)
}

document.addEventListener('nav', () => hydrate(document))
document.addEventListener('contentdecrypted', event => hydrate(event.detail.content))
