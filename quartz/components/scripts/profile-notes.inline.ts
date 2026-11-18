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
    // Hovering or focusing the mark lights its note (marginalia.scss boxes it in the rail), the
    // mirror of a hovered note boxing its mark. The popover runtime skips shown footnotes for this.
    const light = (on: boolean) => () => note.toggleAttribute('data-lit', on)
    mark.addEventListener('pointerenter', light(true), { signal })
    mark.addEventListener('pointerleave', light(false), { signal })
    mark.addEventListener(
      'focusin',
      () => {
        if (mark.matches(':focus-visible')) note.toggleAttribute('data-lit', true)
      },
      { signal },
    )
    mark.addEventListener('focusout', light(false), { signal })
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

  // A rail note hangs from its citation and takes no room, so near a chapter's end it can run
  // into the next chapter's separator. Pad the chapter by the overhang: measure every rail note
  // first, then write.
  const railNotes = [
    ...article.querySelectorAll<HTMLElement>('.profile-note:not(.profile-note-boxed)'),
  ]
  const chapterSections = [
    ...article.querySelectorAll<HTMLElement>("section.collapsible-header[data-level='2']"),
  ]
  const nextChapter = (chapter: HTMLElement) => {
    let el = chapter.nextElementSibling
    while (el && !el.matches("section.collapsible-header[data-level='2']"))
      el = el.nextElementSibling
    return el
  }
  let fitFrame = 0
  const fit = () => {
    fitFrame = 0
    const need = new Map<HTMLElement, number>()
    const room = new Map<HTMLElement, string>()
    for (const note of railNotes) {
      if (getComputedStyle(note).position !== 'absolute') continue
      // An opened long note may run down to the end of the text it sits beside (its chapter's
      // body, or the tldr box) and scrolls inside past that.
      const body = note.classList.contains('profile-note-long')
        ? note.querySelector<HTMLElement>(':scope > .profile-note-body')
        : null
      if (body) {
        const text =
          note.closest<HTMLElement>(".collapsible-header-content[data-level='2']") ??
          note.closest<HTMLElement>('#tldr')
        room.set(
          body,
          text
            ? `${Math.floor(text.getBoundingClientRect().bottom - body.getBoundingClientRect().top)}px`
            : '',
        )
      }
      const chapter = note.closest<HTMLElement>("section.collapsible-header[data-level='2']")
      const next = chapter && nextChapter(chapter)
      if (!chapter || !next) continue
      const padded = parseFloat(chapter.style.getPropertyValue('--mag-note-clear')) || 0
      const over =
        note.getBoundingClientRect().bottom + 24 - (next.getBoundingClientRect().top - padded)
      need.set(chapter, Math.max(need.get(chapter) ?? 0, over))
    }
    for (const [body, value] of room) {
      if (body.style.getPropertyValue('--note-room') === value) continue
      if (value) body.style.setProperty('--note-room', value)
      else body.style.removeProperty('--note-room')
    }
    for (const chapter of chapterSections) {
      const over = Math.ceil(need.get(chapter) ?? 0)
      const value = over > 0 ? `${over}px` : ''
      if (chapter.style.getPropertyValue('--mag-note-clear') === value) continue
      if (value) chapter.style.setProperty('--mag-note-clear', value)
      else chapter.style.removeProperty('--mag-note-clear')
    }
  }
  const scheduleFit = () => {
    if (!fitFrame) fitFrame = requestAnimationFrame(fit)
  }
  if (railNotes.length && chapterSections.length) {
    scheduleFit()
    void document.fonts?.ready.then(() => {
      if (!signal.aborted) scheduleFit()
    })
    const observer = new ResizeObserver(scheduleFit)
    observer.observe(article)
    for (const note of railNotes) observer.observe(note)
    signal.addEventListener(
      'abort',
      () => {
        observer.disconnect()
        cancelAnimationFrame(fitFrame)
      },
      { once: true },
    )
  }

  // Long notes sit folded to a few lines in the rail; the toggle under them opens the rest.
  for (const button of article.querySelectorAll<HTMLButtonElement>('.profile-note-more')) {
    const note = button.closest<HTMLElement>('.profile-note-long')
    if (!note) continue
    button.addEventListener(
      'click',
      () => {
        const open = !note.hasAttribute('data-expanded')
        note.toggleAttribute('data-expanded', open)
        button.setAttribute('aria-expanded', String(open))
        button.textContent = open ? 'less' : 'more'
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
