// CSS owns sticky placement and note layering. Reveal the running title once its chapter
// heading passes the bar, without changing the space reserved for either heading.
document.addEventListener('nav', () => {
  if (document.body.dataset.layout !== 'newspaper') return
  const controller = new AbortController()
  const chapters = [
    ...document.querySelectorAll<HTMLElement>(
      "article section.collapsible-header[data-level='2'] > .mag-running-head",
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
    if (!controller.signal.aborted) scheduleHeads()
  })
  window.addEventListener('scroll', scheduleHeads, { passive: true, signal: controller.signal })
  window.addEventListener('resize', scheduleHeads, { passive: true, signal: controller.signal })
  window.addCleanup(() => {
    controller.abort()
    cancelAnimationFrame(frame)
  })
})

// Below the rail width each note is folded under its paragraph (marginalia.scss); tapping the
// mark (sidenote word or footnote square) opens or closes it. With the rail, a note already sits
// beside its mark, so the mark keeps its usual behaviour.
document.addEventListener('nav', () => {
  if (document.body.dataset.layout !== 'newspaper') return
  const controller = new AbortController()
  const { signal } = controller
  for (const mark of document.querySelectorAll<HTMLElement>('article [data-note]')) {
    const note = document.getElementById(mark.dataset.note ?? '')
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
          if (event.key === 'Enter' || event.key === ' ') toggle(event)
        },
        { signal },
      )
  }
  window.addCleanup(() => controller.abort())
})
