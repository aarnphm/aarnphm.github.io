// Margin notes and chapter bars are placed by the server render and CSS anchor positioning. This
// only shows a chapter bar's title once its heading has scrolled past, and hides a bar a note
// covers. Both toggle visibility, so nothing moves.
document.addEventListener('nav', () => {
  if (document.body.dataset.layout !== 'newspaper') return
  const controller = new AbortController()
  const chapters = [
    ...document.querySelectorAll<HTMLElement>(
      "article section.collapsible-header[data-level='2'] > .mag-running-head",
    ),
  ].map(head => ({ head, heading: head.parentElement!.querySelector('h2') }))
  const notes = [...document.querySelectorAll<HTMLElement>('article .profile-note')]
  let frame = 0
  const updateHeads = () => {
    frame = 0
    const noteRects = notes
      .filter(note => getComputedStyle(note).position === 'absolute')
      .map(note => note.getBoundingClientRect())
    const states = chapters.map(({ head, heading }) => {
      const rect = head.getBoundingClientRect()
      return {
        head,
        running: heading !== null && heading.getBoundingClientRect().bottom < rect.bottom,
        obscured: noteRects.some(
          note =>
            note.left < rect.right + 8 &&
            note.right > rect.left - 8 &&
            note.top < rect.bottom + 8 &&
            note.bottom > rect.top - 8,
        ),
      }
    })
    for (const { head, running, obscured } of states) {
      head.toggleAttribute('data-running', running)
      head.toggleAttribute('data-obscured', obscured)
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
