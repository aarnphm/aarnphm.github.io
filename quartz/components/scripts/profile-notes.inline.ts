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
