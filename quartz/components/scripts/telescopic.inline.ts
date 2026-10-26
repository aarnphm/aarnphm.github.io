document.addEventListener('nav', () => {
  function open(summary: HTMLElement) {
    const detail = summary.parentElement
    if (!detail?.classList.contains('close')) return
    detail.classList.replace('close', 'open')
    summary.setAttribute('aria-expanded', 'true')
    if (document.activeElement === summary) {
      // :scope keeps an outer .expanded from matching the summary that was just hidden.
      const next = detail.querySelector<HTMLElement>(
        ':scope > .expanded .summary, :scope > .expanded a',
      )
      if (next) next.focus()
      else {
        detail.tabIndex = -1
        detail.focus()
      }
    }
  }

  function summaryFor(event: Event) {
    if (!(event.target instanceof Element) || event.target.closest('a')) return null
    return event.target.closest<HTMLElement>('.telescopic-container .summary')
  }

  function expand(event: MouseEvent) {
    const summary = summaryFor(event)
    if (!summary) return
    if (
      event.type === 'mouseover' &&
      summary.closest('.telescope')?.getAttribute('data-expand') !== 'hover'
    )
      return
    open(summary)
  }

  function onKey(event: KeyboardEvent) {
    if (event.key !== 'Enter' && event.key !== ' ') return
    const summary = summaryFor(event)
    if (!summary) return
    event.preventDefault()
    open(summary)
  }

  document.addEventListener('click', expand)
  document.addEventListener('mouseover', expand)
  document.addEventListener('keydown', onKey)
  window.addCleanup(() => {
    document.removeEventListener('click', expand)
    document.removeEventListener('mouseover', expand)
    document.removeEventListener('keydown', onKey)
  })
})
