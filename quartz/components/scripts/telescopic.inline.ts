document.addEventListener('nav', () => {
  function expand(event: MouseEvent) {
    if (!(event.target instanceof Element)) return
    const summary = event.target.closest<HTMLButtonElement>('.telescopic-container button.summary')
    if (!summary || event.target.closest('a')) return
    const detail = summary.parentElement
    if (!detail?.classList.contains('close')) return
    if (
      event.type === 'mouseover' &&
      summary.closest('.telescope')?.getAttribute('data-expand') !== 'hover'
    )
      return
    detail.classList.replace('close', 'open')
    summary.setAttribute('aria-expanded', 'true')
    if (document.activeElement === summary) {
      const next = detail.querySelector<HTMLElement>('.expanded button.summary, .expanded a')
      if (next) next.focus()
      else {
        detail.tabIndex = -1
        detail.focus()
      }
    }
  }

  document.addEventListener('click', expand)
  document.addEventListener('mouseover', expand)
  window.addCleanup(() => {
    document.removeEventListener('click', expand)
    document.removeEventListener('mouseover', expand)
  })
})
