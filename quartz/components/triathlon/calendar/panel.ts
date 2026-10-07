const calendarAnchor = (): boolean =>
  /^#calendar(?:(?:-|#)(?:list|year|\d{4}(?:-year)?|race-[a-z0-9-]+|training(?:-\d{4}-\d{2}-\d{2})?(?:-(?:list|week|month))?))?$/.test(
    window.location.hash,
  )

export const setupCalendarPanel = (root: HTMLElement): (() => void) | null => {
  if (root.dataset.triView) return null
  const button = root.querySelector<HTMLButtonElement>('.tri-calendar-btn')
  const panel = root.querySelector<HTMLElement>('.tri-calendar-panel')
  const scrim = root.querySelector<HTMLElement>('.tri-calendar-scrim')
  const closeButton = panel?.querySelector<HTMLButtonElement>('.tri-calendar-close')
  if (!button || !panel || !closeButton || !scrim) return null

  const isOpen = (): boolean => root.classList.contains('tri-calendar-open')
  const open = (): void => {
    for (const [openClass, selector] of [
      ['tri-analytics-open', '.tri-analytics .tri-ana-close'],
      ['tri-calc-open', '.tri-calc-close'],
      ['tri-map-open', '.tri-map-close'],
      ['tri-training-open', '.tri-training-close'],
    ]) {
      if (root.classList.contains(openClass))
        root.querySelector<HTMLButtonElement>(selector)?.click()
    }
    root.classList.add('tri-calendar-open')
    panel.setAttribute('aria-hidden', 'false')
    button.setAttribute('aria-expanded', 'true')
    panel.focus({ preventScroll: true })
  }
  const close = (): void => {
    if (!isOpen()) return
    root.classList.remove('tri-calendar-open')
    panel.setAttribute('aria-hidden', 'true')
    button.setAttribute('aria-expanded', 'false')
    if (calendarAnchor()) {
      const url = new URL(window.location.href)
      url.hash = ''
      window.history.replaceState(window.history.state, '', url)
    }
    button.focus({ preventScroll: true })
  }
  const onKey = (event: KeyboardEvent): void => {
    if (event.key !== 'Escape' || !isOpen()) return
    event.preventDefault()
    close()
  }
  const onHashChange = (): void => {
    if (calendarAnchor()) open()
  }

  button.addEventListener('click', open)
  closeButton.addEventListener('click', close)
  scrim.addEventListener('click', close)
  document.addEventListener('keydown', onKey)
  window.addEventListener('hashchange', onHashChange)
  onHashChange()
  return () => {
    button.removeEventListener('click', open)
    closeButton.removeEventListener('click', close)
    scrim.removeEventListener('click', close)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('hashchange', onHashChange)
  }
}
