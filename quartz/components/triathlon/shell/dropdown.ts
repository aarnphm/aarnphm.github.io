export const setupDropdown = (
  root: HTMLElement,
  wrapSel: string,
  btnSel: string,
  panelSel: string,
  openClass: string,
): (() => void) | null => {
  const btn = root.querySelector<HTMLButtonElement>(btnSel)
  const wrap = root.querySelector<HTMLElement>(wrapSel)
  const panel = root.querySelector<HTMLElement>(panelSel)
  if (!btn || !wrap || !panel) return null
  let focusFrame: number | null = null

  const close = (restoreFocus = false) => {
    wrap.classList.remove(openClass)
    panel.setAttribute('aria-hidden', 'true')
    btn.setAttribute('aria-expanded', 'false')
    if (restoreFocus) btn.focus()
  }
  const closeButton =
    root.closest('.sidepanel-container') && !root.dataset.triView
      ? document.createElement('button')
      : null
  const onClose = () => close(true)
  if (closeButton) {
    closeButton.type = 'button'
    closeButton.className = 'tri-dropdown-close'
    closeButton.textContent = '×'
    closeButton.setAttribute('aria-label', `Close ${btn.textContent?.trim() ?? 'panel'}`)
    closeButton.addEventListener('click', onClose)
    const content = panel.querySelector('.tri-gear-scroll') ?? panel
    content.prepend(closeButton)
  }
  const onBtn = () => {
    const open = wrap.classList.toggle(openClass)
    panel.setAttribute('aria-hidden', open ? 'false' : 'true')
    btn.setAttribute('aria-expanded', String(open))
    if (open && closeButton) {
      if (focusFrame !== null) cancelAnimationFrame(focusFrame)
      focusFrame = requestAnimationFrame(() => {
        focusFrame = null
        if (wrap.classList.contains(openClass)) closeButton.focus({ preventScroll: true })
      })
    }
  }
  const onDocClick = (event: MouseEvent) => {
    if (event.target instanceof Node && !wrap.contains(event.target)) close()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape' && wrap.classList.contains(openClass)) close(true)
  }

  btn.addEventListener('click', onBtn)
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)

  return () => {
    if (focusFrame !== null) cancelAnimationFrame(focusFrame)
    closeButton?.removeEventListener('click', onClose)
    closeButton?.remove()
    btn.removeEventListener('click', onBtn)
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('keydown', onKey)
  }
}
