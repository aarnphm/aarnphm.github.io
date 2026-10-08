export interface DropdownOptions {
  /** Holds the trigger and the panel; a click outside it closes the panel. */
  wrap: HTMLElement
  trigger: HTMLButtonElement
  panel: HTMLElement
  /** Toggled on `wrap`; the host's styles show the panel while it is set. */
  openClass: string
  /**
   * Adds a "×" button as the first child of `container` and moves focus to it on open. Use it
   * when the panel covers the page, so a pointer user has a stable way out.
   */
  closeButton?: { className: string; container: HTMLElement }
}

/** A disclosure button and its panel. Returns the cleanup for its listeners. */
export const setupDropdown = ({
  wrap,
  trigger,
  panel,
  openClass,
  closeButton: close,
}: DropdownOptions): (() => void) => {
  let focusFrame: number | null = null
  const isOpen = () => wrap.classList.contains(openClass)
  const setOpen = (open: boolean): void => {
    wrap.classList.toggle(openClass, open)
    panel.setAttribute('aria-hidden', String(!open))
    trigger.setAttribute('aria-expanded', String(open))
  }
  const hide = (restoreFocus = false): void => {
    setOpen(false)
    if (restoreFocus) trigger.focus()
  }

  const closeButton = close ? document.createElement('button') : null
  const onClose = () => hide(true)
  if (close && closeButton) {
    closeButton.type = 'button'
    closeButton.className = close.className
    closeButton.textContent = '×'
    closeButton.setAttribute('aria-label', `Close ${trigger.textContent?.trim() || 'panel'}`)
    closeButton.addEventListener('click', onClose)
    close.container.prepend(closeButton)
  }

  const onTrigger = (): void => {
    const open = !isOpen()
    setOpen(open)
    if (!open || !closeButton) return
    if (focusFrame !== null) cancelAnimationFrame(focusFrame)
    focusFrame = requestAnimationFrame(() => {
      focusFrame = null
      if (isOpen()) closeButton.focus({ preventScroll: true })
    })
  }
  const onDocumentClick = (event: MouseEvent): void => {
    if (event.target instanceof Node && !wrap.contains(event.target)) hide()
  }
  const onKeydown = (event: KeyboardEvent): void => {
    if (event.key !== 'Escape' || !isOpen()) return
    // Later Escape handlers on the page skip a handled event instead of moving focus again.
    event.preventDefault()
    hide(true)
  }

  trigger.addEventListener('click', onTrigger)
  document.addEventListener('click', onDocumentClick)
  document.addEventListener('keydown', onKeydown)
  return () => {
    if (focusFrame !== null) cancelAnimationFrame(focusFrame)
    closeButton?.removeEventListener('click', onClose)
    closeButton?.remove()
    trigger.removeEventListener('click', onTrigger)
    document.removeEventListener('click', onDocumentClick)
    document.removeEventListener('keydown', onKeydown)
  }
}
