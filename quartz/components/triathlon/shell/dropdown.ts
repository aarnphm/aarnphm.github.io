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

  const scroller = panel.querySelector<HTMLElement>(`${panelSel}-scroll`)
  const base = panelSel.slice(1)
  const updateFade = () => {
    if (!scroller) return
    panel.classList.toggle(`${base}--top`, scroller.scrollTop > 4)
    panel.classList.toggle(
      `${base}--more`,
      scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop > 4,
    )
  }

  const close = (restoreFocus = false) => {
    wrap.classList.remove(openClass)
    panel.setAttribute('aria-hidden', 'true')
    btn.setAttribute('aria-expanded', 'false')
    if (restoreFocus) btn.focus()
  }
  const onBtn = () => {
    const open = wrap.classList.toggle(openClass)
    panel.setAttribute('aria-hidden', open ? 'false' : 'true')
    btn.setAttribute('aria-expanded', String(open))
    if (open) updateFade()
  }
  const onDocClick = (event: MouseEvent) => {
    if (event.target instanceof Node && !wrap.contains(event.target)) close()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape' && wrap.classList.contains(openClass)) close(true)
  }

  btn.addEventListener('click', onBtn)
  scroller?.addEventListener('scroll', updateFade, { passive: true })
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)

  return () => {
    btn.removeEventListener('click', onBtn)
    scroller?.removeEventListener('scroll', updateFade)
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('keydown', onKey)
  }
}
