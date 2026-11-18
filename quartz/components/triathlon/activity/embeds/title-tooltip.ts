import { el } from '../../runtime/dom'

export const setupActivityTitleTooltip = (): (() => void) | null => {
  if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return null
  document.body.querySelector('.tri-activity-title-tip')?.remove()
  const tip = el('div', 'tri-gloss tri-activity-title-tip', undefined, {
    role: 'tooltip',
    'aria-hidden': 'true',
  })
  document.body.appendChild(tip)
  let current: HTMLElement | null = null
  const hide = (): void => {
    current = null
    tip.classList.remove('tri-gloss--on')
    tip.setAttribute('aria-hidden', 'true')
  }
  const move = (event: PointerEvent): void => {
    const icon =
      event.target instanceof Element
        ? event.target.closest('.tri-act-head > .tri-ico, .tri-act-head > .tri-act-icon-link')
        : null
    const activity = icon?.closest<HTMLElement>('.tri-act[data-activity-title]') ?? null
    const title = activity?.dataset.activityTitle
    if (!activity || !title) {
      hide()
      return
    }
    if (activity !== current) {
      current = activity
      tip.textContent = title
      tip.classList.add('tri-gloss--on')
      tip.setAttribute('aria-hidden', 'false')
    }
    const rect = tip.getBoundingClientRect()
    const offset = 12
    const edge = 8
    const preferredLeft = event.clientX + offset
    const preferredTop = event.clientY + offset
    const left =
      preferredLeft + rect.width <= window.innerWidth - edge
        ? preferredLeft
        : event.clientX - offset - rect.width
    const top =
      preferredTop + rect.height <= window.innerHeight - edge
        ? preferredTop
        : event.clientY - offset - rect.height
    tip.style.left = `${Math.min(Math.max(edge, left), window.innerWidth - edge - rect.width)}px`
    tip.style.top = `${Math.min(Math.max(edge, top), window.innerHeight - edge - rect.height)}px`
  }
  document.body.addEventListener('pointermove', move, { passive: true })
  document.body.addEventListener('pointerleave', hide)
  return () => {
    document.body.removeEventListener('pointermove', move)
    document.body.removeEventListener('pointerleave', hide)
    tip.remove()
  }
}
