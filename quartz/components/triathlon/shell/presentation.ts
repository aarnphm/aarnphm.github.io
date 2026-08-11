import type { TriathlonContext } from '../runtime/context'
import { applyI18n } from '../runtime/dom'
import { toggleTriUnit } from '../runtime/preferences'

export const setupPaceUnit = (
  root: HTMLElement,
  context: TriathlonContext,
): (() => void) | null => {
  const buttons = root.querySelectorAll<HTMLButtonElement>('.tri-pace-unit')
  const cells = root.querySelectorAll<HTMLElement>('.tri-pace [data-kph]')
  if (buttons.length === 0 || cells.length === 0) return null
  const sync = () => {
    const mph = context.presentation.distance === 'imperial'
    for (const b of buttons) b.textContent = mph ? 'mph' : 'km/h'
    for (const c of cells) c.textContent = (mph ? c.dataset.mph : c.dataset.kph) ?? ''
  }
  const onClick = () => toggleTriUnit(context.preferences)
  for (const b of buttons) b.addEventListener('click', onClick)
  window.addEventListener('tri:unit', sync)
  sync()
  return () => {
    for (const b of buttons) b.removeEventListener('click', onClick)
    window.removeEventListener('tri:unit', sync)
  }
}

export const setupI18n = (root: HTMLElement, context: TriathlonContext): (() => void) => {
  const apply = (): void => applyI18n(root, context.presentation)
  apply()
  window.addEventListener('tri:locale', apply)
  return () => window.removeEventListener('tri:locale', apply)
}
