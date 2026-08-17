import type { RoughAnnotation } from 'rough-notation/lib/model'
import { annotate } from 'rough-notation'
import type { TriathlonContext } from '../runtime/context'
import { raceDistanceValue } from '../../../util/triathlon-card'
import { toggleTriUnit } from '../runtime/preferences'

export const setupCheat = (root: HTMLElement, context: TriathlonContext): (() => void) | null => {
  const unit = root.querySelector<HTMLButtonElement>('.tri-cheat-unit')
  const cells = root.querySelectorAll<HTMLElement>('.tri-cheat td[data-km]')
  if (!unit || cells.length === 0) return null

  const target = root.querySelector<HTMLElement>('.tri-cheat-target')
  let ann: RoughAnnotation | null = null
  let showTimer = 0
  if (target) {
    const color = getComputedStyle(root).getPropertyValue('--tri-accent').trim() || '#fc4c02'
    ann = annotate(target, {
      type: 'circle',
      color,
      strokeWidth: 1.6,
      padding: 5,
      animationDuration: 800,
      iterations: 2,
    })
    const a = ann
    showTimer = window.setTimeout(() => a.show(), 200)
  }

  const sync = () => {
    const mi = context.presentation.distance === 'imperial'
    unit.textContent = mi ? 'mi' : 'km'
    for (const c of cells)
      c.textContent = raceDistanceValue(context.presentation, Number(c.dataset.km))
  }
  const onClick = () => toggleTriUnit(context.preferences)
  unit.addEventListener('click', onClick)
  window.addEventListener('tri:unit', sync)
  sync()
  return () => {
    window.clearTimeout(showTimer)
    unit.removeEventListener('click', onClick)
    window.removeEventListener('tri:unit', sync)
    ann?.remove()
  }
}
