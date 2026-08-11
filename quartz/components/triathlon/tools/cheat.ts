import type { RoughAnnotation } from 'rough-notation/lib/model'
import { annotate } from 'rough-notation'
import type { ActivityKind } from '../../../plugins/stores/strava'
import type { TriathlonContext } from '../runtime/context'
import { dist } from '../../../util/triathlon-card'
import { distCombined } from '../../../util/triathlon-card'
import { KM_TO_MI } from '../../../util/triathlon-card'
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

  const dists = root.querySelectorAll<HTMLElement>('.tri-dist[data-km]')
  const sync = () => {
    const mi = context.presentation.distance === 'imperial'
    unit.textContent = mi ? 'mi' : 'km'
    for (const c of cells) {
      if (!mi) {
        c.textContent = c.dataset.km ?? ''
      } else {
        const v = Number(c.dataset.km) * KM_TO_MI
        c.textContent = v < 10 ? v.toFixed(2) : v.toFixed(1)
      }
    }
    for (const d of dists) {
      const km = Number(d.dataset.km)
      const kind = d.dataset.kind ?? 'combined'
      d.textContent =
        kind === 'combined'
          ? distCombined(context.presentation, km)
          : dist(context.presentation, km, kind as ActivityKind)
    }
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
