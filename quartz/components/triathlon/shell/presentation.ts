import type { TriathlonContext } from '../runtime/context'
import { isActivityKind } from '../../../plugins/stores/strava'
import { dist, distCombined } from '../../../util/triathlon-card'
import { formatEquipmentDistance } from '../../../util/triathlon-equipment'
import { triText } from '../../../util/triathlon-i18n'
import { formatTriathlonMaintenanceDistance } from '../../../util/triathlon-maintenance'
import { applyI18n } from '../runtime/dom'
import { toggleTriUnit } from '../runtime/preferences'

export const setupDistanceUnits = (
  root: HTMLElement,
  context: TriathlonContext,
): (() => void) | null => {
  const values = root.querySelectorAll<HTMLElement>(
    '.tri-unit-distance[data-km], .tri-unit-distance[data-mi]',
  )
  if (values.length === 0) return null
  const sync = (): void => {
    for (const value of values) {
      const kind = value.dataset.kind ?? 'combined'
      if (kind === 'maintenance') {
        const miles = Number(value.dataset.mi)
        if (Number.isFinite(miles))
          value.textContent = formatTriathlonMaintenanceDistance(
            miles,
            context.presentation.distance,
          )
        continue
      }
      const kilometres = Number(value.dataset.km)
      value.textContent =
        kind === 'equipment'
          ? formatEquipmentDistance(kilometres * 1_000, context.presentation.distance)
          : kind === 'combined'
            ? distCombined(context.presentation, kilometres)
            : isActivityKind(kind)
              ? dist(context.presentation, kilometres, kind)
              : value.textContent
    }
  }
  sync()
  window.addEventListener('tri:unit', sync)
  return () => window.removeEventListener('tri:unit', sync)
}

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

const SUMMARY_EASE = 'cubic-bezier(0.23, 1, 0.32, 1)'

export const setupSeasonSummary = (
  root: HTMLElement,
  context: TriathlonContext,
): (() => void) | null => {
  const foot = root.querySelector<HTMLElement>('.tri-foot')
  const button = foot?.querySelector<HTMLButtonElement>('.tri-foot-more')
  const extraLegs = foot
    ? Array.from(foot.querySelectorAll<HTMLElement>('.tri-leg[data-tri-extra="true"]'))
    : []
  if (!foot || !button || extraLegs.length === 0) return null

  let activeAnimations: Animation[] = []

  const clearAnimations = (): void => {
    const running = activeAnimations
    activeAnimations = []
    for (const anim of running) anim.cancel()
    delete foot.dataset.collapsing
    foot.style.removeProperty('height')
    for (const leg of extraLegs) {
      leg.style.removeProperty('left')
      leg.style.removeProperty('top')
    }
  }

  const syncLabel = (expanded: boolean): void => {
    const key = expanded ? 'fewer activities' : 'more activities'
    button.dataset.i18nAriaLabel = key
    button.setAttribute('aria-label', triText(context.presentation.locale, key))
  }

  const onClick = (): void => {
    const nextExpanded = foot.dataset.expanded !== 'true'
    const reducedMotion =
      typeof foot.animate !== 'function' ||
      window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const startFootRect = foot.getBoundingClientRect()
    const startBtnRect = button.getBoundingClientRect()
    const startLegRects = extraLegs.map(leg => leg.getBoundingClientRect())
    const startLegOpacities = extraLegs.map(leg => {
      const visible = foot.dataset.expanded === 'true' || foot.dataset.collapsing === 'true'
      if (!visible) return 0
      const parsed = Number.parseFloat(getComputedStyle(leg).opacity)
      return Number.isFinite(parsed) ? parsed : 1
    })

    clearAnimations()

    foot.dataset.expanded = String(nextExpanded)
    button.setAttribute('aria-expanded', String(nextExpanded))
    syncLabel(nextExpanded)
    window.dispatchEvent(new Event('scrollend'))

    if (reducedMotion) return

    const endFootRect = foot.getBoundingClientRect()
    const endBtnRect = button.getBoundingClientRect()
    const batch: Animation[] = []

    if (nextExpanded) {
      if (Math.abs(endFootRect.height - startFootRect.height) > 0.5) {
        batch.push(
          foot.animate(
            [{ height: `${startFootRect.height}px` }, { height: `${endFootRect.height}px` }],
            { duration: 220, easing: SUMMARY_EASE },
          ),
        )
      }

      const dx = startBtnRect.left - endBtnRect.left
      const dy = startBtnRect.top - endBtnRect.top
      if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
        batch.push(
          button.animate(
            [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'translate(0px, 0px)' }],
            { duration: 220, easing: SUMMARY_EASE },
          ),
        )
      }

      for (let i = 0; i < extraLegs.length; i++) {
        const leg = extraLegs[i]
        const fromOpacity = startLegOpacities[i] > 0.05 ? startLegOpacities[i] : 0
        const fromX = (1 - fromOpacity) * -8
        const delay = fromOpacity > 0.05 ? 0 : i * 18
        batch.push(
          leg.animate(
            [
              { opacity: fromOpacity, transform: `translateX(${fromX}px)` },
              { opacity: 1, transform: 'translateX(0px)' },
            ],
            { duration: 200, delay, easing: SUMMARY_EASE, fill: 'both' },
          ),
        )
      }
    } else {
      for (let i = 0; i < extraLegs.length; i++) {
        const leg = extraLegs[i]
        const rect = startLegRects[i]
        leg.style.left = `${rect.left - startFootRect.left}px`
        leg.style.top = `${rect.top - startFootRect.top}px`
      }
      foot.dataset.collapsing = 'true'

      if (Math.abs(endFootRect.height - startFootRect.height) > 0.5) {
        batch.push(
          foot.animate(
            [{ height: `${startFootRect.height}px` }, { height: `${endFootRect.height}px` }],
            { duration: 190, easing: SUMMARY_EASE },
          ),
        )
      }

      const dx = startBtnRect.left - endBtnRect.left
      const dy = startBtnRect.top - endBtnRect.top
      if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
        batch.push(
          button.animate(
            [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'translate(0px, 0px)' }],
            { duration: 190, easing: SUMMARY_EASE },
          ),
        )
      }

      for (let i = 0; i < extraLegs.length; i++) {
        const leg = extraLegs[i]
        const fromOpacity = startLegOpacities[i] > 0.05 ? startLegOpacities[i] : 1
        const reverseIndex = extraLegs.length - 1 - i
        const delay = reverseIndex * 14
        batch.push(
          leg.animate(
            [
              { opacity: fromOpacity, transform: 'translateX(0px)' },
              { opacity: 0, transform: 'translateX(-8px)' },
            ],
            { duration: 150, delay, easing: SUMMARY_EASE, fill: 'both' },
          ),
        )
      }
    }

    activeAnimations = batch
    void Promise.allSettled(batch.map(anim => anim.finished)).then(() => {
      if (activeAnimations === batch) {
        clearAnimations()
        window.dispatchEvent(new Event('scrollend'))
      }
    })
  }

  syncLabel(foot.dataset.expanded === 'true')
  button.addEventListener('click', onClick)
  return () => {
    clearAnimations()
    button.removeEventListener('click', onClick)
  }
}
