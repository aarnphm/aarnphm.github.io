import type { PowerCurvePoint } from '../../../plugins/stores/strava'
import type { TriathlonContext } from '../runtime/context'
import { powerCurveActivityLinkAttributes } from '../../../util/triathlon-power-activity'

export const syncPowerCurveActivityLink = (
  link: HTMLAnchorElement | null,
  point: PowerCurvePoint | null | undefined,
  currentActivityId?: number | string | null,
): void => {
  if (!link) return
  const attributes = powerCurveActivityLinkAttributes(point, currentActivityId)
  for (const name of [
    'href',
    'data-power-activity-id',
    'data-power-activity-date',
    'aria-disabled',
  ])
    link.removeAttribute(name)
  for (const [name, value] of Object.entries(attributes)) link.setAttribute(name, value)
}

export const setupPowerCurveActivityLinks = (
  scope: HTMLElement,
  context: TriathlonContext,
): (() => void) => {
  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const link = event.target.closest<HTMLAnchorElement>('a[data-power-activity-id]')
    const activityId = link?.dataset.powerActivityId
    const date = link?.dataset.powerActivityDate
    if (!link || !activityId || !date || !scope.contains(link)) return
    const request = { activityId, date, source: link, handled: false }
    context.events.dispatch('powerActivity', request)
    if (request.handled) event.preventDefault()
  }
  scope.addEventListener('click', onClick)
  return () => scope.removeEventListener('click', onClick)
}
