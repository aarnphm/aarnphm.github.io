import type { PowerCurvePoint } from '../plugins/stores/strava'
import { triathlonActivityHref } from './triathlon-date-route'

export interface PowerCurveActivityTarget {
  id: string
  date: string
  href: string
}

export const powerCurveActivityTarget = (
  point: PowerCurvePoint | null | undefined,
): PowerCurveActivityTarget | null => {
  if (point?.activityId == null || point.activityDate == null) return null
  const href = triathlonActivityHref(point.activityDate, point.activityId)
  return href ? { id: String(point.activityId), date: point.activityDate, href } : null
}

export const powerCurveActivityLinkAttributes = (
  point: PowerCurvePoint | null | undefined,
  currentActivityId?: number | string | null,
): Record<string, string> => {
  const target = powerCurveActivityTarget(point)
  if (!target || target.id === String(currentActivityId ?? '')) return { 'aria-disabled': 'true' }
  return {
    href: target.href,
    'data-power-activity-id': target.id,
    'data-power-activity-date': target.date,
  }
}
