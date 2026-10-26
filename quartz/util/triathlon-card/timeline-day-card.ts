import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TimelineActivity } from '../triathlon-timeline'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { buildBattery, buildIcon } from './activity-icons'
import { dist, dur, prettyDate } from './metric-text'

export const timelineActivityValue = (
  presentation: TriathlonPresentation,
  activity: TimelineActivity,
): string =>
  activity.distanceKm > 0
    ? dist(presentation, activity.distanceKm, activity.sport)
    : dur(activity.movingTimeS)

export const buildTimelineDayCard = <N>(
  f: TriNodeFactory<N>,
  dateIso: string,
  activities: readonly TimelineActivity[],
): N => {
  const card = f.el('div', 'tri-timeline-card')
  f.add(card, f.el('span', 'tri-pop-date', prettyDate(dateIso)))
  const day = activities.toSorted((a, b) => b.distanceKm - a.distanceKm)
  if (day.length === 0) {
    const label = triText(f.presentation.locale, 'rest')
    const rest = f.el('span', 'tri-timeline-row tri-timeline-rest', undefined, {
      role: 'group',
      'aria-label': label,
    })
    f.add(rest, buildBattery(f), f.el('span', 'tri-timeline-value', label, { 'data-i18n': 'rest' }))
    f.add(card, rest)
    return card
  }

  const list = f.el('ul', 'tri-timeline-list')
  for (const activity of day) {
    const value = timelineActivityValue(f.presentation, activity)
    const item = f.el('li', 'tri-timeline-item')
    const entry = f.el('span', 'tri-timeline-row tri-timeline-activity', undefined, {
      role: 'group',
      'aria-label': `${triText(f.presentation.locale, activity.sport)} · ${activity.name} · ${value}`,
      title: activity.name,
    })
    f.add(entry, buildIcon(f, activity.sport), f.el('span', 'tri-timeline-value', value))
    f.add(item, entry)
    f.add(list, item)
  }
  f.add(card, list)
  return card
}
