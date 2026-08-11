import type { Analytics } from '../../../plugins/stores/analytics'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { DetailPayload } from '../activity/data'
import { ROUTE_SPORTS, type ActivityKind } from '../../../plugins/stores/strava'
import { dist, dur } from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'
import { buildIcon } from '../activity/primitives'
import {
  activityCommandHints,
  activityResultItem,
  matchesActivityTokens,
  parseActivityQuery,
  sortActivitiesBy,
} from '../analytics/search'
import { el } from '../runtime/dom'
import { gpsSegments } from './model'

export interface MapSearchView {
  nodes: readonly HTMLElement[]
  sport: ActivityKind | null
}

const drawableActivityIds = (details: DetailPayload | null): Set<string> => {
  const ids = new Set<string>()
  for (const id in details?.details ?? {}) {
    const detail = details?.details[id]
    if (detail && gpsSegments(detail).length > 0) ids.add(id)
  }
  return ids
}

export const buildMapSearchView = (
  presentation: TriathlonPresentation,
  query: string,
  analytics: Analytics | null,
  details: DetailPayload | null,
  detailLoaded: boolean,
): MapSearchView => {
  const rawTokens = query.split(/\s+/)
  const { filterSport, filterDate, sortKey, tokens } = parseActivityQuery(rawTokens)
  const sport = ROUTE_SPORTS.find(candidate => candidate === filterSport) ?? null
  const lastToken = rawTokens[rawTokens.length - 1]
  const hints = activityCommandHints(lastToken, 'routes', ROUTE_SPORTS)
  const ids = drawableActivityIds(details)
  const activities = sortActivitiesBy(
    (analytics?.activities ?? []).filter(activity => {
      if (!ids.has(String(activity.id))) return false
      if (filterSport && activity.sport !== filterSport) return false
      if (filterDate && (activity.date < filterDate.start || activity.date > filterDate.end))
        return false
      return (
        tokens.length === 0 ||
        matchesActivityTokens(
          `${activity.name} ${activity.sport} ${activity.date}`.toLowerCase(),
          tokens,
        )
      )
    }),
    sortKey,
  )
  const nodes: HTMLElement[] = []
  if (hints.length > 0) {
    const group = el('div', 'tri-ana-rgroup')
    group.appendChild(el('div', 'tri-ana-rlabel', 'suggestions'))
    group.append(...hints)
    nodes.push(group)
  }
  if (activities.length > 0) {
    const group = el('div', 'tri-ana-rgroup')
    group.appendChild(el('div', 'tri-ana-rlabel', triText(presentation.locale, 'routes')))
    for (const activity of activities.slice(0, 50)) {
      const head = el('span', 'tri-ana-ritem-h')
      head.append(
        buildIcon(presentation, activity.sport),
        el('span', '', activity.name || activity.sport),
      )
      const sub = `${activity.date} · ${dist(presentation, activity.distanceKm, activity.sport)} · ${dur(activity.movingTimeS)}`
      const item = activityResultItem(head, sub)
      item.dataset.id = String(activity.id)
      group.appendChild(item)
    }
    nodes.push(group)
  }
  if (activities.length === 0 && hints.length === 0)
    nodes.push(el('div', 'tri-ana-empty', detailLoaded ? 'no routes' : 'loading'))
  return { nodes, sport }
}
