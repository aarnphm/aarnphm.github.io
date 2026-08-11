import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { DetailPayload } from '../activity/data'
import { ROUTE_SPORTS, type ActivityKind } from '../../../plugins/stores/strava'
import { buildOverview, readRouteSport, type Overview } from './model'

export interface OverviewProvider {
  current(): Overview
  clear(): void
}

export interface RouteSportFilter {
  sync(sports: ReadonlySet<ActivityKind>): void
}

export const createRouteSportFilter = (buttons: readonly HTMLButtonElement[]): RouteSportFilter => {
  const sync = (sports: ReadonlySet<ActivityKind>): void => {
    for (const button of buttons) {
      const sport = readRouteSport(button.dataset.sport)
      button.setAttribute('aria-pressed', String(sport != null && sports.has(sport)))
    }
  }
  return { sync }
}

export const createOverviewProvider = (
  presentation: () => TriathlonPresentation,
  details: () => DetailPayload | null,
  sports: () => ReadonlySet<ActivityKind>,
): OverviewProvider => {
  const cache = new Map<string, Overview>()
  return {
    current: () => {
      const enabled = sports()
      const key = ROUTE_SPORTS.filter(sport => enabled.has(sport)).join(',') || 'none'
      const cached = cache.get(key)
      if (cached) return cached
      const overview = buildOverview(presentation(), details(), enabled)
      cache.set(key, overview)
      return overview
    },
    clear: () => cache.clear(),
  }
}
