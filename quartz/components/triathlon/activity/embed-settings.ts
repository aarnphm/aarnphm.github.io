import type { DayCardExtras } from '../../../util/triathlon-card'
import { parseExcludedActivityIds } from '../../../util/triathlon-card'
import { parseTriathlonTraceSettings } from '../../../util/triathlon-trace-settings'

export const dayExtrasFromDataset = (data: DOMStringMap): DayCardExtras => {
  const excludedActivityIds = parseExcludedActivityIds(data.triathlonFilter)
  const settings = parseTriathlonTraceSettings(data.triathlonSettings)
  return {
    location: data.triathlonLoc,
    event: data.triathlonEvent,
    sport: data.triathlonSport as DayCardExtras['sport'],
    activityId: data.triathlonActivityId,
    ...(excludedActivityIds.length > 0 ? { excludedActivityIds } : {}),
    ...(settings ? { settings } : {}),
    analytics: data.triathlonAnalytics === '1',
    expanded: data.triathlonExpanded === '1',
    embedded: data.triathlonEmbedded === '1',
    dateHref: data.triathlonDateHref,
  }
}
