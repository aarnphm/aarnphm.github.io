import type { CriticalPowerEstimate } from '../../plugins/stores/critical-power'
import type {
  ActivityHealth,
  ActivityKind,
  PowerCurvePoint,
  StravaActivityDetail,
  StravaZones,
  SwimTrendPoint,
} from '../../plugins/stores/strava'
import type { TriathlonDailyAnalytics } from '../triathlon-day-analytics'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriathlonTraceSettings } from '../triathlon-trace-settings'

export interface TriNodeFactory<N> {
  presentation: TriathlonPresentation
  el: (tag: string, cls?: string, text?: string, attrs?: Record<string, string>) => N
  math: (cls: string, text: string) => N
  svg: (tag: string, attrs: Record<string, string | number>) => N
  add: (parent: N, ...children: N[]) => void
}

export type DayCardExtras = {
  location?: string
  event?: string
  sport?: ActivityKind
  activityId?: string
  excludedActivityIds?: readonly string[]
  settings?: TriathlonTraceSettings
  analytics?: boolean
  expanded?: boolean
  embedded?: boolean
  dateHref?: string
  dayRouteHref?: string
}

export type DayCardPayload = {
  details: Record<string, StravaActivityDetail>
  swimTrend?: SwimTrendPoint[]
  health: Record<string, ActivityHealth>
  dailyAnalytics?: TriathlonDailyAnalytics
}

export const dayCardActivitiesExpanded = (extras: DayCardExtras): boolean =>
  extras.settings?.expanded ?? Boolean(extras.sport || extras.activityId || extras.expanded)

export const parseExcludedActivityIds = (value: string | undefined): string[] => {
  const filter = value?.startsWith('filter=') ? value.slice('filter='.length) : value
  if (!filter || !/^\d+(?:&\d+)*$/.test(filter)) return []
  return [...new Set(filter.split('&'))]
}

export interface DetailCtx {
  zones: StravaZones | null
  curveRef: PowerCurvePoint[]
  curveYearRef: PowerCurvePoint[]
  runCurveRef: PowerCurvePoint[]
  runCurveYearRef: PowerCurvePoint[]
  curveYear: number | null
  criticalPower: CriticalPowerEstimate | null
  criticalPowerYear: CriticalPowerEstimate | null
  ftp: number | null
  goalFtp: number | null
  vt1: number | null
}
