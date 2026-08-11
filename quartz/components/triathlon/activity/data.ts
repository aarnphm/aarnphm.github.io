import type { MatchedRidesBlock } from '../../../plugins/stores/matched-rides'
import type { MatchedRunsBlock } from '../../../plugins/stores/matched-runs'
import type { ActivityHealth } from '../../../plugins/stores/strava'
import type { PowerCurvePoint } from '../../../plugins/stores/strava'
import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { StravaZones } from '../../../plugins/stores/strava'
import type { SwimTrendPoint } from '../../../plugins/stores/strava'
import type { DetailCtx } from '../../../util/triathlon-card'

export type DetailPayload = {
  details: Record<string, StravaActivityDetail>
  swimTrend?: SwimTrendPoint[]
  health: Record<string, ActivityHealth>
  zones?: StravaZones
  powerCurveRef?: PowerCurvePoint[]
  powerCurveYearRef?: PowerCurvePoint[]
  powerCurveYear?: number | null
  ftp?: number | null
  goalFtp?: number | null
  vt1Hr?: number | null
  matchedRuns?: MatchedRunsBlock
  matchedRides?: MatchedRidesBlock
}

export const detailContextFromPayload = (payload?: DetailPayload | null): DetailCtx => ({
  zones: payload?.zones ?? null,
  curveRef: payload?.powerCurveRef ?? [],
  curveYearRef: payload?.powerCurveYearRef ?? [],
  curveYear: payload?.powerCurveYear ?? null,
  ftp: payload?.ftp ?? null,
  goalFtp: payload?.goalFtp ?? null,
  vt1: payload?.vt1Hr ?? null,
})
