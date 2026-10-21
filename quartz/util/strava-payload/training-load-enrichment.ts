import type { ActivityDistributionPoint, ActivitySummary } from '../../plugins/stores/analytics'
import {
  calculateActivityExerciseLoad,
  calculateActivityIntensityFactor,
  calculateActivityTrainingEffect,
  type StravaPayload,
} from '../../plugins/stores/strava'

export function enrichCalculatedIntensityFactors(
  payload: StravaPayload,
  activities: readonly Pick<ActivitySummary, 'id' | 'paceIntensityFactor'>[],
  ftp: number | null,
  lactateThresholdHr: number | null,
): void {
  const paceIntensityFactors = new Map(
    activities.flatMap(activity =>
      activity.paceIntensityFactor == null
        ? []
        : [[activity.id, activity.paceIntensityFactor] as const],
    ),
  )
  for (const detail of Object.values(payload.details))
    detail.calculatedIntensityFactor = calculateActivityIntensityFactor(
      detail,
      paceIntensityFactors.get(detail.id) ?? null,
      ftp,
      lactateThresholdHr,
    )
}

export function enrichRunPaceZones(
  payload: StravaPayload,
  distributions: {
    activities: readonly Pick<ActivityDistributionPoint, 'id' | 'paceZoneSeconds'>[]
    paceZoneBoundsSPerKm: readonly number[]
    tenKmRaceTimeS: number | null
  },
): void {
  const boundsSPerKm = distributions.paceZoneBoundsSPerKm
  const tenKmRaceTimeS = distributions.tenKmRaceTimeS
  const validReference =
    boundsSPerKm.length === 5 &&
    boundsSPerKm.every(value => Number.isFinite(value) && value > 0) &&
    tenKmRaceTimeS != null &&
    Number.isFinite(tenKmRaceTimeS) &&
    tenKmRaceTimeS > 0
  const zonesByActivity = new Map(
    distributions.activities.flatMap(activity =>
      activity.paceZoneSeconds == null ? [] : [[activity.id, activity.paceZoneSeconds] as const],
    ),
  )
  for (const detail of Object.values(payload.details)) {
    const zoneSeconds = zonesByActivity.get(detail.id)
    detail.runPaceZones =
      detail.sport === 'run' &&
      validReference &&
      zoneSeconds?.length === boundsSPerKm.length + 1 &&
      zoneSeconds.every(value => Number.isFinite(value) && value >= 0) &&
      zoneSeconds.some(value => value > 0)
        ? { zoneSeconds: [...zoneSeconds], boundsSPerKm: [...boundsSPerKm], tenKmRaceTimeS }
        : null
  }
}

export function enrichCalculatedExerciseLoads(payload: StravaPayload): void {
  for (const detail of Object.values(payload.details))
    detail.calculatedExerciseLoad = calculateActivityExerciseLoad(detail)
}

export function enrichCalculatedTrainingEffects(payload: StravaPayload): void {
  for (const detail of Object.values(payload.details))
    detail.calculatedTrainingEffect = calculateActivityTrainingEffect(detail)
}
