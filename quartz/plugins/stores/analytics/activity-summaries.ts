import type { Act, SwimActivityMetrics } from './primitives'
import type { ActivitySummary, AnalyticsInputs } from './schema'
import {
  isTreatment,
  normalizeKind,
  round,
  type StravaActivityDetail,
  type StravaRawCache,
} from '../strava'

/** Activity lists that the activity table reads: training sessions plus walks, strength, treatments, yoga and sauna. */
interface ActivitySummaryScope {
  cache: StravaRawCache
  inputs: AnalyticsInputs
  acts: Act[]
  sinceDay: string | null
  saunaActivityIds: ReadonlySet<number>
  saunaDetails: StravaActivityDetail[]
  supplementalLoadById: ReadonlyMap<number, number>
  loadById: ReadonlyMap<number, number>
  paceIntensityFactorById: ReadonlyMap<number, number>
  swimMetrics: ReadonlyMap<number, SwimActivityMetrics>
}

export function buildActivitySummaries(scope: ActivitySummaryScope): ActivitySummary[] {
  const {
    cache,
    inputs,
    acts,
    sinceDay,
    saunaActivityIds,
    saunaDetails,
    supplementalLoadById,
    loadById,
    paceIntensityFactorById,
    swimMetrics,
  } = scope
  const walkSummaries: ActivitySummary[] = Object.values(cache.activities)
    .filter(
      a =>
        !saunaActivityIds.has(a.id) &&
        normalizeKind(a.sportType) === 'walk' &&
        !isTreatment(a.sportType, a.name),
    )
    .filter(a => !sinceDay || a.startDateLocal.slice(0, 10) >= sinceDay)
    .map(a => ({
      id: a.id,
      date: a.startDateLocal.slice(0, 10),
      sport: 'walk' as const,
      name: a.name ?? '',
      distanceKm: round(a.distance / 1000, 1),
      movingTimeS: a.movingTime,
      load: 0,
      paceIntensityFactor: null,
      effort: a.sufferScore ?? null,
      cadence: a.averageCadence ?? null,
      strokes: null,
      windKph: inputs.weather?.activities[String(a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(a.id)]?.windGustKph ?? null,
    }))
  const strengthSummaries: ActivitySummary[] = Object.values(cache.activities)
    .filter(
      a =>
        !saunaActivityIds.has(a.id) &&
        normalizeKind(a.sportType) === 'strength' &&
        !isTreatment(a.sportType, a.name),
    )
    .filter(a => !sinceDay || a.startDateLocal.slice(0, 10) >= sinceDay)
    .map(a => ({
      id: a.id,
      date: a.startDateLocal.slice(0, 10),
      sport: 'strength' as const,
      name: a.name ?? '',
      distanceKm: 0,
      movingTimeS: a.movingTime,
      load: supplementalLoadById.get(a.id) ?? 0,
      paceIntensityFactor: null,
      effort: a.sufferScore ?? null,
      cadence: null,
      strokes: null,
      windKph: inputs.weather?.activities[String(a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(a.id)]?.windGustKph ?? null,
    }))
  const treatmentSummaries: ActivitySummary[] = Object.values(cache.activities)
    .filter(a => !saunaActivityIds.has(a.id) && isTreatment(a.sportType, a.name))
    .filter(a => !sinceDay || a.startDateLocal.slice(0, 10) >= sinceDay)
    .map(a => ({
      id: a.id,
      date: a.startDateLocal.slice(0, 10),
      sport: 'treatment' as const,
      name: a.name ?? '',
      distanceKm: 0,
      movingTimeS: a.movingTime,
      load: 0,
      paceIntensityFactor: null,
      effort: a.sufferScore ?? null,
      cadence: null,
      strokes: null,
      windKph: inputs.weather?.activities[String(a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(a.id)]?.windGustKph ?? null,
    }))
  const yogaSummaries: ActivitySummary[] = Object.values(cache.activities)
    .filter(
      a =>
        !saunaActivityIds.has(a.id) &&
        normalizeKind(a.sportType) === 'yoga' &&
        !isTreatment(a.sportType, a.name),
    )
    .filter(a => !sinceDay || a.startDateLocal.slice(0, 10) >= sinceDay)
    .map(a => ({
      id: a.id,
      date: a.startDateLocal.slice(0, 10),
      sport: 'yoga' as const,
      name: a.name ?? '',
      distanceKm: 0,
      movingTimeS: a.movingTime,
      load: supplementalLoadById.get(a.id) ?? 0,
      paceIntensityFactor: null,
      effort: a.sufferScore ?? null,
      cadence: null,
      strokes: null,
      windKph: inputs.weather?.activities[String(a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(a.id)]?.windGustKph ?? null,
    }))
  const saunaSummaries: ActivitySummary[] = saunaDetails.map(detail => ({
    id: detail.id,
    date: detail.date,
    sport: 'sauna',
    name: detail.name,
    distanceKm: 0,
    movingTimeS: detail.movingTimeS,
    load: 0,
    paceIntensityFactor: null,
    effort: null,
    cadence: null,
    strokes: null,
    windKph: detail.windKph,
    windDir: detail.windDir,
    windGustKph: detail.windGustKph,
  }))
  const activities: ActivitySummary[] = acts
    .map<ActivitySummary>(act => ({
      id: act.a.id,
      date: act.day,
      sport: act.sport,
      virtual:
        inputs.activityDetails?.[String(act.a.id)]?.virtual ??
        act.a.sportType.startsWith('Virtual'),
      commute: act.a.commute === true,
      treadmill:
        act.sport === 'run' && (act.a.trainer === true || /\btreadmill\b/i.test(act.a.name)),
      name: act.a.name ?? '',
      distanceKm: act.distanceKm,
      movingTimeS: act.a.movingTime,
      load: round(loadById.get(act.a.id) ?? 0, 1),
      paceIntensityFactor: paceIntensityFactorById.get(act.a.id) ?? null,
      effort: act.a.sufferScore ?? null,
      cadence: act.a.averageCadence ?? null,
      strokes: swimMetrics.get(act.a.id)?.strokes ?? null,
      windKph: inputs.weather?.activities[String(act.a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(act.a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(act.a.id)]?.windGustKph ?? null,
    }))
    .concat(walkSummaries, strengthSummaries, treatmentSummaries, yogaSummaries, saunaSummaries)
    .sort((p, q) => q.date.localeCompare(p.date))
  return activities
}
