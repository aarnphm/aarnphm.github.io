import {
  ACTIVITY_DEVICES,
  ACTIVITY_KINDS,
  type ActivityDevice,
  type ActivityKind,
  type RawStravaActivityDetail,
} from './strava/raw-and-kinds'
export {
  ACTIVITY_KINDS,
  SPORT_ORDER,
  isTreatment,
  normalizeActivityDevice,
  normalizeKind,
  normalizeSport,
} from './strava/raw-and-kinds'
export type {
  ActivityDevice,
  ActivityHeartRate,
  ActivityHeartRateTracePoint,
  ActivityKind,
  ActivityThermalSource,
  RawStravaActivity,
  RawStravaActivityDetail,
  RawStravaAnalysisRange,
  RawStravaRunSplit,
  Sport,
  StravaGear,
  StravaMapPoint,
  StravaRawCache,
  StravaStreams,
  StravaZones,
} from './strava/raw-and-kinds'
// haversineMeters stays on the interface: the matched-ride and matched-run fixtures build distances with it.
export { haversineMeters, round } from './strava/stream-primitives'
export { applyActivityTracking, resolveActivityHeartRate } from './strava/stream-selection'
export type {
  ActivityClimbSource,
  ActivityDistanceEffort,
  PowerCurvePoint,
} from './strava/efforts-and-curves'
export type { GarminVerification } from './strava/provider-metadata'
export type { ActivityCyclingDynamics, ActivityGearShift } from './strava/cycling-mechanics'
export type { ActivityAnalysisKind, ActivityAnalysisRange } from './strava/climbs-and-ranges'
export {
  GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD,
  calculateCyclingPerformanceCondition,
} from './strava/detail-streams'
export { calculateAnaerobicPowerEstimate, emptyHealth, emptyPayload } from './strava/activity-types'
export type {
  ActivityAnalyses,
  ActivityHealth,
  StravaActivityDetail,
  StravaDay,
  StravaPayload,
  SwimActivityInterval,
  SwimTrendPoint,
} from './strava/activity-types'
export {
  ACTIVITY_LOAD_INTENSITY_FACTOR_CAP,
  calculateActivityExerciseLoad,
  calculateActivityIntensityFactor,
  calculateActivityTrainingEffect,
  calculateExerciseLoad,
  calculateHeartRateTss,
} from './strava/training-load'
export {
  applyManualFueling,
  applyManualMoves,
  applyManualSauna,
  applyManualStrength,
} from './strava/manual-enrichment'
export { buildPayload } from './strava/build-payload'

export const ROUTE_SPORTS: readonly ActivityKind[] = ['bike', 'run', 'swim', 'walk']

export const isActivityKind = (value: unknown): value is ActivityKind =>
  ACTIVITY_KINDS.some(kind => kind === value)

export const SPORT_ICON: Record<ActivityKind, string[]> = {
  run: [
    'M15 7C16.1046 7 17 6.10457 17 5C17 3.89543 16.1046 3 15 3C13.8954 3 13 3.89543 13 5C13 6.10457 13.8954 7 15 7Z',
    'M12.6129 8.26709L9.30469 12.4023L13.4399 16.5376L11.3723 21.0863',
    'M6.41016 9.50741L9.79704 6.19922L12.613 8.26683L15.5078 11.5751H19.2295',
    'M8.89055 15.7104L7.64998 16.5375H4.3418',
  ],
  bike: [
    'M9 17.5a3.5 3.5 0 1 0-7 0a3.5 3.5 0 1 0 7 0',
    'M22 17.5a3.5 3.5 0 1 0-7 0a3.5 3.5 0 1 0 7 0',
    'M16 5a1 1 0 1 0-2 0a1 1 0 1 0 2 0',
    'M12 17.5V14l-3-3 4-3 2 3h2',
  ],
  swim: [
    'M18 6a2 2 0 1 0-4 0a2 2 0 1 0 4 0',
    'M3 13l6-2 4 2.5',
    'M2 18c1.5 1.4 3 1.4 4.5 0s3-1.4 4.5 0 3 1.4 4.5 0',
  ],
  strength: [
    'M7.4 7H4.6C4.26863 7 4 7.26863 4 7.6V16.4C4 16.7314 4.26863 17 4.6 17H7.4C7.73137 17 8 16.7314 8 16.4V7.6C8 7.26863 7.73137 7 7.4 7Z',
    'M19.4 7H16.6C16.2686 7 16 7.26863 16 7.6V16.4C16 16.7314 16.2686 17 16.6 17H19.4C19.7314 17 20 16.7314 20 16.4V7.6C20 7.26863 19.7314 7 19.4 7Z',
    'M1 14.4V9.6C1 9.26863 1.26863 9 1.6 9H3.4C3.73137 9 4 9.26863 4 9.6V14.4C4 14.7314 3.73137 15 3.4 15H1.6C1.26863 15 1 14.7314 1 14.4Z',
    'M23 14.4V9.6C23 9.26863 22.7314 9 22.4 9H20.6C20.2686 9 20 9.26863 20 9.6V14.4C20 14.7314 20.2686 15 20.6 15H22.4C22.7314 15 23 14.7314 23 14.4Z',
    'M8 12H16',
  ],
  walk: [
    'M15 4a2 2 0 1 0-4 0a2 2 0 1 0 4 0',
    'M12.6 6.4L11 13L8 16.5L7.5 21',
    'M11 13l3.5 2.5-.5 4',
    'M12 8.5l5 2.5',
    'M17 9.5L18.5 21',
    'M11.8 8H10a1.5 1.5 0 0 0-1.5 1.5v2a1.5 1.5 0 0 0 1.5 1.5h.6',
  ],
  yoga: [
    'M14 4.5a2 2 0 1 0-4 0a2 2 0 1 0 4 0',
    'M12 6.5v6.5',
    'M12 8.5l-5 5.5',
    'M12 8.5l5 5.5',
    'M4 17.5h16',
    'M4 17.5c2-3.5 5-4.5 8-4.5s6 1 8 4.5',
  ],
  treatment: [
    'M7 14.5a2 2 0 1 0-4 0a2 2 0 1 0 4 0',
    'M7 15.5h6.5l2.5-3.5 2.5 3.5',
    'M2 18.5h20',
    'M21 5.5a2 2 0 1 0-4 0a2 2 0 1 0 4 0',
    'M19 7.5v2l-4.5 4',
    'M19 9.5l1.5 3.5',
  ],
  sauna: [
    'M9 5a2 2 0 1 0-4 0a2 2 0 1 0 4 0',
    'M7 7v6h5.5v3.5',
    'M7 9.5l3.5 3.5',
    'M3 16.5h11',
    'M5 16.5V21M12 16.5V21',
    'M17 5c1 1 1 2 0 3s-1 2 0 3',
    'M20.5 7c1 1 1 2 0 3s-1 2 0 3',
  ],
}

export const hasFetchedActivityDetail = (
  detail: Partial<RawStravaActivityDetail> | undefined,
): detail is RawStravaActivityDetail =>
  detail !== undefined &&
  Object.hasOwn(detail, 'description') &&
  typeof detail.fetchedAt === 'number' &&
  Number.isFinite(detail.fetchedAt) &&
  Array.isArray(detail.laps) &&
  Array.isArray(detail.segmentEfforts) &&
  Array.isArray(detail.splitsMetric) &&
  Array.isArray(detail.splitsStandard) &&
  [...detail.splitsMetric, ...detail.splitsStandard].every(split =>
    Object.hasOwn(split, 'averageHeartrate'),
  )

export const isActivityDevice = (value: unknown): value is ActivityDevice =>
  ACTIVITY_DEVICES.some(device => device === value)

export const prefersActivityDeviceThermal = (sport: ActivityKind): boolean =>
  sport === 'run' || sport === 'walk'
