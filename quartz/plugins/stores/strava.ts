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
    'M14 4a1 1 0 1 0-2 0a1 1 0 1 0 2 0',
    'M7 21l3-4',
    'M16 21l-2-4l-3-3l1-6',
    'M6 12l2-3l4-1l3 3l3 1',
  ],
  yoga: [
    'M16.22 23H5.5a5.978 5.978 0 01-2.265-.443C1.928 22.021.878 21.03.354 19.766a4.593 4.593 0 01.161-3.881l5.991-12.98a.983.983 0 01.02-.042C7.159 1.646 8.289.737 9.634.296a6.02 6.02 0 014.132.147c1.307.536 2.357 1.527 2.881 2.791a4.592 4.592 0 01-.161 3.881L16.076 8h5.861a2 2 0 011.816 2.838l-4.809 10.42A3 3 0 0116.22 23zM13.006 2.293a4.02 4.02 0 00-2.75-.096c-.887.29-1.573.867-1.944 1.569L3.957 13.2a6.02 6.02 0 013.808.243c1.093.448 2.007 1.216 2.584 2.195l4.329-9.38.02-.043a2.594 2.594 0 00.1-2.215c-.3-.727-.93-1.353-1.792-1.707zM8.983 17.708A2.63 2.63 0 008.8 17c-.302-.727-.932-1.354-1.793-1.707a4.02 4.02 0 00-2.75-.096c-.89.291-1.579.872-1.949 1.577A2.594 2.594 0 002.201 19c.302.727.932 1.354 1.793 1.707.473.194.987.293 1.506.293h10.72a1 1 0 00.908-.58L21.938 10h-6.785l-3.516 7.619a4.099 4.099 0 01-7.388.115l-.143-.287 1.788-.894.144.287a2.099 2.099 0 002.945.868z',
  ],
  treatment: [
    'M12.5 0a3.5 3.5 0 012.51 5.936c.654 2.723.711 5.556.162 8.303l-.019.091 4.48 1.493A2 2 0 0121 17.721V24h-2v-6.28l-6.153-2.05.364-1.823a17.335 17.335 0 00-.026-6.915 3.513 3.513 0 01-1.665-.073 39.226 39.226 0 01-1.949 14.588l-.394 1.186A2 2 0 017.279 24H1v-2h6.28l.395-1.186a37.226 37.226 0 001.903-12.46L5.71 11.032 8.6 13.2l-1.2 1.6-2.89-2.167a2 2 0 01.061-3.245l5.25-3.635A3.5 3.5 0 0112.5 0zm0 2a1.5 1.5 0 100 3 1.5 1.5 0 000-3z',
  ],
  sauna: [
    'M14 14.76V5a2 2 0 0 0-4 0v9.76a4 4 0 1 0 4 0Z',
    'M12 9v7',
    'M18 6c1 1 1 2 0 3s-1 2 0 3',
    'M21 4c1 1 1 2 0 3s-1 2 0 3',
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
