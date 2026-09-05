import type { RunPaceZoneDistribution } from '../../util/run-pace-zones'
import type { SwimLocation, SwimStroke } from './apple'
import type {
  GarminActivity,
  GarminActivityMatch,
  GarminCache,
  GarminClimbSegment,
  GarminCyclingDynamics,
  GarminFueling,
  GarminGearShift,
  GarminRiderPosition,
  GarminRunWalkData,
  GarminRunningDynamicsSummary,
  GarminStreams,
} from './garmin'
import type { OuraCache, OuraDaily, OuraHeartRateSample } from './oura'
import type {
  ActivityTrackingEntry,
  ManualFuelingEntry,
  ManualSaunaEntry,
  ManualStrengthEntry,
  StrengthExercise,
} from './tracking'
import type {
  WahooActivityMatch,
  WahooCache,
  WahooCyclingDynamics,
  WahooStreams,
  WahooSummitSegment,
} from './wahoo'
import type {
  WeatherActivity,
  WeatherCache,
  WeatherRelativeHumidityProvenance,
  WeatherTemperatureSample,
} from './weather'
import {
  buildActivityEnvironment,
  type GardenApparentWindEstimate,
  type GardenEnvironmentEstimate,
  type GardenUvScore,
} from '../../util/activity-environment'
import {
  parseActivityProviderReports,
  type NativeActivityReports,
} from '../../util/activity-provider-reports'
import { applyGardenUvCalibration } from '../../util/activity-uv-score'
import {
  estimateWahooCyclingStamina,
  type CyclingStaminaEstimate,
} from '../../util/cycling-stamina'
import { localDateTimeUtcMs, localIsoDay } from '../../util/local-date'
import { latestProviderSync } from '../../util/provider-sync'
import { rawMapRouteSegments, type MapRoutePoint } from '../../util/triathlon-map-route'
import {
  CRITICAL_POWER_DURATIONS_S,
  fitCriticalPower,
  type CriticalPowerAnchor,
  type CriticalPowerEstimate,
} from './critical-power'
import {
  emptyGarminFueling,
  hasGarminFueling,
  matchGarminActivity,
  matchGarminHeartRateActivity,
  matchGarminTrainingEffectActivity,
} from './garmin'
import { matchWahooActivity } from './wahoo'

export type Sport = 'swim' | 'bike' | 'run'

export type ActivityKind = Sport | 'strength' | 'walk' | 'yoga' | 'treatment' | 'sauna'

export const ACTIVITY_KINDS: readonly ActivityKind[] = [
  'swim',
  'bike',
  'run',
  'strength',
  'walk',
  'yoga',
  'treatment',
  'sauna',
]
export const SPORT_ORDER: readonly Sport[] = ['swim', 'bike', 'run']
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

const DAY_MS = 86_400_000
const WINDOW_DAYS = 364
const ROUTE_POINTS = 140

export interface StravaAuth {
  refreshToken: string
  obtainedAt: number
}

export interface RawStravaActivity {
  id: number
  name: string
  sportType: string
  distance: number
  movingTime: number
  elapsedTime: number
  totalElevationGain: number
  startDate: string
  startDateLocal: string
  averageSpeed: number
  averageHeartrate?: number
  maxHeartrate?: number
  averageWatts?: number
  weightedAverageWatts?: number
  maxWatts?: number
  kilojoules?: number
  deviceWatts?: boolean
  averageCadence?: number
  sufferScore?: number
  averageTemp?: number
  calories?: number
  deviceName?: string
}

export interface RawStravaAnalysisRange {
  id: string
  name: string
  elapsedTime: number
  movingTime: number
  startDate: string | null
  distance: number
  startIndex: number | null
  endIndex: number | null
  totalElevationGain: number | null
  averageSpeed: number | null
  averageHeartrate: number | null
  averageWatts: number | null
  averageCadence: number | null
}

export interface RawStravaRunSplit {
  split: number
  distance: number
  elapsedTime: number
  movingTime: number
  averageSpeed: number
  elevationDifference: number | null
  paceZone: number | null
}

export interface RawStravaActivityDetail {
  description?: string | null
  fetchedAt?: number
  calories: number | null
  laps: RawStravaAnalysisRange[]
  segmentEfforts: RawStravaAnalysisRange[]
  splitsMetric: RawStravaRunSplit[]
  splitsStandard: RawStravaRunSplit[]
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
  Array.isArray(detail.splitsStandard)

export interface StravaStreams {
  time?: number[]
  latlng: [number, number][]
  altitude: number[]
  distance: number[]
  watts?: number[]
  heartrate?: number[]
  cadence?: number[]
}

export interface StravaZones {
  hr: number[]
  power: number[]
  ftp: number | null
}

export interface StravaRawCache {
  version?: number
  athleteId: number
  auth: StravaAuth
  lastSync: number
  lastActivityStart: number
  activities: Record<string, RawStravaActivity>
  activityDetails?: Record<string, RawStravaActivityDetail>
  streams?: Record<string, StravaStreams>
  geo?: Record<string, string>
  zones?: StravaZones
}

export type StravaMapPoint = MapRoutePoint

export interface PowerCurvePoint {
  s: number
  w: number
  activityId?: number
  activityDate?: string
}

export interface CyclingDistanceEffort {
  label: string
  targetDistanceM: number
  elapsedTimeS: number
  averageSpeedKph: number
  averageHeartRate: number | null
  elevationDeltaM: number
}

export interface CyclingPowerEffort {
  durationS: number
  averageWatts: number
  wattsPerKg: number | null
  averageHeartRate: number | null
  elevationDeltaM: number
}

export type ActivityClimbSource =
  | 'garmin-climbpro'
  | 'wahoo-summit-segment'
  | 'wahoo-summit-freeride'

export interface CyclingClimbEffort {
  source: ActivityClimbSource
  name: string
  durationS: number
  distanceM: number
  elevationGainM: number
  averageGradePct: number
  averageSpeedKph: number
  averageHeartRate: number | null
  averageWatts: number | null
  wattsPerKg: number | null
  vamMPerHour: number
}

export interface CyclingBestEfforts {
  weightKg: number | null
  weightDate: string | null
  distance: CyclingDistanceEffort[]
  power: CyclingPowerEffort[]
  climbs: CyclingClimbEffort[]
}

export interface StravaSportTotals {
  sport: Sport
  count: number
  distanceKm: number
  movingTimeS: number
  elevationM: number
}

export interface StravaDayItem {
  id: number
  sport: ActivityKind
  distanceKm: number
  durationS: number
}

export interface StravaDay {
  date: string
  durationS: number
  items: StravaDayItem[]
  dominant: ActivityKind | null
}

export type ActivityThermalSource = 'core-app' | 'core-fit'

export interface StravaRoutePoint {
  x: number
  y: number
  d: number
  alt: number
  w: number
  hr: number
  cad: number
  rightPowerPct?: number | null
  stamina: number | null
  potentialStamina: number | null
  resp: number | null
  muscleOxygenPct?: number | null
  tempC: number | null
  heatStrainIndex: number | null
  heatStrainSource: ActivityThermalSource | null
  coreTemperatureC: number | null
  coreTemperatureSource: ActivityThermalSource | null
  skinTemperatureC: number | null
  skinTemperatureSource: ActivityThermalSource | null
  lat: number
  lng: number
  elapsedS: number
  speedKph: number
  performanceCondition?: number | null
  strideLengthM?: number | null
  verticalRatioPct?: number | null
  groundContactTimeMs?: number | null
  groundContactBalanceLeftPct?: number | null
  verticalOscillationCm?: number | null
  stepSpeedLossMps?: number | null
  stepSpeedLossPct?: number | null
  impactLoadFactor?: number | null
}

export interface ActivityGearShift {
  elapsedS: number
  distanceKm: number
  frontGearNum: number
  frontTeeth: number
  rearGearNum: number
  rearTeeth: number
}

export interface ActivityRiderPositionChange {
  elapsedS: number
  distanceKm: number
  position: GarminRiderPosition
}

export interface ActivityCyclingDynamics {
  elapsedS: number[]
  distanceKm: number[]
  leftPedalSmoothness: (number | null)[]
  rightPedalSmoothness: (number | null)[]
  leftTorqueEffectiveness: (number | null)[]
  rightTorqueEffectiveness: (number | null)[]
  leftPowerPhaseStart: (number | null)[]
  leftPowerPhaseEnd: (number | null)[]
  rightPowerPhaseStart: (number | null)[]
  rightPowerPhaseEnd: (number | null)[]
  positionChanges: ActivityRiderPositionChange[]
  seatedTimeS: number | null
  standingTimeS: number | null
}

export type ActivityAnalysisKind = 'lap' | 'segment' | 'climb'

interface ActivityAnalysisRangeBase {
  id: string
  label: string
  startElapsedS: number
  endElapsedS: number
  startDistanceKm: number
  endDistanceKm: number
  durationS: number
  movingTimeS?: number
  distanceKm: number
  elevationGainM: number | null
  averageSpeedKph: number | null
  averageHeartRate: number | null
  averageWatts: number | null
  averageCadence: number | null
}

export type ActivityAnalysisRange = ActivityAnalysisRangeBase &
  ({ kind: 'lap' | 'segment' } | { kind: 'climb'; source: ActivityClimbSource })

export interface ActivityRunSplit {
  split: number
  distanceKm: number
  elapsedTimeS: number
  movingTimeS: number
  averageSpeedKph: number
  elevationDifferenceM: number | null
  paceZone: number | null
}

export type ActivityHealth = Omit<OuraDaily, 'date'> & {
  windKph?: number | null
  windDir?: string | null
  windDirDeg?: number | null
  windGustKph?: number | null
}

export function emptyHealth(): ActivityHealth {
  return {
    readiness: null,
    sleepScore: null,
    hrv: null,
    rhr: null,
    sleepDurationS: null,
    tempDeviationC: null,
    totalCalories: null,
    activeCalories: null,
  }
}

export interface GarminVerification {
  activityId: string
  name: string | null
  sourceDevice: string | null
  startDate: string
  startDiffS: number
  distanceM: number | null
  distanceDeltaM: number | null
  distanceDeltaPct: number | null
  movingTimeS: number | null
  movingTimeDeltaS: number | null
  elapsedTimeS: number | null
  elapsedTimeDeltaS: number | null
  totalCalories: number | null
  caloriesDelta: number | null
  avgHeartRate: number | null
  avgHeartRateDelta: number | null
  avgPower: number | null
  avgPowerDelta: number | null
  avgCadence: number | null
  normalizedPower: number | null
  maxPower: number | null
  totalWorkKJ: number | null
  totalWorkDeltaKJ: number | null
  trainingStressScore: number | null
  intensityFactor: number | null
  trainingEffectActivityId: string | null
  aerobicTrainingEffect: number | null
  anaerobicTrainingEffect: number | null
  exerciseLoad: number | null
  trainingEffectLabel: string | null
  aerobicTrainingEffectMessage: string | null
  anaerobicTrainingEffectMessage: string | null
  runningDynamics?: GarminRunningDynamicsSummary | null
}

export type ActivityComputer = 'garmin' | 'wahoo'

export type ActivityDevice = 'apple-watch-ultra-3' | 'garmin-forerunner-970'

export const ACTIVITY_DEVICES: readonly ActivityDevice[] = [
  'apple-watch-ultra-3',
  'garmin-forerunner-970',
]

export const isActivityDevice = (value: unknown): value is ActivityDevice =>
  ACTIVITY_DEVICES.some(device => device === value)

export type ActivityStaminaTrace =
  | { source: 'garmin'; method: 'garmin-native'; ftpWatts: null; maxHeartRateBpm: null }
  | {
      source: 'garden-estimate'
      method: CyclingStaminaEstimate['method']
      ftpWatts: number
      maxHeartRateBpm: number
    }

export const GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD = 'garden-cycling-performance-condition-v1'

export interface CyclingPerformanceConditionSample {
  elapsedS: number
  value: number
}

export interface CyclingPerformanceConditionEstimate {
  method: typeof GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD
  ftpWatts: number
  lactateThresholdHeartRateBpm: number
  restingHeartRateBpm: number
  windowSeconds: number
  samples: CyclingPerformanceConditionSample[]
}

export type ActivityPerformanceConditionTrace =
  | { source: 'garmin'; method: 'garmin-native' }
  | ({ source: 'garden-estimate' } & Omit<CyclingPerformanceConditionEstimate, 'samples'>)

export interface ActivityHeartRate {
  avgHr: number | null
  maxHr: number | null
  stream: number[]
}

export interface ActivityHeartRateTracePoint {
  distanceKm: number
  elapsedS: number
  heartRate: number | null
  heatStrainIndex: number | null
  heatStrainSource: ActivityThermalSource | null
  coreTemperatureC: number | null
  coreTemperatureSource: ActivityThermalSource | null
  skinTemperatureC: number | null
  skinTemperatureSource: ActivityThermalSource | null
}

export interface ActivityFueling extends GarminFueling {
  sodiumLossMg: number | null
  source: 'garmin' | 'garmin+wahoo' | 'manual' | 'wahoo'
}

export interface ActivityStrength {
  volumeKg: number | null
  totalSets: number | null
  totalReps: number | null
  exercises: StrengthExercise[]
  source: 'manual' | 'strava'
}

export interface ActivitySauna {
  time: string
  temperatureC: number
  humidityPct: number
  cooldown: ManualSaunaEntry['cooldown']
  heatTrainingLoad: number | null
  heartRateSource: 'oura' | null
  source: 'manual'
}

export interface ActivityPowerWithoutZeros {
  avgWatts: number | null
  powerZones: number[] | null
  powerHist: number[] | null
}

export interface CalculatedIntensityFactor {
  value: number
  source: 'pace' | 'power' | 'heart-rate'
}

export interface CalculatedExerciseLoad {
  value: number
  source: CalculatedIntensityFactor['source'] | 'garmin'
}

export interface AnaerobicPowerEstimate {
  effect: number
  effortCount: number
  stimulus: number
  criticalPowerWatts: number
  wPrimeKilojoules: number
}

export type CalculatedTrainingEffectAnaerobicEvidence =
  | { source: 'heart-rate'; seconds: number }
  | { source: 'pace'; weightedSeconds: number }
  | ({ source: 'power' } & AnaerobicPowerEstimate)

export interface CalculatedTrainingEffect {
  aerobic: number
  anaerobic: number
  evidence: {
    aerobic: { source: 'relative-effort' | 'exercise-load'; load: number }
    anaerobic: CalculatedTrainingEffectAnaerobicEvidence
  }
}

export interface ActivityAnalyses {
  native: NativeActivityReports
  derived: {
    environment: GardenEnvironmentEstimate | null
    uvScore: GardenUvScore | null
    apparentWind: GardenApparentWindEstimate | null
  }
}

export interface StravaActivityDetail {
  id: number
  virtual?: boolean
  distanceSource?: 'garmin'
  sport: ActivityKind
  name: string
  date: string
  start: string
  distanceKm: number
  movingTimeS: number
  elapsedTimeS: number
  maxSpeedKph: number | null
  elevationM: number
  avgHr: number | null
  maxHr: number | null
  avgWatts: number | null
  npWatts: number | null
  maxWatts: number | null
  kilojoules: number | null
  deviceWatts: boolean
  avgCadence: number | null
  sufferScore: number | null
  calories: number | null
  deviceTemperatureC: number | null
  ambientTemperatureC: number | null
  windKph: number | null
  windDir: string | null
  windDirDeg: number | null
  windGustKph: number | null
  averageRelativeHumidityPct: number | null
  relativeHumidityProvenance: WeatherRelativeHumidityProvenance | null
  location: string | null
  fueling: ActivityFueling | null
  strength: ActivityStrength | null
  sauna: ActivitySauna | null
  garmin: GarminVerification | null
  computer: ActivityComputer | null
  device: ActivityDevice | null
  staminaTrace: ActivityStaminaTrace | null
  performanceConditionTrace: ActivityPerformanceConditionTrace | null
  calculatedIntensityFactor: CalculatedIntensityFactor | null
  calculatedExerciseLoad: CalculatedExerciseLoad | null
  anaerobicPowerEstimate: AnaerobicPowerEstimate | null
  calculatedTrainingEffect: CalculatedTrainingEffect | null
  gearShifts: ActivityGearShift[]
  cyclingDynamics: ActivityCyclingDynamics | null
  runWalk: GarminRunWalkData | null
  route: StravaRoutePoint[]
  heartRateTrace: ActivityHeartRateTracePoint[]
  mapRoute: StravaMapPoint[][]
  analysisRanges: ActivityAnalysisRange[]
  runSplitsMetric: ActivityRunSplit[]
  runSplitsStandard: ActivityRunSplit[]
  runPaceZones: RunPaceZoneDistribution | null
  minAlt: number
  maxAlt: number
  descentM: number
  hrZones: number[] | null
  powerZones: number[] | null
  powerHist: number[] | null
  powerWithoutZeros: ActivityPowerWithoutZeros | null
  powerCurve: PowerCurvePoint[] | null
  activityCriticalPower: CriticalPowerEstimate | null
  bestEfforts: CyclingBestEfforts | null
  strokes?: Record<string, number> | null
  strokeCount: number | null
  strokeRateSpm: number | null
  swimPaceSPer100m: number | null
  swimPaceSource: SwimPaceSource | null
  swimDurationS: number | null
  swimIntervals: SwimActivityInterval[]
  swimLocation: SwimLocation | null
  waterTemperatureC: number | null
  analyses: ActivityAnalyses
}

export type SwimPaceSource = 'stroke' | 'active' | 'moving'

export interface SwimActivityInterval {
  startElapsedS: number
  endElapsedS: number
  distanceM: number
  durationS: number
  cumulativeDistanceM: number
  paceSPer100m: number | null
  strokeCount: number | null
  strokeTimeS: number | null
  strokeRateSpm: number | null
  stroke: SwimStroke | null
}

export interface SwimTrendPoint {
  id: number
  date: string
  start: string
  paceSPer100m: number | null
  paceSource: SwimPaceSource | null
  strokeRateSpm: number | null
}

export interface StravaPayload {
  generatedAt: number
  athleteId: number
  totalKm: number
  totalTimeS: number
  totalCount: number
  totals: StravaSportTotals[]
  strengthTotal: { count: number; movingTimeS: number }
  days: StravaDay[]
  details: Record<string, StravaActivityDetail>
  swimTrend: SwimTrendPoint[]
  health: Record<string, ActivityHealth>
  zones: StravaZones
  powerCurveRef: PowerCurvePoint[]
  powerCurveYearRef: PowerCurvePoint[]
  powerCurveYear: number | null
  criticalPower: CriticalPowerEstimate | null
  criticalPowerYear: CriticalPowerEstimate | null
}

export function normalizeSport(sportType: string): Sport | null {
  switch (sportType) {
    case 'Run':
    case 'TrailRun':
    case 'VirtualRun':
      return 'run'
    case 'Ride':
    case 'VirtualRide':
    case 'MountainBikeRide':
    case 'GravelRide':
    case 'EBikeRide':
      return 'bike'
    case 'Swim':
    case 'OpenWaterSwim':
      return 'swim'
    default:
      return null
  }
}

export function normalizeKind(sportType: string): ActivityKind | null {
  switch (sportType) {
    case 'WeightTraining':
    case 'Workout':
    case 'Crossfit':
      return 'strength'
    case 'Walk':
    case 'Hike':
      return 'walk'
    case 'Yoga':
    case 'Pilates':
      return 'yoga'
    case 'PhysicalTherapy':
    case 'Physiotherapy':
      return 'treatment'
    default:
      return normalizeSport(sportType)
  }
}

const TREATMENT_TYPES = new Set(['PhysicalTherapy', 'Physiotherapy'])
const TREATMENT_NAME_RE = /\b(physio|physiotherapy|physical[ -]?therapy|treatment|rehab|massage)\b/i

export function isTreatment(sportType: string, name: string | null | undefined): boolean {
  return TREATMENT_TYPES.has(sportType) || TREATMENT_NAME_RE.test(name ?? '')
}

export function round(value: number, dp: number): number {
  const f = 10 ** dp
  return Math.round(value * f) / f
}

export const calculateActivityIntensityFactor = (
  activity: Pick<StravaActivityDetail, 'sport' | 'avgHr' | 'npWatts' | 'deviceWatts' | 'garmin'>,
  paceIntensityFactor: number | null,
  ftp: number | null,
  lactateThresholdHr: number | null,
): CalculatedIntensityFactor | null => {
  if (
    activity.garmin?.intensityFactor != null ||
    activity.sport === 'treatment' ||
    activity.sport === 'sauna'
  )
    return null
  if (
    activity.sport === 'bike' &&
    activity.deviceWatts &&
    activity.npWatts != null &&
    activity.npWatts > 0 &&
    ftp != null &&
    ftp > 0
  )
    return { value: round(activity.npWatts / ftp, 3), source: 'power' }
  if (
    (activity.sport === 'run' || activity.sport === 'swim') &&
    paceIntensityFactor != null &&
    paceIntensityFactor > 0
  )
    return { value: round(paceIntensityFactor, 3), source: 'pace' }
  if (
    activity.avgHr != null &&
    activity.avgHr > 0 &&
    lactateThresholdHr != null &&
    lactateThresholdHr > 0
  )
    return { value: round(activity.avgHr / lactateThresholdHr, 3), source: 'heart-rate' }
  return null
}

export const ACTIVITY_LOAD_INTENSITY_FACTOR_CAP = 1.15

export const calculateHeartRateTss = (
  averageHeartRate: number,
  movingTimeS: number,
  restingHeartRate: number,
  thresholdHeartRate: number,
  maximumHeartRate: number,
  sex: 'M' | 'F',
): number | null => {
  if (
    !Number.isFinite(averageHeartRate) ||
    !Number.isFinite(movingTimeS) ||
    !Number.isFinite(restingHeartRate) ||
    !Number.isFinite(thresholdHeartRate) ||
    !Number.isFinite(maximumHeartRate) ||
    movingTimeS <= 0 ||
    restingHeartRate <= 0 ||
    averageHeartRate <= restingHeartRate ||
    thresholdHeartRate <= restingHeartRate ||
    maximumHeartRate <= thresholdHeartRate ||
    averageHeartRate > maximumHeartRate
  )
    return null
  const reserve = maximumHeartRate - restingHeartRate
  const averageReserve = (averageHeartRate - restingHeartRate) / reserve
  const thresholdReserve = (thresholdHeartRate - restingHeartRate) / reserve
  const exponent = sex === 'M' ? 1.92 : 1.67
  const impulse = (heartRateReserve: number): number =>
    heartRateReserve * Math.exp(exponent * heartRateReserve)
  const load = 100 * (movingTimeS / 3600) * (impulse(averageReserve) / impulse(thresholdReserve))
  return round(load + Number.EPSILON * load, 1)
}

export const calculateExerciseLoad = (
  intensityFactor: number,
  movingTimeS: number,
): number | null => {
  if (
    !Number.isFinite(intensityFactor) ||
    intensityFactor <= 0 ||
    !Number.isFinite(movingTimeS) ||
    movingTimeS <= 0
  )
    return null
  const intensity = Math.min(intensityFactor, ACTIVITY_LOAD_INTENSITY_FACTOR_CAP)
  const load = intensity * intensity * (movingTimeS / 3600) * 100
  return round(load + Number.EPSILON * load, 1)
}

export const calculateActivityExerciseLoad = (
  activity: Pick<StravaActivityDetail, 'movingTimeS' | 'garmin' | 'calculatedIntensityFactor'>,
): CalculatedExerciseLoad | null => {
  if (activity.garmin?.exerciseLoad != null) return null
  const source =
    activity.garmin?.intensityFactor != null
      ? 'garmin'
      : (activity.calculatedIntensityFactor?.source ?? null)
  const intensityFactor =
    activity.garmin?.intensityFactor ?? activity.calculatedIntensityFactor?.value
  if (source == null || intensityFactor == null) return null
  const value = calculateExerciseLoad(intensityFactor, activity.movingTimeS)
  return value == null ? null : { value, source }
}

type TrainingEffectScale = readonly [
  readonly [number, number],
  ...ReadonlyArray<readonly [number, number]>,
]

const RELATIVE_EFFORT_TRAINING_EFFECT_SCALE: TrainingEffectScale = [
  [0, 0],
  [4, 1],
  [11, 2],
  [30, 3],
  [75, 4],
  [150, 5],
]

const EXERCISE_LOAD_TRAINING_EFFECT_SCALE: TrainingEffectScale = [
  [0, 0],
  [8, 1],
  [20, 2],
  [40, 3],
  [75, 4],
  [130, 5],
]

const HIGH_HEART_RATE_TRAINING_EFFECT_SCALE: TrainingEffectScale = [
  [0, 0],
  [30, 0.1],
  [60, 0.5],
  [120, 1],
  [240, 2],
  [480, 2.5],
]

const HIGH_INTENSITY_INTERVAL_TRAINING_EFFECT_SCALE: TrainingEffectScale = [
  [0, 0],
  [10, 0.5],
  [30, 1],
  [60, 2],
  [120, 3],
  [240, 4],
  [480, 5],
]

const ANAEROBIC_POWER_WINDOW_S = 5
const ANAEROBIC_POWER_INTERVAL_MIN_S = 10
const ANAEROBIC_POWER_INTERVAL_MAX_S = 120
const ANAEROBIC_W_PRIME_MIN_FRACTION = 0.1
const ANAEROBIC_SESSION_DURATION_EXPONENT = 0.6
const ANAEROBIC_TRAINING_EFFECT_RATE = 0.24

const trainingEffectFromScale = (value: number, scale: TrainingEffectScale): number => {
  if (!Number.isFinite(value) || value <= scale[0][0]) return scale[0][1]
  for (let index = 1; index < scale.length; index++) {
    const [upperValue, upperEffect] = scale[index]
    if (value > upperValue) continue
    const [lowerValue, lowerEffect] = scale[index - 1]
    const fraction = (value - lowerValue) / (upperValue - lowerValue)
    return lowerEffect + fraction * (upperEffect - lowerEffect)
  }
  return scale[scale.length - 1][1]
}

export const calculateAnaerobicPowerEstimate = (
  watts: ArrayLike<number>,
  movingTimeS: number,
  powerModel: Pick<CriticalPowerEstimate, 'criticalPowerWatts' | 'wPrimeJoules'> | null,
): AnaerobicPowerEstimate | null => {
  const criticalPowerWatts = powerModel?.criticalPowerWatts
  const wPrimeJoules = powerModel?.wPrimeJoules
  if (
    criticalPowerWatts == null ||
    !Number.isFinite(criticalPowerWatts) ||
    criticalPowerWatts <= 0 ||
    wPrimeJoules == null ||
    !Number.isFinite(wPrimeJoules) ||
    wPrimeJoules <= 0 ||
    !Number.isFinite(movingTimeS) ||
    movingTimeS <= 0 ||
    watts.length < ANAEROBIC_POWER_INTERVAL_MIN_S
  )
    return null

  const smoothed = new Float64Array(watts.length)
  let rollingWatts = 0
  for (let index = 0; index < watts.length; index++) {
    const value = watts[index]
    rollingWatts += Number.isFinite(value) ? Math.max(0, value) : 0
    if (index >= ANAEROBIC_POWER_WINDOW_S) {
      const expired = watts[index - ANAEROBIC_POWER_WINDOW_S]
      rollingWatts -= Number.isFinite(expired) ? Math.max(0, expired) : 0
    }
    smoothed[index] = rollingWatts / Math.min(index + 1, ANAEROBIC_POWER_WINDOW_S)
  }

  let rawStimulus = 0
  let effortCount = 0
  let intervalStart = -1
  let intervalWorkJoules = 0
  for (let index = 0; index <= smoothed.length; index++) {
    const value = index < smoothed.length ? smoothed[index] : 0
    if (value > criticalPowerWatts) {
      if (intervalStart < 0) intervalStart = index
      intervalWorkJoules += value - criticalPowerWatts
      continue
    }
    if (intervalStart < 0) continue
    const durationS = index - intervalStart
    if (
      durationS >= ANAEROBIC_POWER_INTERVAL_MIN_S &&
      durationS <= ANAEROBIC_POWER_INTERVAL_MAX_S
    ) {
      const depletionFraction = intervalWorkJoules / wPrimeJoules
      if (depletionFraction > ANAEROBIC_W_PRIME_MIN_FRACTION) {
        rawStimulus +=
          (depletionFraction - ANAEROBIC_W_PRIME_MIN_FRACTION) /
          (1 - ANAEROBIC_W_PRIME_MIN_FRACTION)
        effortCount++
      }
    }
    intervalStart = -1
    intervalWorkJoules = 0
  }
  const stimulus =
    rawStimulus / Math.pow(Math.max(1, movingTimeS / 3600), ANAEROBIC_SESSION_DURATION_EXPONENT)
  const effect = 5 * (1 - Math.exp(-ANAEROBIC_TRAINING_EFFECT_RATE * stimulus))
  return {
    effect: round(effect, 2),
    effortCount,
    stimulus: round(stimulus, 3),
    criticalPowerWatts: round(criticalPowerWatts, 1),
    wPrimeKilojoules: round(wPrimeJoules / 1_000, 1),
  }
}

interface TrainingEffectPaceEffort {
  durationS: number
  distanceM: number
}

const swimTrainingEffectPaceEfforts = (
  intervals: readonly SwimActivityInterval[],
): TrainingEffectPaceEffort[] => {
  const efforts: TrainingEffectPaceEffort[] = []
  let current: (TrainingEffectPaceEffort & { endElapsedS: number }) | null = null
  for (const interval of intervals) {
    if (!current || interval.startElapsedS - current.endElapsedS > 12) {
      if (current) efforts.push(current)
      current = {
        durationS: interval.durationS,
        distanceM: interval.distanceM,
        endElapsedS: interval.endElapsedS,
      }
      continue
    }
    current.durationS += interval.durationS
    current.distanceM += interval.distanceM
    current.endElapsedS = interval.endElapsedS
  }
  if (current) efforts.push(current)
  return efforts
}

const trainingEffectPaceEfforts = (
  activity: Pick<StravaActivityDetail, 'sport' | 'analysisRanges' | 'swimIntervals'>,
): TrainingEffectPaceEffort[] => {
  if (activity.sport === 'run')
    return activity.analysisRanges.flatMap(range =>
      range.kind !== 'lap' ||
      range.durationS < 10 ||
      range.durationS > 180 ||
      range.averageSpeedKph == null ||
      range.averageSpeedKph <= 0
        ? []
        : [
            {
              durationS: range.durationS,
              distanceM: (range.averageSpeedKph / 3.6) * range.durationS,
            },
          ],
    )
  return activity.sport === 'swim' ? swimTrainingEffectPaceEfforts(activity.swimIntervals) : []
}

const calculatedAnaerobicTrainingEffect = (
  activity: Pick<
    StravaActivityDetail,
    | 'sport'
    | 'distanceKm'
    | 'movingTimeS'
    | 'calculatedIntensityFactor'
    | 'anaerobicPowerEstimate'
    | 'hrZones'
    | 'analysisRanges'
    | 'swimPaceSPer100m'
    | 'swimIntervals'
  >,
): { score: number; evidence: CalculatedTrainingEffectAnaerobicEvidence } => {
  const highHeartRateS = (activity.hrZones ?? []).slice(-2).reduce((sum, value) => sum + value, 0)
  const heartRateEffect = trainingEffectFromScale(
    highHeartRateS,
    HIGH_HEART_RATE_TRAINING_EFFECT_SCALE,
  )
  let best: { score: number; evidence: CalculatedTrainingEffectAnaerobicEvidence } = {
    score: heartRateEffect,
    evidence: { source: 'heart-rate', seconds: highHeartRateS },
  }
  const powerEstimate = activity.anaerobicPowerEstimate
  if (powerEstimate && powerEstimate.effect > best.score)
    best = { score: powerEstimate.effect, evidence: { source: 'power', ...powerEstimate } }
  const intensityFactor = activity.calculatedIntensityFactor
  if (
    (activity.sport !== 'run' && activity.sport !== 'swim') ||
    intensityFactor?.source !== 'pace' ||
    intensityFactor.value <= 0 ||
    intensityFactor.value > 1.5 ||
    activity.movingTimeS <= 0
  )
    return best
  const averageSpeedMps =
    activity.sport === 'swim' && activity.swimPaceSPer100m != null && activity.swimPaceSPer100m > 0
      ? 100 / activity.swimPaceSPer100m
      : (activity.distanceKm * 1_000) / activity.movingTimeS
  const thresholdSpeedMps = averageSpeedMps / intensityFactor.value
  if (!Number.isFinite(thresholdSpeedMps) || thresholdSpeedMps <= 0) return best
  const intervalLoad = trainingEffectPaceEfforts(activity).reduce((sum, effort) => {
    if (effort.durationS <= 0 || effort.distanceM <= 0) return sum
    const relativeIntensity = effort.distanceM / effort.durationS / thresholdSpeedMps
    if (relativeIntensity <= 1.05 || relativeIntensity > 1.5) return sum
    const intensityWeight = Math.min(1, (relativeIntensity - 1.05) / 0.25)
    return sum + Math.min(120, effort.durationS) * intensityWeight
  }, 0)
  const paceEffect = trainingEffectFromScale(
    intervalLoad,
    HIGH_INTENSITY_INTERVAL_TRAINING_EFFECT_SCALE,
  )
  return paceEffect > best.score
    ? { score: paceEffect, evidence: { source: 'pace', weightedSeconds: round(intervalLoad, 1) } }
    : best
}

export const calculateActivityTrainingEffect = (
  activity: Pick<
    StravaActivityDetail,
    | 'sport'
    | 'distanceKm'
    | 'movingTimeS'
    | 'sufferScore'
    | 'garmin'
    | 'calculatedIntensityFactor'
    | 'calculatedExerciseLoad'
    | 'anaerobicPowerEstimate'
    | 'hrZones'
    | 'analysisRanges'
    | 'swimPaceSPer100m'
    | 'swimIntervals'
  >,
): CalculatedTrainingEffect | null => {
  const garmin = activity.garmin
  if (
    garmin?.aerobicTrainingEffect != null ||
    garmin?.anaerobicTrainingEffect != null ||
    garmin?.aerobicTrainingEffectMessage != null ||
    garmin?.anaerobicTrainingEffectMessage != null
  )
    return null
  const relativeEffort =
    activity.sufferScore != null && activity.sufferScore > 0 ? activity.sufferScore : null
  const aerobicLoad = relativeEffort ?? activity.calculatedExerciseLoad?.value ?? null
  if (aerobicLoad == null || !Number.isFinite(aerobicLoad) || aerobicLoad < 0) return null
  const aerobic = trainingEffectFromScale(
    aerobicLoad,
    relativeEffort == null
      ? EXERCISE_LOAD_TRAINING_EFFECT_SCALE
      : RELATIVE_EFFORT_TRAINING_EFFECT_SCALE,
  )
  const anaerobic = calculatedAnaerobicTrainingEffect(activity)
  return {
    aerobic: round(aerobic, 1),
    anaerobic: round(anaerobic.score, 1),
    evidence: {
      aerobic: {
        source: relativeEffort == null ? 'exercise-load' : 'relative-effort',
        load: round(aerobicLoad, 1),
      },
      anaerobic: anaerobic.evidence,
    },
  }
}

export function haversineMeters(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const dLat = (lat2 - lat1) * 111320
  const dLng = (lng2 - lng1) * 111320 * Math.cos((lat1 * Math.PI) / 180)
  return Math.hypot(dLat, dLng)
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0
  const s = [...xs].sort((p, q) => p - q)
  const m = Math.floor(s.length / 2)
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2
}

export function inferRouteHome(starts: [number, number][]): [number, number] | null {
  if (starts.length < 6) return null
  const seedLat = median(starts.map(p => p[0]))
  const seedLng = median(starts.map(p => p[1]))
  const near = starts.filter(p => haversineMeters(p[0], p[1], seedLat, seedLng) <= 200)
  if (near.length < 6) return null
  return [
    near.reduce((s, p) => s + p[0], 0) / near.length,
    near.reduce((s, p) => s + p[1], 0) / near.length,
  ]
}

function cleanAltitude(alt: number[]): number[] {
  const n = alt.length
  const filled = alt.slice()
  let last = filled.find(x => x > 0.5) ?? 0
  for (let i = 0; i < n; i++) {
    if (filled[i] > 0.5) last = filled[i]
    else filled[i] = last
  }
  const w = 4
  if (n <= w * 2 + 1) return filled
  const out = filled.slice()
  for (let i = 0; i < n; i++) {
    let sum = 0
    let count = 0
    for (let j = Math.max(0, i - w); j <= Math.min(n - 1, i + w); j++) {
      sum += filled[j]
      count++
    }
    out[i] = sum / count
  }
  return out
}

function sampleIndices(lo: number, hi: number, maxPoints: number): number[] {
  if (hi < lo) return []
  const span = hi - lo + 1
  const stride = Math.max(1, Math.ceil(span / maxPoints))
  const idx: number[] = []
  for (let i = lo; i <= hi; i += stride) idx.push(i)
  if (idx[idx.length - 1] !== hi) idx.push(hi)
  return idx
}

function sampleIndicesWithRequired(
  lo: number,
  hi: number,
  maxPoints: number,
  required: number[],
): number[] {
  if (hi < lo) return []
  const anchors = required.filter(index => index >= lo && index <= hi)
  return [...new Set([...sampleIndices(lo, hi, maxPoints), ...anchors])].sort((a, b) => a - b)
}

function emptyTotals(): StravaSportTotals[] {
  return SPORT_ORDER.map(sport => ({
    sport,
    count: 0,
    distanceKm: 0,
    movingTimeS: 0,
    elevationM: 0,
  }))
}

function toHealth(o: OuraDaily): ActivityHealth {
  return {
    readiness: o.readiness,
    sleepScore: o.sleepScore,
    hrv: o.hrv,
    rhr: o.rhr,
    sleepDurationS: o.sleepDurationS,
    tempDeviationC: o.tempDeviationC,
    totalCalories: o.totalCalories,
    activeCalories: o.activeCalories,
  }
}

function avgPos(arr: number[]): number | null {
  let sum = 0
  let count = 0
  for (const x of arr) {
    if (x > 0) {
      sum += x
      count++
    }
  }
  return count ? Math.round(sum / count) : null
}

function maxPos(arr: number[]): number | null {
  let m = 0
  for (const x of arr) if (x > m) m = x
  return m > 0 ? Math.round(m) : null
}

function roundPos(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? Math.round(value) : null
}

function hasPositive(values: number[] | undefined): boolean {
  return values?.some(value => value > 0) ?? false
}

function positiveCount(values: number[] | undefined): number {
  return values?.filter(value => value > 0).length ?? 0
}

function streamQuality(streams: StravaStreams | GarminStreams | undefined): number {
  if (!streams) return 0
  const channels =
    (streams.latlng.length >= 2 ? 1 : 0) +
    (streams.altitude.length > 0 ? 1 : 0) +
    (streams.distance.length > 0 ? 1 : 0) +
    (hasPositive(streams.heartrate) ? 1 : 0) +
    (hasPositive(streams.cadence) ? 1 : 0) +
    (hasPositive(streams.watts) ? 1 : 0)
  return (
    channels * 10_000 +
    streams.latlng.length +
    streams.altitude.length +
    streams.distance.length +
    positiveCount(streams.heartrate) +
    positiveCount(streams.cadence) +
    positiveCount(streams.watts)
  )
}

function selectStreams(
  strava: StravaStreams | undefined,
  match: GarminActivityMatch | null,
  garmin: GarminCache | null,
  activity?: RawStravaActivity,
): StravaStreams | GarminStreams | undefined {
  const fromGarmin = match ? garmin?.streams?.[match.activity.id] : undefined
  if (activity && match && fromGarmin && activity.sportType.startsWith('Virtual')) {
    const alignment = timedStreamAlignment(fromGarmin)
    if (!alignment || fromGarmin.latlng.length !== alignment.time.length) return strava
    const offsetS = (Date.parse(match.activity.startDate) - Date.parse(activity.startDate)) / 1_000
    const endS = activity.elapsedTime || activity.movingTime
    const indices = alignment.time.flatMap((time, index) => {
      const elapsedS = time + offsetS
      return elapsedS >= 0 && elapsedS <= endS ? [index] : []
    })
    if (indices.length < 2) return strava
    const time = indices.map(index => alignment.time[index] + offsetS)
    const telemetry = (key: 'watts' | 'heartrate' | 'cadence'): number[] => {
      const values = strava?.[key]
      const samples =
        values && strava?.time?.length === values.length
          ? strava.time.map((elapsedS, index) => ({ elapsedS, value: values[index] }))
          : []
      return indices.map(
        (index, position) =>
          timedNullableMetricAt(samples, time[position], 2.5) ?? fromGarmin[key]?.[index] ?? 0,
      )
    }
    return {
      time,
      latlng: indices.map(index => fromGarmin.latlng[index]),
      altitude: indices.map(index => fromGarmin.altitude[index]),
      distance: indices.map(index => fromGarmin.distance[index]),
      watts: telemetry('watts'),
      heartrate: telemetry('heartrate'),
      cadence: telemetry('cadence'),
    }
  }
  return streamQuality(fromGarmin) > streamQuality(strava) ? fromGarmin : strava
}

export function applyActivityTracking(
  cache: StravaRawCache | null,
  garmin: GarminCache | null,
  entries: readonly ActivityTrackingEntry[],
): StravaRawCache | null {
  if (!cache || entries.length === 0) return cache
  const activities = { ...cache.activities }
  const streams = { ...cache.streams }
  for (const entry of entries) {
    const id = String(entry.activityId)
    const activity = cache.activities[id]
    if (!activity || !entry.virtual) continue
    const sport = normalizeKind(activity.sportType)
    if (sport !== 'bike' && sport !== 'run') continue
    const match = matchGarminActivity(activity, sport, garmin, entry.garminActivityId)
    const garminDistance = match?.activity.distanceM
    const distance =
      garminDistance != null && Number.isFinite(garminDistance) && garminDistance > 0
        ? garminDistance
        : activity.distance
    activities[id] = {
      ...activity,
      sportType: sport === 'bike' ? 'VirtualRide' : 'VirtualRun',
      distance,
      averageSpeed: activity.movingTime > 0 ? distance / activity.movingTime : 0,
      totalElevationGain: match?.activity.metrics.totalAscentM ?? activity.totalElevationGain,
    }
    const original = cache.streams?.[id]
    const fromGarmin = match ? garmin?.streams?.[match.activity.id] : undefined
    const alignment = timedStreamAlignment(fromGarmin)
    if (!match || !original?.time?.length || !alignment) continue
    const offsetS = (Date.parse(match.activity.startDate) - Date.parse(activity.startDate)) / 1_000
    const samples = alignment.time.map((time, index) => ({
      elapsedS: time + offsetS,
      value: alignment.distance[index],
    }))
    const altitudeSamples =
      fromGarmin?.altitude.length === alignment.time.length
        ? alignment.time.map((time, index) => ({
            elapsedS: time + offsetS,
            value: fromGarmin.altitude[index],
          }))
        : []
    streams[id] = {
      ...original,
      latlng: [],
      altitude: original.time.map(time => timedMetricAt(altitudeSamples, time) ?? 0),
      distance: original.time.map(time => timedMetricAt(samples, time) ?? 0),
    }
  }
  return { ...cache, activities, streams }
}

function selectEffortStreams(
  strava: StravaStreams | undefined,
  selected: StravaStreams | GarminStreams | undefined,
): StravaStreams | GarminStreams | undefined {
  return strava?.time?.length ? strava : selected
}

export function resolveActivityHeartRate(
  a: RawStravaActivity,
  sport: ActivityKind,
  selectedStreams: StravaStreams | GarminStreams | undefined,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
): ActivityHeartRate {
  const selectedHr = selectedStreams?.heartrate ?? []
  const stravaAvg = roundPos(a.averageHeartrate) ?? avgPos(selectedHr)
  const stravaMax = roundPos(a.maxHeartrate) ?? maxPos(selectedHr)
  const garminStream = garminMatch
    ? (garmin?.streams?.[garminMatch.activity.id]?.heartrate ?? [])
    : []
  const metrics = garminMatch?.activity.metrics
  const garminAvg = roundPos(metrics?.avgHeartRate) ?? avgPos(garminStream)
  const garminMax = roundPos(metrics?.maxHeartRate) ?? maxPos(garminStream)

  if (sport === 'run' && (garminAvg != null || garminMax != null))
    return {
      avgHr: garminAvg ?? stravaAvg,
      maxHr: garminMax ?? stravaMax,
      stream: hasPositive(garminStream) ? garminStream : selectedHr,
    }

  return { avgHr: stravaAvg, maxHr: stravaMax, stream: selectedHr }
}

const MILE_M = 1609.344
const MAX_EFFORT_TIMELINE_S = (7 * DAY_MS) / 1000
const POWER_EFFORT_SECS = [
  5, 15, 30, 60, 120, 180, 300, 480, 600, 900, 1200, 1800, 2700, 3600, 7200,
]
const DISTANCE_EFFORTS = [
  ['5 mile', 5 * MILE_M],
  ['10K', 10_000],
  ['10 mile', 10 * MILE_M],
  ['20K', 20_000],
  ['30K', 30_000],
  ['40K', 40_000],
  ['50K', 50_000],
  ['80K', 80_000],
  ['50 mile', 50 * MILE_M],
  ['90K', 90_000],
  ['100K', 100_000],
  ['100 mile', 100 * MILE_M],
  ['180K', 180_000],
] as const

interface EffortTimeline {
  distanceM: Float64Array
  altitudeM: Float64Array
  watts: Float64Array
  wattsObserved: Uint8Array
  heartRate: Float64Array
}

interface BestPowerWindow {
  durationS: number
  start: number
  end: number
  averageWatts: number
}

function effortTimeline(
  streams: StravaStreams | GarminStreams | undefined,
  movingTimeS: number,
): EffortTimeline | null {
  if (!streams) return null
  const sampleCount = Math.max(
    streams.distance.length,
    streams.altitude.length,
    streams.watts?.length ?? 0,
    streams.heartrate?.length ?? 0,
  )
  if (sampleCount < 2) return null
  const rawTime = 'time' in streams ? streams.time : undefined
  if ('time' in streams && rawTime?.length !== sampleCount) return null
  if (!rawTime && Math.abs(sampleCount - movingTimeS) / Math.max(1, movingTimeS) > 0.15) return null
  const sampleSeconds = new Int32Array(sampleCount)
  let previousSecond = 0
  for (let i = 0; i < sampleCount; i++) {
    const raw = rawTime ? rawTime[i] : i
    const second = Number.isFinite(raw) ? Math.max(previousSecond, Math.round(raw)) : previousSecond
    sampleSeconds[i] = second
    previousSecond = second
  }
  if (previousSecond < 1 || previousSecond > MAX_EFFORT_TIMELINE_S) return null

  const length = previousSecond + 1
  const distanceM = new Float64Array(length)
  const altitudeM = new Float64Array(length)
  const watts = new Float64Array(length)
  const wattsObserved = new Uint8Array(length)
  const heartRate = new Float64Array(length)
  const distanceSet = new Uint8Array(length)
  const altitudeSet = new Uint8Array(length)
  const wattCount = new Uint16Array(length)
  const heartRateCount = new Uint16Array(length)
  const initialAltitude = streams.altitude.find(Number.isFinite) ?? 0

  for (let i = 0; i < sampleCount; i++) {
    const second = sampleSeconds[i]
    const distance = streams.distance[i]
    if (Number.isFinite(distance)) {
      distanceM[second] = Math.max(0, distance)
      distanceSet[second] = 1
    }
    const altitude = streams.altitude[i]
    if (Number.isFinite(altitude)) {
      altitudeM[second] = altitude
      altitudeSet[second] = 1
    }
    const power = streams.watts?.[i]
    if (Number.isFinite(power)) {
      watts[second] += Math.max(0, power ?? 0)
      wattCount[second]++
      wattsObserved[second] = 1
    }
    const hr = streams.heartrate?.[i]
    if (Number.isFinite(hr) && (hr ?? 0) > 0) {
      heartRate[second] += hr ?? 0
      heartRateCount[second]++
    }
  }

  let distance = 0
  let altitude = initialAltitude
  for (let second = 0; second < length; second++) {
    if (distanceSet[second]) distance = Math.max(distance, distanceM[second])
    distanceM[second] = distance
    if (altitudeSet[second]) altitude = altitudeM[second]
    altitudeM[second] = altitude
    if (wattCount[second]) watts[second] /= wattCount[second]
    if (heartRateCount[second]) heartRate[second] /= heartRateCount[second]
  }
  return { distanceM, altitudeM, watts, wattsObserved, heartRate }
}

function sumPrefix(values: Float64Array): Float64Array {
  const prefix = new Float64Array(values.length + 1)
  for (let i = 0; i < values.length; i++) prefix[i + 1] = prefix[i] + values[i]
  return prefix
}

const CYCLING_PERFORMANCE_CONDITION_WINDOW_S = 6 * 60
const CYCLING_PERFORMANCE_CONDITION_POWER_WINDOW_S = 30
const CYCLING_PERFORMANCE_CONDITION_SAMPLE_STEP_S = 15
const CYCLING_PERFORMANCE_CONDITION_MIN_COVERAGE = 0.85

export function calculateCyclingPerformanceCondition(
  watts: ArrayLike<number>,
  wattsObserved: ArrayLike<number>,
  heartRate: ArrayLike<number>,
  ftpWatts: number | null,
  lactateThresholdHeartRateBpm: number | null,
  restingHeartRateBpm: number | null,
): CyclingPerformanceConditionEstimate | null {
  const length = Math.min(watts.length, wattsObserved.length, heartRate.length)
  if (
    length <= CYCLING_PERFORMANCE_CONDITION_WINDOW_S ||
    ftpWatts == null ||
    !Number.isFinite(ftpWatts) ||
    ftpWatts <= 0 ||
    lactateThresholdHeartRateBpm == null ||
    !Number.isFinite(lactateThresholdHeartRateBpm) ||
    restingHeartRateBpm == null ||
    !Number.isFinite(restingHeartRateBpm) ||
    restingHeartRateBpm <= 0 ||
    lactateThresholdHeartRateBpm <= restingHeartRateBpm
  )
    return null

  const powerSum = new Float64Array(length + 1)
  const powerCount = new Uint32Array(length + 1)
  const heartRateSum = new Float64Array(length + 1)
  const heartRateCount = new Uint32Array(length + 1)
  for (let second = 0; second < length; second++) {
    const power = watts[second]
    const powerIsObserved = wattsObserved[second] > 0 && Number.isFinite(power)
    const bpm = heartRate[second]
    const heartRateIsObserved = Number.isFinite(bpm) && bpm > restingHeartRateBpm
    powerSum[second + 1] = powerSum[second] + (powerIsObserved ? Math.max(0, power) : 0)
    powerCount[second + 1] = powerCount[second] + (powerIsObserved ? 1 : 0)
    heartRateSum[second + 1] = heartRateSum[second] + (heartRateIsObserved ? bpm : 0)
    heartRateCount[second + 1] = heartRateCount[second] + (heartRateIsObserved ? 1 : 0)
  }

  const rollingPowerFourth = new Float64Array(length)
  const rollingPowerValid = new Uint8Array(length)
  for (let second = 0; second < length; second++) {
    const start = Math.max(0, second - CYCLING_PERFORMANCE_CONDITION_POWER_WINDOW_S + 1)
    const span = second - start + 1
    const observed = powerCount[second + 1] - powerCount[start]
    if (observed / span < CYCLING_PERFORMANCE_CONDITION_MIN_COVERAGE) continue
    const averageWatts = (powerSum[second + 1] - powerSum[start]) / observed
    rollingPowerFourth[second] = averageWatts ** 4
    rollingPowerValid[second] = 1
  }
  const rollingPowerFourthSum = sumPrefix(rollingPowerFourth)
  const rollingPowerCount = new Uint32Array(length + 1)
  for (let second = 0; second < length; second++)
    rollingPowerCount[second + 1] = rollingPowerCount[second] + rollingPowerValid[second]

  const thresholdHeartRateReserve = lactateThresholdHeartRateBpm - restingHeartRateBpm
  const samples: CyclingPerformanceConditionSample[] = []
  for (
    let second = CYCLING_PERFORMANCE_CONDITION_WINDOW_S;
    second < length;
    second += CYCLING_PERFORMANCE_CONDITION_SAMPLE_STEP_S
  ) {
    const start = second - CYCLING_PERFORMANCE_CONDITION_WINDOW_S + 1
    const powerSamples = rollingPowerCount[second + 1] - rollingPowerCount[start]
    const heartRateSamples = heartRateCount[second + 1] - heartRateCount[start]
    if (
      powerSamples / CYCLING_PERFORMANCE_CONDITION_WINDOW_S <
        CYCLING_PERFORMANCE_CONDITION_MIN_COVERAGE ||
      heartRateSamples / CYCLING_PERFORMANCE_CONDITION_WINDOW_S <
        CYCLING_PERFORMANCE_CONDITION_MIN_COVERAGE
    )
      continue
    const normalizedPower = Math.pow(
      (rollingPowerFourthSum[second + 1] - rollingPowerFourthSum[start]) / powerSamples,
      0.25,
    )
    const averageHeartRate = (heartRateSum[second + 1] - heartRateSum[start]) / heartRateSamples
    const powerIntensity = normalizedPower / ftpWatts
    const heartRateIntensity = (averageHeartRate - restingHeartRateBpm) / thresholdHeartRateReserve
    if (
      !Number.isFinite(powerIntensity) ||
      !Number.isFinite(heartRateIntensity) ||
      powerIntensity < 0.3 ||
      heartRateIntensity < 0.3 ||
      heartRateIntensity > 1.5
    )
      continue
    const value = Math.min(20, Math.max(-20, 100 * (powerIntensity / heartRateIntensity - 1)))
    samples.push({ elapsedS: second, value: round(value, 1) })
  }
  if (samples.length < 2) return null
  return {
    method: GARDEN_CYCLING_PERFORMANCE_CONDITION_METHOD,
    ftpWatts,
    lactateThresholdHeartRateBpm,
    restingHeartRateBpm,
    windowSeconds: CYCLING_PERFORMANCE_CONDITION_WINDOW_S,
    samples,
  }
}

function positivePrefixes(values: Float64Array): [Float64Array, Uint32Array] {
  const sum = new Float64Array(values.length + 1)
  const count = new Uint32Array(values.length + 1)
  for (let i = 0; i < values.length; i++) {
    const value = values[i]
    sum[i + 1] = sum[i] + (value > 0 ? value : 0)
    count[i + 1] = count[i] + (value > 0 ? 1 : 0)
  }
  return [sum, count]
}

function averagePositive(
  sum: Float64Array,
  count: Uint32Array,
  start: number,
  end: number,
): number | null {
  const n = count[end] - count[start]
  return n > 0 ? Math.round((sum[end] - sum[start]) / n) : null
}

function bestPowerWindows(
  timeline: EffortTimeline,
  durations: readonly number[],
): BestPowerWindow[] {
  const prefix = sumPrefix(timeline.watts)
  const windows: BestPowerWindow[] = []
  for (const durationS of durations) {
    if (durationS > timeline.watts.length) break
    let bestSum = -1
    let bestStart = 0
    for (let start = 0; start + durationS <= timeline.watts.length; start++) {
      const sum = prefix[start + durationS] - prefix[start]
      if (sum > bestSum) {
        bestSum = sum
        bestStart = start
      }
    }
    windows.push({
      durationS,
      start: bestStart,
      end: bestStart + durationS,
      averageWatts: Math.floor(Math.max(0, bestSum / durationS)),
    })
  }
  return windows
}

function bestObservedPowerWindows(
  timeline: EffortTimeline,
  durations: readonly number[],
): BestPowerWindow[] {
  const powerPrefix = sumPrefix(timeline.watts)
  const observedPrefix = new Uint32Array(timeline.wattsObserved.length + 1)
  for (let index = 0; index < timeline.wattsObserved.length; index++)
    observedPrefix[index + 1] = observedPrefix[index] + timeline.wattsObserved[index]

  const windows: BestPowerWindow[] = []
  for (const durationS of durations) {
    if (durationS > timeline.watts.length) break
    let bestSum = -1
    let bestStart = 0
    for (let start = 0; start + durationS <= timeline.watts.length; start++) {
      const end = start + durationS
      if (observedPrefix[end] - observedPrefix[start] !== durationS) continue
      const sum = powerPrefix[end] - powerPrefix[start]
      if (sum > bestSum) {
        bestSum = sum
        bestStart = start
      }
    }
    if (bestSum < 0) continue
    windows.push({
      durationS,
      start: bestStart,
      end: bestStart + durationS,
      averageWatts: round(bestSum / durationS, 2),
    })
  }
  return windows
}

function meanMaxCurve(timeline: EffortTimeline | null): PowerCurvePoint[] {
  if (!timeline) return []
  const durations = Array.from({ length: timeline.watts.length }, (_, index) => index + 1)
  return bestPowerWindows(timeline, durations).map(window => ({
    s: window.durationS,
    w: window.averageWatts,
  }))
}

const MAX_SPEED_WINDOW_S = 3
const MAX_ACCEL_MPS2 = 3

function maxSpeedKph(timeline: EffortTimeline | null): number | null {
  if (!timeline) return null
  const distanceM = timeline.distanceM
  if (distanceM.length < 2) return null
  const speeds = new Float64Array(distanceM.length - 1)
  let allowed = 0
  for (let second = 0; second < speeds.length; second++) {
    allowed = Math.min(distanceM[second + 1] - distanceM[second], allowed + MAX_ACCEL_MPS2)
    speeds[second] = allowed
  }
  const windowS = Math.min(MAX_SPEED_WINDOW_S, speeds.length)
  let bestM = 0
  let sumM = 0
  for (let second = 0; second < speeds.length; second++) {
    sumM += speeds[second]
    if (second >= windowS) sumM -= speeds[second - windowS]
    if (second >= windowS - 1) bestM = Math.max(bestM, sumM)
  }
  return bestM > 0 ? round((bestM / windowS) * 3.6, 1) : null
}

function distanceBestEfforts(timeline: EffortTimeline): CyclingDistanceEffort[] {
  const [hrSum, hrCount] = positivePrefixes(timeline.heartRate)
  const efforts: CyclingDistanceEffort[] = []
  for (const [label, targetDistanceM] of DISTANCE_EFFORTS) {
    if (timeline.distanceM[timeline.distanceM.length - 1] - timeline.distanceM[0] < targetDistanceM)
      break
    let best: CyclingDistanceEffort | null = null
    let bestElapsed = Infinity
    let end = 1
    for (let start = 0; start < timeline.distanceM.length - 1; start++) {
      if (end <= start) end = start + 1
      const target = timeline.distanceM[start] + targetDistanceM
      while (end < timeline.distanceM.length && timeline.distanceM[end] < target) end++
      if (end >= timeline.distanceM.length) break
      const previous = Math.max(start, end - 1)
      const spanM = timeline.distanceM[end] - timeline.distanceM[previous]
      const fraction = spanM > 0 ? (target - timeline.distanceM[previous]) / spanM : 1
      const elapsed = previous - start + Math.min(1, Math.max(0, fraction))
      if (elapsed <= 0 || elapsed >= bestElapsed) continue
      const endAltitude =
        timeline.altitudeM[previous] +
        (timeline.altitudeM[end] - timeline.altitudeM[previous]) * fraction
      bestElapsed = elapsed
      best = {
        label,
        targetDistanceM,
        elapsedTimeS: Math.ceil(elapsed),
        averageSpeedKph: round((targetDistanceM / 1000 / elapsed) * 3600, 3),
        averageHeartRate: averagePositive(hrSum, hrCount, start, end + 1),
        elevationDeltaM: round(endAltitude - timeline.altitudeM[start], 1),
      }
    }
    if (best) efforts.push(best)
  }
  return efforts
}

function powerBestEfforts(timeline: EffortTimeline, weightKg: number | null): CyclingPowerEffort[] {
  const [hrSum, hrCount] = positivePrefixes(timeline.heartRate)
  return bestPowerWindows(timeline, POWER_EFFORT_SECS).map(window => ({
    durationS: window.durationS,
    averageWatts: window.averageWatts,
    wattsPerKg: weightKg != null && weightKg > 0 ? round(window.averageWatts / weightKg, 2) : null,
    averageHeartRate: averagePositive(hrSum, hrCount, window.start, window.end),
    elevationDeltaM: round(
      timeline.altitudeM[Math.max(window.start, window.end - 1)] - timeline.altitudeM[window.start],
      1,
    ),
  }))
}

interface ActivityClimbSegment {
  source: ActivityClimbSource
  name: string
  startDate: string
  endDate: string
  distanceM: number
  durationS: number
  elevationGainM: number | null
  avgGradePct: number | null
  avgSpeedMps: number | null
  avgHeartRate: number | null
  avgPower: number | null
  avgCadence: number | null
}

function cyclingClimbEfforts(
  segments: ActivityClimbSegment[],
  weightKg: number | null,
): CyclingClimbEffort[] {
  return segments.flatMap(segment => {
    const durationS = segment.durationS
    const elevationGainM = segment.elevationGainM ?? 0
    if (durationS <= 0 || segment.distanceM <= 0 || elevationGainM <= 0) return []
    const averageWatts = segment.avgPower
    return [
      {
        source: segment.source,
        name: segment.name,
        durationS: round(durationS, 1),
        distanceM: round(segment.distanceM, 1),
        elevationGainM: round(elevationGainM, 1),
        averageGradePct:
          segment.avgGradePct ?? round((elevationGainM / segment.distanceM) * 100, 1),
        averageSpeedKph:
          segment.avgSpeedMps != null
            ? round(segment.avgSpeedMps * 3.6, 1)
            : round((segment.distanceM / durationS) * 3.6, 1),
        averageHeartRate: segment.avgHeartRate,
        averageWatts,
        wattsPerKg:
          averageWatts != null && weightKg != null && weightKg > 0
            ? round(averageWatts / weightKg, 2)
            : null,
        vamMPerHour: Math.round((elevationGainM / durationS) * 3600),
      },
    ]
  })
}

function garminClimbSegments(segments: GarminClimbSegment[]): ActivityClimbSegment[] {
  return segments.map((segment, index) => ({
    source: 'garmin-climbpro',
    name: `Climb ${index + 1}`,
    startDate: segment.startDate,
    endDate: segment.endDate,
    distanceM: segment.distanceM,
    durationS: segment.durationS,
    elevationGainM: segment.elevationGainM,
    avgGradePct: segment.avgGradePct,
    avgSpeedMps: segment.avgSpeedMps,
    avgHeartRate: segment.avgHeartRate,
    avgPower: segment.avgPower,
    avgCadence: segment.avgCadence,
  }))
}

function wahooClimbSegments(segments: WahooSummitSegment[]): ActivityClimbSegment[] {
  return segments.map((segment, index) => ({
    source: segment.feature === 'summit-segment' ? 'wahoo-summit-segment' : 'wahoo-summit-freeride',
    name: segment.name ? `Summit ${segment.name}` : `Summit ${index + 1}`,
    startDate: segment.startDate,
    endDate: segment.endDate,
    distanceM: segment.distanceM,
    durationS: segment.durationS,
    elevationGainM: segment.elevationGainM,
    avgGradePct: segment.avgGradePct,
    avgSpeedMps: segment.avgSpeedMps,
    avgHeartRate: segment.avgHeartRate,
    avgPower: segment.avgPower,
    avgCadence: segment.avgCadence,
  }))
}

function activityClimbSegments(
  sport: ActivityKind,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
  wahooMatch: WahooActivityMatch | null,
  wahoo: WahooCache | null,
): ActivityClimbSegment[] {
  if (sport !== 'bike') return []
  if (wahooMatch) return wahooClimbSegments(wahoo?.summitSegments[wahooMatch.activity.id] ?? [])
  if (garminMatch) return garminClimbSegments(garmin?.climbs?.[garminMatch.activity.id] ?? [])
  return []
}

function zoneTimes(stream: number[], uppers: number[], countZero: boolean): number[] {
  const counts = Array.from({ length: uppers.length + 1 }, () => 0)
  for (const raw of stream) {
    if (raw <= 0 && !countZero) continue
    const v = raw > 0 ? raw : 0
    let z = uppers.length
    for (let i = 0; i < uppers.length; i++)
      if (v <= uppers[i]) {
        z = i
        break
      }
    counts[z]++
  }
  return counts
}

function durationZoneTimes(stream: number[], uppers: number[], movingTimeS: number): number[] {
  const counts = zoneTimes(stream, uppers, false)
  const total = counts.reduce((sum, count) => sum + count, 0)
  if (total <= 0 || movingTimeS <= 0) return counts
  const scale = movingTimeS / total
  return counts.map(count => Math.round(count * scale))
}

function powerHistogram(w: number[], countZero = true, bin = 25): number[] {
  let maxB = 0
  for (const raw of w) {
    if (raw <= 0 && !countZero) continue
    const b = Math.floor((raw > 0 ? raw : 0) / bin)
    if (b > maxB) maxB = b
  }
  const out = Array.from({ length: maxB + 1 }, () => 0)
  for (const raw of w) {
    if (raw <= 0 && !countZero) continue
    out[Math.floor((raw > 0 ? raw : 0) / bin)]++
  }
  return out
}

function deriveHrBounds(hrmax: number): number[] {
  return [0.6, 0.7, 0.8, 0.9].map(p => Math.round(hrmax * p))
}

function derivePowerBounds(ftp: number): number[] {
  return [0.55, 0.75, 0.9, 1.05, 1.2, 1.5].map(p => Math.round(ftp * p))
}

interface PowerCurveSource {
  activityId: number
  activityDate: string
  curve: PowerCurvePoint[]
}

function mergeMaxCurves(sources: PowerCurveSource[]): PowerCurvePoint[] {
  const best = new Map<number, PowerCurvePoint>()
  for (const source of sources)
    for (const point of source.curve) {
      const current = best.get(point.s)
      if (current && current.w >= point.w) continue
      best.set(point.s, {
        s: point.s,
        w: point.w,
        activityId: source.activityId,
        activityDate: source.activityDate,
      })
    }
  return [...best.values()].sort((left, right) => left.s - right.s)
}

function delta(
  garmin: number | null | undefined,
  strava: number | null | undefined,
): number | null {
  return garmin != null && strava != null ? Math.round(garmin - strava) : null
}

function deltaFloat(
  garmin: number | null | undefined,
  strava: number | null | undefined,
  dp: number,
): number | null {
  return garmin != null && strava != null ? round(garmin - strava, dp) : null
}

interface ActivityWeight {
  kg: number
  date: string
}

function activityWeight(
  garmin: GarminCache | null,
  activity: RawStravaActivity,
): ActivityWeight | null {
  const samples = garmin?.weight
  if (!samples?.length) return null
  const activityDate = activity.startDateLocal.slice(0, 10)
  const startMs = Date.parse(activity.startDate)
  let sameDayBefore: ActivityWeight | null = null
  let sameDayAfter: ActivityWeight | null = null
  let sameDayBeforeTs = -Infinity
  let sameDayAfterTs = Infinity
  for (const sample of samples) {
    if (sample.weightKg == null || !Number.isFinite(sample.weightKg) || sample.weightKg <= 0)
      continue
    const weight = { kg: sample.weightKg, date: sample.date }
    if (sample.date !== activityDate) continue
    if (sample.ts <= startMs && sample.ts > sameDayBeforeTs) {
      sameDayBefore = weight
      sameDayBeforeTs = sample.ts
    } else if (sample.ts > startMs && sample.ts < sameDayAfterTs) {
      sameDayAfter = weight
      sameDayAfterTs = sample.ts
    }
  }
  return sameDayBefore ?? sameDayAfter
}

function activityRestingHeartRate(oura: OuraCache | null, date: string): number | null {
  const valid = (value: number | null | undefined): value is number =>
    value != null && Number.isFinite(value) && value >= 25 && value <= 120
  const sameDay = oura?.days[date]?.rhr
  if (valid(sameDay)) return sameDay
  const end = Date.parse(`${date}T00:00:00Z`)
  if (!Number.isFinite(end)) return null
  const start = end - 27 * DAY_MS
  const values = Object.values(oura?.days ?? {}).flatMap(day => {
    const timestamp = Date.parse(`${day.date}T00:00:00Z`)
    return timestamp >= start && timestamp <= end && valid(day.rhr) ? [day.rhr] : []
  })
  return values.length > 0 ? median(values) : null
}

function garminVerification(
  a: RawStravaActivity,
  match: GarminActivityMatch | null,
  trainingEffectMatch: GarminActivityMatch | null,
): GarminVerification | null {
  if (!match) return null
  const activity = match.activity
  const metrics = activity.metrics
  const trainingEffectMetrics = trainingEffectMatch?.activity.metrics ?? metrics
  const distanceDeltaM = delta(activity.distanceM, a.distance)
  return {
    activityId: activity.id,
    name: activity.name,
    sourceDevice: activity.sourceDevice,
    startDate: activity.startDate,
    startDiffS: Math.round(match.startDiffMs / 1000),
    distanceM: activity.distanceM,
    distanceDeltaM,
    distanceDeltaPct:
      distanceDeltaM != null && a.distance > 0
        ? round((distanceDeltaM / a.distance) * 100, 1)
        : null,
    movingTimeS: activity.movingTimeS,
    movingTimeDeltaS: delta(activity.movingTimeS, a.movingTime),
    elapsedTimeS: activity.elapsedTimeS,
    elapsedTimeDeltaS: delta(activity.elapsedTimeS, a.elapsedTime),
    totalCalories: metrics.totalCalories,
    caloriesDelta: delta(metrics.totalCalories, a.calories),
    avgHeartRate: metrics.avgHeartRate,
    avgHeartRateDelta: delta(metrics.avgHeartRate, a.averageHeartrate),
    avgPower: metrics.avgPower,
    avgPowerDelta: delta(metrics.avgPower, a.averageWatts),
    avgCadence: metrics.avgCadence,
    normalizedPower: metrics.normalizedPower,
    maxPower: metrics.maxPower,
    totalWorkKJ: metrics.totalWorkKJ,
    totalWorkDeltaKJ: deltaFloat(metrics.totalWorkKJ, a.kilojoules, 1),
    trainingStressScore: metrics.trainingStressScore,
    intensityFactor: metrics.intensityFactor,
    trainingEffectActivityId: trainingEffectMatch?.activity.id ?? null,
    aerobicTrainingEffect: trainingEffectMetrics.aerobicTrainingEffect,
    anaerobicTrainingEffect: trainingEffectMetrics.anaerobicTrainingEffect,
    exerciseLoad: trainingEffectMetrics.exerciseLoad,
    trainingEffectLabel: trainingEffectMetrics.trainingEffectLabel,
    aerobicTrainingEffectMessage: trainingEffectMetrics.aerobicTrainingEffectMessage,
    anaerobicTrainingEffectMessage: trainingEffectMetrics.anaerobicTrainingEffectMessage,
    runningDynamics: activity.runningDynamics,
  }
}

function activityComputer(
  sport: ActivityKind,
  garminMatch: GarminActivityMatch | null,
  wahooMatch: WahooActivityMatch | null,
): ActivityComputer | null {
  if (sport !== 'bike') return null
  if (wahooMatch) return 'wahoo'
  return garminMatch ? 'garmin' : null
}

export function normalizeActivityDevice(value: string | null | undefined): ActivityDevice | null {
  if (!value) return null
  const normalized = value
    .trim()
    .toLocaleLowerCase()
    .replaceAll(/[^a-z0-9]+/g, '-')
  if (normalized.includes('apple-watch-ultra-3') || normalized.includes('appl-watch-ultra-3'))
    return 'apple-watch-ultra-3'
  if (normalized.includes('forerunner-970')) return 'garmin-forerunner-970'
  return null
}

export const prefersActivityDeviceThermal = (sport: ActivityKind): boolean =>
  sport === 'run' || sport === 'walk'

function activityDevice(
  sport: ActivityKind,
  activity: RawStravaActivity,
  garminMatch: GarminActivityMatch | null,
): ActivityDevice | null {
  if (sport !== 'run' && sport !== 'walk' && sport !== 'swim') return null
  return (
    normalizeActivityDevice(activity.deviceName) ??
    normalizeActivityDevice(garminMatch?.activity.sourceDevice)
  )
}

function temperatureAt(samples: WeatherTemperatureSample[], elapsedS: number): number | null {
  if (samples.length === 0) return null
  if (elapsedS <= samples[0].elapsedS) return samples[0].temperatureC
  for (let i = 1; i < samples.length; i++) {
    const previous = samples[i - 1]
    const next = samples[i]
    if (elapsedS > next.elapsedS) continue
    const span = next.elapsedS - previous.elapsedS
    if (span <= 0) return next.temperatureC
    const fraction = (elapsedS - previous.elapsedS) / span
    return previous.temperatureC + (next.temperatureC - previous.temperatureC) * fraction
  }
  return samples[samples.length - 1].temperatureC
}

interface TimedMetricSample {
  elapsedS: number
  value: number
}

interface TimedNullableMetricSample {
  elapsedS: number
  value: number | null
}

interface RespirationProjectionModel {
  intercept: number
  slope: number
}

const RESPIRATION_PROJECTION_STEP_S = 0.5
const RESPIRATION_PROJECTION_MAX_GAP_S = 5
const RESPIRATION_CALIBRATION_MIN_SAMPLES = 30
const RESPIRATION_PROJECTION_MIN_BRPM = 12
const RESPIRATION_PROJECTION_MAX_BRPM = 60

function garminRespirationProjectionModel(
  garmin: GarminCache | null,
): RespirationProjectionModel | null {
  if (!garmin?.streams) return null
  let count = 0
  let heartRateSum = 0
  let respirationSum = 0
  let heartRateSquaredSum = 0
  let crossProductSum = 0
  for (const [id, activity] of Object.entries(garmin.activities)) {
    if (activity.sport !== 'bike') continue
    const stream = garmin.streams[id]
    const heartRate = stream?.heartrate
    const respiration = stream?.respiration
    if (!heartRate || !respiration || heartRate.length !== respiration.length) continue
    for (let index = 0; index < heartRate.length; index++) {
      const heartRateValue = heartRate[index]
      const respirationValue = respiration[index]
      if (
        !Number.isFinite(heartRateValue) ||
        heartRateValue < 35 ||
        heartRateValue > 240 ||
        !Number.isFinite(respirationValue) ||
        respirationValue < 8 ||
        respirationValue > 70
      )
        continue
      count += 1
      heartRateSum += heartRateValue
      respirationSum += respirationValue
      heartRateSquaredSum += heartRateValue * heartRateValue
      crossProductSum += heartRateValue * respirationValue
    }
  }
  if (count < RESPIRATION_CALIBRATION_MIN_SAMPLES) return null
  const denominator = count * heartRateSquaredSum - heartRateSum * heartRateSum
  if (!Number.isFinite(denominator) || denominator <= 0) return null
  const slope = (count * crossProductSum - heartRateSum * respirationSum) / denominator
  const intercept = (respirationSum - slope * heartRateSum) / count
  if (!Number.isFinite(slope) || slope <= 0 || slope > 1 || !Number.isFinite(intercept)) return null
  return { intercept, slope }
}

function projectedRespiration(model: RespirationProjectionModel, heartRate: number): number {
  return Math.min(
    RESPIRATION_PROJECTION_MAX_BRPM,
    Math.max(RESPIRATION_PROJECTION_MIN_BRPM, model.intercept + model.slope * heartRate),
  )
}

function projectWahooRespiration(
  stream: WahooStreams,
  startOffsetS: number,
  model: RespirationProjectionModel,
): TimedMetricSample[] {
  if (stream.time.length !== stream.heartrate.length) return []
  const samples: TimedMetricSample[] = []
  let previousElapsedS: number | null = null
  let previousHeartRate: number | null = null
  for (let index = 0; index < stream.time.length; index++) {
    const elapsedS = stream.time[index]
    const heartRate = stream.heartrate[index]
    if (
      !Number.isFinite(elapsedS) ||
      typeof heartRate !== 'number' ||
      !Number.isFinite(heartRate) ||
      heartRate < 35 ||
      heartRate > 240
    ) {
      previousElapsedS = null
      previousHeartRate = null
      continue
    }
    if (previousElapsedS == null || previousHeartRate == null || elapsedS <= previousElapsedS) {
      samples.push({
        elapsedS: elapsedS + startOffsetS,
        value: projectedRespiration(model, heartRate),
      })
      previousElapsedS = elapsedS
      previousHeartRate = heartRate
      continue
    }
    const spanS = elapsedS - previousElapsedS
    if (spanS > RESPIRATION_PROJECTION_MAX_GAP_S) {
      samples.push({
        elapsedS: elapsedS + startOffsetS,
        value: projectedRespiration(model, heartRate),
      })
      previousElapsedS = elapsedS
      previousHeartRate = heartRate
      continue
    }
    for (
      let projectedElapsedS = previousElapsedS + RESPIRATION_PROJECTION_STEP_S;
      projectedElapsedS < elapsedS;
      projectedElapsedS += RESPIRATION_PROJECTION_STEP_S
    ) {
      const fraction = (projectedElapsedS - previousElapsedS) / spanS
      const projectedHeartRate = previousHeartRate + (heartRate - previousHeartRate) * fraction
      samples.push({
        elapsedS: projectedElapsedS + startOffsetS,
        value: projectedRespiration(model, projectedHeartRate),
      })
    }
    samples.push({
      elapsedS: elapsedS + startOffsetS,
      value: projectedRespiration(model, heartRate),
    })
    previousElapsedS = elapsedS
    previousHeartRate = heartRate
  }
  return samples
}

interface ActivityMetricSamples {
  rightBalance: TimedMetricSample[]
  stamina: TimedMetricSample[]
  potentialStamina: TimedMetricSample[]
  staminaTrace: ActivityStaminaTrace | null
  respiration: TimedMetricSample[]
  performanceCondition: TimedNullableMetricSample[]
  strideLengthCm: TimedNullableMetricSample[]
  verticalRatioPct: TimedNullableMetricSample[]
  verticalOscillationCm: TimedNullableMetricSample[]
  groundContactBalanceLeftPct: TimedNullableMetricSample[]
  groundContactTimeMs: TimedNullableMetricSample[]
  stepSpeedLossMps: TimedNullableMetricSample[]
  stepSpeedLossPct: TimedNullableMetricSample[]
  impactLoadFactor: TimedNullableMetricSample[]
  muscleOxygenPercent: TimedMetricSample[]
  heatStrainIndex: TimedNullableMetricSample[]
  coreTemperatureC: TimedNullableMetricSample[]
  skinTemperatureC: TimedNullableMetricSample[]
}

function timedMetricAt(samples: TimedMetricSample[], elapsedS: number): number | null {
  if (samples.length === 0) return null
  if (elapsedS <= samples[0].elapsedS) return samples[0].value
  for (let index = 1; index < samples.length; index++) {
    const previous = samples[index - 1]
    const next = samples[index]
    if (elapsedS > next.elapsedS) continue
    const span = next.elapsedS - previous.elapsedS
    if (span <= 0) return next.value
    const fraction = (elapsedS - previous.elapsedS) / span
    return previous.value + (next.value - previous.value) * fraction
  }
  return samples[samples.length - 1].value
}

const THERMAL_SAMPLE_MAX_DISTANCE_S = 90

function timedNullableMetricAt(
  samples: TimedNullableMetricSample[],
  elapsedS: number,
  maxDistanceS: number,
): number | null {
  if (samples.length === 0) return null
  let low = 0
  let high = samples.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (samples[middle].elapsedS < elapsedS) low = middle + 1
    else high = middle
  }
  const previous = samples[low - 1]
  const next = samples[low]
  if (next?.elapsedS === elapsedS) return next.value
  if (!previous && !next) return null
  const nearest =
    previous && next
      ? elapsedS - previous.elapsedS <= next.elapsedS - elapsedS
        ? previous
        : next
      : (previous ?? next)
  if (!nearest || Math.abs(nearest.elapsedS - elapsedS) > maxDistanceS) return null
  if (
    !previous ||
    !next ||
    previous.value == null ||
    next.value == null ||
    next.elapsedS === previous.elapsedS ||
    next.elapsedS - previous.elapsedS > maxDistanceS * 2
  )
    return nearest.value
  const fraction = (elapsedS - previous.elapsedS) / (next.elapsedS - previous.elapsedS)
  return previous.value + (next.value - previous.value) * fraction
}

function nativeThermalAt(
  samples: ActivityMetricSamples,
  elapsedS: number,
): Pick<
  ActivityHeartRateTracePoint,
  | 'heatStrainIndex'
  | 'heatStrainSource'
  | 'coreTemperatureC'
  | 'coreTemperatureSource'
  | 'skinTemperatureC'
  | 'skinTemperatureSource'
> {
  const heatStrainIndex = timedNullableMetricAt(
    samples.heatStrainIndex,
    elapsedS,
    THERMAL_SAMPLE_MAX_DISTANCE_S,
  )
  const coreTemperatureC = timedNullableMetricAt(
    samples.coreTemperatureC,
    elapsedS,
    THERMAL_SAMPLE_MAX_DISTANCE_S,
  )
  const skinTemperatureC = timedNullableMetricAt(
    samples.skinTemperatureC,
    elapsedS,
    THERMAL_SAMPLE_MAX_DISTANCE_S,
  )
  return {
    heatStrainIndex: heatStrainIndex == null ? null : round(heatStrainIndex, 1),
    heatStrainSource: heatStrainIndex == null ? null : 'core-fit',
    coreTemperatureC: coreTemperatureC == null ? null : round(coreTemperatureC, 2),
    coreTemperatureSource: coreTemperatureC == null ? null : 'core-fit',
    skinTemperatureC: skinTemperatureC == null ? null : round(skinTemperatureC, 2),
    skinTemperatureSource: skinTemperatureC == null ? null : 'core-fit',
  }
}

const RUNNING_DYNAMICS_SAMPLE_MAX_DISTANCE_S = 15
const PERFORMANCE_CONDITION_SAMPLE_MAX_DISTANCE_S = 15

function performanceConditionAt(
  samples: TimedNullableMetricSample[],
  elapsedS: number,
): number | null {
  const value = timedNullableMetricAt(
    samples,
    elapsedS,
    PERFORMANCE_CONDITION_SAMPLE_MAX_DISTANCE_S,
  )
  return value == null ? null : round(value, 1)
}

function estimatedPerformanceConditionAt(
  samples: CyclingPerformanceConditionSample[],
  elapsedS: number,
): number | null {
  if (samples.length === 0 || elapsedS < samples[0].elapsedS) return null
  const value = timedMetricAt(samples, elapsedS)
  return value == null ? null : round(value, 1)
}

function nativeRunDynamicsAt(
  samples: ActivityMetricSamples,
  elapsedS: number,
): Pick<
  StravaRoutePoint,
  | 'strideLengthM'
  | 'verticalRatioPct'
  | 'verticalOscillationCm'
  | 'groundContactBalanceLeftPct'
  | 'groundContactTimeMs'
  | 'stepSpeedLossMps'
  | 'stepSpeedLossPct'
  | 'impactLoadFactor'
> {
  const at = (values: TimedNullableMetricSample[]): number | null =>
    timedNullableMetricAt(values, elapsedS, RUNNING_DYNAMICS_SAMPLE_MAX_DISTANCE_S)
  const strideLengthCm = at(samples.strideLengthCm)
  const verticalRatioPct = at(samples.verticalRatioPct)
  const verticalOscillationCm = at(samples.verticalOscillationCm)
  const groundContactBalanceLeftPct = at(samples.groundContactBalanceLeftPct)
  const groundContactTimeMs = at(samples.groundContactTimeMs)
  const stepSpeedLossMps = at(samples.stepSpeedLossMps)
  const stepSpeedLossPct = at(samples.stepSpeedLossPct)
  const impactLoadFactor = at(samples.impactLoadFactor)
  return {
    strideLengthM: strideLengthCm == null ? null : round(strideLengthCm / 100, 3),
    verticalRatioPct: verticalRatioPct == null ? null : round(verticalRatioPct, 1),
    verticalOscillationCm: verticalOscillationCm == null ? null : round(verticalOscillationCm, 1),
    groundContactBalanceLeftPct:
      groundContactBalanceLeftPct == null ? null : round(groundContactBalanceLeftPct, 1),
    groundContactTimeMs: groundContactTimeMs == null ? null : round(groundContactTimeMs, 1),
    stepSpeedLossMps: stepSpeedLossMps == null ? null : round(stepSpeedLossMps, 4),
    stepSpeedLossPct: stepSpeedLossPct == null ? null : round(stepSpeedLossPct, 2),
    impactLoadFactor: impactLoadFactor == null ? null : round(impactLoadFactor, 2),
  }
}

type GarminMetricStreamKey =
  | 'rightBalance'
  | 'stamina'
  | 'potentialStamina'
  | 'respiration'
  | 'muscleOxygenPercent'
  | 'heatStrainIndex'
  | 'coreTemperatureC'
  | 'skinTemperatureC'

type GarminNullableMetricStreamKey =
  | 'performanceCondition'
  | 'strideLengthCm'
  | 'verticalRatioPct'
  | 'verticalOscillationCm'
  | 'groundContactBalanceLeftPct'
  | 'groundContactTimeMs'
  | 'stepSpeedLossMps'
  | 'stepSpeedLossPct'
  | 'impactLoadFactor'

function activityGarminMetricSamples(
  activity: RawStravaActivity,
  match: GarminActivityMatch | null,
  garmin: GarminCache | null,
): ActivityMetricSamples {
  const empty = (): ActivityMetricSamples => ({
    rightBalance: [],
    stamina: [],
    potentialStamina: [],
    staminaTrace: null,
    respiration: [],
    performanceCondition: [],
    strideLengthCm: [],
    verticalRatioPct: [],
    verticalOscillationCm: [],
    groundContactBalanceLeftPct: [],
    groundContactTimeMs: [],
    stepSpeedLossMps: [],
    stepSpeedLossPct: [],
    impactLoadFactor: [],
    muscleOxygenPercent: [],
    heatStrainIndex: [],
    coreTemperatureC: [],
    skinTemperatureC: [],
  })
  if (!match) return empty()
  const stream = garmin?.streams?.[match.activity.id]
  const time = stream?.time
  if (!time) return empty()
  const startOffsetS =
    (Date.parse(match.activity.startDate) - Date.parse(activity.startDate)) / 1000
  if (!Number.isFinite(startOffsetS)) return empty()

  const collect = (
    key: GarminMetricStreamKey,
    valid: (value: number) => boolean,
  ): TimedMetricSample[] => {
    const values = stream?.[key]
    if (!values || time.length !== values.length) return []
    const samples: TimedMetricSample[] = []
    for (let index = 0; index < time.length; index++) {
      if (!Number.isFinite(time[index]) || !Number.isFinite(values[index])) continue
      if (!valid(values[index])) continue
      samples.push({ elapsedS: time[index] + startOffsetS, value: values[index] })
    }
    return samples.sort((left, right) => left.elapsedS - right.elapsedS)
  }

  const collectNullable = (
    key:
      | GarminNullableMetricStreamKey
      | 'heatStrainIndex'
      | 'coreTemperatureC'
      | 'skinTemperatureC',
    valid: (value: number) => boolean,
  ): TimedNullableMetricSample[] => {
    const values = stream?.[key]
    if (!values || time.length !== values.length) return []
    const samples: TimedNullableMetricSample[] = []
    for (let index = 0; index < time.length; index++) {
      if (!Number.isFinite(time[index])) continue
      const value = values[index]
      samples.push({
        elapsedS: time[index] + startOffsetS,
        value: typeof value === 'number' && Number.isFinite(value) && valid(value) ? value : null,
      })
    }
    return samples.sort((left, right) => left.elapsedS - right.elapsedS)
  }

  const stamina = collect('stamina', value => value >= 0 && value <= 100)
  const potentialStamina = collect('potentialStamina', value => value >= 0 && value <= 100)
  return {
    rightBalance: collect('rightBalance', value => value >= 0 && value <= 100),
    stamina,
    potentialStamina,
    staminaTrace:
      stamina.length >= 2 && potentialStamina.length >= 2
        ? { source: 'garmin', method: 'garmin-native', ftpWatts: null, maxHeartRateBpm: null }
        : null,
    respiration: collect('respiration', value => value > 0),
    performanceCondition: collectNullable(
      'performanceCondition',
      value => value >= -20 && value <= 20,
    ),
    strideLengthCm: collectNullable('strideLengthCm', value => value >= 20 && value <= 300),
    verticalRatioPct: collectNullable('verticalRatioPct', value => value >= 0 && value <= 50),
    verticalOscillationCm: collectNullable(
      'verticalOscillationCm',
      value => value >= 1 && value <= 30,
    ),
    groundContactBalanceLeftPct: collectNullable(
      'groundContactBalanceLeftPct',
      value => value >= 0 && value <= 100,
    ),
    groundContactTimeMs: collectNullable(
      'groundContactTimeMs',
      value => value >= 50 && value <= 1_000,
    ),
    stepSpeedLossMps: collectNullable('stepSpeedLossMps', value => value >= 0 && value <= 5),
    stepSpeedLossPct: collectNullable('stepSpeedLossPct', value => value >= 0 && value <= 100),
    impactLoadFactor: collectNullable('impactLoadFactor', value => value >= 0 && value <= 10),
    muscleOxygenPercent: collect('muscleOxygenPercent', value => value >= 0 && value <= 100),
    heatStrainIndex: collectNullable('heatStrainIndex', value => value >= 0 && value <= 20),
    coreTemperatureC: collectNullable('coreTemperatureC', value => value >= 25 && value <= 45),
    skinTemperatureC: collectNullable('skinTemperatureC', value => value >= 0 && value <= 50),
  }
}

function activityMetricSamples(
  activity: RawStravaActivity,
  sport: ActivityKind,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
  wahooMatch: WahooActivityMatch | null,
  wahoo: WahooCache | null,
  wahooStamina: CyclingStaminaEstimate | null,
  respirationProjection: RespirationProjectionModel | null,
): ActivityMetricSamples {
  const samples = activityGarminMetricSamples(activity, garminMatch, garmin)
  if (!wahooMatch) return samples
  const stream = wahoo?.streams[wahooMatch.activity.id]
  if (!stream || stream.time.length === 0) return samples
  const startOffsetS =
    (Date.parse(wahooMatch.activity.startDate) - Date.parse(activity.startDate)) / 1000
  if (!Number.isFinite(startOffsetS)) return samples
  const collect = (
    values: readonly (number | null)[],
    valid: (value: number) => boolean,
  ): TimedMetricSample[] => {
    if (values.length !== stream.time.length) return []
    const collected: TimedMetricSample[] = []
    for (let index = 0; index < stream.time.length; index++) {
      const elapsedS = stream.time[index]
      const value = values[index]
      if (!Number.isFinite(elapsedS) || typeof value !== 'number' || !Number.isFinite(value))
        continue
      if (!valid(value)) continue
      collected.push({ elapsedS: elapsedS + startOffsetS, value })
    }
    return collected.sort((left, right) => left.elapsedS - right.elapsedS)
  }
  const collectThermal = (
    values: readonly (number | null)[],
    valid: (value: number) => boolean,
  ): TimedNullableMetricSample[] => {
    if (values.length !== stream.time.length) return []
    const collected: TimedNullableMetricSample[] = []
    for (let index = 0; index < stream.time.length; index++) {
      const elapsedS = stream.time[index]
      const value = values[index]
      if (!Number.isFinite(elapsedS)) continue
      collected.push({
        elapsedS: elapsedS + startOffsetS,
        value: typeof value === 'number' && Number.isFinite(value) && valid(value) ? value : null,
      })
    }
    return collected.sort((left, right) => left.elapsedS - right.elapsedS)
  }
  const directWahooRespiration = collect(stream.respiration, value => value > 0)
  const wahooRespiration =
    directWahooRespiration.length > 0 || sport !== 'bike' || !respirationProjection
      ? directWahooRespiration
      : projectWahooRespiration(stream, startOffsetS, respirationProjection)
  const estimatedStamina =
    samples.staminaTrace == null && wahooStamina
      ? {
          stamina: wahooStamina.samples.map(sample => ({
            elapsedS: sample.elapsedS + startOffsetS,
            value: sample.stamina,
          })),
          potentialStamina: wahooStamina.samples.map(sample => ({
            elapsedS: sample.elapsedS + startOffsetS,
            value: sample.potentialStamina,
          })),
          staminaTrace: {
            source: 'garden-estimate',
            method: wahooStamina.method,
            ftpWatts: wahooStamina.ftpWatts,
            maxHeartRateBpm: wahooStamina.maxHeartRateBpm,
          } satisfies ActivityStaminaTrace,
        }
      : null
  return {
    ...samples,
    stamina: estimatedStamina?.stamina ?? samples.stamina,
    potentialStamina: estimatedStamina?.potentialStamina ?? samples.potentialStamina,
    staminaTrace: estimatedStamina?.staminaTrace ?? samples.staminaTrace,
    rightBalance:
      samples.rightBalance.length > 0
        ? samples.rightBalance
        : collect(stream.rightBalance, value => value >= 0 && value <= 100),
    respiration: samples.respiration.length > 0 ? samples.respiration : wahooRespiration,
    muscleOxygenPercent:
      samples.muscleOxygenPercent.length > 0
        ? samples.muscleOxygenPercent
        : collect(stream.muscleOxygenPercent, value => value >= 0 && value <= 100),
    heatStrainIndex: samples.heatStrainIndex.some(sample => sample.value != null)
      ? samples.heatStrainIndex
      : collectThermal(stream.heatStrainIndex, value => value >= 0 && value <= 20),
    coreTemperatureC: samples.coreTemperatureC.some(sample => sample.value != null)
      ? samples.coreTemperatureC
      : collectThermal(stream.coreTemperatureC, value => value >= 25 && value <= 45),
    skinTemperatureC: samples.skinTemperatureC.some(sample => sample.value != null)
      ? samples.skinTemperatureC
      : collectThermal(stream.skinTemperatureC, value => value >= 0 && value <= 50),
  }
}

interface TimedStreamAlignment {
  streams: StravaStreams | GarminStreams
  time: number[]
  distance: number[]
}

interface ProjectedAnalysis {
  ranges: ActivityAnalysisRange[]
  boundaryIndices: number[]
}

function timedStreamAlignment(
  streams: StravaStreams | GarminStreams | undefined,
): TimedStreamAlignment | null {
  if (!streams || !('time' in streams) || !streams.time) return null
  const { time, distance } = streams
  if (time.length < 2 || distance.length !== time.length) return null
  let previousTime = -Infinity
  let previousDistance = 0
  const alignedDistance: number[] = []
  for (let index = 0; index < time.length; index++) {
    const elapsedS = time[index]
    const distanceM = distance[index]
    if (!Number.isFinite(elapsedS) || elapsedS < previousTime || !Number.isFinite(distanceM))
      return null
    previousTime = elapsedS
    previousDistance = Math.max(previousDistance, distanceM)
    alignedDistance.push(previousDistance)
  }
  return { streams, time, distance: alignedDistance }
}

function wahooTimedStreamAlignment(streams: WahooStreams | undefined): TimedStreamAlignment | null {
  if (!streams || streams.time.length < 2 || streams.distance.length !== streams.time.length)
    return null
  const distance: number[] = []
  let previousDistance = 0
  for (const value of streams.distance) {
    if (typeof value === 'number' && Number.isFinite(value))
      previousDistance = Math.max(previousDistance, value)
    distance.push(previousDistance)
  }
  const normalized: GarminStreams = { time: [...streams.time], latlng: [], altitude: [], distance }
  return timedStreamAlignment(normalized)
}

function nativeThermalTimeline(samples: ActivityMetricSamples): number[] {
  const elapsed = new Set<number>()
  for (const sample of [
    ...samples.heatStrainIndex,
    ...samples.coreTemperatureC,
    ...samples.skinTemperatureC,
  ])
    elapsed.add(sample.elapsedS)
  return [...elapsed].sort((left, right) => left - right)
}

function projectRouteLessHeartRateTrace(
  sport: ActivityKind,
  streams: StravaStreams | GarminStreams | undefined,
  heartRate: ActivityHeartRate,
  metricSamples: ActivityMetricSamples,
): ActivityHeartRateTracePoint[] {
  const time = streams?.time?.length ? streams.time : nativeThermalTimeline(metricSamples)
  if (time.length < 2) return []
  const values =
    heartRate.stream.length === time.length
      ? heartRate.stream
      : Array.from({ length: time.length }, () => 0)
  let previousTime = -Infinity
  for (const elapsedS of time) {
    if (!Number.isFinite(elapsedS) || elapsedS < 0 || elapsedS < previousTime) return []
    previousTime = elapsedS
  }
  if (time[time.length - 1] <= time[0]) return []
  const swimAlignment = sport === 'swim' ? timedStreamAlignment(streams) : null
  if (sport === 'swim' && !swimAlignment) return []
  const available = values.map(value => Number.isFinite(value) && value > 0)
  const hasThermal =
    metricSamples.heatStrainIndex.some(sample => sample.value != null) ||
    metricSamples.coreTemperatureC.some(sample => sample.value != null) ||
    metricSamples.skinTemperatureC.some(sample => sample.value != null)
  if (available.filter(Boolean).length < 2 && !hasThermal) return []
  const required: number[] = []
  let peakIndex = -1
  for (let index = 1; index < available.length; index++)
    if (available[index] !== available[index - 1]) required.push(index - 1, index)
  for (let index = 0; index < values.length; index++)
    if (available[index] && (peakIndex < 0 || values[index] > values[peakIndex])) peakIndex = index
  if (peakIndex >= 0) required.push(peakIndex)
  return sampleIndicesWithRequired(0, values.length - 1, ROUTE_POINTS, required).map(index => {
    const elapsedS = time[index]
    return {
      distanceKm: swimAlignment ? round(swimAlignment.distance[index] / 1_000, 3) : 0,
      elapsedS: round(elapsedS, 3),
      heartRate: available[index] ? Math.round(values[index]) : null,
      ...nativeThermalAt(metricSamples, elapsedS),
    }
  })
}

function alignedDistanceAt(alignment: TimedStreamAlignment, elapsedS: number): number {
  const { time, distance } = alignment
  if (elapsedS <= time[0]) return distance[0]
  if (elapsedS >= time[time.length - 1]) return distance[distance.length - 1]
  let low = 0
  let high = time.length - 1
  while (low + 1 < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (time[middle] <= elapsedS) low = middle
    else high = middle
  }
  const span = time[high] - time[low]
  if (span <= 0) return distance[high]
  const fraction = (elapsedS - time[low]) / span
  return distance[low] + (distance[high] - distance[low]) * fraction
}

function projectedGearShifts(
  activity: RawStravaActivity,
  sourceStartDate: string,
  shifts: readonly GarminGearShift[],
  alignment: TimedStreamAlignment | null,
): ActivityGearShift[] {
  const activityStartMs = Date.parse(activity.startDate)
  const sourceStartMs = Date.parse(sourceStartDate)
  if (!Number.isFinite(activityStartMs) || !Number.isFinite(sourceStartMs)) return []
  const durationS = Math.max(activity.elapsedTime, activity.movingTime, 1)
  return shifts.flatMap(shift => {
    const timestampMs = Date.parse(shift.timestamp)
    if (!Number.isFinite(timestampMs)) return []
    const elapsedS = (timestampMs - activityStartMs) / 1000
    const sourceElapsedS = (timestampMs - sourceStartMs) / 1000
    if (sourceElapsedS < -60 || sourceElapsedS > durationS + 60) return []
    const distanceM = alignment
      ? alignedDistanceAt(alignment, sourceElapsedS)
      : (Math.min(durationS, Math.max(0, elapsedS)) / durationS) * activity.distance
    return [
      {
        elapsedS: round(Math.min(durationS, Math.max(0, elapsedS)), 3),
        distanceKm: round(Math.min(activity.distance, Math.max(0, distanceM)) / 1_000, 3),
        frontGearNum: shift.frontGearNum,
        frontTeeth: shift.frontTeeth,
        rearGearNum: shift.rearGearNum,
        rearTeeth: shift.rearTeeth,
      },
    ]
  })
}

function activityGearShifts(
  activity: RawStravaActivity,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
  wahooMatch: WahooActivityMatch | null,
  wahoo: WahooCache | null,
): ActivityGearShift[] {
  if (garminMatch) {
    const projected = projectedGearShifts(
      activity,
      garminMatch.activity.startDate,
      garmin?.gearShifts?.[garminMatch.activity.id] ?? [],
      timedStreamAlignment(garmin?.streams?.[garminMatch.activity.id]),
    )
    if (projected.length > 0) return projected
  }
  if (!wahooMatch) return []
  return projectedGearShifts(
    activity,
    wahooMatch.activity.startDate,
    wahoo?.gearShifts[wahooMatch.activity.id] ?? [],
    wahooTimedStreamAlignment(wahoo?.streams[wahooMatch.activity.id]),
  )
}

const cyclingDynamicsMetricKeys = [
  'leftPedalSmoothness',
  'rightPedalSmoothness',
  'leftTorqueEffectiveness',
  'rightTorqueEffectiveness',
  'leftPowerPhaseStart',
  'leftPowerPhaseEnd',
  'rightPowerPhaseStart',
  'rightPowerPhaseEnd',
] as const

function validCyclingDynamicsSamples(
  dynamics: GarminCyclingDynamics | WahooCyclingDynamics,
): boolean {
  const length = dynamics.time.length
  return (
    length >= 2 &&
    dynamics.distance.length === length &&
    cyclingDynamicsMetricKeys.every(key => dynamics[key].length === length)
  )
}

function projectCyclingDynamics(
  activity: RawStravaActivity,
  sourceStartDate: string,
  dynamics: GarminCyclingDynamics | WahooCyclingDynamics,
): ActivityCyclingDynamics | null {
  const startOffsetS = (Date.parse(sourceStartDate) - Date.parse(activity.startDate)) / 1000
  if (!Number.isFinite(startOffsetS)) return null
  const hasSamples = validCyclingDynamicsSamples(dynamics)
  const maxDistanceM = Math.max(activity.distance, 0)
  const metric = (key: (typeof cyclingDynamicsMetricKeys)[number]): (number | null)[] =>
    hasSamples
      ? dynamics[key].map(value =>
          value == null || !Number.isFinite(value) ? null : round(value, 1),
        )
      : []
  const positionChanges = dynamics.positionChanges
    .map(change => ({
      elapsedS: round(Math.max(0, change.elapsedS + startOffsetS), 3),
      distanceKm: round(Math.min(maxDistanceM, Math.max(0, change.distanceM)) / 1_000, 3),
      position: change.position,
    }))
    .filter(
      (change, index, changes) => index === 0 || changes[index - 1].position !== change.position,
    )
  if (!hasSamples && positionChanges.length === 0) return null
  return {
    elapsedS: hasSamples
      ? dynamics.time.map(value => round(Math.max(0, value + startOffsetS), 3))
      : [],
    distanceKm: hasSamples
      ? dynamics.distance.map(value => round(Math.min(maxDistanceM, Math.max(0, value)) / 1_000, 3))
      : [],
    leftPedalSmoothness: metric('leftPedalSmoothness'),
    rightPedalSmoothness: metric('rightPedalSmoothness'),
    leftTorqueEffectiveness: metric('leftTorqueEffectiveness'),
    rightTorqueEffectiveness: metric('rightTorqueEffectiveness'),
    leftPowerPhaseStart: metric('leftPowerPhaseStart'),
    leftPowerPhaseEnd: metric('leftPowerPhaseEnd'),
    rightPowerPhaseStart: metric('rightPowerPhaseStart'),
    rightPowerPhaseEnd: metric('rightPowerPhaseEnd'),
    positionChanges,
    seatedTimeS:
      dynamics.seatedTimeS == null || !Number.isFinite(dynamics.seatedTimeS)
        ? null
        : round(dynamics.seatedTimeS, 3),
    standingTimeS:
      dynamics.standingTimeS == null || !Number.isFinite(dynamics.standingTimeS)
        ? null
        : round(dynamics.standingTimeS, 3),
  }
}

function cyclingDynamicsScore(dynamics: ActivityCyclingDynamics | null): number {
  if (!dynamics) return 0
  return (
    cyclingDynamicsMetricKeys.reduce(
      (total, key) => total + dynamics[key].filter(value => value != null).length,
      0,
    ) + dynamics.positionChanges.length
  )
}

function activityCyclingDynamics(
  activity: RawStravaActivity,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
  wahooMatch: WahooActivityMatch | null,
  wahoo: WahooCache | null,
): ActivityCyclingDynamics | null {
  const garminDynamics = garminMatch
    ? garmin?.cyclingDynamics?.[garminMatch.activity.id]
    : undefined
  const wahooDynamics = wahooMatch ? wahoo?.cyclingDynamics[wahooMatch.activity.id] : undefined
  const projectedGarmin = garminDynamics
    ? projectCyclingDynamics(activity, garminMatch?.activity.startDate ?? '', garminDynamics)
    : null
  const projectedWahoo = wahooDynamics
    ? projectCyclingDynamics(activity, wahooMatch?.activity.startDate ?? '', wahooDynamics)
    : null
  return cyclingDynamicsScore(projectedWahoo) > cyclingDynamicsScore(projectedGarmin)
    ? projectedWahoo
    : projectedGarmin
}

function activityRunWalk(
  sport: ActivityKind,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
): GarminRunWalkData | null {
  if (sport !== 'run' || !garminMatch) return null
  return garmin?.runWalks?.[garminMatch.activity.id] ?? null
}

function nearestElapsedIndex(time: number[], elapsedS: number): number {
  let lo = 0
  let hi = time.length - 1
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2)
    if (time[mid] < elapsedS) lo = mid + 1
    else hi = mid
  }
  if (lo === 0) return 0
  return Math.abs(time[lo] - elapsedS) < Math.abs(time[lo - 1] - elapsedS) ? lo : lo - 1
}

function validStreamIndex(value: number | null, length: number): value is number {
  return value != null && Number.isInteger(value) && value >= 0 && value < length
}

function rangeIndices(
  raw: RawStravaAnalysisRange,
  alignment: TimedStreamAlignment,
  activityStartMs: number,
  fallbackStartElapsedS: number | null,
): [number, number] | null {
  if (
    validStreamIndex(raw.startIndex, alignment.time.length) &&
    validStreamIndex(raw.endIndex, alignment.time.length) &&
    raw.endIndex >= raw.startIndex
  )
    return [raw.startIndex, raw.endIndex]

  const startMs = raw.startDate ? Date.parse(raw.startDate) : NaN
  const startElapsedS = Number.isFinite(startMs)
    ? (startMs - activityStartMs) / 1000
    : fallbackStartElapsedS
  if (startElapsedS == null || !Number.isFinite(startElapsedS)) return null
  const endElapsedS = startElapsedS + raw.elapsedTime
  return [
    nearestElapsedIndex(alignment.time, startElapsedS),
    nearestElapsedIndex(alignment.time, endElapsedS),
  ]
}

function nullableRound(value: number | null, dp = 1): number | null {
  return value == null || !Number.isFinite(value) ? null : round(value, dp)
}

function projectStravaAnalysisRange(
  kind: 'lap' | 'segment',
  raw: RawStravaAnalysisRange,
  index: number,
  bounds: [number, number],
  alignment: TimedStreamAlignment,
): ActivityAnalysisRange {
  const [startIndex, endIndex] = bounds
  return {
    kind,
    id: `${kind}:${raw.id}`,
    label: raw.name || `${kind === 'lap' ? 'Lap' : 'Segment'} ${index + 1}`,
    startElapsedS: round(alignment.time[startIndex], 3),
    endElapsedS: round(alignment.time[endIndex], 3),
    startDistanceKm: round(alignment.distance[startIndex] / 1000, 3),
    endDistanceKm: round(alignment.distance[endIndex] / 1000, 3),
    durationS: raw.elapsedTime,
    movingTimeS: raw.movingTime,
    distanceKm: round(raw.distance / 1000, 3),
    elevationGainM: nullableRound(raw.totalElevationGain),
    averageSpeedKph: nullableRound(raw.averageSpeed == null ? null : raw.averageSpeed * 3.6, 2),
    averageHeartRate: nullableRound(raw.averageHeartrate),
    averageWatts: nullableRound(raw.averageWatts),
    averageCadence: nullableRound(raw.averageCadence),
  }
}

function projectRunSplits(splits: RawStravaRunSplit[] | undefined): ActivityRunSplit[] {
  return (splits ?? []).map(split => ({
    split: split.split,
    distanceKm: round(split.distance / 1000, 3),
    elapsedTimeS: split.elapsedTime,
    movingTimeS: split.movingTime,
    averageSpeedKph: round(split.averageSpeed * 3.6, 2),
    elevationDifferenceM: nullableRound(split.elevationDifference),
    paceZone: nullableRound(split.paceZone, 0),
  }))
}

function projectAnalysisRanges(
  a: RawStravaActivity,
  detail: RawStravaActivityDetail | undefined,
  streams: StravaStreams | GarminStreams | undefined,
  climbs: ActivityClimbSegment[],
): ProjectedAnalysis {
  const alignment = timedStreamAlignment(streams)
  if (!alignment) return { ranges: [], boundaryIndices: [] }
  const activityStartMs = Date.parse(a.startDate)
  if (!Number.isFinite(activityStartMs)) return { ranges: [], boundaryIndices: [] }
  const ranges: ActivityAnalysisRange[] = []
  const boundaryIndices: number[] = []
  let lapElapsedS = 0

  for (const [index, raw] of (detail?.laps ?? []).entries()) {
    const bounds = rangeIndices(raw, alignment, activityStartMs, lapElapsedS)
    lapElapsedS += raw.elapsedTime
    if (!bounds || bounds[1] < bounds[0]) continue
    ranges.push(projectStravaAnalysisRange('lap', raw, index, bounds, alignment))
    boundaryIndices.push(...bounds)
  }
  for (const [index, raw] of (detail?.segmentEfforts ?? []).entries()) {
    const bounds = rangeIndices(raw, alignment, activityStartMs, null)
    if (!bounds || bounds[1] < bounds[0]) continue
    ranges.push(projectStravaAnalysisRange('segment', raw, index, bounds, alignment))
    boundaryIndices.push(...bounds)
  }

  const activityEndElapsedS = alignment.time[alignment.time.length - 1]
  for (const [index, climb] of climbs.entries()) {
    const startMs = Date.parse(climb.startDate)
    const endMs = Date.parse(climb.endDate)
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) continue
    const unclampedStartS = (startMs - activityStartMs) / 1000
    const unclampedEndS = (endMs - activityStartMs) / 1000
    if (unclampedEndS < 0 || unclampedStartS > activityEndElapsedS) continue
    const startIndex = nearestElapsedIndex(
      alignment.time,
      Math.max(0, Math.min(activityEndElapsedS, unclampedStartS)),
    )
    const endIndex = nearestElapsedIndex(
      alignment.time,
      Math.max(0, Math.min(activityEndElapsedS, unclampedEndS)),
    )
    if (endIndex < startIndex) continue
    ranges.push({
      kind: 'climb',
      source: climb.source,
      id: `${climb.source}:${index + 1}:${climb.startDate}`,
      label: climb.name,
      startElapsedS: round(alignment.time[startIndex], 3),
      endElapsedS: round(alignment.time[endIndex], 3),
      startDistanceKm: round(alignment.distance[startIndex] / 1000, 3),
      endDistanceKm: round(alignment.distance[endIndex] / 1000, 3),
      durationS: climb.durationS,
      distanceKm: round(climb.distanceM / 1000, 3),
      elevationGainM: nullableRound(climb.elevationGainM),
      averageSpeedKph: nullableRound(climb.avgSpeedMps == null ? null : climb.avgSpeedMps * 3.6, 2),
      averageHeartRate: nullableRound(climb.avgHeartRate),
      averageWatts: nullableRound(climb.avgPower),
      averageCadence: nullableRound(climb.avgCadence),
    })
    boundaryIndices.push(startIndex, endIndex)
  }

  const order: Record<ActivityAnalysisKind, number> = { lap: 0, segment: 1, climb: 2 }
  if (a.sportType.startsWith('Virtual'))
    for (const range of ranges) {
      if (range.kind === 'climb') continue
      range.distanceKm = round(Math.max(0, range.endDistanceKm - range.startDistanceKm), 3)
      const durationS = range.movingTimeS ?? range.durationS
      range.averageSpeedKph =
        durationS > 0 ? round((range.distanceKm * 3_600) / durationS, 2) : null
      range.elevationGainM = null
    }
  ranges.sort(
    (left, right) =>
      left.startElapsedS - right.startElapsedS ||
      order[left.kind] - order[right.kind] ||
      left.id.localeCompare(right.id),
  )
  return { ranges, boundaryIndices }
}

function routeElapsedSeconds(
  streams: StravaStreams | GarminStreams,
  movingTimeS: number,
): number[] | null {
  const timed = timedStreamAlignment(streams)
  if (timed && timed.time.length === streams.latlng.length) return timed.time
  if (
    streams.latlng.length >= 2 &&
    Math.abs(streams.latlng.length - movingTimeS) / Math.max(1, movingTimeS) <= 0.15
  )
    return Array.from({ length: streams.latlng.length }, (_, index) => index)
  return null
}

function localSpeedKph(time: number[], distance: number[]): number[] {
  const speedKph = Array.from({ length: time.length }, () => 0)
  const monotonicDistance: number[] = []
  let previousDistance = 0
  for (const value of distance) {
    previousDistance = Math.max(previousDistance, Number.isFinite(value) ? value : 0)
    monotonicDistance.push(previousDistance)
  }
  let windowStart = 0
  let previousMps = 0
  for (let index = 1; index < time.length; index++) {
    while (windowStart + 1 < index && time[index] - time[windowStart] > MAX_SPEED_WINDOW_S)
      windowStart++
    const elapsedS = time[index] - time[windowStart]
    const distanceM = monotonicDistance[index] - monotonicDistance[windowStart]
    const rawMps = elapsedS > 0 && distanceM > 0 ? distanceM / elapsedS : 0
    const sampleElapsedS = Math.max(0, Math.min(1, time[index] - time[index - 1]))
    previousMps = Math.min(rawMps, previousMps + MAX_ACCEL_MPS2 * sampleElapsedS)
    speedKph[index] = round(previousMps * 3.6, 1)
  }
  return speedKph
}

export function privateRouteBounds(
  latlng: [number, number][],
  home: [number, number] | null,
): [number, number] {
  let lo = 0
  let hi = latlng.length - 1
  if (home) {
    while (
      lo < latlng.length &&
      haversineMeters(latlng[lo][0], latlng[lo][1], home[0], home[1]) <= 200
    )
      lo++
    while (hi > lo && haversineMeters(latlng[hi][0], latlng[hi][1], home[0], home[1]) <= 200) hi--
  }
  return hi - lo >= 1 ? [lo, hi] : [0, latlng.length - 1]
}

function garminActivityFueling(match: GarminActivityMatch | null): ActivityFueling | null {
  if (!match || !hasGarminFueling(match.activity.fueling)) return null
  return {
    ...match.activity.fueling,
    sourceDevice: match.activity.fueling.sourceDevice ?? match.activity.sourceDevice,
    sodiumLossMg: null,
    source: 'garmin',
  }
}

function wahooActivityFueling(match: WahooActivityMatch | null): ActivityFueling | null {
  if (!match) return null
  const { fluidMl, sodiumMg } = match.activity.sweatLoss
  if (fluidMl == null && sodiumMg == null) return null
  return {
    ...emptyGarminFueling(match.activity.sourceDevice),
    sweatLossMl: fluidMl,
    sodiumLossMg: sodiumMg,
    source: 'wahoo',
  }
}

function activityFueling(
  garminMatch: GarminActivityMatch | null,
  wahooMatch: WahooActivityMatch | null,
): ActivityFueling | null {
  const garminFueling = garminActivityFueling(garminMatch)
  const wahooFueling = wahooActivityFueling(wahooMatch)
  if (!wahooFueling) return garminFueling
  if (!garminFueling) return wahooFueling
  return {
    ...garminFueling,
    sweatLossMl: wahooFueling.sweatLossMl ?? garminFueling.sweatLossMl,
    sodiumLossMg: wahooFueling.sodiumLossMg,
    source: 'garmin+wahoo',
  }
}

function projectDetail(
  a: RawStravaActivity,
  sport: ActivityKind,
  streams: StravaStreams | GarminStreams | undefined,
  effortStreams: StravaStreams | GarminStreams | undefined,
  heartRate: ActivityHeartRate,
  metricSamples: ActivityMetricSamples,
  gearShifts: ActivityGearShift[],
  cyclingDynamics: ActivityCyclingDynamics | null,
  runWalk: GarminRunWalkData | null,
  weather: WeatherCache['activities'][string] | undefined,
  weatherAttribution: WeatherCache['attribution'],
  uvCalibration: WeatherCache['uvCalibration'],
  generatedAt: number,
  geo: string | undefined,
  fueling: ActivityFueling | null,
  garmin: GarminVerification | null,
  computer: ActivityComputer | null,
  device: ActivityDevice | null,
  weight: ActivityWeight | null,
  rawDetail: RawStravaActivityDetail | undefined,
  climbs: ActivityClimbSegment[],
  hrBounds: number[],
  powerBounds: number[],
  ftpWatts: number | null,
  lactateThresholdHeartRateBpm: number | null,
  restingHeartRateBpm: number | null,
  anaerobicPowerModel: Pick<CriticalPowerEstimate, 'criticalPowerWatts' | 'wPrimeJoules'> | null,
  home: [number, number] | null,
  powerCurve: PowerCurvePoint[] | undefined,
  activityCriticalPower: CriticalPowerEstimate | null,
): StravaActivityDetail {
  const route: StravaRoutePoint[] = []
  let mapRoute: StravaMapPoint[][] = []
  const analysis = projectAnalysisRanges(a, rawDetail, effortStreams, climbs)
  let minAlt = 0
  let maxAlt = 0
  let ascentM = 0
  let descentM = 0
  let environment: GardenEnvironmentEstimate | null = null
  let apparentWind: GardenApparentWindEstimate | null = null
  const temperatureSeries = weather?.temperatureSeries ?? []
  const fallbackTemperatureC = weather?.temperatureC ?? a.averageTemp ?? null
  const mapLatlng = streams?.latlng ?? []
  const mapTime = streams
    ? (routeElapsedSeconds(streams, a.movingTime) ??
      Array.from({ length: mapLatlng.length }, (_, index) => index))
    : null
  if (
    mapRoute.length === 0 &&
    streams &&
    mapTime &&
    mapLatlng.length >= 2 &&
    streams.distance.length === mapLatlng.length
  ) {
    const [mapLo, mapHi] = privateRouteBounds(mapLatlng, home)
    mapRoute = rawMapRouteSegments(mapLatlng, streams.distance, mapTime, mapLo, mapHi)
  }
  const timedEffort = timedStreamAlignment(effortStreams)
  const hasEffortPower = effortStreams?.watts?.some(value => value > 0) ?? false
  const timeline =
    sport === 'bike' || hasEffortPower ? effortTimeline(effortStreams, a.movingTime) : null
  const nativePerformanceCondition =
    (sport === 'run' || sport === 'bike') &&
    metricSamples.performanceCondition.filter(sample => sample.value != null).length >= 2
  const estimatedPerformanceCondition =
    sport === 'bike' && !nativePerformanceCondition && timeline
      ? calculateCyclingPerformanceCondition(
          timeline.watts,
          timeline.wattsObserved,
          timeline.heartRate,
          ftpWatts,
          lactateThresholdHeartRateBpm,
          restingHeartRateBpm,
        )
      : null
  const performanceConditionTrace: ActivityPerformanceConditionTrace | null =
    nativePerformanceCondition
      ? { source: 'garmin', method: 'garmin-native' }
      : estimatedPerformanceCondition
        ? {
            source: 'garden-estimate',
            method: estimatedPerformanceCondition.method,
            ftpWatts: estimatedPerformanceCondition.ftpWatts,
            lactateThresholdHeartRateBpm:
              estimatedPerformanceCondition.lactateThresholdHeartRateBpm,
            restingHeartRateBpm: estimatedPerformanceCondition.restingHeartRateBpm,
            windowSeconds: estimatedPerformanceCondition.windowSeconds,
          }
        : null
  const routeStreams =
    timedEffort && timedEffort.streams.latlng.length === timedEffort.time.length
      ? timedEffort.streams
      : streams
  const routeTime = routeStreams ? routeElapsedSeconds(routeStreams, a.movingTime) : null
  const latlng = routeTime ? (routeStreams?.latlng ?? []) : []
  const cadStream = routeStreams?.cadence ?? []
  if (routeStreams && routeTime && latlng.length >= 2) {
    const altitude = cleanAltitude(routeStreams.altitude)
    const distance: number[] = []
    let previousDistance = 0
    for (const value of routeStreams.distance) {
      previousDistance = Math.max(previousDistance, Number.isFinite(value) ? value : 0)
      distance.push(previousDistance)
    }
    const watts = routeStreams.watts ?? []
    const routeHrStream = routeStreams.heartrate ?? []
    const speedKph = localSpeedKph(routeTime, distance)
    if (weather?.activityId === a.id) {
      const result = buildActivityEnvironment({
        activityId: a.id,
        elapsedTimeS: a.elapsedTime > 0 ? a.elapsedTime : a.movingTime,
        movingTimeS: a.movingTime,
        timeS: routeTime,
        distanceM: distance,
        latlng,
        weather,
        attribution: weatherAttribution,
        computedAt: generatedAt,
      })
      environment = result.environment
      apparentWind = result.apparentWind
    }
    let ascent = 0
    let descent = 0
    for (let i = 1; i < altitude.length; i++) {
      const delta = altitude[i] - altitude[i - 1]
      if (delta > 0) ascent += delta
      else descent -= delta
    }
    ascentM = Math.round(ascent)
    descentM = Math.round(descent)
    const [lo0, hi0] = privateRouteBounds(latlng, home)
    const requiredIndices = routeStreams === effortStreams ? analysis.boundaryIndices : []
    const idx = sampleIndicesWithRequired(lo0, hi0, ROUTE_POINTS, requiredIndices)
    let sumLat = 0
    let sumLng = 0
    for (const i of idx) {
      sumLat += latlng[i][0]
      sumLng += latlng[i][1]
    }
    const meanLat = sumLat / idx.length
    const meanLng = sumLng / idx.length
    const cosLat = Math.cos((meanLat * Math.PI) / 180)
    const xs = idx.map(i => (latlng[i][1] - meanLng) * cosLat)
    const ys = idx.map(i => latlng[i][0] - meanLat)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const span = Math.max(maxX - minX, maxY - minY) || 1
    const offX = (1 - (maxX - minX) / span) / 2
    const offY = (1 - (maxY - minY) / span) / 2
    const alts = idx.map(i => altitude[i] ?? 0)
    minAlt = round(Math.min(...alts), 1)
    maxAlt = round(Math.max(...alts), 1)
    idx.forEach((i, k) => {
      const elapsedS = routeTime[i]
      const temperatureC = temperatureAt(temperatureSeries, elapsedS) ?? fallbackTemperatureC
      const rightPowerPct = timedMetricAt(metricSamples.rightBalance, elapsedS)
      const stamina = timedMetricAt(metricSamples.stamina, elapsedS)
      const potentialStamina = timedMetricAt(metricSamples.potentialStamina, elapsedS)
      const respiration = timedMetricAt(metricSamples.respiration, elapsedS)
      const muscleOxygen = timedMetricAt(metricSamples.muscleOxygenPercent, elapsedS)
      const thermal = nativeThermalAt(metricSamples, elapsedS)
      const performanceCondition = estimatedPerformanceCondition
        ? estimatedPerformanceConditionAt(estimatedPerformanceCondition.samples, elapsedS)
        : nativePerformanceCondition
          ? performanceConditionAt(metricSamples.performanceCondition, elapsedS)
          : null
      const runDynamics = nativeRunDynamicsAt(metricSamples, elapsedS)
      route.push({
        x: round((xs[k] - minX) / span + offX, 4),
        y: round((ys[k] - minY) / span + offY, 4),
        d: round((distance[i] ?? 0) / 1000, 3),
        alt: round(alts[k], 1),
        w: Math.round(watts[i] ?? 0),
        hr: Math.round(routeHrStream[i] ?? 0),
        cad: Math.round(cadStream[i] ?? 0),
        ...(rightPowerPct == null ? {} : { rightPowerPct: round(rightPowerPct, 1) }),
        stamina: stamina == null ? null : round(stamina, 1),
        potentialStamina: potentialStamina == null ? null : round(potentialStamina, 1),
        resp: respiration == null ? null : round(respiration, 1),
        muscleOxygenPct: muscleOxygen == null ? null : round(muscleOxygen, 1),
        tempC: temperatureC == null ? null : round(temperatureC, 1),
        ...thermal,
        performanceCondition,
        ...runDynamics,
        lat: round(latlng[i][0], 5),
        lng: round(latlng[i][1], 5),
        elapsedS: round(elapsedS, 3),
        speedKph: speedKph[i] ?? 0,
      })
    })
  }
  const wFull = (a.sportType.startsWith('Virtual') ? effortStreams?.watts : streams?.watts) ?? []
  const hasHr = heartRate.stream.some(v => v > 0)
  const hasW = wFull.some(v => v > 0)
  const avgWattsWithoutZeros = hasW ? avgPos(wFull) : roundPos(garmin?.avgPower)
  const powerWithoutZeros =
    sport === 'bike' && avgWattsWithoutZeros != null
      ? {
          avgWatts: avgWattsWithoutZeros,
          powerZones: hasW && powerBounds.length > 0 ? zoneTimes(wFull, powerBounds, false) : null,
          powerHist: hasW ? powerHistogram(wFull, false) : null,
        }
      : null
  const elapsedTimeline =
    timeline && effortStreams && 'time' in effortStreams && effortStreams.time?.length
      ? timeline
      : null
  return {
    id: a.id,
    virtual: a.sportType.startsWith('Virtual'),
    sport,
    name: a.name,
    date: a.startDateLocal.slice(0, 10),
    start: a.startDate,
    distanceKm: round(a.distance / 1000, sport === 'swim' ? 3 : 1),
    movingTimeS: a.movingTime,
    elapsedTimeS: a.elapsedTime > 0 ? a.elapsedTime : a.movingTime,
    maxSpeedKph: maxSpeedKph(timeline),
    elevationM: ascentM,
    avgHr: heartRate.avgHr,
    maxHr: heartRate.maxHr,
    avgWatts: a.averageWatts != null ? Math.round(a.averageWatts) : null,
    npWatts: a.weightedAverageWatts != null ? Math.round(a.weightedAverageWatts) : null,
    maxWatts: a.maxWatts != null ? Math.round(a.maxWatts) : null,
    kilojoules: a.kilojoules != null ? Math.round(a.kilojoules) : null,
    deviceWatts: a.deviceWatts === true,
    avgCadence:
      a.averageCadence != null
        ? Math.round(a.averageCadence)
        : (avgPos(cadStream) ?? roundPos(garmin?.avgCadence)),
    sufferScore: a.sufferScore != null ? Math.round(a.sufferScore) : null,
    calories:
      a.calories != null
        ? Math.round(a.calories)
        : rawDetail?.calories != null
          ? Math.round(rawDetail.calories)
          : (garmin?.totalCalories ?? null),
    deviceTemperatureC: a.averageTemp != null ? round(a.averageTemp, 1) : null,
    ambientTemperatureC:
      environment?.summary.averageAmbientTemperatureC ?? weather?.temperatureC ?? null,
    windKph: weather?.windKph ?? null,
    windDir: weather?.windDir ?? null,
    windDirDeg: weather?.windDirDeg ?? null,
    windGustKph: weather?.windGustKph ?? null,
    averageRelativeHumidityPct: weather?.averageRelativeHumidityPct ?? null,
    relativeHumidityProvenance: weather?.relativeHumidityProvenance ?? null,
    location: geo ?? null,
    fueling,
    strength: null,
    sauna: null,
    garmin,
    computer,
    device,
    staminaTrace: metricSamples.staminaTrace,
    performanceConditionTrace,
    calculatedIntensityFactor: null,
    calculatedExerciseLoad: null,
    anaerobicPowerEstimate:
      sport === 'bike' && hasEffortPower && timeline
        ? calculateAnaerobicPowerEstimate(timeline.watts, a.movingTime, anaerobicPowerModel)
        : null,
    calculatedTrainingEffect: null,
    gearShifts,
    cyclingDynamics,
    runWalk,
    route,
    heartRateTrace:
      route.length >= 2
        ? []
        : projectRouteLessHeartRateTrace(sport, streams, heartRate, metricSamples),
    mapRoute,
    analysisRanges: analysis.ranges,
    runSplitsMetric: sport === 'run' ? projectRunSplits(rawDetail?.splitsMetric) : [],
    runSplitsStandard: sport === 'run' ? projectRunSplits(rawDetail?.splitsStandard) : [],
    runPaceZones: null,
    minAlt,
    maxAlt,
    descentM,
    hrZones:
      hasHr && hrBounds.length > 0
        ? durationZoneTimes(heartRate.stream, hrBounds, a.movingTime)
        : null,
    powerZones: hasW && powerBounds.length > 0 ? zoneTimes(wFull, powerBounds, true) : null,
    powerHist: hasW ? powerHistogram(wFull) : null,
    powerWithoutZeros,
    powerCurve: hasEffortPower && timeline ? (powerCurve ?? meanMaxCurve(timeline)) : null,
    activityCriticalPower,
    bestEfforts:
      sport === 'bike' && (elapsedTimeline || climbs.length > 0)
        ? {
            weightKg: weight?.kg ?? null,
            weightDate: weight?.date ?? null,
            distance: elapsedTimeline ? distanceBestEfforts(elapsedTimeline) : [],
            power:
              elapsedTimeline && hasEffortPower
                ? powerBestEfforts(elapsedTimeline, weight?.kg ?? null)
                : [],
            climbs: cyclingClimbEfforts(climbs, weight?.kg ?? null),
          }
        : null,
    strokeCount: null,
    strokeRateSpm: null,
    swimPaceSPer100m: null,
    swimPaceSource: null,
    swimDurationS: null,
    swimIntervals: [],
    swimLocation: null,
    waterTemperatureC: null,
    analyses: {
      native: parseActivityProviderReports(
        rawDetail?.description ?? null,
        a.id,
        rawDetail?.fetchedAt ?? 0,
      ),
      derived: {
        environment,
        uvScore: environment
          ? applyGardenUvCalibration(environment, uvCalibration, generatedAt)
          : null,
        apparentWind,
      },
    },
  }
}

export function emptyPayload(athleteId = 0): StravaPayload {
  return {
    generatedAt: 0,
    athleteId,
    totalKm: 0,
    totalTimeS: 0,
    totalCount: 0,
    totals: emptyTotals(),
    strengthTotal: { count: 0, movingTimeS: 0 },
    days: [],
    details: {},
    swimTrend: [],
    health: {},
    zones: { hr: [], power: [], ftp: null },
    powerCurveRef: [],
    powerCurveYearRef: [],
    powerCurveYear: null,
    criticalPower: null,
    criticalPowerYear: null,
  }
}

export function applyManualFueling(
  payload: StravaPayload,
  entries: readonly ManualFuelingEntry[],
): void {
  for (const entry of entries) {
    const detail = payload.details[String(entry.activityId)]
    if (!detail || detail.date !== entry.date) continue
    detail.fueling = {
      ...emptyGarminFueling(),
      caloriesConsumed: entry.caloriesConsumed,
      sodiumLossMg: null,
      source: 'manual',
    }
  }
}

export function applyManualStrength(
  payload: StravaPayload,
  entries: readonly ManualStrengthEntry[],
): void {
  for (const entry of entries) {
    const detail = payload.details[String(entry.activityId)]
    if (!detail || detail.date !== entry.date || detail.sport !== 'strength') continue
    detail.strength = {
      volumeKg: entry.volumeKg,
      totalSets: entry.totalSets,
      totalReps: entry.totalReps,
      exercises: entry.exercises,
      source: 'manual',
    }
  }
}

const saunaHeartRate = (
  entry: ManualSaunaEntry,
  samples: readonly OuraHeartRateSample[],
  timeZone?: string,
): { avgHr: number | null; maxHr: number | null; trace: ActivityHeartRateTracePoint[] } => {
  const startMs = localDateTimeUtcMs(entry.date, entry.time, timeZone)
  const endMs = startMs + entry.durationS * 1_000
  const matched = samples
    .flatMap(sample => {
      const timestampMs = Date.parse(sample.timestamp)
      return Number.isFinite(timestampMs) && timestampMs >= startMs && timestampMs < endMs
        ? [{ timestampMs, bpm: sample.bpm }]
        : []
    })
    .sort((left, right) => left.timestampMs - right.timestampMs)
  const values = matched.map(sample => sample.bpm)
  return {
    avgHr: values.length
      ? Math.round(values.reduce((total, value) => total + value, 0) / values.length)
      : null,
    maxHr: values.length ? Math.max(...values) : null,
    trace: matched.map(sample => ({
      distanceKm: 0,
      elapsedS: Math.round((sample.timestampMs - startMs) / 1_000),
      heartRate: sample.bpm,
      heatStrainIndex: null,
      heatStrainSource: null,
      coreTemperatureC: null,
      coreTemperatureSource: null,
      skinTemperatureC: null,
      skinTemperatureSource: null,
    })),
  }
}

const refreshStravaDay = (day: StravaDay): void => {
  day.durationS = day.items.reduce((total, item) => total + item.durationS, 0)
  day.dominant =
    day.items.reduce<StravaDayItem | null>(
      (best, item) => (item.distanceKm > (best?.distanceKm ?? -1) ? item : best),
      null,
    )?.sport ?? null
}

const applySaunaWeather = (
  detail: StravaActivityDetail,
  entry: ManualSaunaEntry,
  weather: WeatherCache | null | undefined,
): void => {
  if (!weather) return
  const exact =
    entry.stravaActivityId == null ? undefined : weather.activities[String(entry.stravaActivityId)]
  const activityWeather =
    exact?.date === entry.date ? exact : nearestSameDayWeatherAt(weather, entry.date, detail.start)
  const dayWeather = weather.days[entry.date]
  detail.ambientTemperatureC ??= activityWeather?.temperatureC ?? null
  detail.windKph ??= activityWeather?.windKph ?? dayWeather?.windKph ?? null
  detail.windDir ??= activityWeather?.windDir ?? dayWeather?.windDir ?? null
  detail.windDirDeg ??= activityWeather?.windDirDeg ?? dayWeather?.windDirDeg ?? null
  detail.windGustKph ??= activityWeather?.windGustKph ?? dayWeather?.windGustKph ?? null
  detail.averageRelativeHumidityPct ??= activityWeather?.averageRelativeHumidityPct ?? null
  detail.relativeHumidityProvenance ??= activityWeather?.relativeHumidityProvenance ?? null
}

const manualSaunaGarminActivity = (
  entry: ManualSaunaEntry,
  garmin: GarminCache | null | undefined,
): GarminActivity | null => {
  if (entry.garminActivityId == null || !garmin) return null
  const id = String(entry.garminActivityId)
  const activity = garmin.activities[id] ?? garmin.activities[`connect:${id}`]
  return activity?.startDateLocal.slice(0, 10) === entry.date ? activity : null
}

const manualSaunaGarminVerification = (
  detail: StravaActivityDetail,
  activity: GarminActivity,
): GarminVerification => {
  const metrics = activity.metrics
  const trainingEffect = {
    trainingEffectActivityId: activity.id,
    aerobicTrainingEffect: metrics.aerobicTrainingEffect,
    anaerobicTrainingEffect: metrics.anaerobicTrainingEffect,
    exerciseLoad: metrics.exerciseLoad,
    trainingEffectLabel: metrics.trainingEffectLabel,
    aerobicTrainingEffectMessage: metrics.aerobicTrainingEffectMessage,
    anaerobicTrainingEffectMessage: metrics.anaerobicTrainingEffectMessage,
  }
  if (detail.garmin) return { ...detail.garmin, ...trainingEffect }
  const distanceM = detail.distanceKm * 1_000
  const distanceDeltaM = delta(activity.distanceM, distanceM)
  return {
    activityId: activity.id,
    name: activity.name,
    sourceDevice: activity.sourceDevice,
    startDate: activity.startDate,
    startDiffS: Math.round(
      Math.abs(Date.parse(activity.startDate) - Date.parse(detail.start)) / 1_000,
    ),
    distanceM: activity.distanceM,
    distanceDeltaM,
    distanceDeltaPct:
      distanceDeltaM != null && distanceM > 0 ? round((distanceDeltaM / distanceM) * 100, 1) : null,
    movingTimeS: activity.movingTimeS,
    movingTimeDeltaS: delta(activity.movingTimeS, detail.movingTimeS),
    elapsedTimeS: activity.elapsedTimeS,
    elapsedTimeDeltaS: delta(activity.elapsedTimeS, detail.elapsedTimeS),
    totalCalories: metrics.totalCalories,
    caloriesDelta: delta(metrics.totalCalories, detail.calories),
    avgHeartRate: metrics.avgHeartRate,
    avgHeartRateDelta: delta(metrics.avgHeartRate, detail.avgHr),
    avgPower: metrics.avgPower,
    avgPowerDelta: delta(metrics.avgPower, detail.avgWatts),
    avgCadence: metrics.avgCadence,
    normalizedPower: metrics.normalizedPower,
    maxPower: metrics.maxPower,
    totalWorkKJ: metrics.totalWorkKJ,
    totalWorkDeltaKJ: deltaFloat(metrics.totalWorkKJ, detail.kilojoules, 1),
    trainingStressScore: metrics.trainingStressScore,
    intensityFactor: metrics.intensityFactor,
    ...trainingEffect,
    runningDynamics: activity.runningDynamics,
  }
}

export function applyManualSauna(
  payload: StravaPayload,
  entries: readonly ManualSaunaEntry[],
  heartRateSamples: readonly OuraHeartRateSample[],
  timeZone?: string,
  weather?: WeatherCache | null,
  garmin?: GarminCache | null,
): void {
  for (const entry of entries) {
    const heartRate = saunaHeartRate(entry, heartRateSamples, timeZone)
    const sauna: ActivitySauna = {
      time: entry.time,
      temperatureC: entry.temperatureC,
      humidityPct: entry.humidityPct,
      cooldown: entry.cooldown,
      heatTrainingLoad: entry.heatTrainingLoad,
      heartRateSource: heartRate.trace.length > 0 ? 'oura' : null,
      source: 'manual',
    }
    if (entry.stravaActivityId != null) {
      const id = String(entry.stravaActivityId)
      const detail = payload.details[id]
      const day = payload.days.find(candidate => candidate.date === entry.date)
      const item = day?.items.find(candidate => candidate.id === entry.stravaActivityId)
      if (!detail || detail.date !== entry.date || !day || !item) continue
      const garminActivity = manualSaunaGarminActivity(entry, garmin)
      const hasActivityHeartRate =
        detail.avgHr != null ||
        detail.maxHr != null ||
        detail.heartRateTrace.some(point => point.heartRate != null)
      if (detail.sport !== 'sauna') {
        const total = payload.totals.find(candidate => candidate.sport === detail.sport)
        if (total) {
          total.count = Math.max(0, total.count - 1)
          total.distanceKm = round(Math.max(0, total.distanceKm - detail.distanceKm), 1)
          total.movingTimeS = Math.max(0, total.movingTimeS - detail.movingTimeS)
          total.elevationM = Math.max(0, total.elevationM - detail.elevationM)
        }
        if (detail.sport === 'strength') {
          payload.strengthTotal.count = Math.max(0, payload.strengthTotal.count - 1)
          payload.strengthTotal.movingTimeS = Math.max(
            0,
            payload.strengthTotal.movingTimeS - detail.movingTimeS,
          )
        }
      }
      if (garminActivity) {
        detail.garmin = manualSaunaGarminVerification(detail, garminActivity)
        detail.calculatedTrainingEffect = null
      }
      detail.sport = 'sauna'
      detail.computer = null
      detail.device = null
      detail.runWalk = null
      detail.name = entry.title ?? detail.name
      detail.distanceKm = 0
      if (!hasActivityHeartRate) {
        detail.avgHr = heartRate.avgHr
        detail.maxHr = heartRate.maxHr
        detail.heartRateTrace = heartRate.trace
      }
      detail.strength = null
      detail.sauna = hasActivityHeartRate ? { ...sauna, heartRateSource: null } : sauna
      applySaunaWeather(detail, entry, weather)
      item.sport = 'sauna'
      item.distanceKm = 0
      payload.totalKm = round(
        payload.totals.reduce((total, candidate) => total + candidate.distanceKm, 0),
        1,
      )
      refreshStravaDay(day)
      continue
    }
    const id = String(entry.id)
    if (payload.details[id]) continue
    const startMs = localDateTimeUtcMs(entry.date, entry.time, timeZone)
    payload.details[id] = {
      id: entry.id,
      sport: 'sauna',
      name: entry.title ?? 'sauna',
      date: entry.date,
      start: new Date(startMs).toISOString(),
      distanceKm: 0,
      movingTimeS: entry.durationS,
      elapsedTimeS: entry.durationS,
      maxSpeedKph: null,
      elevationM: 0,
      avgHr: heartRate.avgHr,
      maxHr: heartRate.maxHr,
      avgWatts: null,
      npWatts: null,
      maxWatts: null,
      kilojoules: null,
      deviceWatts: false,
      avgCadence: null,
      sufferScore: null,
      calories: null,
      deviceTemperatureC: null,
      ambientTemperatureC: null,
      windKph: null,
      windDir: null,
      windDirDeg: null,
      windGustKph: null,
      averageRelativeHumidityPct: null,
      relativeHumidityProvenance: null,
      location: null,
      fueling: null,
      strength: null,
      sauna,
      garmin: null,
      computer: null,
      device: null,
      staminaTrace: null,
      performanceConditionTrace: null,
      calculatedIntensityFactor: null,
      calculatedExerciseLoad: null,
      anaerobicPowerEstimate: null,
      calculatedTrainingEffect: null,
      gearShifts: [],
      cyclingDynamics: null,
      runWalk: null,
      route: [],
      heartRateTrace: heartRate.trace,
      mapRoute: [],
      analysisRanges: [],
      runSplitsMetric: [],
      runSplitsStandard: [],
      runPaceZones: null,
      minAlt: 0,
      maxAlt: 0,
      descentM: 0,
      hrZones: null,
      powerZones: null,
      powerHist: null,
      powerWithoutZeros: null,
      powerCurve: null,
      activityCriticalPower: null,
      bestEfforts: null,
      strokeCount: null,
      strokeRateSpm: null,
      swimPaceSPer100m: null,
      swimPaceSource: null,
      swimDurationS: null,
      swimIntervals: [],
      swimLocation: null,
      waterTemperatureC: null,
      analyses: {
        native: { myWindsock: null, pelotan: null },
        derived: { environment: null, uvScore: null, apparentWind: null },
      },
    }
    applySaunaWeather(payload.details[id], entry, weather)
    const garminActivity = manualSaunaGarminActivity(entry, garmin)
    if (garminActivity)
      payload.details[id].garmin = manualSaunaGarminVerification(
        payload.details[id],
        garminActivity,
      )
    const day = payload.days.find(candidate => candidate.date === entry.date) ?? {
      date: entry.date,
      durationS: 0,
      items: [],
      dominant: null,
    }
    if (!payload.days.includes(day)) payload.days.push(day)
    day.items.push({ id: entry.id, sport: 'sauna', distanceKm: 0, durationS: entry.durationS })
    refreshStravaDay(day)
    payload.totalCount += 1
    payload.totalTimeS += entry.durationS
  }
  payload.days.sort((left, right) => left.date.localeCompare(right.date))
}

function nearestSameDayWeatherAt(
  weather: WeatherCache,
  date: string,
  start: string,
): WeatherActivity | undefined {
  const startMs = Date.parse(start)
  if (!Number.isFinite(startMs)) return undefined
  let nearest: WeatherActivity | undefined
  let nearestDifferenceMs = Number.POSITIVE_INFINITY
  for (const candidate of Object.values(weather.activities)) {
    if (candidate.date !== date) continue
    const candidateStartMs = Date.parse(candidate.start)
    if (!Number.isFinite(candidateStartMs)) continue
    const differenceMs = Math.abs(candidateStartMs - startMs)
    if (differenceMs >= nearestDifferenceMs) continue
    nearest = candidate
    nearestDifferenceMs = differenceMs
  }
  return nearest
}

function nearestSameDayWeather(
  weather: WeatherCache,
  activity: RawStravaActivity,
): WeatherActivity | undefined {
  return nearestSameDayWeatherAt(weather, activity.startDateLocal.slice(0, 10), activity.startDate)
}

export function buildPayload(
  cache: StravaRawCache | null,
  oura: OuraCache | null,
  garmin: GarminCache | null,
  since?: string,
  weather?: WeatherCache | null,
  inputFtp?: number | null,
  inputHrBounds?: number[] | null,
  timeZone?: string,
  wahoo?: WahooCache | null,
  inputMaxHeartRate?: number | null,
  inputLactateThresholdHeartRate?: number | null,
  inputGeneratedAt?: number,
  activityTracking: readonly ActivityTrackingEntry[] = [],
): StravaPayload {
  if (!cache) return emptyPayload()
  const originalCache = cache
  cache = applyActivityTracking(cache, garmin, activityTracking) ?? cache
  const trackingById = new Map(activityTracking.map(entry => [entry.activityId, entry]))
  const resolveGarmin = (activity: RawStravaActivity, sport: ActivityKind) =>
    matchGarminActivity(
      originalCache.activities[String(activity.id)] ?? activity,
      sport,
      garmin,
      trackingById.get(activity.id)?.garminActivityId,
    )

  const generatedAt = latestProviderSync(inputGeneratedAt, cache, oura, garmin, weather, wahoo)

  const sinceDay = since && /^\d{4}-\d{2}-\d{2}$/.test(since) ? since : null
  const allActivities = Object.values(cache.activities)
    .map(a => ({
      a,
      sport: isTreatment(a.sportType, a.name)
        ? ('treatment' as ActivityKind)
        : normalizeKind(a.sportType),
    }))
    .filter((x): x is { a: RawStravaActivity; sport: ActivityKind } => x.sport !== null)
  const activities = allActivities
    .filter(x => !sinceDay || x.a.startDateLocal.slice(0, 10) >= sinceDay)
    .sort((p, q) => p.a.startDateLocal.localeCompare(q.a.startDateLocal))

  if (activities.length === 0) return { ...emptyPayload(cache.athleteId), generatedAt }

  const garminMatches = new Map<string, GarminActivityMatch | null>()
  const garminTrainingEffectMatches = new Map<string, GarminActivityMatch | null>()
  const garminHeartRateMatches = new Map<string, GarminActivityMatch | null>()
  const wahooMatches = new Map<string, WahooActivityMatch | null>()
  const selectedStreams = new Map<string, StravaStreams | GarminStreams | undefined>()
  const heartRates = new Map<string, ActivityHeartRate>()
  const respirationProjection = garminRespirationProjectionModel(garmin)
  for (const { a, sport } of activities) {
    const id = String(a.id)
    const original = originalCache.activities[id] ?? a
    const match = resolveGarmin(a, sport)
    const trainingEffectMatch =
      trackingById.get(a.id)?.garminActivityId != null
        ? match
        : matchGarminTrainingEffectActivity(a, sport, garmin, match)
    const hrMatch = matchGarminHeartRateActivity(a, sport, garmin)
    const wahooMatch = matchWahooActivity(original, sport, wahoo ?? null)
    const streams = selectStreams(cache.streams?.[id], match, garmin, a)
    garminMatches.set(id, match)
    garminTrainingEffectMatches.set(id, trainingEffectMatch)
    garminHeartRateMatches.set(id, hrMatch)
    wahooMatches.set(id, wahooMatch)
    selectedStreams.set(id, streams)
    heartRates.set(
      id,
      resolveActivityHeartRate(
        a,
        sport,
        a.sportType.startsWith('Virtual')
          ? selectEffortStreams(cache.streams?.[id], streams)
          : streams,
        hrMatch,
        garmin,
      ),
    )
  }

  const totals = emptyTotals()
  const strengthTotal = { count: 0, movingTimeS: 0 }
  const byDate = new Map<string, StravaDayItem[]>()
  for (const { a, sport } of activities) {
    const t = totals.find(x => x.sport === sport)
    if (t) {
      t.count += 1
      t.distanceKm += a.distance / 1000
      t.movingTimeS += a.movingTime
      t.elevationM += a.totalElevationGain
    }
    if (sport === 'strength') {
      strengthTotal.count += 1
      strengthTotal.movingTimeS += a.movingTime
    }

    const date = a.startDateLocal.slice(0, 10)
    const items = byDate.get(date) ?? []
    items.push({
      id: a.id,
      sport,
      distanceKm: round(a.distance / 1000, 1),
      durationS: a.movingTime,
    })
    byDate.set(date, items)
  }

  const dayMs = (iso: string): number => Date.parse(`${iso}T00:00:00Z`)
  const firstMs = dayMs(activities[0].a.startDateLocal.slice(0, 10))
  const lastActMs = dayMs(activities[activities.length - 1].a.startDateLocal.slice(0, 10))
  const end = generatedAt ? dayMs(localIsoDay(generatedAt, timeZone)) : lastActMs
  const start = sinceDay ? dayMs(sinceDay) : Math.max(firstMs, end - (WINDOW_DAYS - 1) * DAY_MS)
  const days: StravaDay[] = []
  for (let ms = start; ms <= end; ms += DAY_MS) {
    const date = new Date(ms).toISOString().slice(0, 10)
    const items = byDate.get(date) ?? []
    const dominant = items.reduce<StravaDayItem | null>(
      (best, item) => (item.distanceKm > (best?.distanceKm ?? -1) ? item : best),
      null,
    )
    days.push({
      date,
      durationS: items.reduce((s, item) => s + item.durationS, 0),
      items,
      dominant: dominant?.sport ?? null,
    })
  }

  const finalized = totals.map(t => ({
    ...t,
    distanceKm: round(t.distanceKm, 1),
    elevationM: Math.round(t.elevationM),
  }))

  let hrmax = 0
  for (const { a } of activities) {
    const maxHr = heartRates.get(String(a.id))?.maxHr
    if ((maxHr ?? 0) > hrmax) hrmax = maxHr ?? 0
  }
  if (hrmax < 100) hrmax = 190
  const recentCut = end - 41 * DAY_MS
  const powerCurveYear = new Date(end).getUTCFullYear()
  const yearCut = dayMs(`${powerCurveYear}-01-01`)
  const detailIds = new Set(activities.map(({ a }) => String(a.id)))
  let best20 = 0
  const powerCurves = new Map<string, PowerCurvePoint[]>()
  const recentCurves: PowerCurveSource[] = []
  const yearCurves: PowerCurveSource[] = []
  const activityCriticalPowers = new Map<string, CriticalPowerEstimate>()
  const recentCriticalPowerAnchors: CriticalPowerAnchor[] = []
  const yearCriticalPowerAnchors: CriticalPowerAnchor[] = []
  for (const { a, sport } of allActivities) {
    if (sport !== 'bike') continue
    const id = String(a.id)
    const activityDay = dayMs(a.startDateLocal.slice(0, 10))
    const inRecentWindow = activityDay >= recentCut && activityDay <= end
    const inYear = activityDay >= yearCut && activityDay <= end
    if (!detailIds.has(id) && !inRecentWindow && !inYear) continue
    const selected =
      selectedStreams.get(id) ??
      selectStreams(cache.streams?.[id], resolveGarmin(a, sport), garmin, a)
    if (!selectedStreams.has(id)) selectedStreams.set(id, selected)
    const streams = selectEffortStreams(cache.streams?.[id], selected)
    if (!streams?.watts?.some(v => v > 0)) continue
    const timeline = effortTimeline(streams, a.movingTime)
    const c = meanMaxCurve(timeline)
    powerCurves.set(id, c)
    if (detailIds.has(id)) {
      const p20 = c.find(p => p.s === 1200)
      if (p20 && p20.w > best20) best20 = p20.w
    }
    const source = { activityId: a.id, activityDate: a.startDateLocal.slice(0, 10), curve: c }
    if (inRecentWindow) recentCurves.push(source)
    if (inYear) yearCurves.push(source)
    if (a.deviceWatts && timeline) {
      const activityDate = a.startDateLocal.slice(0, 10)
      const anchors = bestObservedPowerWindows(timeline, CRITICAL_POWER_DURATIONS_S).map(
        (window): CriticalPowerAnchor => ({
          durationS: window.durationS,
          meanPowerWatts: window.averageWatts,
          activityId: a.id,
          activityDate,
          startElapsedS: window.start,
          endElapsedS: window.end,
        }),
      )
      const activityCriticalPower = fitCriticalPower(
        anchors,
        'activity',
        activityDate,
        activityDate,
      )
      if (activityCriticalPower) activityCriticalPowers.set(id, activityCriticalPower)
      if (inRecentWindow) recentCriticalPowerAnchors.push(...anchors)
      if (inYear) yearCriticalPowerAnchors.push(...anchors)
    }
  }
  const powerCurveRef = mergeMaxCurves(recentCurves)
  const powerCurveYearRef = mergeMaxCurves(yearCurves)
  const windowTo = new Date(end).toISOString().slice(0, 10)
  const criticalPower = fitCriticalPower(
    recentCriticalPowerAnchors,
    'six-weeks',
    new Date(recentCut).toISOString().slice(0, 10),
    windowTo,
  )
  const criticalPowerYear = fitCriticalPower(
    yearCriticalPowerAnchors,
    'calendar-year',
    `${powerCurveYear}-01-01`,
    windowTo,
  )
  const powerCurveActivityIds = new Set([
    ...[...powerCurveRef, ...powerCurveYearRef].flatMap(point =>
      point.activityId == null ? [] : [String(point.activityId)],
    ),
    ...[criticalPower, criticalPowerYear].flatMap(estimate =>
      estimate ? estimate.anchors.map(anchor => String(anchor.activityId)) : [],
    ),
  ])
  const projectedActivities = [
    ...activities,
    ...allActivities.filter(
      ({ a }) => !detailIds.has(String(a.id)) && powerCurveActivityIds.has(String(a.id)),
    ),
  ]
  const ftp = inputFtp ?? cache.zones?.ftp ?? (best20 > 0 ? Math.round(best20 * 0.95) : null)
  const hrBounds = inputHrBounds?.length
    ? inputHrBounds
    : cache.zones?.hr?.length
      ? cache.zones.hr
      : deriveHrBounds(hrmax)
  const powerBounds =
    inputFtp != null
      ? derivePowerBounds(inputFtp)
      : cache.zones?.power?.length
        ? cache.zones.power
        : ftp != null
          ? derivePowerBounds(ftp)
          : []
  const wahooStamina = estimateWahooCyclingStamina(
    wahoo ?? null,
    garmin,
    ftp,
    inputMaxHeartRate ?? hrmax,
  )

  const starts: [number, number][] = []
  for (const { a } of activities) {
    if (a.sportType.startsWith('Virtual')) continue
    const ll = selectedStreams.get(String(a.id))?.latlng
    if (ll && ll.length >= 2) starts.push([ll[0][0], ll[0][1]])
  }
  const home = inferRouteHome(starts)

  const details: Record<string, StravaActivityDetail> = {}
  for (const { a, sport } of projectedActivities) {
    const id = String(a.id)
    const garminMatch = garminMatches.get(id) ?? resolveGarmin(a, sport)
    const garminTrainingEffectMatch =
      trackingById.get(a.id)?.garminActivityId != null
        ? garminMatch
        : (garminTrainingEffectMatches.get(id) ??
          matchGarminTrainingEffectActivity(a, sport, garmin, garminMatch))
    const garminHeartRateMatch =
      garminHeartRateMatches.get(id) ?? matchGarminHeartRateActivity(a, sport, garmin)
    const original = originalCache.activities[id] ?? a
    const wahooMatch = wahooMatches.get(id) ?? matchWahooActivity(original, sport, wahoo ?? null)
    const climbs = activityClimbSegments(
      sport,
      garminMatch,
      garmin,
      trackingById.get(a.id)?.virtual ? null : wahooMatch,
      wahoo ?? null,
    )
    const selectedStream = selectedStreams.get(id)
    const exactWeather = weather?.activities[id]
    const usesRouteLessWeather =
      (sport === 'swim' || sport === 'strength') &&
      (!selectedStream || selectedStream.latlng.length < 2)
    const activityWeather = a.sportType.startsWith('Virtual')
      ? undefined
      : (exactWeather ??
        (weather && usesRouteLessWeather ? nearestSameDayWeather(weather, a) : undefined))
    details[String(a.id)] = projectDetail(
      a,
      sport,
      selectedStream,
      selectEffortStreams(cache.streams?.[id], selectedStream),
      heartRates.get(id) ??
        resolveActivityHeartRate(a, sport, selectedStream, garminHeartRateMatch, garmin),
      activityMetricSamples(
        a,
        sport,
        garminMatch,
        garmin,
        wahooMatch,
        wahoo ?? null,
        wahooMatch ? (wahooStamina.get(wahooMatch.activity.id) ?? null) : null,
        respirationProjection,
      ),
      activityGearShifts(a, garminMatch, garmin, wahooMatch, wahoo ?? null),
      activityCyclingDynamics(a, garminMatch, garmin, wahooMatch, wahoo ?? null),
      activityRunWalk(sport, garminMatch, garmin),
      activityWeather,
      weather?.attribution ?? null,
      weather?.uvCalibration ?? null,
      generatedAt,
      a.sportType.startsWith('Virtual') ? undefined : cache.geo?.[String(a.id)],
      activityFueling(garminMatch, wahooMatch),
      garminVerification(original, garminMatch, garminTrainingEffectMatch),
      activityComputer(sport, garminMatch, wahooMatch),
      activityDevice(sport, a, garminMatch),
      activityWeight(garmin, a),
      cache.activityDetails?.[id],
      climbs,
      hrBounds,
      powerBounds,
      ftp,
      inputLactateThresholdHeartRate ?? null,
      activityRestingHeartRate(oura, a.startDateLocal.slice(0, 10)),
      criticalPowerYear ?? criticalPower,
      a.sportType.startsWith('Virtual') ? null : home,
      powerCurves.get(id),
      activityCriticalPowers.get(id) ?? null,
    )
    if (trackingById.get(a.id)?.virtual && garminMatch) {
      const detail = details[id]
      const metrics = garminMatch.activity.metrics
      if (garminMatch.activity.distanceM != null && garminMatch.activity.distanceM > 0)
        detail.distanceSource = 'garmin'
      detail.elevationM = metrics.totalAscentM ?? detail.elevationM
      detail.descentM = metrics.totalDescentM ?? detail.descentM
      const alignment = timedStreamAlignment(cache.streams?.[id])
      if (alignment) {
        const distanceSamples = alignment.time.map((elapsedS, index) => ({
          elapsedS,
          value: alignment.distance[index],
        }))
        const distanceKmAt = (elapsedS: number) =>
          round((timedMetricAt(distanceSamples, elapsedS) ?? 0) / 1_000, 3)
        for (const shift of detail.gearShifts) shift.distanceKm = distanceKmAt(shift.elapsedS)
        if (detail.cyclingDynamics) {
          detail.cyclingDynamics.distanceKm = detail.cyclingDynamics.elapsedS.map(distanceKmAt)
          for (const change of detail.cyclingDynamics.positionChanges)
            change.distanceKm = distanceKmAt(change.elapsedS)
        }
      }
    }
  }

  const health: Record<string, ActivityHealth> = {}
  if (oura) for (const [date, o] of Object.entries(oura.days)) health[date] = toHealth(o)
  if (weather)
    for (const [date, w] of Object.entries(weather.days)) {
      const h = health[date] ?? emptyHealth()
      health[date] = {
        ...h,
        windKph: h.windKph ?? w.windKph,
        windDir: h.windDir ?? w.windDir,
        windDirDeg: h.windDirDeg ?? w.windDirDeg,
        windGustKph: h.windGustKph ?? w.windGustKph,
      }
    }

  return {
    generatedAt,
    athleteId: cache.athleteId,
    totalKm: round(
      finalized.reduce((s, t) => s + t.distanceKm, 0),
      1,
    ),
    totalTimeS: activities.reduce((s, { a }) => s + a.movingTime, 0),
    totalCount: activities.length,
    totals: finalized,
    strengthTotal,
    days,
    details,
    swimTrend: [],
    health,
    zones: { hr: hrBounds, power: powerBounds, ftp },
    powerCurveRef,
    powerCurveYearRef,
    powerCurveYear,
    criticalPower,
    criticalPowerYear,
  }
}
