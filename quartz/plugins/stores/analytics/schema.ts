import type { BestEffortsBlock } from '../../../util/best-efforts'
import type { SleepMetrics } from '../../../util/sleep-metrics'
import type { SwimPowerCurveBlock } from '../../../util/swim-power'
import type { WeeklyTargetRange } from '../../../util/weekly-target-range'
import type { CoreBodyTemperatureCache } from '../core-body-temperature'
import type { GarminCache } from '../garmin'
import type { GarminHealthDay } from '../garmin-health'
import type { ActivityKind, Sport, StravaActivityDetail, StravaZones } from '../strava'
import type { WahooData } from '../wahoo'
import type { WeatherCache } from '../weather'
import type { BodyBlock, LabTests } from './body-and-labs'
import type { CalibrationBlock, CalibrationProjection } from './calibration-and-lactate'
import type { CoreTemperatureOrigin, DistributionsBlock } from './distributions'
import type { PowerCurveBlock, PowerCurveInput } from './power-to-weight'
import type { RaceReadiness } from './readiness'
import type { RecoveryBlock } from './recovery'
import type { LactateThresholdBlock } from './thresholds'
import type { SportTrend } from './trends'
import type { CardioDir, CardioKey, FtpHypothesis, Vo2maxBlock } from './vo2-and-ftp'
import { AppleCache } from '../apple'
import { OuraCache, type OuraHealthDay } from '../oura'
import { RaceEvent, TrackEntry, TrainingExclusion } from '../tracking'
import {
  clamp,
  type AnalyticsMeta,
  type DailyPoint,
  type RiskBlock,
  type SportBest,
  type ThresholdEstimate,
  type TrainingAction,
  type WeeklyPoint,
} from './primitives'

export interface ActivitySummary {
  id: number
  date: string
  sport: ActivityKind
  virtual?: boolean
  commute?: boolean
  treadmill?: boolean
  name: string
  distanceKm: number
  movingTimeS: number
  load: number
  paceIntensityFactor: number | null
  effort: number | null
  cadence: number | null
  strokes: Record<string, number> | null
  windKph: number | null
  windDir: string | null
  windGustKph: number | null
}

export interface AnalyticsInputs {
  oura?: OuraCache | null
  apple?: AppleCache | null
  core?: CoreBodyTemperatureCache | null
  wahoo?: WahooData | null
  garmin?: GarminCache | null
  weather?: WeatherCache | null
  weights?: TrackEntry[]
  events?: RaceEvent[]
  dexa?: unknown
  vo2labs?: unknown
  ftp?: number | null
  powerCurve?: PowerCurveInput
  zones?: StravaZones | null
  activityDetails?: Readonly<Record<string, StravaActivityDetail>>
  since?: string
  timeZone?: string
  generatedAt?: number
}

export type RadarAxisKey =
  | 'sprint'
  | 'threshold'
  | 'endurance'
  | 'climb'
  | 'stride'
  | 'cadence'
  | 'recovery'
  | 'oscillation'

export interface RadarAxis {
  key: RadarAxisKey
  label: string
  score: number | null
  proj: number | null
  projRawValue?: number | null
  rawValue: number | null
  rawUnit: string
  lo: number
  hi: number
}

export interface AbilityTrendPoint {
  date: string
  sprint: number | null
  threshold: number | null
  endurance: number | null
  cadence: number | null
  climb?: number | null
  stride?: number | null
  recovery?: number | null
  oscillation?: number | null
}

export interface SportAbilities {
  sport: Sport
  axes: RadarAxis[]
  area: number | null
  history: AbilityTrendPoint[]
}

export interface AbilitiesBlock {
  sports: SportAbilities[]
}

export interface CardioMetric {
  key: CardioKey
  label: string
  value: number | null
  unit: string
  slopePerWeek: number | null
  dir: CardioDir
  sampleSize: number
  note: string
}

export interface CardioBlock {
  metrics: CardioMetric[]
  rhrSeries: { date: string; rhr: number }[]
  hrvSeries: { date: string; hrv: number }[]
  efSeries: { date: string; ef: number; sport: Sport }[]
  decouplingSeries: { date: string; pct: number }[]
}

export type HeatAcclimatisationState = 'unexposed' | 'acclimating' | 'maintaining' | 'decaying'
export type HeatConfidence = 'none' | 'low' | 'moderate'
export type HeatTemperatureSource = 'core' | 'weatherkit' | 'strava' | 'manual-sauna'

export type HeatSauna = NonNullable<StravaActivityDetail['sauna']> & { durationMinutes: number }

export interface HeatActivityPoint {
  id: number
  date: string
  startedAt: string
  sport: 'bike' | 'run' | 'sauna'
  name: string
  temperatureC: number | null
  heatStrainIndex: number | null
  sauna: HeatSauna | null
  source: HeatTemperatureSource
  coreOrigin: CoreTemperatureOrigin | null
  observedMinutes: number
  hotMinutes: number
  dose: number
}

export interface HeatDay {
  date: string
  temperatureC: number | null
  heatStrainIndex: number | null
  source: HeatTemperatureSource | 'mixed' | null
  observedMinutes: number
  hotMinutes: number
  saunaMinutes: number
  saunaHtl: number | null
  dose: number
  acclimatisationPct: number
}

export interface HeatMethod {
  eligibleSports: ['bike', 'run', 'sauna']
  hotThresholdC: number
  heatStrainThreshold: number
  targetMinutesPerDay: number
  targetDays: number
  decayGraceDays: number
  decayPerDay: number
  coverageDays: number
  htlPerExposure: number
  passiveHtlCapPerDay: number
  note: string
}

export interface HeatBlock {
  currentPct: number | null
  state: HeatAcclimatisationState
  confidence: HeatConfidence
  coveragePct: number
  eligibleMinutes: number
  observedMinutes: number
  latestTemperatureC: number | null
  lastObservedDate: string | null
  lastHeatDate: string | null
  heatMinutes14d: number
  heatDays14d: number
  saunaMinutes14d: number
  saunaHtl14d: number | null
  saunaSessions14d: number
  sourceCounts: Record<HeatTemperatureSource, number>
  coreSourceCounts: Record<CoreTemperatureOrigin, number>
  activities: HeatActivityPoint[]
  series: HeatDay[]
  method: HeatMethod
}

export interface EngineBlock {
  vo2max: Vo2maxBlock
  lactateThreshold: LactateThresholdBlock
  abilities: AbilitiesBlock
  cardio: CardioBlock
  ftpHypothesis: FtpHypothesis | null
}

export interface DataFeedInputs {
  oura?: OuraCache | null
  apple?: AppleCache | null
  weather?: WeatherCache | null
  garmin?: GarminCache | null
  weights?: TrackEntry[]
  trainingExclusions?: TrainingExclusion[]
  zones?: StravaZones | null
  activityDetails?: Readonly<Record<string, StravaActivityDetail>>
  generatedAt?: number
}

/** Trend, race readiness and pace projections fitted on the recent window alone. */
export interface RecentEstimates {
  /** First day of the window; the window ends at `meta.today`. */
  from: string
  trends: SportTrend[]
  races: RaceReadiness[]
  projections: CalibrationProjection[]
}

export interface Analytics {
  meta: AnalyticsMeta
  calibration: CalibrationBlock
  thresholds: ThresholdEstimate[]
  daily: DailyPoint[]
  weekly: WeeklyPoint[]
  trends: SportTrend[]
  bests: SportBest[]
  risk: RiskBlock
  races: RaceReadiness[]
  loadShare: Record<Sport, number>
  body: BodyBlock
  recovery: RecoveryBlock
  powerCurve: PowerCurveBlock
  swimPowerCurve: SwimPowerCurveBlock
  /** All-time ranks: the 60-day range view leaves this block whole. */
  bestEfforts: BestEffortsBlock
  heat: HeatBlock
  distributions: DistributionsBlock
  engine: EngineBlock
  events: RaceEvent[]
  activities: ActivitySummary[]
  weakestSport: Sport
  actions: TrainingAction[]
  tests: LabTests
  /** The 60-day range swaps these in for `trends`, `races` and the calibration projections. */
  recent: RecentEstimates
}

export const norm01 = (v: number, lo: number, hi: number): number =>
  clamp((v - lo) / (hi - lo), 0, 1)
export const metricBounds = (values: readonly number[]): [number, number] | null => {
  if (values.length === 0) return null
  const sorted = [...values].sort((left, right) => left - right)
  if (sorted.length === 1) {
    const spread = Math.max(Math.abs(sorted[0]) * 0.1, 0.1)
    return [sorted[0] - spread, sorted[0] + spread]
  }
  const at = (fraction: number): number => {
    const index = (sorted.length - 1) * fraction
    const lower = Math.floor(index)
    const upper = Math.ceil(index)
    const weight = index - lower
    return sorted[lower] * (1 - weight) + sorted[upper] * weight
  }
  const low = at(0.1)
  const high = at(0.9)
  if (high > low) return [low, high]
  const spread = Math.max(Math.abs(low) * 0.1, 0.1)
  return [low - spread, high + spread]
}
export const personalMetricScore = (
  value: number | null,
  bounds: [number, number] | null,
  lowerIsHigher: boolean,
): number | null => {
  if (value == null || bounds == null) return null
  const normalized = norm01(value, bounds[0], bounds[1])
  return Math.round((lowerIsHigher ? 1 - normalized : normalized) * 100)
}

export const DAY_FIELDS = [
  'kind',
  'date',
  'sessions',
  'km',
  'hours',
  'load',
  'swimLoad',
  'bikeLoad',
  'runLoad',
  'ctl',
  'atl',
  'tsb',
  'swimCtl',
  'bikeCtl',
  'runCtl',
  'warmup',
  'readiness',
  'sleepScore',
  'hrv',
  'rhr',
  'sleepDurationS',
  'napDurationS',
  'napCount',
  'garminHealth',
  'ouraHealth',
  'sleepMetrics',
  'tempDeviationC',
  'totalCalories',
  'activeCalories',
  'intakeKcal',
  'weightKg',
  'bmi',
  'ffmi',
  'bodyFatPct',
  'bodyWaterPct',
  'muscleMassKg',
  'boneMassKg',
  'bmr',
  'windKph',
  'windDir',
  'windGustKph',
  'readinessNext',
  'hrvNext',
] as const

export const ACTIVITY_FIELDS = [
  'kind',
  'id',
  'date',
  'sport',
  'name',
  'distanceKm',
  'movingTimeS',
  'elapsedTimeS',
  'elevationM',
  'avgHr',
  'maxHr',
  'avgWatts',
  'weightedWatts',
  'deviceWatts',
  'cadence',
  'strokes',
  'calories',
  'sufferScore',
  'deviceTemperatureC',
  'ambientTemperatureC',
  'weatherImpactPct',
  'weatherImpactSource',
  'uvExposureKind',
  'uvScore',
  'uvSeverity',
  'ambientSed',
  'avgUvIndex',
  'peakUvIndex',
  'avgCloudCoverPct',
  'gardenHeadwindSharePct',
  'gardenApparentAirRatio',
  'gardenWindCoveragePct',
  'windKph',
  'windDir',
  'windGustKph',
  'sauna',
  'skipTraining',
  'vGap',
  'intensity',
  'load',
  'pp30',
  'pp60',
  'pp300',
  'pp1200',
  'ps30',
  'ps60',
  'ps300',
  'ps1200',
  'ef',
  'decoupling',
] as const

export const WEEK_FIELDS = [
  'kind',
  'weekStart',
  'complete',
  'sessions',
  'load',
  'km',
  'hours',
  'swimKm',
  'bikeKm',
  'runKm',
  'swimHours',
  'bikeHours',
  'runHours',
  'effort',
  'effortSessions',
  'loadRange',
  'effortRange',
  'ramp',
  'monotony',
  'strain',
] as const

export interface FeedDayRow {
  kind: 'day'
  date: string
  sessions: number
  km: number
  hours: number
  load: number
  swimLoad: number
  bikeLoad: number
  runLoad: number
  ctl: number
  atl: number
  tsb: number
  swimCtl: number
  bikeCtl: number
  runCtl: number
  warmup: boolean
  readiness: number | null
  sleepScore: number | null
  hrv: number | null
  rhr: number | null
  sleepDurationS: number | null
  napDurationS?: number | null
  napCount?: number | null
  garminHealth?: GarminHealthDay | null
  ouraHealth?: OuraHealthDay | null
  sleepMetrics: SleepMetrics | null
  tempDeviationC: number | null
  totalCalories: number | null
  activeCalories: number | null
  intakeKcal: number | null
  weightKg: number | null
  bmi: number | null
  ffmi: number | null
  bodyFatPct: number | null
  bodyWaterPct: number | null
  muscleMassKg: number | null
  boneMassKg: number | null
  windKph: number | null
  windDir: string | null
  windGustKph: number | null
  readinessNext: number | null
  hrvNext: number | null
}

export interface FeedActivityRow {
  kind: 'activity'
  id: number
  date: string
  sport: ActivityKind
  name: string
  distanceKm: number
  movingTimeS: number
  elapsedTimeS: number
  elevationM: number
  avgHr: number | null
  maxHr: number | null
  avgWatts: number | null
  weightedWatts: number | null
  deviceWatts: boolean | null
  cadence: number | null
  strokes: Record<string, number> | null
  calories: number | null
  sufferScore: number | null
  deviceTemperatureC: number | null
  ambientTemperatureC: number | null
  weatherImpactPct: number | null
  weatherImpactSource: 'mywindsock' | null
  uvExposureKind: 'pelotan-score' | 'garden-score' | 'garden-sed' | null
  uvScore: number | null
  uvSeverity: string | null
  ambientSed: number | null
  avgUvIndex: number | null
  peakUvIndex: number | null
  avgCloudCoverPct: number | null
  gardenHeadwindSharePct: number | null
  gardenApparentAirRatio: number | null
  gardenWindCoveragePct: number | null
  windKph: number | null
  windDir: string | null
  windGustKph: number | null
  sauna: StravaActivityDetail['sauna']
  skipTraining: boolean
  vGap: number
  intensity: number | null
  load: number
  pp30: number | null
  pp60: number | null
  pp300: number | null
  pp1200: number | null
  ps30: number | null
  ps60: number | null
  ps300: number | null
  ps1200: number | null
  ef: number | null
  decoupling: number | null
}

export interface FeedWeekRow {
  kind: 'week'
  weekStart: string
  complete: boolean
  sessions: number
  load: number
  km: number
  hours: number
  swimKm: number
  bikeKm: number
  runKm: number
  swimHours: number
  bikeHours: number
  runHours: number
  effort: number
  effortSessions: number
  loadRange: WeeklyTargetRange | null
  effortRange: WeeklyTargetRange | null
  ramp: number | null
  monotony: number | null
  strain: number | null
}
