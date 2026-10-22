import type {
  GardenApparentWindEstimate,
  GardenEnvironmentEstimate,
  GardenUvScore,
} from '../../../util/activity-environment'
import type { NativeActivityReports } from '../../../util/activity-provider-reports'
import type { CyclingIntensityTrace } from '../../../util/cycling-intensity'
import type { CyclingPowerTrace } from '../../../util/cycling-power'
import type { CyclingTorqueTrace } from '../../../util/cycling-torque'
import type { HeartRatePhysiology } from '../../../util/heart-rate-physiology'
import type { RunHeartCostTrace, RunStrideCadenceTrace } from '../../../util/run-mechanics'
import type { RunPaceZoneDistribution } from '../../../util/run-pace-zones'
import type { StaminaLedgerStart } from '../../../util/stamina-ledger'
import type { SwimPhysiology } from '../../../util/swim-physiology'
import type { SwimPowerEstimate } from '../../../util/swim-power'
import type { SwimProvenance } from '../../../util/swim-provenance'
import type { WalkPowerEstimate } from '../../../util/walk-power'
import type { SwimLocation, SwimStroke } from '../apple'
import type { CriticalPowerAnchor, CriticalPowerEstimate } from '../critical-power'
import type { GarminRunWalkData } from '../garmin'
import type { OuraDaily } from '../oura'
import type { ActivityMove, ManualSaunaEntry, StrengthExercise } from '../tracking'
import type { WeatherRelativeHumidityProvenance } from '../weather'
import type { ActivityAnalysisRange, ActivityRunSplit } from './climbs-and-ranges'
import type { ActivityCyclingDynamics, ActivityGearShift } from './cycling-mechanics'
import type { ActivityPerformanceConditionTrace } from './detail-streams'
import type { ActivityBestEfforts, PowerCurvePoint } from './efforts-and-curves'
import type { ActivityFueling, GarminVerification } from './provider-metadata'
import {
  SPORT_ORDER,
  type ActivityComputer,
  type ActivityDevice,
  type ActivityEquipment,
  type ActivityHeartRateTracePoint,
  type ActivityKind,
  type ActivitySource,
  type ActivityStaminaTrace,
  type Sport,
  type StravaMapPoint,
  type StravaRoutePoint,
  type StravaZones,
  type WahooVerification,
} from './raw-and-kinds'
import { round } from './stream-primitives'

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

export interface ActivityStrength {
  volumeKg: number | null
  totalSets: number | null
  totalReps: number | null
  totalsSource?: 'manual'
  exercises: StrengthExercise[]
  source:
    | 'manual'
    | 'strava'
    | 'strava-description'
    | 'routine-title'
    | 'routine-description'
    | 'trainingpeaks-description'
    | 'trainingpeaks-routine'
  prescription?: {
    rounds: number
    workS: number
    restS: number
    durationS: number
    optionalRoundRestS?: number
    trainingPeaksWorkoutId?: string
  }
}

export interface ActivityMoveList {
  entries: ActivityMove[]
  source: 'manual'
}

export interface ActivitySauna {
  time: string
  location?: ManualSaunaEntry['location']
  temperatureC: number
  humidityPct: number
  cooldown: ManualSaunaEntry['cooldown']
  lapPhases?: ManualSaunaEntry['lapPhases']
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
  source: CalculatedIntensityFactor['source'] | 'garmin' | 'wahoo'
}

export interface AnaerobicPowerEstimate {
  effect: number
  effortCount: number
  stimulus: number
  sprintEffortCount?: number
  sprintStimulus?: number
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
  equipment?: ActivityEquipment
  sources?: ActivitySource[]
  virtual?: boolean
  commute?: boolean
  distanceSource?: 'garmin' | 'strava'
  wahoo?: WahooVerification
  cyclingTorque?: CyclingTorqueTrace | null
  cyclingIntensityTrace?: CyclingIntensityTrace | null
  cyclingPowerTrace?: CyclingPowerTrace | null
  runStrideCadence?: RunStrideCadenceTrace | null
  runHeartCost?: RunHeartCostTrace | null
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
  moves?: ActivityMoveList
  sauna: ActivitySauna | null
  garmin: GarminVerification | null
  computer: ActivityComputer | null
  computerOverride?: string
  device: ActivityDevice | null
  staminaTrace: ActivityStaminaTrace | null
  staminaLedger?: StaminaLedgerStart | null
  performanceConditionTrace: ActivityPerformanceConditionTrace | null
  heartRatePhysiology?: HeartRatePhysiology | null
  walkPower?: WalkPowerEstimate | null
  swimPhysiology?: SwimPhysiology | null
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
  powerCurveWeight?: { kg: number; date: string; source: 'garmin' }
  criticalPowerAnchors?: CriticalPowerAnchor[]
  activityCriticalPower: CriticalPowerEstimate | null
  bestEfforts: ActivityBestEfforts | null
  strokes?: Record<string, number> | null
  strokeCount: number | null
  strokeRateSpm: number | null
  swimPaceSPer100m: number | null
  swimPaceSource: SwimPaceSource | null
  swimDurationS: number | null
  swimIntervals: SwimActivityInterval[]
  swimProvenance?: SwimProvenance
  swimPower?: SwimPowerEstimate | null
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
  runPowerCurveRef: PowerCurvePoint[]
  runPowerCurveYearRef: PowerCurvePoint[]
  powerCurveYear: number | null
  criticalPower: CriticalPowerEstimate | null
  criticalPowerYear: CriticalPowerEstimate | null
}

const ANAEROBIC_POWER_WINDOW_S = 5
const ANAEROBIC_POWER_INTERVAL_MIN_S = 10
const ANAEROBIC_POWER_INTERVAL_MAX_S = 120
const ANAEROBIC_W_PRIME_MIN_FRACTION = 0.1
const ANAEROBIC_SESSION_DURATION_EXPONENT = 0.6
const ANAEROBIC_TRAINING_EFFECT_RATE = 0.24

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
  let sprintRawStimulus = 0
  let effortCount = 0
  let sprintEffortCount = 0
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
        const effortStimulus =
          (depletionFraction - ANAEROBIC_W_PRIME_MIN_FRACTION) /
          (1 - ANAEROBIC_W_PRIME_MIN_FRACTION)
        rawStimulus += effortStimulus
        effortCount++
        if (
          durationS <= 30 &&
          criticalPowerWatts + intervalWorkJoules / durationS >= 1.5 * criticalPowerWatts
        ) {
          sprintRawStimulus += effortStimulus
          sprintEffortCount++
        }
      }
    }
    intervalStart = -1
    intervalWorkJoules = 0
  }
  const stimulus =
    rawStimulus / Math.pow(Math.max(1, movingTimeS / 3600), ANAEROBIC_SESSION_DURATION_EXPONENT)
  const sprintStimulus =
    sprintRawStimulus /
    Math.pow(Math.max(1, movingTimeS / 3600), ANAEROBIC_SESSION_DURATION_EXPONENT)
  const effect = 5 * (1 - Math.exp(-ANAEROBIC_TRAINING_EFFECT_RATE * stimulus))
  return {
    effect: round(effect, 2),
    effortCount,
    stimulus: round(stimulus, 3),
    sprintEffortCount,
    sprintStimulus: round(sprintStimulus, 3),
    criticalPowerWatts: round(criticalPowerWatts, 1),
    wPrimeKilojoules: round(wPrimeJoules / 1_000, 1),
  }
}

export function emptyTotals(): StravaSportTotals[] {
  return SPORT_ORDER.map(sport => ({
    sport,
    count: 0,
    distanceKm: 0,
    movingTimeS: 0,
    elevationM: 0,
  }))
}

export function toHealth(o: OuraDaily): ActivityHealth {
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
    runPowerCurveRef: [],
    runPowerCurveYearRef: [],
    powerCurveYear: null,
    criticalPower: null,
    criticalPowerYear: null,
  }
}

export const refreshStravaDay = (day: StravaDay): void => {
  day.durationS = day.items.reduce((total, item) => total + item.durationS, 0)
  day.dominant =
    day.items.reduce<StravaDayItem | null>(
      (best, item) => (item.distanceKm > (best?.distanceKm ?? -1) ? item : best),
      null,
    )?.sport ?? null
}
