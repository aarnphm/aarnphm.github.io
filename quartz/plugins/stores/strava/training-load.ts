import type {
  CalculatedExerciseLoad,
  CalculatedIntensityFactor,
  CalculatedTrainingEffect,
  CalculatedTrainingEffectAnaerobicEvidence,
  StravaActivityDetail,
  SwimActivityInterval,
} from './activity-types'
import { round } from './stream-primitives'

export const calculateActivityIntensityFactor = (
  activity: Pick<
    StravaActivityDetail,
    'sport' | 'commute' | 'avgHr' | 'npWatts' | 'deviceWatts' | 'garmin' | 'wahoo'
  >,
  paceIntensityFactor: number | null,
  ftp: number | null,
  lactateThresholdHr: number | null,
): CalculatedIntensityFactor | null => {
  if (
    activity.wahoo?.metrics.intensityFactor != null ||
    activity.garmin?.intensityFactor != null ||
    activity.commute === true ||
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
  activity: Pick<
    StravaActivityDetail,
    'commute' | 'movingTimeS' | 'garmin' | 'wahoo' | 'calculatedIntensityFactor'
  >,
): CalculatedExerciseLoad | null => {
  if (activity.commute === true || activity.garmin?.exerciseLoad != null) return null
  const source =
    activity.wahoo?.metrics.intensityFactor != null
      ? 'wahoo'
      : activity.garmin?.intensityFactor != null
        ? 'garmin'
        : (activity.calculatedIntensityFactor?.source ?? null)
  const intensityFactor =
    activity.wahoo?.metrics.intensityFactor ??
    activity.garmin?.intensityFactor ??
    activity.calculatedIntensityFactor?.value
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
