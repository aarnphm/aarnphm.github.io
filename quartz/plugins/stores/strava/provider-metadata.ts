import type { OuraCache } from '../oura'
import type { WahooActivityMatch, WahooStreams } from '../wahoo'
import type { WeatherActivity, WeatherCache, WeatherTemperatureSample } from '../weather'
import {
  emptyGarminFueling,
  hasGarminFueling,
  type GarminActivityMatch,
  type GarminCache,
  type GarminFueling,
  type GarminRunningDynamicsSummary,
  type GarminRunWalkData,
} from '../garmin'
import {
  DAY_MS,
  normalizeActivityDevice,
  type ActivityComputer,
  type ActivityDevice,
  type ActivityKind,
  type RawStravaActivity,
} from './raw-and-kinds'
import { delta, deltaFloat, median, round, type TimedMetricSample } from './stream-primitives'

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

export interface ActivityFueling extends GarminFueling {
  sodiumLossMg: number | null
  source: 'garmin' | 'garmin+wahoo' | 'manual' | 'wahoo'
}

export interface ActivityWeight {
  kg: number
  date: string
}

export function activityWeight(
  garmin: GarminCache | null,
  activity: RawStravaActivity,
): ActivityWeight | null {
  const samples = garmin?.weight
  if (!samples?.length) return null
  const activityDate = activity.startDateLocal.slice(0, 10)
  const startMs = Date.parse(activity.startDate)
  let sameDayBefore: ActivityWeight | null = null
  let sameDayAfter: ActivityWeight | null = null
  let earlier: ActivityWeight | null = null
  let sameDayBeforeTs = -Infinity
  let sameDayAfterTs = Infinity
  let earlierTs = -Infinity
  for (const sample of samples) {
    if (
      sample.weightKg == null ||
      !Number.isFinite(sample.weightKg) ||
      sample.weightKg <= 0 ||
      !Number.isFinite(sample.ts) ||
      !/^\d{4}-\d{2}-\d{2}$/.test(sample.date)
    )
      continue
    const weight = { kg: sample.weightKg, date: sample.date }
    if (sample.date < activityDate && sample.ts <= startMs && sample.ts > earlierTs) {
      earlier = weight
      earlierTs = sample.ts
    }
    if (sample.date !== activityDate) continue
    if (sample.ts <= startMs && sample.ts > sameDayBeforeTs) {
      sameDayBefore = weight
      sameDayBeforeTs = sample.ts
    } else if (sample.ts > startMs && sample.ts < sameDayAfterTs) {
      sameDayAfter = weight
      sameDayAfterTs = sample.ts
    }
  }
  return sameDayBefore ?? sameDayAfter ?? earlier
}

export function activityRestingHeartRate(oura: OuraCache | null, date: string): number | null {
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

export function garminVerification(
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

export function activityComputer(
  sport: ActivityKind,
  garminMatch: GarminActivityMatch | null,
  wahooMatch: WahooActivityMatch | null,
): ActivityComputer | null {
  if (sport !== 'bike') return null
  if (wahooMatch) return 'wahoo'
  return garminMatch ? 'garmin' : null
}

export function activityDevice(
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

export function temperatureAt(
  samples: WeatherTemperatureSample[],
  elapsedS: number,
): number | null {
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

export interface RespirationProjectionModel {
  intercept: number
  slope: number
}

const RESPIRATION_PROJECTION_STEP_S = 0.5
const RESPIRATION_PROJECTION_MAX_GAP_S = 5
const RESPIRATION_CALIBRATION_MIN_SAMPLES = 30
const RESPIRATION_PROJECTION_MIN_BRPM = 12
const RESPIRATION_PROJECTION_MAX_BRPM = 60

export function garminRespirationProjectionModel(
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

export function projectWahooRespiration(
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

export function activityRunWalk(
  sport: ActivityKind,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
): GarminRunWalkData | null {
  if (sport !== 'run' || !garminMatch) return null
  return garmin?.runWalks?.[garminMatch.activity.id] ?? null
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

export function activityFueling(
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

export function nearestSameDayWeatherAt(
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

export function nearestSameDayWeather(
  weather: WeatherCache,
  activity: RawStravaActivity,
): WeatherActivity | undefined {
  return nearestSameDayWeatherAt(weather, activity.startDateLocal.slice(0, 10), activity.startDate)
}
