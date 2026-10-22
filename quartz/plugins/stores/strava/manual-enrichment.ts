import type { OuraHeartRateSample } from '../oura'
import type {
  ManualFuelingEntry,
  ManualMovesEntry,
  ManualSaunaEntry,
  ManualStrengthEntry,
} from '../tracking'
import type { WeatherCache } from '../weather'
import type { ActivityHeartRateTracePoint } from './raw-and-kinds'
import { localDateTimeUtcMs } from '../../../util/local-date'
import { emptyGarminFueling, type GarminActivity, type GarminCache } from '../garmin'
import {
  refreshStravaDay,
  type ActivitySauna,
  type StravaActivityDetail,
  type StravaPayload,
} from './activity-types'
import { nearestSameDayWeatherAt, type GarminVerification } from './provider-metadata'
import { delta, deltaFloat, round } from './stream-primitives'

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
    if (
      !detail ||
      detail.date !== entry.date ||
      (detail.sport !== 'strength' && detail.sport !== 'sauna')
    )
      continue
    if (entry.exercises.length === 0 && detail.strength?.prescription) {
      detail.strength = {
        ...detail.strength,
        volumeKg: entry.volumeKg,
        totalSets: entry.totalSets,
        totalReps: entry.totalReps,
        totalsSource: 'manual',
      }
      continue
    }
    detail.strength = {
      volumeKg: entry.volumeKg,
      totalSets: entry.totalSets,
      totalReps: entry.totalReps,
      exercises: entry.exercises,
      source: 'manual',
    }
  }
}

export function applyManualMoves(
  payload: StravaPayload,
  entries: readonly ManualMovesEntry[],
): void {
  for (const entry of entries) {
    const detail = payload.details[String(entry.activityId)]
    if (!detail || detail.date !== entry.date) continue
    detail.moves = { entries: entry.moves, source: 'manual' }
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
      location: entry.location ?? null,
      temperatureC: entry.temperatureC,
      humidityPct: entry.humidityPct,
      cooldown: entry.cooldown,
      ...(entry.lapPhases ? { lapPhases: entry.lapPhases } : {}),
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
