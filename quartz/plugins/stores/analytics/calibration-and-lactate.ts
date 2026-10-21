import type { GarminCache, GarminLactateThresholdValue } from '../garmin'
import type { PowerCurveBlock } from './power-to-weight'
import type {
  LactateThresholdBlock,
  LactateThresholdDeviceEstimate,
  LactateThresholdHeartRate,
  LactateThresholdSportEstimate,
} from './thresholds'
import { round, SPORT_ORDER, type Sport } from '../strava'
import {
  ATHLETE,
  CALIBRATION_PROJECTION_DAYS,
  CALIBRATION_WINDOW_DAYS,
  DAY_MS,
  dayMs,
  type Act,
  type ThresholdEstimate,
} from './primitives'
import { trendSpeedRatio, type SportTrend } from './trends'

export type CalibrationDirection = 'faster' | 'slower' | 'flat' | 'unknown'

export interface SportCalibrationPoint {
  date: string
  value: number
}

export interface SportCalibration {
  sport: Sport
  unit: string
  average: number | null
  projected: number | null
  previous: number | null
  delta: number | null
  deltaPct: number | null
  projectedDelta: number | null
  projectedDeltaPct: number | null
  direction: CalibrationDirection
  sampleSize: number
  previousSampleSize: number
  latestDate: string | null
  points: SportCalibrationPoint[]
}

export interface CalibrationSportVolume {
  sport: Sport
  currentKm: number
  previousKm: number
  deltaKm: number
  currentHours: number
  previousHours: number
  deltaHours: number
  currentLoad: number
  previousLoad: number
  deltaLoad: number
}

export interface CalibrationVolume {
  currentKm: number
  previousKm: number
  deltaKm: number
  currentHours: number
  previousHours: number
  deltaHours: number
  currentLoad: number
  previousLoad: number
  deltaLoad: number
  sports: CalibrationSportVolume[]
}

export interface CalibrationBlock {
  asOf: string
  windowDays: number
  projectionDays: number
  paces: SportCalibration[]
  volume: CalibrationVolume
}

export type CalibrationProjection = Pick<
  SportCalibration,
  'sport' | 'projected' | 'projectedDelta' | 'projectedDeltaPct'
>

const humanPaceValue = (sport: Sport, v: number): number => {
  if (sport === 'swim') return round(100 / v, 1)
  if (sport === 'run') return round(1000 / v, 1)
  return round(v * 3.6, 1)
}

const weightedHumanPace = (sport: Sport, acts: Act[]): number | null => {
  let weightedSpeed = 0
  let weight = 0
  for (const act of acts) {
    if (!(act.vGap > 0) || !(act.a.movingTime > 0)) continue
    weightedSpeed += act.vGap * act.a.movingTime
    weight += act.a.movingTime
  }
  return weight > 0 ? humanPaceValue(sport, weightedSpeed / weight) : null
}

const fasterPct = (
  sport: Sport,
  current: number | null,
  previous: number | null,
): number | null => {
  if (current == null || previous == null || !(current > 0) || !(previous > 0)) return null
  const raw = sport === 'bike' ? (current - previous) / previous : (previous - current) / previous
  return round(raw * 100, 1)
}

const directionOf = (pct: number | null): CalibrationDirection => {
  if (pct == null) return 'unknown'
  if (Math.abs(pct) < 0.25) return 'flat'
  return pct > 0 ? 'faster' : 'slower'
}

const thresholdHuman = (threshold: ThresholdEstimate): number | null =>
  threshold.vThr > 0 ? humanPaceValue(threshold.sport, threshold.vThr) : null

export const declaredLactateThresholdHeartRate = (): LactateThresholdHeartRate | null =>
  ATHLETE.lt == null ? null : { value: ATHLETE.lt, unit: 'bpm', source: 'declared' }

const usableGarminThreshold = (
  sample: GarminLactateThresholdValue | null | undefined,
  today: string,
): sample is GarminLactateThresholdValue =>
  sample != null &&
  Number.isFinite(sample.value) &&
  sample.value > 0 &&
  /^\d{4}-\d{2}-\d{2}$/.test(sample.date) &&
  Number.isFinite(Date.parse(sample.date)) &&
  new Date(sample.date).toISOString().slice(0, 10) === sample.date &&
  sample.date <= today

const lactateThresholdHistory = (
  latest: GarminLactateThresholdValue | null | undefined,
  history: readonly GarminLactateThresholdValue[],
  today: string,
): GarminLactateThresholdValue[] => {
  const days = new Map(
    history.filter(point => usableGarminThreshold(point, today)).map(point => [point.date, point]),
  )
  if (usableGarminThreshold(latest, today)) days.set(latest.date, latest)
  return [...days.values()].sort((left, right) => left.date.localeCompare(right.date))
}

/** Latest valid Garmin running LTHR, otherwise the declared anchor. */
export const resolveLactateThresholdHeartRate = (
  garmin: GarminCache | null | undefined,
  today: string,
): LactateThresholdHeartRate | null => {
  const native = garmin?.runningLactateThreshold
  const heartRate = lactateThresholdHistory(
    native?.heartRateBpm,
    native?.history?.heartRateBpm ?? [],
    today,
  ).at(-1)
  return heartRate
    ? { value: heartRate.value, unit: 'bpm', source: 'garmin', sport: 'run', date: heartRate.date }
    : declaredLactateThresholdHeartRate()
}

// Garmin derives running threshold pace from its own VO2max estimate. Use it only when it agrees
// with the heart-rate-anchored estimate from the athlete's runs.
const GARMIN_LACTATE_PACE_TOLERANCE = 0.05

const runLactateThreshold = (
  threshold: ThresholdEstimate | undefined,
  garmin: GarminCache | null | undefined,
  today: string,
): LactateThresholdSportEstimate | null => {
  const native = garmin?.runningLactateThreshold
  const speed = lactateThresholdHistory(
    native?.speedMps,
    native?.history?.speedMps ?? [],
    today,
  ).at(-1)
  const modelled = threshold && threshold.conf !== 'prior' ? threshold : null
  const deltaPct = speed && modelled ? round((speed.value / modelled.vThr - 1) * 100, 1) : null
  const device: LactateThresholdDeviceEstimate | null = speed
    ? {
        value: humanPaceValue('run', speed.value),
        date: speed.date,
        deltaPct,
        accepted:
          modelled == null || Math.abs(deltaPct ?? 0) <= GARMIN_LACTATE_PACE_TOLERANCE * 100,
      }
    : null
  if (device?.accepted)
    return {
      sport: 'run',
      unit: 's/km',
      value: device.value,
      conf: null,
      source: 'garmin',
      date: device.date,
      sampleSize: 1,
      heartRateAnchor: modelled?.heartRateAnchor ?? null,
      device,
    }
  if (!threshold) return null
  return {
    sport: 'run',
    unit: 's/km',
    value: humanPaceValue('run', threshold.vThr),
    conf: threshold.conf,
    source: threshold.method,
    date: null,
    sampleSize: threshold.heartRateAnchor?.windows ?? threshold.sampleSize,
    heartRateAnchor: threshold.heartRateAnchor,
    device,
  }
}

const bikeLactateThreshold = (
  powerCurve: PowerCurveBlock | null,
): LactateThresholdSportEstimate | null => {
  const base = { sport: 'bike', unit: 'W', heartRateAnchor: null, device: null } as const
  const recent = powerCurve?.criticalPower
  if (recent)
    return {
      ...base,
      value: Math.round(recent.criticalPowerWatts),
      conf: recent.confidence === 'medium' ? 'firm' : 'low',
      source: 'critical-power',
      date: recent.windowTo,
      sampleSize: recent.independentEffortCount,
    }
  // A calendar-year fit can predate current fitness, so it is never firm.
  const year = powerCurve?.criticalPowerYear
  if (year)
    return {
      ...base,
      value: Math.round(year.criticalPowerWatts),
      conf: 'low',
      source: 'critical-power-year',
      date: year.windowTo,
      sampleSize: year.independentEffortCount,
    }
  const twentyMinute = powerCurve?.estimatedFtp ?? powerCurve?.estimatedFtpYear
  if (twentyMinute)
    return {
      ...base,
      value: twentyMinute.watts,
      conf: 'low',
      source: 'twenty-minute-power',
      date: twentyMinute.anchor.activityDate ?? null,
      sampleSize: 1,
    }
  if (powerCurve?.ftp == null) return null
  return {
    ...base,
    value: powerCurve.ftp,
    conf: null,
    source: 'declared-ftp',
    date: null,
    sampleSize: 0,
  }
}

export const buildLactateThreshold = (
  thresholds: ReadonlyMap<Sport, ThresholdEstimate>,
  today: string,
  garmin: GarminCache | null | undefined,
  powerCurve: PowerCurveBlock | null,
): LactateThresholdBlock => {
  const native = garmin?.runningLactateThreshold
  const heartRateSeries = lactateThresholdHistory(
    native?.heartRateBpm,
    native?.history?.heartRateBpm ?? [],
    today,
  )
  const swim = thresholds.get('swim')
  const sports = [
    swim
      ? ({
          sport: 'swim',
          unit: 's/100m',
          value: humanPaceValue('swim', swim.vThr),
          conf: swim.conf,
          source: swim.method,
          date: null,
          sampleSize: swim.sampleSize,
          heartRateAnchor: null,
          device: null,
        } satisfies LactateThresholdSportEstimate)
      : null,
    bikeLactateThreshold(powerCurve),
    runLactateThreshold(thresholds.get('run'), garmin, today),
  ]
  return {
    heartRate: resolveLactateThresholdHeartRate(garmin, today),
    heartRateSeries,
    sports: sports.filter((sport): sport is LactateThresholdSportEstimate => sport != null),
  }
}

const projectedHuman = (
  average: number | null,
  trend: SportTrend | undefined,
  projectionDays: number,
): number | null => {
  if (average == null || !(average > 0)) return null
  if (!trend || trend.level == null || !(trend.level > 0)) return average
  const end =
    trend.forecast[trend.forecast.length - 1]?.value ??
    (trend.slopePerWeek == null
      ? trend.level
      : trend.level + trend.slopePerWeek * (projectionDays / 7))
  if (!(end > 0)) return average
  const ratio = trendSpeedRatio(trend, trend.level, end)
  return round(trend.invert ? average / ratio : average * ratio, 1)
}

export const calibrationProjection = (
  sport: Sport,
  average: number | null,
  trend: SportTrend | undefined,
): CalibrationProjection => {
  const projected = projectedHuman(average, trend, CALIBRATION_PROJECTION_DAYS)
  return {
    sport,
    projected,
    projectedDelta: projected != null && average != null ? round(projected - average, 1) : null,
    projectedDeltaPct: fasterPct(sport, projected, average),
  }
}

type VolumeBucket = {
  km: number
  seconds: number
  load: number
  sports: Record<Sport, { km: number; seconds: number; load: number }>
}

const emptySportVolume = (): { km: number; seconds: number; load: number } => ({
  km: 0,
  seconds: 0,
  load: 0,
})

const emptyVolumeBucket = (): VolumeBucket => ({
  km: 0,
  seconds: 0,
  load: 0,
  sports: { swim: emptySportVolume(), bike: emptySportVolume(), run: emptySportVolume() },
})

const volumeForWindow = (
  acts: Act[],
  loadById: Map<number, number>,
  fromMs: number,
  toMs: number,
): VolumeBucket => {
  const bucket = emptyVolumeBucket()
  for (const act of acts) {
    const ms = dayMs(act.day)
    if (ms < fromMs || ms > toMs) continue
    const load = loadById.get(act.a.id) ?? 0
    bucket.km += act.distanceKm
    bucket.seconds += act.a.movingTime
    bucket.load += load
    const sport = bucket.sports[act.sport]
    sport.km += act.distanceKm
    sport.seconds += act.a.movingTime
    sport.load += load
  }
  return bucket
}

const volumeBlock = (current: VolumeBucket, previous: VolumeBucket): CalibrationVolume => ({
  currentKm: round(current.km, 1),
  previousKm: round(previous.km, 1),
  deltaKm: round(current.km - previous.km, 1),
  currentHours: round(current.seconds / 3600, 1),
  previousHours: round(previous.seconds / 3600, 1),
  deltaHours: round((current.seconds - previous.seconds) / 3600, 1),
  currentLoad: round(current.load, 1),
  previousLoad: round(previous.load, 1),
  deltaLoad: round(current.load - previous.load, 1),
  sports: SPORT_ORDER.map(sport => {
    const cur = current.sports[sport]
    const prev = previous.sports[sport]
    return {
      sport,
      currentKm: round(cur.km, 1),
      previousKm: round(prev.km, 1),
      deltaKm: round(cur.km - prev.km, 1),
      currentHours: round(cur.seconds / 3600, 1),
      previousHours: round(prev.seconds / 3600, 1),
      deltaHours: round((cur.seconds - prev.seconds) / 3600, 1),
      currentLoad: round(cur.load, 1),
      previousLoad: round(prev.load, 1),
      deltaLoad: round(cur.load - prev.load, 1),
    }
  }),
})

export function buildCalibration(
  acts: Act[],
  thresholds: Map<Sport, ThresholdEstimate>,
  trends: Map<Sport, SportTrend>,
  loadById: Map<number, number>,
  today: string,
  todayMs: number,
): CalibrationBlock {
  const curFrom = todayMs - (CALIBRATION_WINDOW_DAYS - 1) * DAY_MS
  const prevFrom = curFrom - CALIBRATION_WINDOW_DAYS * DAY_MS
  const prevTo = curFrom - DAY_MS
  const paces = SPORT_ORDER.map(sport => {
    const mine = acts.filter(act => act.sport === sport && act.a.commute !== true)
    const currentActs = mine.filter(act => {
      const ms = dayMs(act.day)
      return ms >= curFrom && ms <= todayMs
    })
    const previousActs = mine.filter(act => {
      const ms = dayMs(act.day)
      return ms >= prevFrom && ms <= prevTo
    })
    const th = thresholds.get(sport)
    const average = weightedHumanPace(sport, currentActs) ?? (th ? thresholdHuman(th) : null)
    const previous = weightedHumanPace(sport, previousActs)
    const delta = average != null && previous != null ? round(average - previous, 1) : null
    const deltaPct = fasterPct(sport, average, previous)
    return {
      ...calibrationProjection(sport, average, trends.get(sport)),
      unit: th?.unit ?? (sport === 'bike' ? 'km/h' : sport === 'swim' ? 's/100m' : 's/km'),
      average,
      previous,
      delta,
      deltaPct,
      direction: directionOf(deltaPct),
      sampleSize: currentActs.length,
      previousSampleSize: previousActs.length,
      latestDate: mine.length ? mine[mine.length - 1].day : null,
      points: mine
        .slice(-90)
        .map(act => ({ date: act.day, value: humanPaceValue(sport, act.vGap) })),
    }
  })
  const currentVolume = volumeForWindow(acts, loadById, curFrom, todayMs)
  const previousVolume = volumeForWindow(acts, loadById, prevFrom, prevTo)
  return {
    asOf: today,
    windowDays: CALIBRATION_WINDOW_DAYS,
    projectionDays: CALIBRATION_PROJECTION_DAYS,
    paces,
    volume: volumeBlock(currentVolume, previousVolume),
  }
}

export const emptyCalibration = (today: string): CalibrationBlock => ({
  asOf: today,
  windowDays: CALIBRATION_WINDOW_DAYS,
  projectionDays: CALIBRATION_PROJECTION_DAYS,
  paces: [],
  volume: volumeBlock(emptyVolumeBucket(), emptyVolumeBucket()),
})
