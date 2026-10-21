import type { PowerCurvePoint, StravaActivityDetail } from '../../plugins/stores/strava'
import type { Locale } from '../triathlon-presentation'
import {
  powerCurvePointAt,
  powerCurveSecondsAt,
  powerCurveWattsAt,
  type PowerCurveData,
} from '../power-curve'
import { axisNumber, niceStep } from './chart-geometry'
import { dlabel } from './metric-text'

export type PowerCurveHover = {
  index: number
  durationS: number
  watts: number
  referenceWatts: number | null
  xPct: number
}

const POWER_CURVE_PATH_POINTS = 1_024

export const powerCurveFraction = (
  seconds: number,
  minSeconds: number,
  maxSeconds: number,
): number => {
  if (minSeconds <= 0 || maxSeconds <= minSeconds) return 0
  const value = Math.min(maxSeconds, Math.max(minSeconds, seconds))
  return (Math.log(value) - Math.log(minSeconds)) / (Math.log(maxSeconds) - Math.log(minSeconds))
}

export const POWER_CURVE_EXTRA_AXIS_MARKERS = [180, 360, 720]
const POWER_CURVE_AXIS_MARKERS = [5, 10, 20, 30, 120, ...POWER_CURVE_EXTRA_AXIS_MARKERS]
const POWER_CURVE_ENDPOINT_GAP = 0.12
const EMBEDDED_POWER_CURVE_PRECISE_ENDPOINT_GAP = 0.14

export const powerCurveDurationTicks = (
  minSeconds: number,
  maxSeconds: number,
  durations: readonly number[],
): number[] =>
  [...new Set([...durations, ...POWER_CURVE_AXIS_MARKERS, maxSeconds])]
    .filter(seconds => seconds >= minSeconds && seconds <= maxSeconds)
    .sort((left, right) => left - right)
    .filter(
      seconds =>
        seconds === maxSeconds ||
        powerCurveFraction(seconds, minSeconds, maxSeconds) <= 1 - POWER_CURVE_ENDPOINT_GAP,
    )

export const embeddedPowerCurveDurationTicks = (
  minSeconds: number,
  maxSeconds: number,
  durations: readonly number[],
): number[] => {
  const endpointGap = dlabel(maxSeconds).endsWith('s')
    ? EMBEDDED_POWER_CURVE_PRECISE_ENDPOINT_GAP
    : POWER_CURVE_ENDPOINT_GAP
  return powerCurveDurationTicks(minSeconds, maxSeconds, durations).filter(
    seconds =>
      seconds !== 10 &&
      seconds !== 20 &&
      (seconds === maxSeconds ||
        powerCurveFraction(seconds, minSeconds, maxSeconds) <= 1 - endpointGap),
  )
}

const nearestPowerCurveIndex = (curve: PowerCurveData, seconds: number): number => {
  let low = 0
  let high = curve.length - 1
  while (low < high) {
    const mid = Math.floor((low + high) / 2)
    if (powerCurveSecondsAt(curve, mid) < seconds) low = mid + 1
    else high = mid
  }
  if (
    low > 0 &&
    Math.abs(Math.log(powerCurveSecondsAt(curve, low - 1)) - Math.log(seconds)) <
      Math.abs(Math.log(powerCurveSecondsAt(curve, low)) - Math.log(seconds))
  )
    return low - 1
  return low
}

export const powerCurvePathPoints = (curve: PowerCurvePoint[]): PowerCurvePoint[] => {
  if (curve.length <= POWER_CURVE_PATH_POINTS) return curve
  const minSeconds = curve[0].s
  const maxSeconds = curve[curve.length - 1].s
  const points: PowerCurvePoint[] = []
  let previousIndex = -1
  for (let sample = 0; sample < POWER_CURVE_PATH_POINTS; sample++) {
    const fraction = sample / (POWER_CURVE_PATH_POINTS - 1)
    const seconds = Math.exp(
      Math.log(minSeconds) + fraction * (Math.log(maxSeconds) - Math.log(minSeconds)),
    )
    const index = nearestPowerCurveIndex(curve, seconds)
    if (index === previousIndex) continue
    points.push(curve[index])
    previousIndex = index
  }
  if (previousIndex !== curve.length - 1) points.push(curve[curve.length - 1])
  return points
}

export const powerCurveHoverAt = (
  curve: PowerCurveData,
  reference: PowerCurveData,
  pointerFraction: number,
): PowerCurveHover | null => {
  if (curve.length < 2) return null
  const fraction = Math.min(1, Math.max(0, pointerFraction))
  const minSeconds = powerCurveSecondsAt(curve, 0)
  const maxSeconds = powerCurveSecondsAt(curve, curve.length - 1)
  const targetSeconds = Math.exp(
    Math.log(minSeconds) + fraction * (Math.log(maxSeconds) - Math.log(minSeconds)),
  )
  const index = nearestPowerCurveIndex(curve, targetSeconds)
  const point = powerCurvePointAt(curve, index)
  let referenceWatts: number | null = null
  let low = 0
  let high = reference.length - 1
  while (low <= high) {
    const mid = Math.floor((low + high) / 2)
    if (powerCurveSecondsAt(reference, mid) < point.s) low = mid + 1
    else if (powerCurveSecondsAt(reference, mid) > point.s) high = mid - 1
    else {
      referenceWatts = powerCurveWattsAt(reference, mid)
      break
    }
  }
  return {
    index,
    durationS: point.s,
    watts: point.w,
    referenceWatts,
    xPct: powerCurveFraction(point.s, minSeconds, maxSeconds) * 100,
  }
}

export type PowerCurveRange = 'six-weeks' | 'year'

export const powerCurveWeight = (
  detail: StravaActivityDetail,
): StravaActivityDetail['powerCurveWeight'] => {
  const weight = detail.powerCurveWeight
  const eligibleDate = (date: string): boolean =>
    /^\d{4}-\d{2}-\d{2}$/.test(date) && date <= detail.date
  if (weight && Number.isFinite(weight.kg) && weight.kg > 0 && eligibleDate(weight.date))
    return weight
  const kg = detail.bestEfforts?.weightKg
  const date = detail.bestEfforts?.weightDate
  return kg != null && Number.isFinite(kg) && kg > 0 && date != null && eligibleDate(date)
    ? { kg, date, source: 'garmin' }
    : undefined
}

export const powerCurveValueText = (
  watts: number,
  weightKg: number | null,
  locale: Locale,
  compact = false,
): string => {
  const relative = weightKg != null && Number.isFinite(weightKg) && weightKg > 0
  const value = relative ? watts / weightKg : watts
  return `${value.toLocaleString(locale === 'fr' ? 'fr-CA' : 'en-US', {
    minimumFractionDigits: relative ? 2 : 0,
    maximumFractionDigits: relative ? 2 : 1,
  })}${compact ? '' : ' '}${relative ? 'W/kg' : 'W'}`
}

export const POWER_CURVE_WATT_STEP = 200

export const powerCurveAxisTicks = (
  maxWatts: number,
  weightKg: number | null,
  locale: Locale,
): { label: string; watts: number }[] => {
  const mass = weightKg != null && Number.isFinite(weightKg) && weightKg > 0 ? weightKg : null
  const max = maxWatts / (mass ?? 1)
  const step = mass == null ? POWER_CURVE_WATT_STEP : niceStep(max, 4)
  if (!Number.isFinite(max) || max <= 0 || !Number.isFinite(step) || step <= 0) return []
  return Array.from({ length: Math.floor(max / step + 1e-9) + 1 }, (_, index) => {
    const value = index * step
    return {
      label:
        value === 0
          ? '0'
          : mass == null
            ? `${axisNumber(value, step)}w`
            : `${value.toLocaleString(locale === 'fr' ? 'fr-CA' : 'en-US', { maximumFractionDigits: 2 })} W/kg`,
      watts: value * (mass ?? 1),
    }
  })
}

export const normalizePowerCurvePoints = (
  curve: readonly PowerCurvePoint[] | null,
): PowerCurvePoint[] => {
  if (!curve) return []
  const sorted = curve
    .filter(
      point => Number.isFinite(point.s) && point.s > 0 && Number.isFinite(point.w) && point.w >= 0,
    )
    .map(point => ({
      s: point.s,
      w: point.w,
      ...(point.activityId == null ? {} : { activityId: point.activityId }),
      ...(point.activityDate == null ? {} : { activityDate: point.activityDate }),
    }))
    .sort((a, b) => a.s - b.s)
  const points: PowerCurvePoint[] = []
  for (const point of sorted) {
    if (points[points.length - 1]?.s === point.s) continue
    points.push(point)
  }
  return points
}

export const nearestPowerCurvePoint = (
  points: PowerCurveData,
  durationS: number,
): PowerCurvePoint | null => {
  if (
    points.length === 0 ||
    !Number.isFinite(durationS) ||
    durationS < powerCurveSecondsAt(points, 0) ||
    durationS > powerCurveSecondsAt(points, points.length - 1)
  )
    return null
  return powerCurvePointAt(points, nearestPowerCurveIndex(points, durationS))
}

export const nearestPowerCurveValue = (points: PowerCurveData, durationS: number): number | null =>
  nearestPowerCurvePoint(points, durationS)?.w ?? null
