import { round, type ActivityHeartRate, type StravaRawCache } from '../strava'
import {
  ATHLETE,
  clamp,
  DAY_MS,
  dayMs,
  mean,
  olsSlope,
  oneHz,
  TANAKA_A,
  TANAKA_B,
  type Act,
} from './primitives'
import {
  DANIELS_A,
  DANIELS_B,
  DANIELS_C,
  VO2_CEIL,
  VO2_FLOOR,
  type ObservedVo2Point,
  type Vo2Point,
  type Vo2TrendSummary,
} from './vo2-and-ftp'

export const hrMaxOf = (
  acts: Act[],
  cache: StravaRawCache,
  age: number,
  heartRateById: Map<number, ActivityHeartRate>,
): { hrMax: number; source: 'observed' | 'declared' | 'tanaka' } => {
  let obs = 0
  for (const { a } of acts) {
    const resolved = heartRateById.get(a.id)
    if (resolved?.maxHr != null && resolved.maxHr > obs) obs = resolved.maxHr
    const hr = resolved?.stream ?? cache.streams?.[String(a.id)]?.heartrate
    if (hr) for (const v of hr) if (v > obs) obs = v
  }
  const base = ATHLETE.hrMax ?? TANAKA_A - TANAKA_B * age
  return obs >= base
    ? { hrMax: obs, source: 'observed' }
    : { hrMax: Math.round(base), source: ATHLETE.hrMax != null ? 'declared' : 'tanaka' }
}

const danielsVo2 = (vms: number): number => {
  const vmin = vms * 60
  return DANIELS_A + DANIELS_B * vmin + DANIELS_C * vmin * vmin
}

export function runVo2Of(
  acts: Act[],
  cache: StravaRawCache,
  hrMax: number,
  hrRest: number | null,
  heartRateById: Map<number, ActivityHeartRate>,
): number | null {
  const vs: number[] = []
  const hs: number[] = []
  let vMaxObs = 0
  const qual: Act[] = []
  for (const x of acts) {
    if (x.sport !== 'run' || x.a.movingTime < 600) continue
    const st = cache.streams?.[String(x.a.id)]
    const hr = st?.heartrate ?? []
    const dist = st?.distance ?? []
    const n = Math.min(hr.length, dist.length)
    if (n < 60 || !oneHz(n, x.a.movingTime)) continue
    qual.push(x)
    let sum = 0
    for (let i = 1; i < n; i++) {
      const dv = dist[i] - dist[i - 1]
      sum += dv
      if (i > 10) sum -= dist[i - 10] - dist[i - 11]
      if (i >= 10) {
        const v = sum / 10
        const h = hr[i]
        if (v >= 1.5 && v <= 7 && h >= 90 && h <= hrMax) {
          vs.push(v)
          hs.push(h)
          if (v > vMaxObs) vMaxObs = v
        }
      }
    }
  }
  if (!qual.length) return null
  if (vs.length >= 120 && vMaxObs > 0) {
    const b = olsSlope(vs, hs)
    if (b != null && b > 0) {
      const a = mean(hs) - b * mean(vs)
      const vvo2 = clamp((hrMax - a) / b, vMaxObs, 1.4 * vMaxObs)
      return round(clamp(danielsVo2(vvo2), VO2_FLOOR, VO2_CEIL), 1)
    }
  }
  if (hrRest == null || hrRest >= hrMax) return null
  const ests: number[] = []
  for (const x of qual) {
    const ah = heartRateById.get(x.a.id)?.avgHr ?? x.a.averageHeartrate
    if (ah == null || ah <= hrRest) continue
    const frac = (ah - hrRest) / (hrMax - hrRest)
    if (frac <= 0.4) continue
    ests.push(danielsVo2(x.vGap) / frac)
  }
  return ests.length ? round(clamp(mean(ests), VO2_FLOOR, VO2_CEIL), 1) : null
}

const VO2_METHOD_PRIORITY: Record<ObservedVo2Point['method'], number> = {
  lab: 3,
  garmin: 2,
  apple: 1,
}

export const preferObservedVo2 = (
  current: ObservedVo2Point | null,
  candidate: ObservedVo2Point,
): ObservedVo2Point => {
  if (!current || candidate.date > current.date) return candidate
  if (
    candidate.date === current.date &&
    VO2_METHOD_PRIORITY[candidate.method] > VO2_METHOD_PRIORITY[current.method]
  )
    return candidate
  return current
}

const VO2_TREND_DAYS = 84
const VO2_TREND_STABLE_PER_WEEK = 0.1

export const buildVo2TrendSummary = (trend: Vo2Point[]): Vo2TrendSummary => {
  const latest = trend[trend.length - 1]
  if (!latest)
    return {
      direction: null,
      slopePerWeek: null,
      change28d: null,
      sampleSize: 0,
      spanDays: 0,
      method: 'none',
    }
  const cutoff = dayMs(latest.weekStart) - VO2_TREND_DAYS * DAY_MS
  const comparable = trend.filter(
    point => point.method === latest.method && dayMs(point.weekStart) >= cutoff,
  )
  const first = comparable[0]
  const spanDays = first
    ? Math.round((dayMs(latest.weekStart) - dayMs(first.weekStart)) / DAY_MS)
    : 0
  if (comparable.length < 2 || spanDays < 7)
    return {
      direction: null,
      slopePerWeek: null,
      change28d: null,
      sampleSize: comparable.length,
      spanDays,
      method: latest.method,
    }
  const firstMs = dayMs(first.weekStart)
  const xs = comparable.map(point => (dayMs(point.weekStart) - firstMs) / (7 * DAY_MS))
  const slope = olsSlope(
    xs,
    comparable.map(point => point.vo2max),
  )
  if (slope == null)
    return {
      direction: null,
      slopePerWeek: null,
      change28d: null,
      sampleSize: comparable.length,
      spanDays,
      method: latest.method,
    }
  return {
    direction:
      Math.abs(slope) < VO2_TREND_STABLE_PER_WEEK
        ? 'stable'
        : slope > 0
          ? 'improving'
          : 'declining',
    slopePerWeek: round(slope, 2),
    change28d: round(slope * 4, 1),
    sampleSize: comparable.length,
    spanDays,
    method: latest.method,
  }
}
