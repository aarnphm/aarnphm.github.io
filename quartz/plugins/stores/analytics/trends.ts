import { round, type Sport, type StravaStreams } from '../strava'
import {
  DAY_MS,
  dayMs,
  type Act,
  type ThresholdEstimate,
  type TrendMethod,
  type TrendSignal,
} from './primitives'
import {
  heartRateTrendObservations,
  TREND_MIN_SPAN_DAYS,
  type TrendObservation,
} from './thresholds'

export interface SportTrendForecastPoint {
  date: string
  value: number
  lo: number
  hi: number
}

export interface SportTrendSession {
  date: string
  /** In the trend's signal: at the reference heart rate when the trend is heart-rate anchored. */
  value: number
  /** Steady-window mean before the heart-rate correction. */
  steady: number | null
  heartRate: number | null
  windows: number
  /** Huber-weighted: the filter counted this session at reduced weight. */
  outlier: boolean
}

export interface SportTrend {
  sport: Sport
  unit: string
  invert: boolean
  method: TrendMethod
  signal: TrendSignal
  referenceHeartRate: number | null
  /** Change in speed or power per beat, in percent, from steady windows within sessions. */
  heartRateSlopePct: number | null
  stale: boolean
  sampleSize: number
  spanDays: number
  level: number | null
  slopePerWeek: number | null
  etaNow: number | null
  daysSinceLastEffort: number | null
  forecast: SportTrendForecastPoint[]
  /** Smoothed daily level of every fitted day, ending today. */
  history: SportTrendForecastPoint[]
  sessions: SportTrendSession[]
  /** The forecast issued fourteen days ago, from then to today. */
  issued: SportTrendForecastPoint[]
  /** Days whose smoothed level fell inside the 80% band issued fourteen days before. */
  hindcast: { hits: number; days: number } | null
}

const TREND_FORECAST_DAYS = 14
// The analytics "last 60 days" range; trend and readiness refit on it.
export const RECENT_RANGE_DAYS = 60
const TREND_DAMP = 0.94
const TREND_Z = 1.28
// Innovations past 2.5 standard deviations get Huber weights, so a broken heart-rate strap or a GPS
// jump moves the trend by a bounded amount.
const TREND_HUBER = 2.5

// Maximum-likelihood grid for day-to-day noise (log units) and the level and slope random walks.
const TREND_DAY_SD = [0.01, 0.015, 0.02, 0.03, 0.04, 0.055, 0.075, 0.1, 0.14, 0.2, 0.28]
const TREND_LEVEL_Q = [0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
const TREND_SLOPE_Q = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

interface TrendState {
  l: number
  s: number
  p00: number
  p01: number
  p11: number
}

interface TrendNoise {
  r: number
  ql: number
  qs: number
}

const predictTrend = (x: TrendState, phi: number, noise: TrendNoise): TrendState => ({
  l: x.l + phi * x.s,
  s: phi * x.s,
  p00: x.p00 + 2 * phi * x.p01 + phi * phi * x.p11 + noise.ql,
  p01: phi * x.p01 + phi * phi * x.p11,
  p11: phi * phi * x.p11 + noise.qs,
})

/**
 * Local linear trend stepped one calendar day at a time, so gaps between sessions widen the
 * uncertainty. Returns the predicted and filtered state of every day.
 */
function filterTrend(
  observations: readonly TrendObservation[],
  byDay: ReadonlyMap<number, number[]>,
  days: number,
  noise: TrendNoise,
): { predicted: TrendState[]; filtered: TrendState[]; logLik: number; outliers: Set<number> } {
  let x: TrendState = { l: observations[0].y, s: 0, p00: 1, p01: 0, p11: 1e-4 }
  const predicted: TrendState[] = []
  const filtered: TrendState[] = []
  const outliers = new Set<number>()
  let logLik = 0
  for (let k = 0; k < days; k++) {
    if (k > 0) x = predictTrend(x, 1, noise)
    predicted.push(x)
    for (const i of byDay.get(k) ?? []) {
      const o = observations[i]
      const v = o.y - x.l
      let r = noise.r + o.windowVar
      let S = x.p00 + r
      const z = Math.abs(v) / Math.sqrt(S)
      if (z > TREND_HUBER) {
        r *= (z / TREND_HUBER) ** 2
        S = x.p00 + r
        outliers.add(i)
      }
      const k0 = x.p00 / S
      const k1 = x.p01 / S
      x = {
        l: x.l + k0 * v,
        s: x.s + k1 * v,
        p00: x.p00 * (1 - k0),
        p01: x.p01 * (1 - k0),
        p11: x.p11 - k1 * x.p01,
      }
      logLik -= 0.5 * (Math.log(S) + (v * v) / S)
    }
    filtered.push(x)
  }
  return { predicted, filtered, logLik, outliers }
}

/** Rauch–Tung–Striebel pass: each day's level and its variance given every session. */
function smoothTrend(
  predicted: readonly TrendState[],
  filtered: readonly TrendState[],
): { l: number; v: number }[] {
  const last = filtered.length - 1
  let sm = filtered[last]
  const out = filtered.map(f => ({ l: f.l, v: f.p00 }))
  for (let k = last - 1; k >= 0; k--) {
    const f = filtered[k]
    const p = predicted[k + 1]
    const det = p.p00 * p.p11 - p.p01 * p.p01
    // Without process noise the predicted covariance can go singular; keep the filtered state.
    if (!(det > 1e-30)) {
      sm = f
      continue
    }
    // C = Pf·Fᵀ·Pp⁻¹ with F = [[1, 1], [0, 1]].
    const a00 = f.p00 + f.p01
    const a10 = f.p01 + f.p11
    const i00 = p.p11 / det
    const i01 = -p.p01 / det
    const i11 = p.p00 / det
    const c00 = a00 * i00 + f.p01 * i01
    const c01 = a00 * i01 + f.p01 * i11
    const c10 = a10 * i00 + f.p11 * i01
    const c11 = a10 * i01 + f.p11 * i11
    const dl = sm.l - p.l
    const ds = sm.s - p.s
    const d00 = sm.p00 - p.p00
    const d01 = sm.p01 - p.p01
    const d11 = sm.p11 - p.p11
    const e00 = c00 * d00 + c01 * d01
    const e01 = c00 * d01 + c01 * d11
    const e10 = c10 * d00 + c11 * d01
    const e11 = c10 * d01 + c11 * d11
    sm = {
      l: f.l + c00 * dl + c01 * ds,
      s: f.s + c10 * dl + c11 * ds,
      p00: f.p00 + e00 * c00 + e01 * c01,
      p01: f.p01 + e00 * c10 + e01 * c11,
      p11: f.p11 + e10 * c10 + e11 * c11,
    }
    out[k] = { l: sm.l, v: Math.max(0, sm.p00) }
  }
  return out
}

/** Damped projection of the level: the slope decays by TREND_DAMP per day. Index 0 is the state. */
const projectTrend = (
  x: TrendState,
  noise: TrendNoise,
  days: number,
): { l: number; v: number }[] => {
  const out = [{ l: x.l, v: x.p00 }]
  let state = x
  for (let d = 1; d <= days; d++) {
    state = predictTrend(state, TREND_DAMP, noise)
    out.push({ l: state.l, v: state.p00 })
  }
  return out
}

function fitTrendNoise(
  observations: readonly TrendObservation[],
  byDay: ReadonlyMap<number, number[]>,
  days: number,
): TrendNoise {
  let best: TrendNoise = { r: TREND_DAY_SD[0] ** 2, ql: 0, qs: 0 }
  let bestLogLik = -Infinity
  for (const daySd of TREND_DAY_SD)
    for (const ql of TREND_LEVEL_Q)
      for (const qs of TREND_SLOPE_Q) {
        const noise = { r: daySd ** 2, ql, qs }
        const { logLik } = filterTrend(observations, byDay, days, noise)
        if (logLik > bestLogLik) {
          bestLogLik = logLik
          best = noise
        }
      }
  return best
}

/** Speed ratio between two values of a trend. Air drag dominates cycling, so power converts by the cube root. */
export function trendSpeedRatio(trend: SportTrend, from: number, to: number): number {
  if (trend.signal === 'power-at-heart-rate') return Math.cbrt(to / from)
  return trend.invert ? from / to : to / from
}

interface TrendSource extends Pick<
  SportTrend,
  'sport' | 'unit' | 'invert' | 'signal' | 'referenceHeartRate' | 'heartRateSlopePct'
> {
  observations: TrendObservation[]
  /** Converts a fitted log value to the unit the trend reports. */
  human: (y: number) => number
}

/** One sport's observations. The heart-rate normalisation uses every session, whatever window is fitted. */
export function trendSource(
  acts: Act[],
  threshold: ThresholdEstimate,
  sport: Sport,
  streams: Readonly<Record<string, StravaStreams>> | undefined,
): TrendSource {
  const mine = acts.filter(x => x.sport === sport)
  const anchored = heartRateTrendObservations(mine, sport, streams)
  const signal: TrendSignal = !anchored
    ? 'pace'
    : sport === 'bike'
      ? 'power-at-heart-rate'
      : 'pace-at-heart-rate'
  const power = signal === 'power-at-heart-rate'
  return {
    sport,
    unit: power ? 'W' : threshold.unit,
    invert: !power && sport !== 'bike',
    signal,
    referenceHeartRate: anchored?.referenceHeartRate ?? null,
    heartRateSlopePct: anchored ? round(anchored.slope * 100, 2) : null,
    observations:
      anchored?.observations ??
      mine
        .filter(x => x.vGap > 0)
        .map(x => ({
          day: x.day,
          y: Math.log(x.vGap),
          windowVar: 0,
          steady: null,
          heartRate: null,
          windows: 0,
        })),
    human: y => {
      const v = Math.exp(y)
      if (power) return v
      if (sport === 'swim') return 100 / v
      if (sport === 'run') return 1000 / v
      return v * 3.6
    },
  }
}

/** Fits the sessions from `fromMs` to today. */
export function buildTrend(source: TrendSource, today: number, fromMs = -Infinity): SportTrend {
  const { observations: every, human, ...base } = source
  const observations = every.filter(o => dayMs(o.day) >= fromMs)
  const none = (level: number | null, daysSinceLastEffort: number | null): SportTrend => ({
    ...base,
    method: 'none',
    stale: true,
    sampleSize: observations.length,
    spanDays: 0,
    level,
    slopePerWeek: null,
    etaNow: null,
    daysSinceLastEffort,
    forecast: [],
    history: [],
    sessions: [],
    issued: [],
    hindcast: null,
  })
  if (observations.length === 0) return none(null, null)

  const startMs = dayMs(observations[0].day)
  const lastMs = dayMs(observations[observations.length - 1].day)
  const daysSinceLast = Math.round((today - lastMs) / DAY_MS)
  if (daysSinceLast > 45 || observations.length < 3)
    return none(round(human(observations[observations.length - 1].y), 1), daysSinceLast)

  const days = Math.round((today - startMs) / DAY_MS) + 1
  const byDay = new Map<number, number[]>()
  observations.forEach((o, i) => {
    const k = Math.round((dayMs(o.day) - startMs) / DAY_MS)
    byDay.set(k, [...(byDay.get(k) ?? []), i])
  })
  const noise = fitTrendNoise(observations, byDay, days)
  const { predicted, filtered, outliers } = filterTrend(observations, byDay, days, noise)
  const smoothed = smoothTrend(predicted, filtered)
  const lastK = days - 1
  const dateAt = (k: number): string => new Date(startMs + k * DAY_MS).toISOString().slice(0, 10)
  const banded = (k: number, p: { l: number; v: number }): SportTrendForecastPoint => {
    const half = TREND_Z * Math.sqrt(p.v)
    const a = human(p.l - half)
    const b = human(p.l + half)
    return {
      date: dateAt(k),
      value: round(human(p.l), 1),
      lo: round(Math.min(a, b), 1),
      hi: round(Math.max(a, b), 1),
    }
  }

  const issuedAt = lastK - TREND_FORECAST_DAYS
  // Each eligible day asks whether the band issued 14 days earlier held the smoothed level.
  let hits = 0
  let checked = 0
  for (let k = TREND_MIN_SPAN_DAYS + TREND_FORECAST_DAYS; k <= lastK; k++) {
    const p = projectTrend(filtered[k - TREND_FORECAST_DAYS], noise, TREND_FORECAST_DAYS)[
      TREND_FORECAST_DAYS
    ]
    checked++
    if (Math.abs(smoothed[k].l - p.l) <= TREND_Z * Math.sqrt(p.v)) hits++
  }
  const now = filtered[lastK]
  const level = human(now.l)
  return {
    ...base,
    method: 'kalman',
    stale: false,
    sampleSize: observations.length,
    spanDays: Math.round((lastMs - startMs) / DAY_MS),
    level: round(level, 1),
    slopePerWeek: round(human(now.l + 7 * now.s) - level, 2),
    etaNow: round(level, 1),
    daysSinceLastEffort: daysSinceLast,
    forecast: projectTrend(now, noise, TREND_FORECAST_DAYS)
      .slice(1)
      .map((p, i) => banded(lastK + i + 1, p)),
    history: smoothed.map((p, k) => banded(k, p)),
    sessions: observations.map((o, i) => ({
      date: o.day,
      value: round(human(o.y), 1),
      steady: o.steady == null ? null : round(human(o.steady), 1),
      heartRate: o.heartRate == null ? null : Math.round(o.heartRate),
      windows: o.windows,
      outlier: outliers.has(i),
    })),
    issued:
      issuedAt >= 0
        ? projectTrend(filtered[issuedAt], noise, TREND_FORECAST_DAYS).map((p, i) =>
            banded(issuedAt + i, p),
          )
        : [],
    hindcast: checked > 0 ? { hits, days: checked } : null,
  }
}
