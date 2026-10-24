import type { SleepMetrics } from '../../../util/sleep-metrics'
import type { WeeklyTargetRange } from '../../../util/weekly-target-range'
import type { GarminHealthDay } from '../garmin-health'
import type { OuraHealthDay } from '../oura'
import { runGradeFactor, runGradeFactors } from '../../../util/run-grade'
import {
  ACTIVITY_LOAD_INTENSITY_FACTOR_CAP,
  round,
  type RawStravaActivity,
  type Sport,
  type StravaStreams,
} from '../strava'

export type Conf = 'firm' | 'low' | 'prior' | 'stale'
export type TsbZone = 'fresh' | 'neutral' | 'fatigued' | 'deep'
export type AcwrState = 'building' | 'low' | 'ok' | 'caution' | 'high'
export type TrendMethod = 'none' | 'kalman'
/** pace: session average. The heart-rate signals are speed or power at the trend's reference heart rate. */
export type TrendSignal = 'pace' | 'pace-at-heart-rate' | 'power-at-heart-rate'
export const RACE_DISTANCES = ['sprint', 'olympic', 't50', '70.3', 't100', 'ironman'] as const
export type RaceDistance = (typeof RACE_DISTANCES)[number]

export interface MethodConstants {
  ctlTau: number
  atlTau: number
  k42: number
  k7: number
  ifCap: number
  thresholdWindowDays: number
  seededFrom: string
  note: string
}

export interface AnalyticsMeta {
  athleteId: number
  today: string
  windowFrom: string
  windowTo: string
  activityCount: number
  method: MethodConstants
}

export interface DailyPoint {
  date: string
  load: number
  effort: number
  swimLoad: number
  bikeLoad: number
  runLoad: number
  ctl: number
  atl: number
  tsb: number
  swimCtl: number
  bikeCtl: number
  runCtl: number
  readiness: number | null
  hrv: number | null
  rhr: number | null
  sleepScore: number | null
  sleepDurationS: number | null
  napDurationS?: number | null
  napCount?: number | null
  garminHealth?: GarminHealthDay | null
  ouraHealth?: OuraHealthDay | null
  sleepMetrics: SleepMetrics | null
  tempDevC: number | null
  weightKg: number | null
  totalCalories: number | null
  intakeKcal: number | null
  warmup: boolean
}

export interface WeeklyPoint {
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

export interface ThresholdEstimate {
  sport: Sport
  vThr: number
  paceLabel: string
  unit: string
  conf: Conf
  /** heart-rate-anchored: run speed at LTHR; pace-p90: 90th percentile of session averages. */
  method: 'heart-rate-anchored' | 'pace-p90' | 'pace-max' | 'prior'
  sampleSize: number
  staleDays: number
  heartRateAnchor: RunHeartRateAnchor | null
}

export interface SportBest {
  sport: Sport
  count: number
  totalKm: number
  totalTimeS: number
  fastestRate: number | null
  fastestUnit: string
  longestKm: number
  biggestClimbM: number
  bestToDate: { date: string; rate: number }[]
}

export interface RiskBlock {
  ctl: number
  atl: number
  tsb: number
  tsbZone: TsbZone
  rampWeek: number
  acwr: number | null
  acwrState: AcwrState
  monotony: number | null
  strain: number | null
}

export interface TrainingAction {
  text: string
  sourceMetric: string
  value: string
}

export type RecoveryStatus = 'building' | 'low' | 'firm'
export type FlagSeverity = 'info' | 'watch' | 'alert'

export interface RecoveryFlag {
  id: string
  severity: FlagSeverity
  label: string
  detail: string
  metric: string
  value: number | null
  baseline: number | null
}

export interface RecoveryThresholds {
  hrvWatchZ: number
  hrvAlertZ: number
  rhrWatchZ: number
  rhrAlertZ: number
  rhrWatchBpm: number
  rhrAlertBpm: number
  tempWatchC: number
  tempAlertC: number
  sleepTargetS: number
  sleepFloorS: number
  sleepDebtWatchS: number
  sleepDebtAlertS: number
  readinessFloor: number
  readinessOptimal: number
}

export interface Act {
  a: RawStravaActivity
  sport: Sport
  day: string
  distanceKm: number
  vGap: number
}

export interface SupplementalLoadActivity {
  a: RawStravaActivity
  day: string
  load: number
}

export interface EffortBucket {
  effort: number
  sessions: number
}

export interface SwimActivityMetrics {
  paceSPer100m: number | null
  strokeRateSpm: number | null
  strokes: Record<string, number> | null
}

export interface RunningDynamicsMetrics {
  strideLengthM: number | null
  verticalOscillationCm: number | null
}

export const DAY_MS = 86_400_000
export const KG_PER_LB = 0.45359237
export const K42 = 1 - Math.exp(-1 / 42)
export const K7 = 1 - Math.exp(-1 / 7)
export const IF_CAP = ACTIVITY_LOAD_INTENSITY_FACTOR_CAP
export const CALIBRATION_WINDOW_DAYS = 28
export const CALIBRATION_PROJECTION_DAYS = 14
export const ANALYTICS_LOAD_NOTE =
  'Load uses pace and duration for swim, bike, and run. Strength and yoga use threshold-normalized heart-rate TRIMP when exercise and resting heart rate are available.'

export const SPORT_PRIOR: Record<Sport, number> = { swim: 1.3, bike: 6.9, run: 3.3 }
export const LOAD_SHARE_TARGET: Record<Sport, number> = { swim: 0.2, bike: 0.5, run: 0.3 }

export const dayMs = (iso: string): number => Date.parse(`${iso}T00:00:00Z`)
export const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x))
export const mean = (xs: number[]): number =>
  xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0
export const sd = (xs: number[]): number => {
  if (xs.length < 2) return 0
  const m = mean(xs)
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1))
}

export function mmss(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds))
  const m = Math.floor(s / 60)
  const r = s % 60
  return `${m}:${String(r).padStart(2, '0')}`
}

export function weightedPercentile(values: number[], weights: number[], p: number): number {
  const pairs = values.map((v, i) => ({ v, w: weights[i] })).sort((a, b) => a.v - b.v)
  const total = pairs.reduce((s, x) => s + x.w, 0)
  if (total <= 0) return pairs.length ? pairs[pairs.length - 1].v : 0
  const target = p * total
  let acc = 0
  for (const pair of pairs) {
    acc += pair.w
    if (acc >= target) return pair.v
  }
  return pairs[pairs.length - 1].v
}

export function gradeAdjSpeed(
  a: RawStravaActivity,
  sport: Sport,
  stream: StravaStreams | undefined,
): number {
  const v = a.averageSpeed
  if (sport === 'swim') return v
  const distM = a.distance > 0 ? a.distance : 1
  const gOverall = a.totalElevationGain / distM
  if (sport === 'bike') {
    const factor = Math.min(1 + 3.5 * Math.max(gOverall, 0), 1.25)
    return v * factor
  }
  const factors = runGradeFactors(stream)
  if (factors) {
    const dist = stream!.distance
    let weighted = 0
    let length = 0
    for (let i = 1; i < factors.length; i++) {
      const segment = dist[i] - dist[i - 1]
      if (!(segment > 0)) continue
      weighted += factors[i] * segment
      length += segment
    }
    if (length > 0) return v * (weighted / length)
  }
  // Strava reports gain only. Treat the route as a loop that climbs and descends over equal halves.
  return v * ((runGradeFactor(2 * gOverall) + runGradeFactor(-2 * gOverall)) / 2)
}

export interface RunHeartRateAnchor {
  heartRateBpm: number
  windows: number
  runs: number
  maxWindowHeartRateBpm: number
  lookbackDays: number
}

export const RUN_MOVING_MIN_MPS = 1609.344 / (30 * 60)

export const HRV_BASE_SPAN = 28
export const HRV_BASE_MIN = 14
export const ACUTE_SPAN = 7
export const ACUTE_MIN = 3
export const SLEEP_DEBT_SPAN = 14
export const RAMP_HOT = 0.1

export const winValues = (a: (number | null)[], i: number, span: number): number[] => {
  const xs: number[] = []
  for (let j = Math.max(0, i - span + 1); j <= i; j++) {
    const v = a[j]
    if (v != null) xs.push(v)
  }
  return xs
}

export const winStats = (
  a: (number | null)[],
  i: number,
  span: number,
  minN: number,
): { mean: number; sd: number; n: number } | null => {
  const xs = winValues(a, i, span)
  return xs.length >= minN ? { mean: mean(xs), sd: sd(xs), n: xs.length } : null
}

export const winMedian = (
  a: (number | null)[],
  i: number,
  span: number,
  minN: number,
): number | null => {
  const xs = winValues(a, i, span)
  if (xs.length < minN) return null
  xs.sort((p, q) => p - q)
  const m = xs.length >> 1
  return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2
}

export const ATHLETE = {
  sex: 'M' as 'M' | 'F',
  born: '2001-03',
  bornAnchor: '2001-03-01',
  hrMax: 200 as number | null,
  vo2max: 47.8 as number | null,
  ftp: 231 as number | null,
  lt: 174 as number | null,
  goalWeightLb: 170 as number | null,
  goalFTP: 368 as number | null,
  heightCm: 188,
  tenKmRaceTime: '00:46:20',
}

export const goalWeightKg = ATHLETE.goalWeightLb != null ? ATHLETE.goalWeightLb * KG_PER_LB : null

export const TANAKA_A = 208
export const TANAKA_B = 0.7

export const RUN_SPRINT_WIN_S = 30
export const SPRINT_CAP_MS: Record<'swim' | 'run', number> = { swim: 3, run: 12 }

const ONE_HZ_TOL = 0.15
const DECOUPLE_MIN_S = 1200
export const PEAK_WINDOWS = [30, 60, 300, 1200] as const

export const ageOn = (iso: string): number =>
  Math.floor((dayMs(iso) - dayMs(ATHLETE.bornAnchor)) / (365.25 * DAY_MS))

export const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((left, right) => left - right)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

export const olsSlope = (xs: number[], ys: number[]): number | null => {
  if (xs.length < 3) return null
  const mx = mean(xs)
  const my = mean(ys)
  let num = 0
  let den = 0
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - mx) * (ys[i] - my)
    den += (xs[i] - mx) ** 2
  }
  return den === 0 ? null : num / den
}

export const oneHz = (n: number, movingTimeS: number): boolean =>
  movingTimeS > 0 && n > 0 && Math.abs(n / movingTimeS - 1) <= ONE_HZ_TOL

export const peakMean = (xs: number[], w: number): number | null => {
  if (w < 1 || xs.length < w) return null
  let sum = 0
  let best = -Infinity
  for (let i = 0; i < xs.length; i++) {
    sum += xs[i]
    if (i >= w) sum -= xs[i - w]
    if (i >= w - 1 && sum / w > best) best = sum / w
  }
  return Number.isFinite(best) ? best : null
}

export const peakDistRate = (d: number[], w: number): number | null => {
  if (d.length <= w) return null
  let best = -Infinity
  for (let i = w; i < d.length; i++) {
    const r = (d[i] - d[i - w]) / w
    if (r > best) best = r
  }
  return Number.isFinite(best) && best >= 0 ? best : null
}

const np30 = (watts: number[]): number | null => {
  if (watts.length < 30) return null
  let sum = 0
  let acc = 0
  let count = 0
  for (let i = 0; i < watts.length; i++) {
    sum += watts[i]
    if (i >= 30) sum -= watts[i - 30]
    if (i >= 29) {
      acc += (sum / 30) ** 4
      count++
    }
  }
  return count > 0 ? Math.pow(acc / count, 0.25) : null
}

export const efOf = (
  sport: Sport,
  a: RawStravaActivity,
  vGap: number,
  avgHr: number | null | undefined = a.averageHeartrate,
): number | null => {
  const hr = avgHr
  if (hr == null || hr <= 0) return null
  if (sport === 'bike')
    return a.weightedAverageWatts != null ? round(a.weightedAverageWatts / hr, 2) : null
  return vGap > 0 ? round((vGap * 60) / hr, 2) : null
}

export const decouplingOf = (
  sport: Sport,
  stream: StravaStreams | undefined,
  movingTimeS: number,
): number | null => {
  if (sport === 'swim' || !stream || movingTimeS < DECOUPLE_MIN_S) return null
  const hr = stream.heartrate ?? []
  const metric = sport === 'bike' ? (stream.watts ?? []) : stream.distance
  const n = Math.min(hr.length, metric.length)
  if (n < 60 || !oneHz(n, movingTimeS)) return null
  const mid = n >> 1
  const efHalf = (lo: number, hiEnd: number): number | null => {
    const h = mean(hr.slice(lo, hiEnd))
    if (!(h > 0)) return null
    if (sport === 'bike') {
      const np = np30(metric.slice(lo, hiEnd))
      return np != null && np > 0 ? np / h : null
    }
    const span = hiEnd - lo - 1
    if (span < 1) return null
    const v = (metric[hiEnd - 1] - metric[lo]) / span
    return v > 0 ? v / h : null
  }
  const ef1 = efHalf(0, mid)
  const ef2 = efHalf(mid, n)
  if (ef1 == null || ef2 == null || !(ef1 > 0)) return null
  return round(((ef1 - ef2) / ef1) * 100, 1)
}
