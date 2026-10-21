import type { GarminCache, GarminCyclingDynamics } from '../garmin'
import type { PowerCurveBlock } from './power-to-weight'
import {
  calculateFtpHypothesis,
  FTP_HYPOTHESIS_DEFAULTS,
  type FtpHypothesisParams,
} from '../../../util/ftp-hypothesis'
import { criticalPowerAtDuration, type CriticalPowerEstimate } from '../critical-power'
import { round } from '../strava'
import { clamp, DAY_MS, dayMs, median, type Conf } from './primitives'

export type Vo2Method = 'garmin' | 'apple' | 'bike' | 'run' | 'hrratio' | 'lab' | 'none'

export type CardioDir = 'improving' | 'stable' | 'declining' | null
export type CardioKey = 'rhr' | 'hrv' | 'ef' | 'decoupling'

export interface Vo2Point {
  weekStart: string
  vo2max: number
  method: Vo2Method
}

export interface Vo2TrendSummary {
  direction: CardioDir
  slopePerWeek: number | null
  change28d: number | null
  sampleSize: number
  spanDays: number
  method: Vo2Method
}

export interface Vo2Estimate {
  method: Vo2Method
  vo2max: number
  conf: Conf
}

export interface Vo2BikeSource {
  ftpW: number
  ftpSource: 'athlete' | 'strava' | 'derived'
  mapW: number
  weightKg: number
  labBaseline: { date: string; vo2max: number; ftpW: number; weightKg: number } | null
}

export interface Vo2maxBlock {
  value: number | null
  method: Vo2Method
  conf: Conf
  hrMax: number
  hrMaxSource: 'observed' | 'declared' | 'tanaka'
  hrRest: number | null
  chronoAge: number
  fitnessAge: number | null
  ageDeltaYears: number | null
  percentileForAge: number | null
  estimates: Vo2Estimate[]
  trend: Vo2Point[]
  trendSummary: Vo2TrendSummary
  bikeSource: Vo2BikeSource | null
}

export type FtpHypothesisMassSource = 'daily' | 'lab'
export type FtpHypothesisVo2maxSource = 'garmin' | 'apple' | 'lab' | 'default'
export type FtpHypothesisVo2maxSport = 'running' | 'cycling' | 'unknown'
export type FtpHypothesisEfficiencySource = 'literature-prior' | 'measured-metabolic'

export interface FtpHypothesisEfficiencyEvidence {
  valuePct: number
  source: FtpHypothesisEfficiencySource
  conf: Conf
}

export interface FtpHypothesisPedalingEvidence {
  windowFrom: string
  windowTo: string
  activityCount: number
  sampleCount: number
  coveragePct: number
  leftPedalSmoothnessPct: number
  rightPedalSmoothnessPct: number
  leftTorqueEffectivenessPct: number
  rightTorqueEffectivenessPct: number
}

export interface FtpHypothesisPowerEvidence {
  criticalPowerWatts: number | null
  modeled60MinuteWatts: number | null
  confidence: CriticalPowerEstimate['confidence'] | null
  independentEffortCount: number
  declaredFtpWatts: number | null
}

export interface FtpHypothesisSources {
  massDate: string
  massSource: FtpHypothesisMassSource
  vo2maxDate: string
  vo2maxSource: FtpHypothesisVo2maxSource
  vo2maxSport: FtpHypothesisVo2maxSport
  defaultVo2max: number | null
  efficiency: FtpHypothesisEfficiencyEvidence
  pedaling: FtpHypothesisPedalingEvidence | null
  power: FtpHypothesisPowerEvidence
}

export interface FtpHypothesis {
  date: string
  conf: Conf
  massKg: number
  massDate: string
  massSource: FtpHypothesisMassSource
  vo2max: number
  vo2maxDate: string
  vo2maxSource: FtpHypothesisVo2maxSource
  vo2maxSport: FtpHypothesisVo2maxSport
  defaultVo2max: number | null
  crossModalDiscountPct: number
  thresholdPct: number
  grossEfficiencyPct: number
  absoluteVo2: number
  cyclingVo2max: number
  thresholdVo2: number
  metabolicWatts: number
  efficiencyFtp: number
  acsmMapWatts: number
  acsmFtp: number
  ftp: number
  low: number
  high: number
  wattsPerKg: number
  efficiency: FtpHypothesisEfficiencyEvidence
  pedaling: FtpHypothesisPedalingEvidence | null
  power: FtpHypothesisPowerEvidence
  note: string
}

export const VO2_FLOOR = 20
export const VO2_CEIL = 80
export const FRIEND_MED_M: [number, number][] = [
  [25, 48],
  [35, 42.4],
  [45, 37.8],
  [55, 32.6],
  [65, 28.2],
  [75, 24.4],
]
const FRIEND_PCT_M: { age: number; rows: [number, number][] }[] = [
  {
    age: 25,
    rows: [
      [10, 32.1],
      [25, 40.1],
      [50, 48],
      [75, 55.2],
      [90, 61.8],
    ],
  },
  {
    age: 35,
    rows: [
      [10, 30.2],
      [25, 35.9],
      [50, 42.4],
      [75, 49.2],
      [90, 56.5],
    ],
  },
  {
    age: 45,
    rows: [
      [10, 26.8],
      [25, 31.9],
      [50, 37.8],
      [75, 45],
      [90, 52.1],
    ],
  },
  {
    age: 55,
    rows: [
      [10, 22.8],
      [25, 27.1],
      [50, 32.6],
      [75, 39.7],
      [90, 45.6],
    ],
  },
  {
    age: 65,
    rows: [
      [10, 19.8],
      [25, 23.7],
      [50, 28.2],
      [75, 34.5],
      [90, 40.3],
    ],
  },
  {
    age: 75,
    rows: [
      [10, 17.1],
      [25, 20.4],
      [50, 24.4],
      [75, 30.4],
      [90, 36.6],
    ],
  },
]
export const ACSM_WATT_K = 10.8
export const ACSM_BASE = 7
export const MAP_FTP_RATIO = 0.75
export const DANIELS_A = -4.6
export const DANIELS_B = 0.182258
export const DANIELS_C = 0.000104
export const UTH_K = 15.3

const pedalingMetric = (values: readonly (number | null)[], sampleCount: number): number[] =>
  values
    .slice(0, sampleCount)
    .filter((value): value is number => value != null && value >= 0 && value <= 100)

const pedalingActivityEvidence = (
  dynamics: GarminCyclingDynamics,
): {
  sampleCount: number
  validCount: number
  leftPedalSmoothnessPct: number
  rightPedalSmoothnessPct: number
  leftTorqueEffectivenessPct: number
  rightTorqueEffectivenessPct: number
} | null => {
  const sampleCount = dynamics.time.length
  if (sampleCount === 0) return null
  const leftPedalSmoothness = pedalingMetric(dynamics.leftPedalSmoothness, sampleCount)
  const rightPedalSmoothness = pedalingMetric(dynamics.rightPedalSmoothness, sampleCount)
  const leftTorqueEffectiveness = pedalingMetric(dynamics.leftTorqueEffectiveness, sampleCount)
  const rightTorqueEffectiveness = pedalingMetric(dynamics.rightTorqueEffectiveness, sampleCount)
  const metrics = [
    leftPedalSmoothness,
    rightPedalSmoothness,
    leftTorqueEffectiveness,
    rightTorqueEffectiveness,
  ]
  if (metrics.some(values => values.length / sampleCount < 0.8)) return null
  return {
    sampleCount,
    validCount: metrics.reduce((sum, values) => sum + values.length, 0),
    leftPedalSmoothnessPct: median(leftPedalSmoothness),
    rightPedalSmoothnessPct: median(rightPedalSmoothness),
    leftTorqueEffectivenessPct: median(leftTorqueEffectiveness),
    rightTorqueEffectivenessPct: median(rightTorqueEffectiveness),
  }
}

export const buildFtpPedalingEvidence = (
  garmin: GarminCache | null | undefined,
  today: string,
): FtpHypothesisPedalingEvidence | null => {
  if (!garmin?.cyclingDynamics) return null
  const cutoff = dayMs(today) - 42 * DAY_MS
  const activities: {
    date: string
    evidence: NonNullable<ReturnType<typeof pedalingActivityEvidence>>
  }[] = []
  for (const [activityId, dynamics] of Object.entries(garmin.cyclingDynamics)) {
    const activity = garmin.activities[activityId]
    const date = activity?.startDateLocal.slice(0, 10)
    if (!activity || activity.sport !== 'bike' || !date || dayMs(date) < cutoff) continue
    const evidence = pedalingActivityEvidence(dynamics)
    if (evidence) activities.push({ date, evidence })
  }
  if (activities.length === 0) return null
  activities.sort((left, right) => left.date.localeCompare(right.date))
  const sampleCount = activities.reduce((sum, activity) => sum + activity.evidence.sampleCount, 0)
  const validCount = activities.reduce((sum, activity) => sum + activity.evidence.validCount, 0)
  return {
    windowFrom: activities[0].date,
    windowTo: activities.at(-1)?.date ?? activities[0].date,
    activityCount: activities.length,
    sampleCount,
    coveragePct: round((validCount / (sampleCount * 4)) * 100, 1),
    leftPedalSmoothnessPct: round(
      median(activities.map(activity => activity.evidence.leftPedalSmoothnessPct)),
      1,
    ),
    rightPedalSmoothnessPct: round(
      median(activities.map(activity => activity.evidence.rightPedalSmoothnessPct)),
      1,
    ),
    leftTorqueEffectivenessPct: round(
      median(activities.map(activity => activity.evidence.leftTorqueEffectivenessPct)),
      1,
    ),
    rightTorqueEffectivenessPct: round(
      median(activities.map(activity => activity.evidence.rightTorqueEffectivenessPct)),
      1,
    ),
  }
}

export const ftpPowerEvidence = (powerCurve: PowerCurveBlock): FtpHypothesisPowerEvidence => {
  const estimate = powerCurve.criticalPower
  return {
    criticalPowerWatts: estimate ? Math.round(estimate.criticalPowerWatts) : null,
    modeled60MinuteWatts: estimate ? Math.round(criticalPowerAtDuration(estimate, 60 * 60)) : null,
    confidence: estimate?.confidence ?? null,
    independentEffortCount: estimate?.independentEffortCount ?? 0,
    declaredFtpWatts: powerCurve.ftp,
  }
}

export function computeFtpHypothesisFromVo2(
  date: string,
  vo2max: number,
  massKg: number,
  params: FtpHypothesisParams = FTP_HYPOTHESIS_DEFAULTS,
  sources: FtpHypothesisSources = {
    massDate: date,
    massSource: 'lab',
    vo2maxDate: date,
    vo2maxSource: 'lab',
    vo2maxSport: 'running',
    defaultVo2max: vo2max,
    efficiency: { valuePct: params.grossEfficiencyPct, source: 'literature-prior', conf: 'prior' },
    pedaling: null,
    power: {
      criticalPowerWatts: null,
      modeled60MinuteWatts: null,
      confidence: null,
      independentEffortCount: 0,
      declaredFtpWatts: null,
    },
  },
): FtpHypothesis | null {
  const calculation = calculateFtpHypothesis(vo2max, massKg, params)
  if (!calculation) return null
  return {
    date,
    conf: 'low',
    massDate: sources.massDate,
    massSource: sources.massSource,
    vo2maxDate: sources.vo2maxDate,
    vo2maxSource: sources.vo2maxSource,
    vo2maxSport: sources.vo2maxSport,
    defaultVo2max: sources.defaultVo2max == null ? null : round(sources.defaultVo2max, 1),
    ...calculation,
    efficiency: { ...sources.efficiency, valuePct: calculation.grossEfficiencyPct },
    pedaling: sources.pedaling,
    power: sources.power,
    note:
      sources.vo2maxSport === 'cycling'
        ? 'This estimate comes from cycling VO2max.'
        : sources.vo2maxSport === 'running'
          ? 'This estimate comes from running VO2max.'
          : 'This estimate comes from VO2max with unknown sport provenance.',
  }
}

export const invLerp = (tbl: [number, number][], y: number): number => {
  const n = tbl.length
  if (y >= tbl[0][1]) {
    const [x0, y0] = tbl[0]
    const [x1, y1] = tbl[1]
    return x0 + ((y - y0) * (x1 - x0)) / (y1 - y0)
  }
  for (let i = 1; i < n; i++)
    if (y >= tbl[i][1]) {
      const [x0, y0] = tbl[i - 1]
      const [x1, y1] = tbl[i]
      return x0 + ((y - y0) * (x1 - x0)) / (y1 - y0)
    }
  const [x0, y0] = tbl[n - 2]
  const [x1, y1] = tbl[n - 1]
  return x0 + ((y - y0) * (x1 - x0)) / (y1 - y0)
}

export const pctForAge = (vo2: number, age: number): number => {
  let nearest = FRIEND_PCT_M[0]
  for (const b of FRIEND_PCT_M) if (Math.abs(b.age - age) < Math.abs(nearest.age - age)) nearest = b
  const rows = nearest.rows
  let i = 1
  while (i < rows.length - 1 && vo2 > rows[i][1]) i++
  const [p0, v0] = rows[i - 1]
  const [p1, v1] = rows[i]
  const p = v1 === v0 ? p0 : p0 + ((vo2 - v0) * (p1 - p0)) / (v1 - v0)
  return Math.round(clamp(p, 1, 99))
}

export interface GarminVo2Point {
  date: string
  v: number
  generic: number | null
  cycling: number | null
}

export interface ObservedVo2Point {
  date: string
  v: number
  method: 'lab' | 'garmin' | 'apple'
  sport: FtpHypothesisVo2maxSport
}
