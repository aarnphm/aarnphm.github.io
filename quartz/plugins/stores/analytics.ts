import type { GarminCache, GarminWeightSample } from './garmin'
import type { WeatherCache } from './weather'
import { AppleCache } from './apple'
import { matchGarminHeartRateActivity } from './garmin'
import { OuraCache } from './oura'
import {
  type Sport,
  type ActivityKind,
  SPORT_ORDER,
  normalizeSport,
  normalizeKind,
  round,
  resolveActivityHeartRate,
  type ActivityHeartRate,
  type RawStravaActivity,
  type StravaStreams,
  type StravaRawCache,
  type StravaZones,
} from './strava'
import { RaceEvent, TrackEntry } from './tracking'

export interface BodyCompositionDay {
  date: string
  kg: number | null
  bmi: number | null
  bodyFatPct: number | null
  bodyWaterPct: number | null
  muscleMassKg: number | null
  boneMassKg: number | null
}

export interface BodyBlock {
  latestKg: number | null
  latestLbs: number | null
  trendKgPerWeek: number | null
  goalKg: number | null
  goalLbs: number | null
  goalDeltaKg: number | null
  goalEtaWeeks: number | null
  bmi: number | null
  bodyFatPct: number | null
  bodyWaterPct: number | null
  muscleMassKg: number | null
  boneMassKg: number | null
  series: { date: string; ts: number; kg: number }[]
  bmrSeries: { date: string; ts: number; bmr: number }[]
  latestBmr: number | null
  composition: BodyCompositionDay[]
}

export interface ActivitySummary {
  id: number
  date: string
  sport: ActivityKind
  name: string
  distanceKm: number
  movingTimeS: number
  load: number
  cadence: number | null
  windKph: number | null
  windDir: string | null
  windGustKph: number | null
}

export interface AnalyticsInputs {
  oura?: OuraCache | null
  apple?: AppleCache | null
  garmin?: GarminCache | null
  weather?: WeatherCache | null
  weights?: TrackEntry[]
  events?: RaceEvent[]
  since?: string
}

export type Conf = 'firm' | 'low' | 'prior' | 'stale'
export type TsbZone = 'fresh' | 'neutral' | 'fatigued' | 'deep'
export type AcwrState = 'building' | 'low' | 'ok' | 'caution' | 'high'
export type TrendMethod = 'none' | 'ewma' | 'ols'
export type RaceDistance = 'sprint' | 'olympic' | '70.3' | 'ironman'

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
  tempDevC: number | null
  weightKg: number | null
  totalCalories: number | null
  intakeKcal: number | null
  warmup: boolean
}

export interface WeeklyPoint {
  weekStart: string
  load: number
  km: number
  hours: number
  effort: number
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
  sampleSize: number
  staleDays: number
}

export interface SportTrendForecastPoint {
  date: string
  value: number
  lo: number
  hi: number
}

export interface SportTrend {
  sport: Sport
  unit: string
  invert: boolean
  method: TrendMethod
  stale: boolean
  sampleSize: number
  spanDays: number
  level: number | null
  slopePerWeek: number | null
  etaNow: number | null
  note: string
  forecast: SportTrendForecastPoint[]
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

export interface RaceLeg {
  sport: Sport
  legKm: number
  longestKm: number
  coverage: number
  recencyGate: number
  splitS: number
}

export interface RaceReadiness {
  distance: RaceDistance
  legs: RaceLeg[]
  predictedTotalS: number
  score: number
  fitnessReady: number
  bindingLeg: Sport
  bandPct: number
  conf: Conf
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

export interface RecoveryDay {
  date: string
  hrv: number | null
  hrvZ: number | null
  rhr: number | null
  rhrZ: number | null
  sleepS: number | null
  sleepScore: number | null
  sleepDebtS: number | null
  readiness: number | null
  tempDevC: number | null
  warmup: boolean
}

export interface RecoveryBlock {
  status: RecoveryStatus
  baselineDays: number
  hrvLatest: number | null
  hrvBaseline: number | null
  hrvZ: number | null
  hrvCv: number | null
  rhrLatest: number | null
  rhrBaseline: number | null
  rhrZ: number | null
  readinessLatest: number | null
  readinessBaseline: number | null
  lowReadinessStreak: number
  tempDevLatest: number | null
  sleepLatestS: number | null
  sleepBaselineS: number | null
  sleepTargetS: number
  sleepDebtS: number
  shortSleepStreak: number
  series: RecoveryDay[]
  flags: RecoveryFlag[]
  thresholds: RecoveryThresholds
}

export type Vo2Method = 'garmin' | 'apple' | 'bike' | 'run' | 'hrratio' | 'none'
export type RadarAxisKey = 'sprint' | 'threshold' | 'endurance' | 'climb' | 'cadence' | 'recovery'
export type CardioDir = 'improving' | 'stable' | 'declining' | null
export type CardioKey = 'rhr' | 'hrv' | 'ef' | 'decoupling'

export interface Vo2Point {
  weekStart: string
  vo2max: number
  method: Vo2Method
}

export interface Vo2Estimate {
  method: Vo2Method
  vo2max: number
  conf: Conf
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
  note: string
}

export interface RadarAxis {
  key: RadarAxisKey
  label: string
  score: number | null
  rawValue: number | null
  rawUnit: string
  lo: number
  hi: number
}

export interface AbilitiesBlock {
  axes: RadarAxis[]
  area: number | null
}

export interface CardioMetric {
  key: CardioKey
  label: string
  value: number | null
  unit: string
  slopePerWeek: number | null
  dir: CardioDir
  sampleSize: number
  note: string
}

export interface CardioBlock {
  metrics: CardioMetric[]
  rhrSeries: { date: string; rhr: number }[]
  hrvSeries: { date: string; hrv: number }[]
  efSeries: { date: string; ef: number; sport: Sport }[]
  decouplingSeries: { date: string; pct: number }[]
}

export interface EngineBlock {
  vo2max: Vo2maxBlock
  abilities: AbilitiesBlock
  cardio: CardioBlock
}

export interface DataFeedInputs {
  oura?: OuraCache | null
  apple?: AppleCache | null
  weather?: WeatherCache | null
  garmin?: GarminCache | null
  weights?: TrackEntry[]
  zones?: StravaZones | null
}

export interface Analytics {
  meta: AnalyticsMeta
  thresholds: ThresholdEstimate[]
  daily: DailyPoint[]
  weekly: WeeklyPoint[]
  trends: SportTrend[]
  bests: SportBest[]
  risk: RiskBlock
  races: RaceReadiness[]
  loadShare: Record<Sport, number>
  body: BodyBlock
  recovery: RecoveryBlock
  engine: EngineBlock
  events: RaceEvent[]
  activities: ActivitySummary[]
  weakestSport: Sport
  actions: TrainingAction[]
}

interface Act {
  a: RawStravaActivity
  sport: Sport
  day: string
  distanceKm: number
  vGap: number
}

const DAY_MS = 86_400_000
const K42 = 1 - Math.exp(-1 / 42)
const K7 = 1 - Math.exp(-1 / 7)
const IF_CAP = 1.15

const SPORT_PRIOR: Record<Sport, number> = { swim: 1.3, bike: 6.9, run: 3.3 }
const LOAD_SHARE_TARGET: Record<Sport, number> = { swim: 0.2, bike: 0.5, run: 0.3 }

const dayMs = (iso: string): number => Date.parse(`${iso}T00:00:00Z`)
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x))
const mean = (xs: number[]): number => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0)
const sd = (xs: number[]): number => {
  if (xs.length < 2) return 0
  const m = mean(xs)
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1))
}

function mmss(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds))
  const m = Math.floor(s / 60)
  const r = s % 60
  return `${m}:${String(r).padStart(2, '0')}`
}

function weightedPercentile(values: number[], weights: number[], p: number): number {
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

function gradeFactorRun(g: number): number {
  if (g >= 0) return 1 + 8.85 * g + 44 * g * g
  return Math.max(0.83, 1 + 8 * g + 44 * g * g)
}

function segGrades(
  stream: StravaStreams | undefined,
): { grades: number[]; lengths: number[] } | null {
  if (!stream) return null
  const alt = stream.altitude
  const dist = stream.distance
  if (!alt || !dist || alt.length < 2 || dist.length < 2) return null
  const grades: number[] = []
  const lengths: number[] = []
  const n = Math.min(alt.length, dist.length)
  for (let i = 1; i < n; i++) {
    const len = dist[i] - dist[i - 1]
    if (len <= 0) continue
    grades.push(clamp((alt[i] - alt[i - 1]) / len, -0.3, 0.3))
    lengths.push(len)
  }
  if (grades.length < 1) return null
  return { grades, lengths }
}

function gradeAdjSpeed(
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
  const seg = segGrades(stream)
  if (seg) {
    const totalLen = seg.lengths.reduce((s, x) => s + x, 0)
    if (totalLen > 0) {
      const weighted =
        seg.grades.reduce((s, g, i) => s + gradeFactorRun(g) * seg.lengths[i], 0) / totalLen
      return v * weighted
    }
  }
  return v * gradeFactorRun(gOverall)
}

function estimateThreshold(acts: Act[], sport: Sport, today: number): ThresholdEstimate {
  const mine = acts.filter(x => x.sport === sport)
  const lastDay = mine.length ? mine[mine.length - 1].day : null
  const staleDays = lastDay ? Math.round((today - dayMs(lastDay)) / DAY_MS) : 0
  const n = mine.length
  let vThr: number
  let conf: Conf
  if (n >= 4) {
    const values = mine.map(x => x.vGap)
    const weights = mine.map(x => Math.floor(x.a.movingTime / 600) + 1)
    vThr = weightedPercentile(values, weights, 0.9)
    conf = 'firm'
  } else if (n >= 2) {
    vThr = 0.97 * Math.max(...mine.map(x => x.vGap))
    conf = 'low'
  } else {
    vThr = SPORT_PRIOR[sport]
    conf = 'prior'
  }
  if (sport === 'run' && staleDays > 45) conf = 'stale'

  let paceLabel: string
  let unit: string
  if (sport === 'swim') {
    unit = 's/100m'
    paceLabel = mmss(100 / vThr)
  } else if (sport === 'run') {
    unit = 's/km'
    paceLabel = mmss(1000 / vThr)
  } else {
    unit = 'km/h'
    paceLabel = String(round(vThr * 3.6, 0))
  }
  return { sport, vThr: round(vThr, 4), paceLabel, unit, conf, sampleSize: n, staleDays }
}

function activityLoad(act: Act, vThr: number): number {
  const intensity = clamp(act.vGap / vThr, 0, IF_CAP)
  const load = intensity * intensity * (act.a.movingTime / 3600) * 100
  return round(load, 1)
}

function buildDaily(
  acts: Act[],
  loadById: Map<number, number>,
  windowFrom: number,
  windowTo: number,
): DailyPoint[] {
  const byDay = new Map<string, { total: number; swim: number; bike: number; run: number }>()
  for (const act of acts) {
    const load = loadById.get(act.a.id) ?? 0
    const bucket = byDay.get(act.day) ?? { total: 0, swim: 0, bike: 0, run: 0 }
    bucket.total += load
    bucket[act.sport] += load
    byDay.set(act.day, bucket)
  }

  const rows: { date: string; total: number; swim: number; bike: number; run: number }[] = []
  for (let ms = windowFrom; ms <= windowTo; ms += DAY_MS) {
    const date = new Date(ms).toISOString().slice(0, 10)
    const bucket = byDay.get(date) ?? { total: 0, swim: 0, bike: 0, run: 0 }
    rows.push({ date, ...bucket })
  }

  const active = rows.filter(r => r.total > 0)
  const seedRows = active.slice(0, 14)
  const seed = seedRows.length ? mean(seedRows.map(r => r.total)) : 0
  const warmupCut = windowFrom + 42 * DAY_MS

  let ctl = seed
  let atl = seed
  let swimCtl = seed * LOAD_SHARE_TARGET.swim
  let bikeCtl = seed * LOAD_SHARE_TARGET.bike
  let runCtl = seed * LOAD_SHARE_TARGET.run

  const out: DailyPoint[] = []
  for (const row of rows) {
    const tsb = ctl - atl
    out.push({
      date: row.date,
      load: round(row.total, 1),
      swimLoad: round(row.swim, 1),
      bikeLoad: round(row.bike, 1),
      runLoad: round(row.run, 1),
      ctl: round(ctl, 1),
      atl: round(atl, 1),
      tsb: round(tsb, 1),
      swimCtl: round(swimCtl, 1),
      bikeCtl: round(bikeCtl, 1),
      runCtl: round(runCtl, 1),
      readiness: null,
      hrv: null,
      rhr: null,
      sleepScore: null,
      sleepDurationS: null,
      tempDevC: null,
      weightKg: null,
      totalCalories: null,
      intakeKcal: null,
      warmup: dayMs(row.date) < warmupCut,
    })
    ctl += (row.total - ctl) * K42
    atl += (row.total - atl) * K7
    swimCtl += (row.swim - swimCtl) * K42
    bikeCtl += (row.bike - bikeCtl) * K42
    runCtl += (row.run - runCtl) * K42
  }
  return out
}

function buildWeekly(acts: Act[], loadById: Map<number, number>): WeeklyPoint[] {
  const byWeek = new Map<string, { load: number; km: number; seconds: number; effort: number }>()
  for (const act of acts) {
    const ms = dayMs(act.day)
    const dow = new Date(ms).getUTCDay()
    const offset = (dow + 6) % 7
    const weekStart = new Date(ms - offset * DAY_MS).toISOString().slice(0, 10)
    const bucket = byWeek.get(weekStart) ?? { load: 0, km: 0, seconds: 0, effort: 0 }
    bucket.load += loadById.get(act.a.id) ?? 0
    bucket.km += act.distanceKm
    bucket.seconds += act.a.movingTime
    bucket.effort += act.a.sufferScore ?? 0
    byWeek.set(weekStart, bucket)
  }
  const weeks = [...byWeek.keys()].sort()
  return weeks.map((weekStart, i) => {
    const cur = byWeek.get(weekStart)!
    const prev = i > 0 ? byWeek.get(weeks[i - 1])! : null
    const ramp = prev && prev.load > 0 ? round((cur.load - prev.load) / prev.load, 2) : null
    const sessionLoads: number[] = []
    for (const act of acts) {
      const ms = dayMs(act.day)
      const dow = new Date(ms).getUTCDay()
      const offset = (dow + 6) % 7
      const ws = new Date(ms - offset * DAY_MS).toISOString().slice(0, 10)
      if (ws === weekStart) sessionLoads.push(loadById.get(act.a.id) ?? 0)
    }
    const m = mean(sessionLoads)
    const s = sd(sessionLoads)
    const monotony = s > 0 ? round(m / s, 2) : null
    const strain = monotony != null ? round(cur.load * monotony, 0) : null
    return {
      weekStart,
      load: round(cur.load, 1),
      km: round(cur.km, 1),
      hours: round(cur.seconds / 3600, 1),
      effort: round(cur.effort, 0),
      ramp,
      monotony,
      strain,
    }
  })
}

function buildTrend(
  acts: Act[],
  threshold: ThresholdEstimate,
  sport: Sport,
  today: number,
): SportTrend {
  const mine = acts.filter(x => x.sport === sport)
  const unit = threshold.unit
  const invert = sport !== 'bike'
  const toHuman = (v: number): number => {
    if (sport === 'swim') return round(100 / v, 1)
    if (sport === 'run') return round(1000 / v, 1)
    return round(v * 3.6, 1)
  }

  if (mine.length === 0) {
    return {
      sport,
      unit,
      invert,
      method: 'none',
      stale: true,
      sampleSize: 0,
      spanDays: 0,
      level: null,
      slopePerWeek: null,
      etaNow: null,
      note: 'no efforts recorded',
      forecast: [],
    }
  }

  const firstMs = dayMs(mine[0].day)
  const lastMs = dayMs(mine[mine.length - 1].day)
  const spanDays = Math.round((lastMs - firstMs) / DAY_MS)
  const daysSinceLast = Math.round((today - lastMs) / DAY_MS)
  const n = mine.length
  const xs = mine.map(x => (dayMs(x.day) - firstMs) / DAY_MS)
  const ys = mine.map(x => toHuman(x.vGap))
  let maxGap = 0
  for (let i = 1; i < xs.length; i++) maxGap = Math.max(maxGap, xs[i] - xs[i - 1])
  const todayX = (today - firstMs) / DAY_MS

  if (daysSinceLast > 45 || n < 3) {
    return {
      sport,
      unit,
      invert,
      method: 'none',
      stale: true,
      sampleSize: n,
      spanDays,
      level: round(ys[ys.length - 1], 1),
      slopePerWeek: null,
      etaNow: null,
      note: `last effort ${daysSinceLast}d ago`,
      forecast: [],
    }
  }

  if (n >= 6 && spanDays >= 21 && maxGap <= 14) {
    const mx = mean(xs)
    const my = mean(ys)
    let sxx = 0
    let sxy = 0
    for (let i = 0; i < xs.length; i++) {
      sxx += (xs[i] - mx) ** 2
      sxy += (xs[i] - mx) * (ys[i] - my)
    }
    const b = sxx > 0 ? sxy / sxx : 0
    const a = my - b * mx
    const resid = ys.map((y, i) => y - (a + b * xs[i]))
    const dof = Math.max(1, n - 2)
    const se = Math.sqrt(resid.reduce((s, r) => s + r * r, 0) / dof)
    const etaNow = a + b * todayX
    const forecast: SportTrendForecastPoint[] = []
    for (let d = 1; d <= 14; d++) {
      const fx = todayX + d
      const fitted = a + b * fx
      const band = 1.96 * se * Math.sqrt(1 + 1 / n + (fx - mx) ** 2 / (sxx || 1))
      forecast.push({
        date: new Date(today + d * DAY_MS).toISOString().slice(0, 10),
        value: round(fitted, 1),
        lo: round(fitted - band, 1),
        hi: round(fitted + band, 1),
      })
    }
    return {
      sport,
      unit,
      invert,
      method: 'ols',
      stale: false,
      sampleSize: n,
      spanDays,
      level: round(etaNow, 1),
      slopePerWeek: round(b * 7, 2),
      etaNow: round(etaNow, 1),
      note: `${n} efforts over ${spanDays}d`,
      forecast,
    }
  }

  const alpha = 0.3
  let level = ys[0]
  for (let i = 1; i < ys.length; i++) level += alpha * (ys[i] - level)
  const resid = ys.map((y, i) => {
    let l = ys[0]
    for (let j = 1; j <= i; j++) l += alpha * (ys[j] - l)
    return y - l
  })
  const band = 1.5 * sd(resid)
  const firstHalf = mean(ys.slice(0, Math.max(1, Math.floor(ys.length / 2))))
  const secondHalf = mean(ys.slice(Math.floor(ys.length / 2)))
  const perPoint = ys.length > 1 ? (secondHalf - firstHalf) / Math.max(1, ys.length / 2) : 0
  const avgGap = spanDays > 0 && n > 1 ? spanDays / (n - 1) : 7
  const slopePerWeek = avgGap > 0 ? round((perPoint / avgGap) * 7, 2) : 0
  const forecast: SportTrendForecastPoint[] = []
  for (let d = 1; d <= 14; d++) {
    forecast.push({
      date: new Date(today + d * DAY_MS).toISOString().slice(0, 10),
      value: round(level, 1),
      lo: round(level - band, 1),
      hi: round(level + band, 1),
    })
  }
  return {
    sport,
    unit,
    invert,
    method: 'ewma',
    stale: false,
    sampleSize: n,
    spanDays,
    level: round(level, 1),
    slopePerWeek,
    etaNow: round(level, 1),
    note: `${n} efforts, ewma`,
    forecast,
  }
}

function buildBest(acts: Act[], sport: Sport): SportBest {
  const mine = acts.filter(x => x.sport === sport)
  const fastestUnit = sport === 'bike' ? 'km/h' : sport === 'swim' ? 's/100m' : 's/km'
  const toRate = (v: number): number => {
    if (sport === 'swim') return round(100 / v, 1)
    if (sport === 'run') return round(1000 / v, 1)
    return round(v * 3.6, 1)
  }
  const minDistM = sport === 'swim' ? 400 : 0
  const minTimeS = sport === 'swim' ? 0 : 20 * 60
  const better = (rate: number, best: number): boolean =>
    sport === 'bike' ? rate > best : rate < best

  let fastest: number | null = null
  for (const x of mine) {
    if (x.a.distance < minDistM || x.a.movingTime < minTimeS) continue
    const rate = toRate(x.vGap)
    if (fastest === null || better(rate, fastest)) fastest = rate
  }

  const longestKm = mine.reduce((m, x) => Math.max(m, x.distanceKm), 0)
  const biggestClimbM = mine.reduce((m, x) => Math.max(m, x.a.totalElevationGain), 0)
  const totalKm = mine.reduce((s, x) => s + x.distanceKm, 0)
  const totalTimeS = mine.reduce((s, x) => s + x.a.movingTime, 0)

  const bestToDate: { date: string; rate: number }[] = []
  let running: number | null = null
  for (const x of mine) {
    const rate = toRate(x.vGap)
    if (running === null || better(rate, running)) running = rate
    bestToDate.push({ date: x.day, rate: running })
  }

  return {
    sport,
    count: mine.length,
    totalKm: round(totalKm, 1),
    totalTimeS,
    fastestRate: fastest,
    fastestUnit,
    longestKm: round(longestKm, 1),
    biggestClimbM: Math.round(biggestClimbM),
    bestToDate,
  }
}

function tsbZoneOf(tsb: number): TsbZone {
  if (tsb > 5) return 'fresh'
  if (tsb > -10) return 'neutral'
  if (tsb > -30) return 'fatigued'
  return 'deep'
}

function buildRisk(daily: DailyPoint[], weekly: WeeklyPoint[]): RiskBlock {
  if (daily.length === 0) {
    return {
      ctl: 0,
      atl: 0,
      tsb: 0,
      tsbZone: 'neutral',
      rampWeek: 0,
      acwr: null,
      acwrState: 'building',
      monotony: null,
      strain: null,
    }
  }
  const last = daily[daily.length - 1]
  const lastWeek = weekly.length ? weekly[weekly.length - 1] : null
  const rampWeek = lastWeek?.ramp ?? 0
  const acute7 = daily.slice(-7).reduce((s, d) => s + d.load, 0)
  const chronic28 = daily.slice(-28).reduce((s, d) => s + d.load, 0)
  const activeBefore = daily.slice(0, -1).filter(d => d.load > 0).length
  let acwr: number | null
  let acwrState: AcwrState
  if (activeBefore < 14 || chronic28 < 5) {
    acwr = null
    acwrState = 'building'
  } else {
    const value = acute7 / (chronic28 / 4)
    acwr = round(value, 2)
    if (value < 0.8) acwrState = 'low'
    else if (value <= 1.3) acwrState = 'ok'
    else if (value <= 1.5) acwrState = 'caution'
    else acwrState = 'high'
  }
  return {
    ctl: round(last.ctl, 2),
    atl: round(last.atl, 2),
    tsb: round(last.tsb, 2),
    tsbZone: tsbZoneOf(last.tsb),
    rampWeek: round(rampWeek, 2),
    acwr,
    acwrState,
    monotony: lastWeek?.monotony ?? null,
    strain: lastWeek?.strain ?? null,
  }
}

const RACE_LEGS: Record<RaceDistance, Record<Sport, number>> = {
  sprint: { swim: 0.75, bike: 20, run: 5 },
  olympic: { swim: 1.5, bike: 40, run: 10 },
  '70.3': { swim: 1.9, bike: 90, run: 21.1 },
  ironman: { swim: 3.8, bike: 180, run: 42.2 },
}
const RACE_REF: Record<RaceDistance, number> = { sprint: 35, olympic: 50, '70.3': 70, ironman: 90 }
const T_REF_S = 3600
const RIEGEL_K: Record<Sport, number> = { swim: 1.03, bike: 1.05, run: 1.06 }
const RUN_BRICK_FADE: Record<RaceDistance, number> = {
  sprint: 1.02,
  olympic: 1.04,
  '70.3': 1.07,
  ironman: 1.12,
}
const confRank: Record<Conf, number> = { firm: 0, stale: 1, low: 2, prior: 3 }

function recencyGate(staleDays: number): number {
  if (staleDays <= 21) return 1
  if (staleDays <= 42) return 0.6
  return 0.3
}

function legSplitSeconds(distKm: number, vThr: number, k: number): number {
  const dRefKm = (vThr * T_REF_S) / 1000
  if (dRefKm <= 0) return 0
  return T_REF_S * Math.pow(distKm / dRefKm, k)
}

function buildReadiness(
  distance: RaceDistance,
  thresholds: Map<Sport, ThresholdEstimate>,
  bests: Map<Sport, SportBest>,
  ctlNow: number,
): RaceReadiness {
  const legKm = RACE_LEGS[distance]
  const legs: RaceLeg[] = SPORT_ORDER.map(sport => {
    const th = thresholds.get(sport)!
    const longestKm = bests.get(sport)?.longestKm ?? 0
    const coverage = clamp(longestKm / legKm[sport], 0, 1)
    const gate = recencyGate(th.staleDays)
    const raw = legSplitSeconds(legKm[sport], th.vThr, RIEGEL_K[sport])
    const splitS = sport === 'run' ? raw * RUN_BRICK_FADE[distance] : raw
    return {
      sport,
      legKm: legKm[sport],
      longestKm,
      coverage,
      recencyGate: gate,
      splitS: round(splitS, 0),
    }
  })

  const total = legs.reduce((s, l) => s + l.splitS, 0) + 120 + 90
  const fitnessReady = clamp(ctlNow / RACE_REF[distance], 0, 1)
  const gatedCoverage = legs.map(l => l.coverage * l.recencyGate)
  const score = round(100 * (0.45 * fitnessReady + 0.55 * mean(gatedCoverage)), 0)

  let bindingLeg: Sport = legs[0].sport
  let worstGated = Infinity
  for (const l of legs) {
    const g = l.coverage * l.recencyGate
    if (g < worstGated) {
      worstGated = g
      bindingLeg = l.sport
    }
  }

  const widen = SPORT_ORDER.some(sport => {
    const c = thresholds.get(sport)!.conf
    return c === 'low' || c === 'prior' || c === 'stale'
  })
  const bandPct = widen ? 25 : 12
  let conf: Conf = 'firm'
  for (const sport of SPORT_ORDER) {
    const c = thresholds.get(sport)!.conf
    if (confRank[c] > confRank[conf]) conf = c
  }

  return {
    distance,
    legs,
    predictedTotalS: round(total, 0),
    score,
    fitnessReady: round(fitnessReady, 2),
    bindingLeg,
    bandPct,
    conf,
  }
}

function buildActions(
  thresholds: Map<Sport, ThresholdEstimate>,
  bests: Map<Sport, SportBest>,
  daily: DailyPoint[],
  loadShare: Record<Sport, number>,
): { weakest: Sport; actions: TrainingAction[] } {
  const last = daily.length ? daily[daily.length - 1] : null
  const ctlBySport: Record<Sport, number> = {
    swim: last?.swimCtl ?? 0,
    bike: last?.bikeCtl ?? 0,
    run: last?.runCtl ?? 0,
  }
  const olyLeg = RACE_LEGS.olympic
  const coverageBySport: Record<Sport, number> = {
    swim: clamp((bests.get('swim')?.longestKm ?? 0) / olyLeg.swim, 0, 1),
    bike: clamp((bests.get('bike')?.longestKm ?? 0) / olyLeg.bike, 0, 1),
    run: clamp((bests.get('run')?.longestKm ?? 0) / olyLeg.run, 0, 1),
  }

  const staleArr = SPORT_ORDER.map(s => thresholds.get(s)!.staleDays)
  const coverGapArr = SPORT_ORDER.map(s => 1 - coverageBySport[s])
  const negCtlArr = SPORT_ORDER.map(s => -ctlBySport[s])
  const shortfallArr = SPORT_ORDER.map(s => LOAD_SHARE_TARGET[s] - (loadShare[s] ?? 0))

  const zOf = (arr: number[]): number[] => {
    const m = mean(arr)
    const s = sd(arr)
    return arr.map(x => (s > 0 ? (x - m) / s : 0))
  }
  const zStale = zOf(staleArr)
  const zCover = zOf(coverGapArr)
  const zCtl = zOf(negCtlArr)
  const zShort = zOf(shortfallArr)

  const scores = SPORT_ORDER.map((sport, i) => ({
    sport,
    score: zStale[i] + zCover[i] + zCtl[i] + zShort[i],
  }))
  let weakest: Sport = scores[0].sport
  let best = -Infinity
  for (const item of scores) {
    if (item.score > best) {
      best = item.score
      weakest = item.sport
    }
  }

  const verbBySport: Record<Sport, string> = {
    swim: 'return to swimming',
    bike: 'build the bike',
    run: 'return to running',
  }
  const actions: TrainingAction[] = []
  const ordered = [...scores].sort((a, b) => b.score - a.score)
  for (const item of ordered) {
    if (actions.length >= 4) break
    const th = thresholds.get(item.sport)!
    if (th.staleDays > 21) {
      actions.push({
        text: verbBySport[item.sport],
        sourceMetric: `${item.sport} staleDays`,
        value: `${th.staleDays}d`,
      })
    } else if (coverageBySport[item.sport] < 0.6) {
      actions.push({
        text: `extend long ${item.sport}`,
        sourceMetric: `${item.sport} olympicCoverage`,
        value: `${Math.round(coverageBySport[item.sport] * 100)}%`,
      })
    } else {
      actions.push({
        text: `hold ${item.sport} volume`,
        sourceMetric: `${item.sport} ctl`,
        value: String(round(ctlBySport[item.sport], 1)),
      })
    }
  }

  return { weakest, actions }
}

function emptyMeta(athleteId: number, today: string): AnalyticsMeta {
  return {
    athleteId,
    today,
    windowFrom: today,
    windowTo: today,
    activityCount: 0,
    method: {
      ctlTau: 42,
      atlTau: 7,
      k42: K42,
      k7: K7,
      ifCap: IF_CAP,
      thresholdWindowDays: 0,
      seededFrom: 'mean daily load over first 14 active days',
      note: 'pace-derived load; HR, power, and cadence captured per activity',
    },
  }
}

function neutralRisk(): RiskBlock {
  return {
    ctl: 0,
    atl: 0,
    tsb: 0,
    tsbZone: 'neutral',
    rampWeek: 0,
    acwr: null,
    acwrState: 'building',
    monotony: null,
    strain: null,
  }
}

const emptyBody = (): BodyBlock => ({
  latestKg: null,
  latestLbs: null,
  trendKgPerWeek: null,
  goalKg: null,
  goalLbs: null,
  goalDeltaKg: null,
  goalEtaWeeks: null,
  bmi: null,
  bodyFatPct: null,
  bodyWaterPct: null,
  muscleMassKg: null,
  boneMassKg: null,
  series: [],
  bmrSeries: [],
  latestBmr: null,
  composition: [],
})

const katchMcArdleBmr = (weightKg: number, bodyFatPct: number): number =>
  Math.round(370 + 21.6 * weightKg * (1 - bodyFatPct / 100))

function weightTrendPerWeek(series: { date: string; kg: number }[]): number | null {
  const pts = series.map(e => ({ t: dayMs(e.date) / DAY_MS, w: e.kg }))
  if (pts.length < 2) return null
  const mt = mean(pts.map(p => p.t))
  const mw = mean(pts.map(p => p.w))
  let num = 0
  let den = 0
  for (const p of pts) {
    num += (p.t - mt) * (p.w - mw)
    den += (p.t - mt) ** 2
  }
  if (den === 0) return null
  return round((num / den) * 7, 2)
}

const HRV_BASE_SPAN = 28
const HRV_BASE_MIN = 14
const ACUTE_SPAN = 7
const ACUTE_MIN = 3
const SLEEP_DEBT_SPAN = 14
const RAMP_HOT = 0.1
const RECOVERY_THRESHOLDS: RecoveryThresholds = {
  hrvWatchZ: -1,
  hrvAlertZ: -2,
  rhrWatchZ: 1,
  rhrAlertZ: 2,
  rhrWatchBpm: 5,
  rhrAlertBpm: 7,
  tempWatchC: 0.5,
  tempAlertC: 1,
  sleepTargetS: 28800,
  sleepFloorS: 25200,
  sleepDebtWatchS: 18000,
  sleepDebtAlertS: 36000,
  readinessFloor: 70,
  readinessOptimal: 85,
}

const winValues = (a: (number | null)[], i: number, span: number): number[] => {
  const xs: number[] = []
  for (let j = Math.max(0, i - span + 1); j <= i; j++) {
    const v = a[j]
    if (v != null) xs.push(v)
  }
  return xs
}

const winStats = (
  a: (number | null)[],
  i: number,
  span: number,
  minN: number,
): { mean: number; sd: number; n: number } | null => {
  const xs = winValues(a, i, span)
  return xs.length >= minN ? { mean: mean(xs), sd: sd(xs), n: xs.length } : null
}

const winMedian = (a: (number | null)[], i: number, span: number, minN: number): number | null => {
  const xs = winValues(a, i, span)
  if (xs.length < minN) return null
  xs.sort((p, q) => p - q)
  const m = xs.length >> 1
  return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2
}

const lastIdx = (a: (number | null)[]): number => {
  for (let i = a.length - 1; i >= 0; i--) if (a[i] != null) return i
  return -1
}

const streakBack = (a: (number | null)[], from: number, bad: (v: number) => boolean): number => {
  let n = 0
  for (let i = from; i >= 0; i--) {
    const v = a[i]
    if (v == null || !bad(v)) break
    n++
  }
  return n
}

function emptyRecovery(): RecoveryBlock {
  return {
    status: 'building',
    baselineDays: 0,
    hrvLatest: null,
    hrvBaseline: null,
    hrvZ: null,
    hrvCv: null,
    rhrLatest: null,
    rhrBaseline: null,
    rhrZ: null,
    readinessLatest: null,
    readinessBaseline: null,
    lowReadinessStreak: 0,
    tempDevLatest: null,
    sleepLatestS: null,
    sleepBaselineS: null,
    sleepTargetS: RECOVERY_THRESHOLDS.sleepTargetS,
    sleepDebtS: 0,
    shortSleepStreak: 0,
    series: [],
    flags: [],
    thresholds: RECOVERY_THRESHOLDS,
  }
}

function buildRecovery(daily: DailyPoint[], risk: RiskBlock): RecoveryBlock {
  const firstSignal = daily.findIndex(
    d =>
      d.hrv != null ||
      d.rhr != null ||
      d.sleepDurationS != null ||
      d.readiness != null ||
      d.tempDevC != null,
  )
  if (firstSignal < 0) return emptyRecovery()

  const t = RECOVERY_THRESHOLDS
  const lnHrv = daily.map(d => (d.hrv != null && d.hrv > 0 ? Math.log(d.hrv) : null))
  const hrvArr = daily.map(d => d.hrv)
  const rhrArr = daily.map(d => d.rhr)
  const slpArr = daily.map(d => d.sleepDurationS)
  const rdyArr = daily.map(d => d.readiness)
  const tmpArr = daily.map(d => d.tempDevC)

  const hrvZArr: (number | null)[] = []
  const hrvBaseArr: (number | null)[] = []
  const rhrZArr: (number | null)[] = []
  const debtArr: (number | null)[] = []
  for (let i = 0; i < daily.length; i++) {
    const hb = winStats(lnHrv, i, HRV_BASE_SPAN, HRV_BASE_MIN)
    const ha = winStats(lnHrv, i, ACUTE_SPAN, ACUTE_MIN)
    hrvZArr.push(hb && ha && hb.sd > 0 ? round((ha.mean - hb.mean) / hb.sd, 2) : null)
    hrvBaseArr.push(hb ? round(Math.exp(hb.mean), 1) : null)
    const rb = winStats(rhrArr, i, HRV_BASE_SPAN, HRV_BASE_MIN)
    const ra = winStats(rhrArr, i, ACUTE_SPAN, ACUTE_MIN)
    rhrZArr.push(rb && ra && rb.sd > 0 ? round((ra.mean - rb.mean) / rb.sd, 2) : null)
    const nights = winValues(slpArr, i, SLEEP_DEBT_SPAN)
    debtArr.push(
      nights.length
        ? Math.round(nights.reduce((s, x) => s + clamp(t.sleepTargetS - x, 0, t.sleepTargetS), 0))
        : null,
    )
  }

  const last = daily.length - 1
  const hi = lastIdx(hrvZArr)
  const ri = lastIdx(rhrZArr)
  const hl = lastIdx(hrvArr)
  const rl = lastIdx(rhrArr)
  const sl = lastIdx(slpArr)
  const yl = lastIdx(rdyArr)
  const tl = lastIdx(tmpArr)
  const dl = lastIdx(debtArr)

  const hrvLatest = hl >= 0 ? hrvArr[hl] : null
  const hrvZ = hi >= 0 ? hrvZArr[hi] : null
  const hrvBaseline = hi >= 0 ? hrvBaseArr[hi] : null
  const acuteLn = hl >= 0 ? winStats(lnHrv, hl, ACUTE_SPAN, ACUTE_MIN) : null
  const hrvCv = acuteLn && acuteLn.mean !== 0 ? round((100 * acuteLn.sd) / acuteLn.mean, 2) : null
  const rhrLatest = rl >= 0 ? rhrArr[rl] : null
  const rhrZ = ri >= 0 ? rhrZArr[ri] : null
  const rhrBaseStats = ri >= 0 ? winStats(rhrArr, ri, HRV_BASE_SPAN, HRV_BASE_MIN) : null
  const rhrBaseline = rhrBaseStats ? round(rhrBaseStats.mean, 1) : null
  const readinessLatest = yl >= 0 ? rdyArr[yl] : null
  const rdyBase = yl >= 0 ? winStats(rdyArr, yl, HRV_BASE_SPAN, 7) : null
  const readinessBaseline = rdyBase ? round(rdyBase.mean, 0) : null
  const tempDevLatest = tl >= 0 ? tmpArr[tl] : null
  const sleepLatestS = sl >= 0 ? slpArr[sl] : null
  const sleepBaselineS = sl >= 0 ? winMedian(slpArr, sl, HRV_BASE_SPAN, 7) : null
  const sleepDebtS = dl >= 0 ? (debtArr[dl] ?? 0) : 0
  const shortSleepStreak = sl >= 0 ? streakBack(slpArr, sl, v => v < t.sleepFloorS) : 0
  const lowReadinessStreak = yl >= 0 ? streakBack(rdyArr, yl, v => v < t.readinessFloor) : 0

  const baselineDays = winStats(lnHrv, last, HRV_BASE_SPAN, 1)?.n ?? 0
  const status: RecoveryStatus =
    baselineDays >= 14 ? 'firm' : baselineDays >= 7 ? 'low' : 'building'

  const series: RecoveryDay[] = daily.slice(firstSignal).map((d, k) => {
    const i = firstSignal + k
    return {
      date: d.date,
      hrv: d.hrv,
      hrvZ: hrvZArr[i],
      rhr: d.rhr,
      rhrZ: rhrZArr[i],
      sleepS: d.sleepDurationS,
      sleepScore: d.sleepScore,
      sleepDebtS: debtArr[i],
      readiness: d.readiness,
      tempDevC: d.tempDevC,
      warmup: d.warmup,
    }
  })

  const flags: RecoveryFlag[] = []
  const autonomic = status !== 'building'
  const hrvTxt = hrvLatest != null ? `${Math.round(hrvLatest)} ms` : '—'
  const overreaching =
    autonomic &&
    hrvZ != null &&
    hrvZ <= t.hrvWatchZ &&
    (risk.acwrState === 'caution' || risk.acwrState === 'high' || risk.rampWeek > RAMP_HOT)
  if (overreaching)
    flags.push({
      id: 'overreaching',
      severity: 'alert',
      label: 'overreaching risk',
      detail: `hrv ${hrvTxt} sits ${Math.abs(hrvZ).toFixed(1)}σ below baseline while load ramps (acwr ${risk.acwr ?? '—'}, ramp ${Math.round(risk.rampWeek * 100)}%)`,
      metric: 'overreaching',
      value: hrvZ,
      baseline: risk.acwr,
    })
  if (!overreaching && autonomic && hrvZ != null && hrvZ <= t.hrvWatchZ)
    flags.push({
      id: 'hrv-suppressed',
      severity: hrvZ <= t.hrvAlertZ ? 'alert' : 'watch',
      label: 'hrv suppressed',
      detail: `7-day hrv ${hrvTxt} is ${Math.abs(hrvZ).toFixed(1)}σ under the ${hrvBaseline ?? '—'} ms baseline`,
      metric: 'hrv',
      value: hrvLatest,
      baseline: hrvBaseline,
    })
  const rhrDelta = rhrLatest != null && rhrBaseline != null ? rhrLatest - rhrBaseline : null
  if (
    autonomic &&
    ((rhrZ != null && rhrZ >= t.rhrWatchZ) || (rhrDelta != null && rhrDelta >= t.rhrWatchBpm))
  ) {
    const alert =
      (rhrZ != null && rhrZ >= t.rhrAlertZ) || (rhrDelta != null && rhrDelta >= t.rhrAlertBpm)
    flags.push({
      id: 'rhr-elevated',
      severity: alert ? 'alert' : 'watch',
      label: 'resting hr elevated',
      detail: `rhr ${rhrLatest != null ? Math.round(rhrLatest) : '—'} bpm vs ${rhrBaseline ?? '—'} baseline${rhrDelta != null ? ` (${rhrDelta >= 0 ? '+' : ''}${round(rhrDelta, 1)} bpm)` : ''}`,
      metric: 'rhr',
      value: rhrLatest,
      baseline: rhrBaseline,
    })
  }
  const illness =
    autonomic &&
    tempDevLatest != null &&
    tempDevLatest >= t.tempWatchC &&
    ((rhrZ != null && rhrZ >= t.rhrWatchZ) || (hrvZ != null && hrvZ <= t.hrvWatchZ))
  if (illness)
    flags.push({
      id: 'illness-watch',
      severity: 'alert',
      label: 'illness watch',
      detail: `temp +${tempDevLatest.toFixed(1)}°c over baseline with autonomic strain — often precedes symptoms by 24–48 h`,
      metric: 'tempdev',
      value: tempDevLatest,
      baseline: null,
    })
  if (!illness && autonomic && tempDevLatest != null && tempDevLatest >= t.tempWatchC)
    flags.push({
      id: 'temp-elevated',
      severity: tempDevLatest >= t.tempAlertC ? 'alert' : 'watch',
      label: 'temperature elevated',
      detail: `skin temp +${tempDevLatest.toFixed(1)}°c over personal baseline`,
      metric: 'tempdev',
      value: tempDevLatest,
      baseline: null,
    })
  if (sleepDebtS >= t.sleepDebtWatchS)
    flags.push({
      id: 'sleep-debt',
      severity: sleepDebtS >= t.sleepDebtAlertS ? 'alert' : 'watch',
      label: 'sleep debt',
      detail: `${(sleepDebtS / 3600).toFixed(1)} h short of the 8 h target over 14 nights`,
      metric: 'sleepdebt',
      value: sleepDebtS,
      baseline: t.sleepTargetS,
    })
  if (shortSleepStreak >= 2)
    flags.push({
      id: 'short-sleep',
      severity: shortSleepStreak >= 3 ? 'alert' : 'watch',
      label: 'short sleep streak',
      detail: `${shortSleepStreak} nights under the 7 h floor`,
      metric: 'sleepdebt',
      value: shortSleepStreak,
      baseline: t.sleepFloorS,
    })
  if (lowReadinessStreak >= 2)
    flags.push({
      id: 'low-readiness',
      severity: lowReadinessStreak >= 3 ? 'alert' : 'watch',
      label: 'low readiness streak',
      detail: `${lowReadinessStreak} days under ${t.readinessFloor}`,
      metric: 'oreadiness',
      value: readinessLatest,
      baseline: t.readinessFloor,
    })
  if (
    flags.length === 0 &&
    hrvZ != null &&
    hrvZ >= 0 &&
    rhrZ != null &&
    rhrZ <= 0 &&
    readinessLatest != null &&
    readinessLatest >= t.readinessOptimal &&
    sleepDebtS < t.sleepDebtWatchS
  )
    flags.push({
      id: 'well-recovered',
      severity: 'info',
      label: 'well recovered',
      detail: `hrv on baseline, readiness ${Math.round(readinessLatest)}, sleep debt ${(sleepDebtS / 3600).toFixed(1)} h`,
      metric: 'oreadiness',
      value: readinessLatest,
      baseline: t.readinessOptimal,
    })

  const sevRank: Record<FlagSeverity, number> = { alert: 0, watch: 1, info: 2 }
  const idOrder = [
    'overreaching',
    'illness-watch',
    'hrv-suppressed',
    'rhr-elevated',
    'temp-elevated',
    'sleep-debt',
    'short-sleep',
    'low-readiness',
    'well-recovered',
  ]
  flags.sort(
    (a, b) =>
      sevRank[a.severity] - sevRank[b.severity] || idOrder.indexOf(a.id) - idOrder.indexOf(b.id),
  )

  return {
    status,
    baselineDays,
    hrvLatest,
    hrvBaseline,
    hrvZ,
    hrvCv,
    rhrLatest,
    rhrBaseline,
    rhrZ,
    readinessLatest,
    readinessBaseline,
    lowReadinessStreak,
    tempDevLatest,
    sleepLatestS,
    sleepBaselineS,
    sleepTargetS: t.sleepTargetS,
    sleepDebtS,
    shortSleepStreak,
    series,
    flags: flags.slice(0, 6),
    thresholds: t,
  }
}

export const ATHLETE = {
  sex: 'M' as const,
  born: '2001-03',
  bornAnchor: '2001-03-01',
  hrMax: 190 as number | null,
  vo2max: 45 as number | null,
  ftp: 208 as number | null,
  goalWeightLb: 180 as number | null,
  goalFTP: 290 as number | null,
}

const goalWeightKg = ATHLETE.goalWeightLb != null ? ATHLETE.goalWeightLb * 0.45359237 : null

const VO2_FLOOR = 20
const VO2_CEIL = 80
const FRIEND_MED_M: [number, number][] = [
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
const ACSM_WATT_K = 10.8
const ACSM_BASE = 7
const FTP_FROM_P20 = 0.95
const MAP_FTP_RATIO = 0.75
const DANIELS_A = -4.6
const DANIELS_B = 0.182258
const DANIELS_C = 0.000104
const UTH_K = 15.3
const TANAKA_A = 208
const TANAKA_B = 0.7
const COGGAN_SPRINT_WKG: [number, number] = [7, 24]
const COGGAN_FTP_WKG: [number, number] = [1.5, 6.4]
const VAM_ANCHOR: [number, number] = [300, 1500]
const CTL_ANCHOR: [number, number] = [0, 100]
const RUN_SPM_TARGET = 180
const BIKE_RPM_TARGET = 90
const ONE_HZ_TOL = 0.15
const DECOUPLE_MIN_S = 1200
const PEAK_WINDOWS = [30, 60, 300, 1200] as const

const ageOn = (iso: string): number =>
  Math.floor((dayMs(iso) - dayMs(ATHLETE.bornAnchor)) / (365.25 * DAY_MS))

const norm01 = (v: number, lo: number, hi: number): number => clamp((v - lo) / (hi - lo), 0, 1)

const invLerp = (tbl: [number, number][], y: number): number => {
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

const pctForAge = (vo2: number, age: number): number => {
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

const olsSlope = (xs: number[], ys: number[]): number | null => {
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

const oneHz = (n: number, movingTimeS: number): boolean =>
  movingTimeS > 0 && n > 0 && Math.abs(n / movingTimeS - 1) <= ONE_HZ_TOL

const peakMean = (xs: number[], w: number): number | null => {
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

const peakDistRate = (d: number[], w: number): number | null => {
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

const efOf = (
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

const decouplingOf = (
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

const hrMaxOf = (
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

function runVo2Of(
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

function emptyEngine(): EngineBlock {
  return {
    vo2max: {
      value: null,
      method: 'none',
      conf: 'prior',
      hrMax: 0,
      hrMaxSource: 'tanaka',
      hrRest: null,
      chronoAge: 0,
      fitnessAge: null,
      ageDeltaYears: null,
      percentileForAge: null,
      estimates: [],
      trend: [],
      note: 'no power or hr data',
    },
    abilities: { axes: [], area: null },
    cardio: { metrics: [], rhrSeries: [], hrvSeries: [], efSeries: [], decouplingSeries: [] },
  }
}

function buildEngine(
  cache: StravaRawCache,
  acts: Act[],
  daily: DailyPoint[],
  body: BodyBlock,
  ctlNow: number,
  today: string,
  garminVo2: { date: string; v: number }[],
  appleVo2: { date: string; v: number }[],
  heartRateById: Map<number, ActivityHeartRate>,
): EngineBlock {
  const age = ageOn(today)
  const { hrMax, source: hrMaxSource } = hrMaxOf(acts, cache, age, heartRateById)
  const rhrArr = daily.map(d => d.rhr)
  const hrRestRaw = winMedian(rhrArr, daily.length - 1, HRV_BASE_SPAN, 1)
  const hrRest = hrRestRaw != null ? round(hrRestRaw, 0) : null

  const kgAt = new Map<string, number>()
  for (const d of daily) if (d.weightKg != null) kgAt.set(d.date, d.weightKg)
  const bikes = acts.filter(x => x.sport === 'bike')
  const wattsOf = (id: number): number[] => cache.streams?.[String(id)]?.watts ?? []

  let p20: number | null = null
  let p20Day: string | null = null
  let p20Win = 1200
  for (const win of [1200, 600]) {
    for (const b of bikes) {
      const v = peakMean(wattsOf(b.a.id), win)
      if (v != null && (p20 == null || v > p20)) {
        p20 = v
        p20Day = b.day
        p20Win = win
      }
    }
    if (p20 != null) break
  }
  const deviceN = bikes.filter(
    b => b.a.deviceWatts === true && wattsOf(b.a.id).length >= p20Win,
  ).length

  const estimates: Vo2Estimate[] = []
  if (garminVo2.length)
    estimates.push({
      method: 'garmin',
      vo2max: round(clamp(garminVo2[garminVo2.length - 1].v, VO2_FLOOR, VO2_CEIL), 1),
      conf: 'firm',
    })
  if (appleVo2.length)
    estimates.push({
      method: 'apple',
      vo2max: round(clamp(appleVo2[appleVo2.length - 1].v, VO2_FLOOR, VO2_CEIL), 1),
      conf: 'firm',
    })
  const declaredFtp = cache.zones?.ftp ?? null
  let ftp = declaredFtp
  let ftpSrc: 'strava' | 'derived' | null = declaredFtp != null ? 'strava' : null
  let bikeKg: number | null = null
  if (ftp == null && p20 != null) {
    ftp = round(p20 * FTP_FROM_P20, 0)
    ftpSrc = 'derived'
  }
  if (ftp != null) {
    bikeKg = (ftpSrc === 'derived' && p20Day ? kgAt.get(p20Day) : null) ?? body.latestKg
    if (bikeKg != null && bikeKg > 0) {
      const map = ftp / MAP_FTP_RATIO
      estimates.push({
        method: 'bike',
        vo2max: round(clamp((ACSM_WATT_K * map) / bikeKg + ACSM_BASE, VO2_FLOOR, VO2_CEIL), 1),
        conf: ftpSrc === 'strava' ? 'firm' : deviceN >= 3 && p20Win === 1200 ? 'firm' : 'low',
      })
    }
  }
  const runVo2 = runVo2Of(acts, cache, hrMax, hrRest, heartRateById)
  if (runVo2 != null) estimates.push({ method: 'run', vo2max: runVo2, conf: 'low' })
  if (hrRest != null && hrRest > 0)
    estimates.push({
      method: 'hrratio',
      vo2max: round(clamp((UTH_K * hrMax) / hrRest, VO2_FLOOR, VO2_CEIL), 1),
      conf: 'prior',
    })

  const primary = estimates.length ? estimates[0] : null
  const noteOf = (m: Vo2Method): string => {
    if (m === 'garmin') return 'garmin connect (device/manual)'
    if (m === 'apple') return 'apple watch measurement'
    if (m === 'bike')
      return `ftp ${ftp}w${ftpSrc === 'strava' ? ' (strava)' : ''} · map ${ftp != null ? Math.round(ftp / MAP_FTP_RATIO) : '—'}w · ${bikeKg != null ? round(bikeKg, 1) : '—'}kg`
    if (m === 'run') return 'run hr–speed extrapolation'
    if (m === 'hrratio') return 'upper-bound proxy from sleeping rhr'
    return 'no power or hr data'
  }

  const weekOf = (iso: string): string => {
    const ms = dayMs(iso)
    const dow = new Date(ms).getUTCDay()
    return new Date(ms - ((dow + 6) % 7) * DAY_MS).toISOString().slice(0, 10)
  }
  const weeks = new Map<string, { p20: number | null; kg: number | null }>()
  for (const d of daily) {
    const ws = weekOf(d.date)
    const w = weeks.get(ws) ?? { p20: null, kg: null }
    if (d.weightKg != null) w.kg = d.weightKg
    weeks.set(ws, w)
  }
  for (const b of bikes) {
    const v = peakMean(wattsOf(b.a.id), 1200)
    if (v == null) continue
    const ws = weekOf(b.day)
    const w = weeks.get(ws) ?? { p20: null, kg: null }
    if (w.p20 == null || v > w.p20) w.p20 = v
    weeks.set(ws, w)
  }
  const trend: Vo2Point[] = []
  let lastVo2: number | null = null
  let kgCarry: number | null = null
  for (const ws of [...weeks.keys()].sort()) {
    const w = weeks.get(ws)!
    if (w.kg != null) kgCarry = w.kg
    if (w.p20 != null && kgCarry != null && kgCarry > 0)
      lastVo2 = round(
        clamp(
          (ACSM_WATT_K * ((w.p20 * FTP_FROM_P20) / MAP_FTP_RATIO)) / kgCarry + ACSM_BASE,
          VO2_FLOOR,
          VO2_CEIL,
        ),
        1,
      )
    if (lastVo2 != null) trend.push({ weekStart: ws, vo2max: lastVo2, method: 'bike' })
  }

  const vo2 = primary?.vo2max ?? null
  const fitnessAge = vo2 != null ? Math.round(clamp(invLerp(FRIEND_MED_M, vo2), 20, 80)) : null
  const ageDeltaYears = fitnessAge != null ? fitnessAge - age : null
  const percentileForAge = vo2 != null ? pctForAge(vo2, age) : null

  const kgNow = body.latestKg
  let p5: number | null = null
  for (const b of bikes) {
    const v = peakMean(wattsOf(b.a.id), 5)
    if (v != null && (p5 == null || v > p5)) p5 = v
  }
  if (p5 == null)
    for (const b of bikes)
      if (b.a.maxWatts != null && (p5 == null || b.a.maxWatts > p5)) p5 = b.a.maxWatts
  const sprintWkg = p5 != null && kgNow ? p5 / kgNow : null
  const ftpWkg = ftp != null && kgNow ? ftp / kgNow : null
  let vam: number | null = null
  for (const b of bikes) {
    if (b.a.movingTime <= 0 || b.a.totalElevationGain <= 0) continue
    const v = (b.a.totalElevationGain * 3600) / b.a.movingTime
    if (vam == null || v > vam) vam = v
  }
  const bikeRpms = bikes
    .filter(b => b.a.averageCadence != null)
    .map(b => b.a.averageCadence as number)
  const runSpms = acts
    .filter(x => x.sport === 'run' && x.a.averageCadence != null)
    .map(x => (x.a.averageCadence as number) * 2)
  const devs: number[] = []
  if (bikeRpms.length) devs.push(Math.abs(mean(bikeRpms) - BIKE_RPM_TARGET) / BIKE_RPM_TARGET)
  if (runSpms.length) devs.push(Math.abs(mean(runSpms) - RUN_SPM_TARGET) / RUN_SPM_TARGET)
  const cadScore = devs.length ? Math.round(clamp(100 - Math.min(...devs) * 200, 0, 100)) : null
  const rdy14 = winValues(
    daily.map(d => d.readiness),
    daily.length - 1,
    14,
  )
  const hrv14 = winValues(
    daily.map(d => d.hrv),
    daily.length - 1,
    14,
  )
  const recScore = rdy14.length
    ? Math.round(mean(rdy14))
    : hrv14.length
      ? Math.round(norm01(mean(hrv14), 20, 120) * 100)
      : null

  const axes: RadarAxis[] = [
    {
      key: 'sprint',
      label: 'sprint',
      score: sprintWkg != null ? Math.round(norm01(sprintWkg, ...COGGAN_SPRINT_WKG) * 100) : null,
      rawValue: sprintWkg != null ? round(sprintWkg, 1) : null,
      rawUnit: 'w/kg',
      lo: COGGAN_SPRINT_WKG[0],
      hi: COGGAN_SPRINT_WKG[1],
    },
    {
      key: 'threshold',
      label: 'threshold',
      score: ftpWkg != null ? Math.round(norm01(ftpWkg, ...COGGAN_FTP_WKG) * 100) : null,
      rawValue: ftpWkg != null ? round(ftpWkg, 2) : null,
      rawUnit: 'w/kg',
      lo: COGGAN_FTP_WKG[0],
      hi: COGGAN_FTP_WKG[1],
    },
    {
      key: 'endurance',
      label: 'endurance',
      score: Math.round(norm01(ctlNow, ...CTL_ANCHOR) * 100),
      rawValue: round(ctlNow, 0),
      rawUnit: 'ctl',
      lo: CTL_ANCHOR[0],
      hi: CTL_ANCHOR[1],
    },
    {
      key: 'climb',
      label: 'climb',
      score: vam != null ? Math.round(norm01(vam, ...VAM_ANCHOR) * 100) : null,
      rawValue: vam != null ? round(vam, 0) : null,
      rawUnit: 'm/h',
      lo: VAM_ANCHOR[0],
      hi: VAM_ANCHOR[1],
    },
    {
      key: 'cadence',
      label: 'cadence',
      score: cadScore,
      rawValue: bikeRpms.length
        ? round(mean(bikeRpms), 0)
        : runSpms.length
          ? round(mean(runSpms), 0)
          : null,
      rawUnit: bikeRpms.length ? 'rpm' : 'spm',
      lo: 0,
      hi: 100,
    },
    {
      key: 'recovery',
      label: 'recovery',
      score: recScore,
      rawValue: rdy14.length ? round(mean(rdy14), 0) : hrv14.length ? round(mean(hrv14), 0) : null,
      rawUnit: rdy14.length ? 'readiness' : 'ms',
      lo: 0,
      hi: 100,
    },
  ]
  const scored = axes.filter(a => a.score != null).map(a => a.score as number)
  const area = scored.length ? Math.round(mean(scored)) : null

  const last28 = daily.slice(-28)
  const rhrXs: number[] = []
  const rhrYs: number[] = []
  last28.forEach((d, i) => {
    if (d.rhr != null) {
      rhrXs.push(i)
      rhrYs.push(d.rhr)
    }
  })
  const rhrSlope = olsSlope(rhrXs, rhrYs)
  const rhrWeek = rhrSlope != null ? round(rhrSlope * 7, 2) : null
  const rhr7 = winMedian(rhrArr, daily.length - 1, ACUTE_SPAN, 1)

  const hrvAll = daily.map(d => d.hrv)
  const hrvXs: number[] = []
  const hrvYs: number[] = []
  last28.forEach((d, i) => {
    if (d.hrv != null) {
      hrvXs.push(i)
      hrvYs.push(d.hrv)
    }
  })
  const hrvSlope = olsSlope(hrvXs, hrvYs)
  const hrv7m = winStats(hrvAll, daily.length - 1, ACUTE_SPAN, ACUTE_MIN)
  const hrv28m = winStats(hrvAll, daily.length - 1, HRV_BASE_SPAN, 7)
  const hrvRel =
    hrv7m && hrv28m && hrv28m.mean > 0 ? (hrv7m.mean - hrv28m.mean) / hrv28m.mean : null

  const efActs = acts
    .map(x => ({ x, ef: efOf(x.sport, x.a, x.vGap, heartRateById.get(x.a.id)?.avgHr) }))
    .filter((e): e is { x: Act; ef: number } => e.ef != null)
  const efBike = efActs.filter(e => e.x.sport === 'bike')
  const efRun = efActs.filter(e => e.x.sport === 'run')
  const efPool = efBike.length >= 3 ? efBike : efRun.length >= 3 ? efRun : efActs
  const efSlope = olsSlope(
    efPool.map(e => dayMs(e.x.day) / DAY_MS),
    efPool.map(e => e.ef),
  )
  const efMean = efPool.length ? mean(efPool.map(e => e.ef)) : 0
  const efRelWk = efSlope != null && efMean > 0 ? (efSlope * 7) / efMean : null
  const efUnit = efBike.length >= 3 || (efBike.length && efPool === efActs) ? 'w/bpm' : 'm/min/bpm'

  const decActs = acts
    .map(x => ({
      x,
      d:
        x.a.movingTime >= 3600
          ? decouplingOf(x.sport, cache.streams?.[String(x.a.id)], x.a.movingTime)
          : null,
    }))
    .filter((e): e is { x: Act; d: number } => e.d != null)
  const decSlope = olsSlope(
    decActs.map(e => dayMs(e.x.day) / DAY_MS),
    decActs.map(e => e.d),
  )
  const decWeek = decSlope != null ? round(decSlope * 7, 2) : null
  const decLatest = decActs.length ? decActs[decActs.length - 1].d : null

  const metrics: CardioMetric[] = [
    {
      key: 'rhr',
      label: 'resting hr',
      value: rhr7 != null ? round(rhr7, 0) : null,
      unit: 'bpm',
      slopePerWeek: rhrWeek,
      dir:
        rhrWeek == null
          ? null
          : rhrWeek <= -0.25
            ? 'improving'
            : rhrWeek >= 0.25
              ? 'declining'
              : 'stable',
      sampleSize: rhrYs.length,
      note: 'falling is improving',
    },
    {
      key: 'hrv',
      label: 'hrv baseline',
      value: hrv7m ? round(hrv7m.mean, 0) : null,
      unit: 'ms',
      slopePerWeek: hrvSlope != null ? round(hrvSlope * 7, 2) : null,
      dir:
        hrvRel == null
          ? null
          : hrvRel > 0.05
            ? 'improving'
            : hrvRel < -0.05
              ? 'declining'
              : 'stable',
      sampleSize: hrvYs.length,
      note: '7d vs 28d baseline',
    },
    {
      key: 'ef',
      label: 'efficiency',
      value: efPool.length ? efPool[efPool.length - 1].ef : null,
      unit: efUnit,
      slopePerWeek: efSlope != null ? round(efSlope * 7, 4) : null,
      dir:
        efRelWk == null
          ? null
          : efRelWk > 0.005
            ? 'improving'
            : efRelWk < -0.005
              ? 'declining'
              : 'stable',
      sampleSize: efPool.length,
      note: 'output per heartbeat',
    },
    {
      key: 'decoupling',
      label: 'decoupling',
      value: decLatest,
      unit: '%',
      slopePerWeek: decWeek,
      dir:
        decActs.length < 3 || decWeek == null
          ? null
          : decWeek <= -0.5
            ? 'improving'
            : decWeek >= 0.5
              ? 'declining'
              : 'stable',
      sampleSize: decActs.length,
      note:
        decLatest == null
          ? 'needs a 60 min+ session with hr'
          : decLatest < 5
            ? 'coupled — durable aerobic base'
            : decLatest <= 10
              ? 'moderate drift'
              : 'high drift',
    },
  ]

  return {
    vo2max: {
      value: vo2,
      method: primary?.method ?? 'none',
      conf: primary?.conf ?? 'prior',
      hrMax,
      hrMaxSource,
      hrRest,
      chronoAge: age,
      fitnessAge,
      ageDeltaYears,
      percentileForAge,
      estimates,
      trend,
      note: noteOf(primary?.method ?? 'none'),
    },
    abilities: { axes, area },
    cardio: {
      metrics,
      rhrSeries: daily
        .filter(d => d.rhr != null)
        .map(d => ({ date: d.date, rhr: d.rhr as number })),
      hrvSeries: daily
        .filter(d => d.hrv != null)
        .map(d => ({ date: d.date, hrv: d.hrv as number })),
      efSeries: efActs.map(e => ({ date: e.x.day, ef: e.ef, sport: e.x.sport })),
      decouplingSeries: decActs.map(e => ({ date: e.x.day, pct: e.d })),
    },
  }
}

function emptyAnalytics(athleteId: number, today: string): Analytics {
  return {
    meta: emptyMeta(athleteId, today),
    thresholds: [],
    daily: [],
    weekly: [],
    trends: [],
    bests: [],
    risk: neutralRisk(),
    races: [],
    loadShare: { swim: 0, bike: 0, run: 0 },
    body: emptyBody(),
    recovery: emptyRecovery(),
    engine: emptyEngine(),
    events: [],
    activities: [],
    weakestSport: 'run',
    actions: [],
  }
}

export function buildAnalytics(
  cache: StravaRawCache | null,
  inputs: AnalyticsInputs = {},
): Analytics {
  const todayFromSync = cache?.lastSync ? new Date(cache.lastSync).toISOString().slice(0, 10) : null

  if (!cache) return emptyAnalytics(0, todayFromSync ?? '1970-01-01')

  const sinceDay = inputs.since && /^\d{4}-\d{2}-\d{2}$/.test(inputs.since) ? inputs.since : null
  const raw = Object.values(cache.activities)
    .map(a => ({ a, sport: normalizeSport(a.sportType) }))
    .filter((x): x is { a: RawStravaActivity; sport: Sport } => x.sport !== null)
    .filter(x => !sinceDay || x.a.startDateLocal.slice(0, 10) >= sinceDay)
    .sort((p, q) => p.a.startDateLocal.localeCompare(q.a.startDateLocal))

  if (raw.length === 0) {
    const fallback = todayFromSync ?? '1970-01-01'
    return emptyAnalytics(cache.athleteId, fallback)
  }

  const lastActDay = raw[raw.length - 1].a.startDateLocal.slice(0, 10)
  const today = todayFromSync ?? lastActDay
  const todayMs = dayMs(today)
  const heartRateById = new Map<number, ActivityHeartRate>()
  for (const { a, sport } of raw) {
    const stream = cache.streams?.[String(a.id)]
    heartRateById.set(
      a.id,
      resolveActivityHeartRate(
        a,
        sport,
        stream,
        matchGarminHeartRateActivity(a, sport, inputs.garmin ?? null),
        inputs.garmin ?? null,
      ),
    )
  }

  const acts: Act[] = raw.map(({ a, sport }) => ({
    a,
    sport,
    day: a.startDateLocal.slice(0, 10),
    distanceKm: round(a.distance / 1000, 1),
    vGap: gradeAdjSpeed(a, sport, cache.streams?.[String(a.id)]),
  }))

  const thresholdList = SPORT_ORDER.map(sport => estimateThreshold(acts, sport, todayMs))
  const thresholds = new Map<Sport, ThresholdEstimate>(thresholdList.map(t => [t.sport, t]))

  const loadById = new Map<number, number>()
  for (const act of acts) {
    const vThr = thresholds.get(act.sport)!.vThr
    loadById.set(act.a.id, activityLoad(act, vThr))
  }

  const firstDay = sinceDay ?? acts[0].day
  const windowFrom = dayMs(firstDay)
  const windowTo = Math.max(todayMs, dayMs(lastActDay))

  const daily = buildDaily(acts, loadById, windowFrom, windowTo)
  const ouraDays = inputs.oura?.days ?? {}
  const appleDays = inputs.apple?.days ?? {}
  const garminWeightRaw = inputs.garmin?.weight
  const garminSamples = Array.isArray(garminWeightRaw) ? garminWeightRaw : []
  const weightByDate = new Map<string, number>()
  for (const w of inputs.weights ?? []) if (w.weightKg != null) weightByDate.set(w.date, w.weightKg)
  for (const s of garminSamples) if (s.weightKg != null) weightByDate.set(s.date, s.weightKg)
  let carryKg: number | null = null
  for (const d of daily) {
    const o = ouraDays[d.date]
    if (o) {
      d.readiness = o.readiness
      d.hrv = o.hrv
      d.rhr = o.rhr
      d.sleepScore = o.sleepScore
      d.sleepDurationS = o.sleepDurationS
      d.tempDevC = o.tempDeviationC
      d.totalCalories = o.totalCalories
    }
    const w = weightByDate.get(d.date)
    if (w != null) carryKg = w
    d.weightKg = carryKg
    const ap = appleDays[d.date]
    if (ap) {
      if (d.totalCalories == null && ap.burnKcal != null) d.totalCalories = ap.burnKcal
      d.intakeKcal = ap.intakeKcal
      if (ap.weightKg != null && d.weightKg == null) d.weightKg = ap.weightKg
    }
  }
  const weighedDaily = daily.filter(d => d.weightKg != null)
  const latestKg = weighedDaily.length ? weighedDaily[weighedDaily.length - 1].weightKg : null
  const dailySeries = [...weightByDate.entries()]
    .map(([date, kg]) => ({ date, kg }))
    .sort((p, q) => p.date.localeCompare(q.date))
  const trendKgPerWeek = weightTrendPerWeek(dailySeries)
  const garminDates = new Set(garminSamples.map(s => s.date))
  const trackingPts = (inputs.weights ?? [])
    .filter(w => w.weightKg != null && !garminDates.has(w.date))
    .map(w => ({ date: w.date, ts: dayMs(w.date) + 12 * 3_600_000, kg: w.weightKg as number }))
  const garminPts = garminSamples
    .filter(s => s.weightKg != null)
    .map(s => ({ date: s.date, ts: s.ts, kg: s.weightKg as number }))
  const weightSeries = [...trackingPts, ...garminPts].sort((p, q) => p.ts - q.ts)
  const bmrSeries = garminSamples
    .filter(s => s.weightKg != null && s.bodyFatPct != null)
    .map(s => ({
      date: s.date,
      ts: s.ts,
      bmr: katchMcArdleBmr(s.weightKg as number, s.bodyFatPct as number),
    }))
    .sort((p, q) => p.ts - q.ts)
  const latestBmr = bmrSeries.length ? bmrSeries[bmrSeries.length - 1].bmr : null
  const composition = garminSamples.map(s => ({
    date: s.date,
    kg: s.weightKg,
    bmi: s.bmi,
    bodyFatPct: s.bodyFatPct,
    bodyWaterPct: s.bodyWaterPct,
    muscleMassKg: s.muscleMassKg,
    boneMassKg: s.boneMassKg,
  }))
  const lastComp = (
    key: 'bmi' | 'bodyFatPct' | 'bodyWaterPct' | 'muscleMassKg' | 'boneMassKg',
  ): number | null => {
    for (let i = composition.length - 1; i >= 0; i--) {
      const v = composition[i][key]
      if (v != null) return v
    }
    return null
  }
  const goalKg = goalWeightKg
  const goalDeltaKg = latestKg != null && goalKg != null ? round(latestKg - goalKg, 1) : null
  const converging =
    goalDeltaKg != null &&
    trendKgPerWeek != null &&
    Math.abs(trendKgPerWeek) >= 0.05 &&
    goalDeltaKg * trendKgPerWeek < 0
  const goalEtaWeeks = converging
    ? Math.min(104, Math.ceil(Math.abs(goalDeltaKg / trendKgPerWeek)))
    : null
  const body: BodyBlock = {
    latestKg,
    latestLbs: latestKg != null ? round(latestKg / 0.45359237, 1) : null,
    trendKgPerWeek,
    goalKg,
    goalLbs: goalKg != null ? round(goalKg / 0.45359237, 1) : null,
    goalDeltaKg,
    goalEtaWeeks,
    bmi: lastComp('bmi'),
    bodyFatPct: lastComp('bodyFatPct'),
    bodyWaterPct: lastComp('bodyWaterPct'),
    muscleMassKg: lastComp('muscleMassKg'),
    boneMassKg: lastComp('boneMassKg'),
    series: weightSeries,
    bmrSeries,
    latestBmr,
    composition,
  }
  const weekly = buildWeekly(acts, loadById)
  const trends = SPORT_ORDER.map(sport => buildTrend(acts, thresholds.get(sport)!, sport, todayMs))
  const bestList = SPORT_ORDER.map(sport => buildBest(acts, sport))
  const bests = new Map<Sport, SportBest>(bestList.map(b => [b.sport, b]))
  const risk = buildRisk(daily, weekly)
  const recovery = buildRecovery(daily, risk)

  const ctlNow = daily.length ? daily[daily.length - 1].ctl : 0
  const races = (['sprint', 'olympic', '70.3', 'ironman'] as RaceDistance[]).map(distance =>
    buildReadiness(distance, thresholds, bests, ctlNow),
  )

  const recentCut = todayMs - 42 * DAY_MS
  const recentActs = acts.filter(act => dayMs(act.day) >= recentCut)
  const recentLoad = recentActs.reduce((s, act) => s + (loadById.get(act.a.id) ?? 0), 0)
  const loadShare: Record<Sport, number> = { swim: 0, bike: 0, run: 0 }
  if (recentLoad > 0) {
    for (const act of recentActs) loadShare[act.sport] += (loadById.get(act.a.id) ?? 0) / recentLoad
  }

  const appleVo2: { date: string; v: number }[] = []
  for (const d of Object.values(appleDays))
    if (d.vo2max != null) appleVo2.push({ date: d.date, v: d.vo2max })
  appleVo2.sort((p, q) => p.date.localeCompare(q.date))
  const garminVo2: { date: string; v: number }[] = []
  for (const d of Object.values(inputs.garmin?.vo2max ?? {})) {
    const v = d.generic ?? d.cycling
    if (v != null) garminVo2.push({ date: d.date, v })
  }
  garminVo2.sort((p, q) => p.date.localeCompare(q.date))
  const engine = buildEngine(
    cache,
    acts,
    daily,
    body,
    ctlNow,
    today,
    garminVo2,
    appleVo2,
    heartRateById,
  )

  const walkSummaries: ActivitySummary[] = Object.values(cache.activities)
    .filter(a => normalizeKind(a.sportType) === 'walk')
    .filter(a => !sinceDay || a.startDateLocal.slice(0, 10) >= sinceDay)
    .map(a => ({
      id: a.id,
      date: a.startDateLocal.slice(0, 10),
      sport: 'walk' as const,
      name: a.name ?? '',
      distanceKm: round(a.distance / 1000, 1),
      movingTimeS: a.movingTime,
      load: 0,
      cadence: a.averageCadence ?? null,
      windKph: inputs.weather?.activities[String(a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(a.id)]?.windGustKph ?? null,
    }))
  const activities: ActivitySummary[] = acts
    .map(act => ({
      id: act.a.id,
      date: act.day,
      sport: act.sport as ActivityKind,
      name: act.a.name ?? '',
      distanceKm: act.distanceKm,
      movingTimeS: act.a.movingTime,
      load: round(loadById.get(act.a.id) ?? 0, 1),
      cadence: act.a.averageCadence ?? null,
      windKph: inputs.weather?.activities[String(act.a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(act.a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(act.a.id)]?.windGustKph ?? null,
    }))
    .concat(walkSummaries)
    .sort((p, q) => q.date.localeCompare(p.date))

  const { weakest, actions } = buildActions(thresholds, bests, daily, loadShare)

  return {
    meta: {
      athleteId: cache.athleteId,
      today,
      windowFrom: firstDay,
      windowTo: lastActDay,
      activityCount: acts.length,
      method: {
        ctlTau: 42,
        atlTau: 7,
        k42: K42,
        k7: K7,
        ifCap: IF_CAP,
        thresholdWindowDays: 0,
        seededFrom: 'mean daily load over first 14 active days',
        note: 'pace-derived load; HR, power, and cadence captured per activity',
      },
    },
    thresholds: thresholdList,
    daily,
    weekly,
    trends,
    bests: bestList,
    risk,
    races,
    loadShare,
    body,
    recovery,
    engine,
    events: inputs.events ?? [],
    activities,
    weakestSport: weakest,
    actions,
  }
}

export const DAY_FIELDS = [
  'kind',
  'date',
  'sessions',
  'km',
  'hours',
  'load',
  'swimLoad',
  'bikeLoad',
  'runLoad',
  'ctl',
  'atl',
  'tsb',
  'swimCtl',
  'bikeCtl',
  'runCtl',
  'warmup',
  'readiness',
  'sleepScore',
  'hrv',
  'rhr',
  'sleepDurationS',
  'tempDeviationC',
  'totalCalories',
  'activeCalories',
  'intakeKcal',
  'weightKg',
  'bmi',
  'bodyFatPct',
  'bodyWaterPct',
  'muscleMassKg',
  'boneMassKg',
  'bmr',
  'windKph',
  'windDir',
  'windGustKph',
  'readinessNext',
  'hrvNext',
] as const

export const ACTIVITY_FIELDS = [
  'kind',
  'id',
  'date',
  'sport',
  'name',
  'distanceKm',
  'movingTimeS',
  'elapsedTimeS',
  'elevationM',
  'avgHr',
  'maxHr',
  'avgWatts',
  'weightedWatts',
  'deviceWatts',
  'cadence',
  'calories',
  'sufferScore',
  'avgTemp',
  'windKph',
  'windDir',
  'windGustKph',
  'vGap',
  'intensity',
  'load',
  'pp30',
  'pp60',
  'pp300',
  'pp1200',
  'ps30',
  'ps60',
  'ps300',
  'ps1200',
  'ef',
  'decoupling',
] as const

export const WEEK_FIELDS = [
  'kind',
  'weekStart',
  'load',
  'km',
  'hours',
  'effort',
  'ramp',
  'monotony',
  'strain',
] as const

export interface FeedDayRow {
  kind: 'day'
  date: string
  sessions: number
  km: number
  hours: number
  load: number
  swimLoad: number
  bikeLoad: number
  runLoad: number
  ctl: number
  atl: number
  tsb: number
  swimCtl: number
  bikeCtl: number
  runCtl: number
  warmup: boolean
  readiness: number | null
  sleepScore: number | null
  hrv: number | null
  rhr: number | null
  sleepDurationS: number | null
  tempDeviationC: number | null
  totalCalories: number | null
  activeCalories: number | null
  intakeKcal: number | null
  weightKg: number | null
  bmi: number | null
  bodyFatPct: number | null
  bodyWaterPct: number | null
  muscleMassKg: number | null
  boneMassKg: number | null
  windKph: number | null
  windDir: string | null
  windGustKph: number | null
  readinessNext: number | null
  hrvNext: number | null
}

export interface FeedActivityRow {
  kind: 'activity'
  id: number
  date: string
  sport: Sport
  name: string
  distanceKm: number
  movingTimeS: number
  elapsedTimeS: number
  elevationM: number
  avgHr: number | null
  maxHr: number | null
  avgWatts: number | null
  weightedWatts: number | null
  deviceWatts: boolean | null
  cadence: number | null
  calories: number | null
  sufferScore: number | null
  avgTemp: number | null
  windKph: number | null
  windDir: string | null
  windGustKph: number | null
  vGap: number
  intensity: number | null
  load: number
  pp30: number | null
  pp60: number | null
  pp300: number | null
  pp1200: number | null
  ps30: number | null
  ps60: number | null
  ps300: number | null
  ps1200: number | null
  ef: number | null
  decoupling: number | null
}

export interface FeedWeekRow {
  kind: 'week'
  weekStart: string
  load: number
  km: number
  hours: number
  effort: number
  ramp: number | null
  monotony: number | null
  strain: number | null
}

const pickLine = (row: Record<string, unknown>, fields: readonly string[]): string =>
  JSON.stringify(Object.fromEntries(fields.map(k => [k, row[k] ?? null])))

const rd = (x: number | null, dp: number): number | null => (x == null ? null : round(x, dp))

const nextIso = (date: string): string => new Date(dayMs(date) + DAY_MS).toISOString().slice(0, 10)

export function buildDataFeed(
  cache: StravaRawCache | null,
  analytics: Analytics,
  inputs: DataFeedInputs = {},
): string {
  const ouraDays = inputs.oura?.days ?? {}
  const appleDays = inputs.apple?.days ?? {}
  const vThrBySport = new Map(analytics.thresholds.map(t => [t.sport, t.vThr]))
  const sorted = [...analytics.activities].sort(
    (p, q) => p.date.localeCompare(q.date) || p.id - q.id,
  )

  const byDay = new Map<string, { sessions: number; meters: number; seconds: number }>()
  for (const s of sorted) {
    const raw = cache?.activities[String(s.id)]
    const b = byDay.get(s.date) ?? { sessions: 0, meters: 0, seconds: 0 }
    b.sessions += 1
    b.meters += raw?.distance ?? 0
    b.seconds += s.movingTimeS
    byDay.set(s.date, b)
  }

  const windByDate = new Map<string, TrackEntry>()
  for (const w of inputs.weights ?? [])
    if (w.windKph != null || w.windDir != null) windByDate.set(w.date, w)
  const weatherDays = inputs.weather?.days ?? {}
  const garminWeightByDay = new Map<string, GarminWeightSample>()
  const feedWeightRaw = inputs.garmin?.weight
  for (const s of Array.isArray(feedWeightRaw) ? feedWeightRaw : [])
    garminWeightByDay.set(s.date, s)

  const dayLines = analytics.daily.map(d => {
    const o = ouraDays[d.date]
    const next = ouraDays[nextIso(d.date)]
    const ap = appleDays[d.date]
    const gw = garminWeightByDay.get(d.date)
    const agg = byDay.get(d.date)
    const wind = windByDate.get(d.date)
    const weather = weatherDays[d.date]
    return pickLine(
      {
        kind: 'day',
        date: d.date,
        sessions: agg?.sessions ?? 0,
        km: round((agg?.meters ?? 0) / 1000, 2),
        hours: round((agg?.seconds ?? 0) / 3600, 2),
        load: d.load,
        swimLoad: d.swimLoad,
        bikeLoad: d.bikeLoad,
        runLoad: d.runLoad,
        ctl: d.ctl,
        atl: d.atl,
        tsb: d.tsb,
        swimCtl: d.swimCtl,
        bikeCtl: d.bikeCtl,
        runCtl: d.runCtl,
        warmup: d.warmup,
        readiness: o?.readiness ?? null,
        sleepScore: o?.sleepScore ?? null,
        hrv: o?.hrv ?? null,
        rhr: o?.rhr ?? null,
        sleepDurationS: o?.sleepDurationS ?? null,
        tempDeviationC: o?.tempDeviationC ?? null,
        totalCalories: d.totalCalories ?? o?.totalCalories ?? ap?.burnKcal ?? null,
        activeCalories: o?.activeCalories ?? ap?.activeKcal ?? null,
        intakeKcal: d.intakeKcal ?? ap?.intakeKcal ?? null,
        weightKg: d.weightKg ?? ap?.weightKg ?? null,
        bmi: gw?.bmi ?? null,
        bodyFatPct: gw?.bodyFatPct ?? null,
        bodyWaterPct: gw?.bodyWaterPct ?? null,
        muscleMassKg: gw?.muscleMassKg ?? null,
        boneMassKg: gw?.boneMassKg ?? null,
        bmr:
          gw?.weightKg != null && gw?.bodyFatPct != null
            ? katchMcArdleBmr(gw.weightKg, gw.bodyFatPct)
            : null,
        windKph: wind?.windKph ?? weather?.windKph ?? null,
        windDir: wind?.windDir ?? weather?.windDir ?? null,
        windGustKph: weather?.windGustKph ?? null,
        readinessNext: next?.readiness ?? null,
        hrvNext: next?.hrv ?? null,
      },
      DAY_FIELDS,
    )
  })

  const activityLines: string[] = []
  for (const s of sorted) {
    const raw = cache?.activities[String(s.id)]
    if (!raw) continue
    if (s.sport !== 'swim' && s.sport !== 'bike' && s.sport !== 'run') continue
    const stream = cache?.streams?.[String(s.id)]
    const heartRate = resolveActivityHeartRate(
      raw,
      s.sport,
      stream,
      matchGarminHeartRateActivity(raw, s.sport, inputs.garmin ?? null),
      inputs.garmin ?? null,
    )
    const vThr = vThrBySport.get(s.sport) ?? 0
    const vGap = gradeAdjSpeed(raw, s.sport, stream)
    const watts = stream?.watts ?? []
    const distArr = stream?.distance ?? []
    const wattsOk = oneHz(watts.length, raw.movingTime)
    const distOk = s.sport !== 'swim' && oneHz(distArr.length, raw.movingTime)
    const peaks: Record<string, number | null> = {}
    for (const w of PEAK_WINDOWS) {
      peaks[`pp${w}`] = wattsOk ? rd(peakMean(watts, w), 0) : null
      peaks[`ps${w}`] = distOk ? rd(peakDistRate(distArr, w), 2) : null
    }
    activityLines.push(
      pickLine(
        {
          kind: 'activity',
          id: s.id,
          date: s.date,
          sport: s.sport,
          name: s.name,
          distanceKm: round(raw.distance / 1000, 2),
          movingTimeS: raw.movingTime,
          elapsedTimeS: raw.elapsedTime,
          elevationM: round(raw.totalElevationGain, 0),
          avgHr: heartRate.avgHr,
          maxHr: heartRate.maxHr,
          avgWatts: raw.averageWatts ?? null,
          weightedWatts: raw.weightedAverageWatts ?? null,
          deviceWatts: raw.deviceWatts ?? null,
          cadence: s.cadence,
          calories: raw.calories ?? null,
          sufferScore: raw.sufferScore ?? null,
          avgTemp: raw.averageTemp ?? null,
          windKph: s.windKph,
          windDir: s.windDir,
          windGustKph: s.windGustKph,
          vGap: round(vGap, 3),
          intensity: vThr > 0 ? round(vGap / vThr, 3) : null,
          load: s.load,
          ...peaks,
          ef: efOf(s.sport, raw, vGap, heartRate.avgHr),
          decoupling: decouplingOf(s.sport, stream, raw.movingTime),
        },
        ACTIVITY_FIELDS,
      ),
    )
  }

  const weekLines = analytics.weekly.map(w => pickLine({ kind: 'week', ...w }, WEEK_FIELDS))

  const todayMsV = dayMs(analytics.meta.today)
  const bornMs = dayMs(ATHLETE.bornAnchor)
  const ageYears =
    Number.isFinite(todayMsV) && todayMsV > bornMs
      ? Math.floor((todayMsV - bornMs) / (365.25 * DAY_MS))
      : null
  const meta = {
    kind: 'meta',
    v: 1,
    generatedAt: cache?.lastSync ?? 0,
    athleteId: analytics.meta.athleteId,
    today: analytics.meta.today,
    windowFrom: analytics.meta.windowFrom,
    windowTo: analytics.meta.windowTo,
    athlete: {
      sex: ATHLETE.sex,
      born: ATHLETE.born,
      ageYears,
      hrMaxEst:
        ATHLETE.hrMax ?? (ageYears != null ? round(TANAKA_A - TANAKA_B * ageYears, 1) : null),
      weightGoalKg: goalWeightKg,
      ftp: ATHLETE.ftp,
      goalFtp: ATHLETE.goalFTP,
    },
    zones: inputs.zones
      ? { hr: inputs.zones.hr, power: inputs.zones.power, ftp: inputs.zones.ftp }
      : null,
    thresholds: analytics.thresholds.map(t => ({
      sport: t.sport,
      vThr: t.vThr,
      conf: t.conf,
      sampleSize: t.sampleSize,
    })),
    counts: { day: dayLines.length, activity: activityLines.length, week: weekLines.length },
    labels: [
      { key: 'readinessNext', source: 'oura readiness', lagDays: 1 },
      { key: 'hrvNext', source: 'oura hrv', lagDays: 1 },
    ],
    fields: { day: DAY_FIELDS, activity: ACTIVITY_FIELDS, week: WEEK_FIELDS },
    defs: {
      load: 'clamp(vGap/vThr,0,1.15)^2 * movingHours * 100',
      intensity: 'vGap/vThr, unclamped, current threshold applied retroactively',
      vGap: 'grade-adjusted speed m/s',
      pp: 'peak mean watts over 30/60/300/1200s windows, 1hz power streams only',
      ps: 'peak mean m/s over 30/60/300/1200s from distance stream, bike+run only',
      ef: 'bike np/avgHr w/bpm; run+swim 60*vGap/avgHr m/min/bpm',
      decoupling: '(ef first half - ef second half)/ef first half * 100, streams, >=1200s only',
      hrMaxEst: 'declared max hr; 208 - 0.7*age (tanaka 2001) when unset',
      weightKg: 'garmin scale primary, tracking forward-filled, apple fallback',
      bodyComposition:
        'bmi/bodyFatPct/bodyWaterPct/muscleMassKg/boneMassKg from garmin index scale, measurement days only',
      avgWatts: 'strava estimate unless deviceWatts true',
      windDir: 'tracking override, WeatherKit compass fallback',
    },
  }

  return [JSON.stringify(meta), ...dayLines, ...activityLines, ...weekLines].join('\n') + '\n'
}
