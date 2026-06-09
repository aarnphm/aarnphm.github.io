import { AppleCache } from './apple'
import { OuraCache } from './oura'
import {
  type Sport,
  SPORT_ORDER,
  normalizeSport,
  round,
  type RawStravaActivity,
  type StravaStreams,
  type StravaRawCache,
} from './strava'
import { RaceEvent, TrackEntry } from './tracking'

export interface BodyBlock {
  latestKg: number | null
  latestLbs: number | null
  trendKgPerWeek: number | null
  series: { date: string; kg: number }[]
}

export interface ActivitySummary {
  id: number
  date: string
  sport: Sport
  name: string
  distanceKm: number
  movingTimeS: number
  load: number
  cadence: number | null
}

export interface AnalyticsInputs {
  oura?: OuraCache | null
  apple?: AppleCache | null
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
  events: RaceEvent[]
  activities: ActivitySummary[]
  weakestSport: Sport
  headline: string
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
): { weakest: Sport; headline: string; actions: TrainingAction[] } {
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

  const staleLabels = SPORT_ORDER.filter(s => thresholds.get(s)!.staleDays > 45).map(
    s => `${s} ${thresholds.get(s)!.staleDays}d stale`,
  )
  const buildLabels = SPORT_ORDER.filter(s => thresholds.get(s)!.staleDays <= 45)
  const headline = staleLabels.length
    ? `${buildLabels.length ? `${buildLabels.join('/')} build, ` : ''}${staleLabels.join(', ')}`
    : 'balanced build'

  return { weakest, headline, actions }
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
  series: [],
})

function weightTrendPerWeek(entries: TrackEntry[]): number | null {
  const pts = entries
    .filter(e => e.weightKg != null)
    .map(e => ({ t: dayMs(e.date) / DAY_MS, w: e.weightKg as number }))
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
    events: [],
    activities: [],
    weakestSport: 'run',
    headline: '',
    actions: [],
  }
}

export function buildAnalytics(
  cache: StravaRawCache | null,
  inputs: AnalyticsInputs = {},
): Analytics {
  const todayFromSync = cache?.lastSync ? new Date(cache.lastSync).toISOString().slice(0, 10) : null

  if (!cache) return emptyAnalytics(0, todayFromSync ?? '1970-01-01')

  const sinceDay =
    inputs.since && /^\d{4}-\d{2}-\d{2}$/.test(inputs.since) ? inputs.since : null
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
  const weightByDate = new Map<string, number>()
  for (const w of inputs.weights ?? []) if (w.weightKg != null) weightByDate.set(w.date, w.weightKg)
  let carryKg: number | null = null
  for (const d of daily) {
    const o = ouraDays[d.date]
    if (o) {
      d.readiness = o.readiness
      d.hrv = o.hrv
      d.rhr = o.rhr
      d.sleepScore = o.sleepScore
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
  const body: BodyBlock = {
    latestKg,
    latestLbs: latestKg != null ? round(latestKg / 0.45359237, 1) : null,
    trendKgPerWeek: weightTrendPerWeek(inputs.weights ?? []),
    series: (inputs.weights ?? [])
      .filter(w => w.weightKg != null)
      .map(w => ({ date: w.date, kg: w.weightKg as number })),
  }
  const weekly = buildWeekly(acts, loadById)
  const trends = SPORT_ORDER.map(sport => buildTrend(acts, thresholds.get(sport)!, sport, todayMs))
  const bestList = SPORT_ORDER.map(sport => buildBest(acts, sport))
  const bests = new Map<Sport, SportBest>(bestList.map(b => [b.sport, b]))
  const risk = buildRisk(daily, weekly)

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

  const activities: ActivitySummary[] = acts
    .map(act => ({
      id: act.a.id,
      date: act.day,
      sport: act.sport,
      name: act.a.name ?? '',
      distanceKm: act.distanceKm,
      movingTimeS: act.a.movingTime,
      load: round(loadById.get(act.a.id) ?? 0, 1),
      cadence: act.a.averageCadence ?? null,
    }))
    .sort((p, q) => q.date.localeCompare(p.date))

  const { weakest, headline, actions } = buildActions(thresholds, bests, daily, loadShare)

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
    events: inputs.events ?? [],
    activities,
    weakestSport: weakest,
    headline,
    actions,
  }
}
