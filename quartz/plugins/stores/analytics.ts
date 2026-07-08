import type { GarminCache, GarminWeightSample } from './garmin'
import type { WeatherCache } from './weather'
import { localIsoDay } from '../../util/local-date'
import { isRecord, numberValue } from '../../util/type-guards'
import { AppleCache } from './apple'
import { matchGarminHeartRateActivity } from './garmin'
import { OuraCache } from './oura'
import {
  type Sport,
  type ActivityKind,
  SPORT_ORDER,
  normalizeSport,
  normalizeKind,
  isTreatment,
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

export interface DexaRegion {
  fat: number
  lean: number
  bmc: number
}

export interface DexaRecord {
  date: string
  totalLbs: number
  fatLbs: number
  leanLbs: number
  bmcLbs: number
  ffmLbs: number | null
  bodyFat: number
  vatLbs: number | null
  bmd: number | null
  bmdT: number | null
  rmr: number | null
  rsmi: number | null
  ag: number | null
  arms: DexaRegion | null
  legs: DexaRegion | null
  trunk: DexaRegion | null
}

export interface Vo2LabRecord {
  date: string
  value: number
  massKg: number | null
  hrMax: number | null
  hrAtVo2max: number | null
  vt1Hr: number | null
  vt1Kmh: number | null
  vt2Hr: number | null
  vt2Kmh: number | null
  caloriesAtVt1: number | null
  maxKmh: number | null
  ve: number | null
  percentile: number | null
  zonesHr: number[]
  zonesKmh: number[]
  zonesKcal: number[]
  profile: Vo2LabProfile | null
}

export interface Vo2LabProfileStats {
  min: number
  max: number
  avg: number
}

export interface Vo2LabProfileStatsMap {
  vo2: Vo2LabProfileStats | null
  hr: Vo2LabProfileStats | null
  ve: Vo2LabProfileStats | null
  rf: Vo2LabProfileStats | null
  tv: Vo2LabProfileStats | null
}

export interface Vo2LabTargetStep {
  t: number
  kmh: number
}

export interface Vo2LabProfileSample {
  t: number
  vo2: number | null
  hr: number | null
  ve: number | null
  rf: number | null
  tv: number | null
}

export interface Vo2LabProfile {
  durationSec: number
  warmupEndSec: number | null
  cooldownStartSec: number | null
  vt1Sec: number | null
  vo2maxSec: number | null
  stats: Vo2LabProfileStatsMap
  targetKmh: Vo2LabTargetStep[]
  samples: Vo2LabProfileSample[]
}

export interface LabTests {
  dexa: DexaRecord[]
  vo2max: Vo2LabRecord[]
}

export interface BodyBlock {
  latestKg: number | null
  latestLbs: number | null
  trendKgPerWeek: number | null
  goalKg: number | null
  goalLbs: number | null
  goalDeltaKg: number | null
  goalEtaWeeks: number | null
  goalBmr: number | null
  goalLeanBmr: number | null
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
  effort: number | null
  cadence: number | null
  strokes: Record<string, number> | null
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
  dexa?: unknown
  vo2labs?: unknown
  ftp?: number | null
  since?: string
  timeZone?: string
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
  tempDevC: number | null
  weightKg: number | null
  totalCalories: number | null
  intakeKcal: number | null
  warmup: boolean
}

export interface WeeklyPoint {
  weekStart: string
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
  currentTotalS: number
  predictedFastS: number
  predictedSlowS: number
  projected: boolean
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

export type Vo2Method = 'garmin' | 'apple' | 'bike' | 'run' | 'hrratio' | 'lab' | 'none'
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
  proj: number | null
  rawValue: number | null
  rawUnit: string
  lo: number
  hi: number
}

export interface AbilityTrendPoint {
  date: string
  sprint: number | null
  threshold: number | null
  endurance: number | null
  climb: number | null
  cadence: number | null
  recovery: number | null
}

export interface SportAbilities {
  sport: Sport
  axes: RadarAxis[]
  area: number | null
  history: AbilityTrendPoint[]
}

export interface AbilitiesBlock {
  sports: SportAbilities[]
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

export interface FtpHypothesisParams {
  crossModalDiscountPct: number
  thresholdPct: number
  grossEfficiencyPct: number
}

export interface FtpHypothesis {
  date: string
  conf: Conf
  massKg: number
  runningVo2max: number
  crossModalDiscountPct: number
  thresholdPct: number
  grossEfficiencyPct: number
  absoluteRunningVo2: number
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
  note: string
}

export interface EngineBlock {
  vo2max: Vo2maxBlock
  abilities: AbilitiesBlock
  cardio: CardioBlock
  ftpHypothesis: FtpHypothesis | null
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
  calibration: CalibrationBlock
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
  tests: LabTests
}

interface Act {
  a: RawStravaActivity
  sport: Sport
  day: string
  distanceKm: number
  vGap: number
}

const swimStrokesByActivityId = (
  acts: readonly Act[],
  apple: AppleCache | null | undefined,
): Map<number, Record<string, number>> => {
  const byDay = new Map<string, Act>()
  for (const act of acts) {
    if (act.sport !== 'swim') continue
    const cur = byDay.get(act.day)
    if (!cur || act.distanceKm > cur.distanceKm) byDay.set(act.day, act)
  }
  const out = new Map<number, Record<string, number>>()
  for (const [day, act] of byDay) {
    const strokes = apple?.swims?.[day]?.strokes
    if (strokes && Object.keys(strokes).length > 0) out.set(act.a.id, strokes)
  }
  return out
}

const DAY_MS = 86_400_000
const K42 = 1 - Math.exp(-1 / 42)
const K7 = 1 - Math.exp(-1 / 7)
const IF_CAP = 1.15
const CALIBRATION_WINDOW_DAYS = 28
const CALIBRATION_PROJECTION_DAYS = 14

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
  type DayBucket = { total: number; effort: number; swim: number; bike: number; run: number }
  const emptyDay = (): DayBucket => ({ total: 0, effort: 0, swim: 0, bike: 0, run: 0 })
  const byDay = new Map<string, DayBucket>()
  for (const act of acts) {
    const load = loadById.get(act.a.id) ?? 0
    const bucket = byDay.get(act.day) ?? emptyDay()
    bucket.total += load
    bucket.effort += act.a.sufferScore ?? 0
    bucket[act.sport] += load
    byDay.set(act.day, bucket)
  }

  const rows: ({ date: string } & DayBucket)[] = []
  for (let ms = windowFrom; ms <= windowTo; ms += DAY_MS) {
    const date = new Date(ms).toISOString().slice(0, 10)
    const bucket = byDay.get(date) ?? emptyDay()
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
      effort: round(row.effort, 0),
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

type WeeklyBucket = {
  sessions: number
  load: number
  km: number
  seconds: number
  effort: number
  swimKm: number
  bikeKm: number
  runKm: number
  swimSeconds: number
  bikeSeconds: number
  runSeconds: number
}

const emptyWeeklyBucket = (): WeeklyBucket => ({
  sessions: 0,
  load: 0,
  km: 0,
  seconds: 0,
  effort: 0,
  swimKm: 0,
  bikeKm: 0,
  runKm: 0,
  swimSeconds: 0,
  bikeSeconds: 0,
  runSeconds: 0,
})

function buildWeekly(acts: Act[], loadById: Map<number, number>): WeeklyPoint[] {
  const byWeek = new Map<string, WeeklyBucket>()
  for (const act of acts) {
    const ms = dayMs(act.day)
    const dow = new Date(ms).getUTCDay()
    const offset = (dow + 6) % 7
    const weekStart = new Date(ms - offset * DAY_MS).toISOString().slice(0, 10)
    const bucket = byWeek.get(weekStart) ?? emptyWeeklyBucket()
    bucket.sessions += 1
    bucket.load += loadById.get(act.a.id) ?? 0
    bucket.km += act.distanceKm
    bucket.seconds += act.a.movingTime
    bucket.effort += act.a.sufferScore ?? 0
    if (act.sport === 'swim') {
      bucket.swimKm += act.distanceKm
      bucket.swimSeconds += act.a.movingTime
    } else if (act.sport === 'bike') {
      bucket.bikeKm += act.distanceKm
      bucket.bikeSeconds += act.a.movingTime
    } else {
      bucket.runKm += act.distanceKm
      bucket.runSeconds += act.a.movingTime
    }
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
      sessions: cur.sessions,
      load: round(cur.load, 1),
      km: round(cur.km, 1),
      hours: round(cur.seconds / 3600, 1),
      swimKm: round(cur.swimKm, 1),
      bikeKm: round(cur.bikeKm, 1),
      runKm: round(cur.runKm, 1),
      swimHours: round(cur.swimSeconds / 3600, 1),
      bikeHours: round(cur.bikeSeconds / 3600, 1),
      runHours: round(cur.runSeconds / 3600, 1),
      effort: round(cur.effort, 0),
      ramp,
      monotony,
      strain,
    }
  })
}

const TREND_FORECAST_DAYS = 14
const TREND_HALFLIFE_DAYS = 28
const TREND_DAMP = 0.94
const TREND_Z = 1.28

const dampedTrendForecast = (
  today: number,
  etaNow: number,
  slopePerDay: number,
  halfAt: (d: number) => number,
): SportTrendForecastPoint[] => {
  const out: SportTrendForecastPoint[] = []
  let phi = 1
  let cum = 0
  for (let d = 1; d <= TREND_FORECAST_DAYS; d++) {
    phi *= TREND_DAMP
    cum += phi
    const value = etaNow + slopePerDay * cum
    const half = halfAt(d)
    out.push({
      date: new Date(today + d * DAY_MS).toISOString().slice(0, 10),
      value: round(value, 1),
      lo: round(value - half, 1),
      hi: round(value + half, 1),
    })
  }
  return out
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
    const w = xs.map(x => 0.5 ** ((todayX - x) / TREND_HALFLIFE_DAYS))
    const wSum = w.reduce((s, v) => s + v, 0)
    const mx = xs.reduce((s, x, i) => s + w[i] * x, 0) / wSum
    const my = ys.reduce((s, y, i) => s + w[i] * y, 0) / wSum
    let sxx = 0
    let sxy = 0
    for (let i = 0; i < n; i++) {
      sxx += w[i] * (xs[i] - mx) ** 2
      sxy += w[i] * (xs[i] - mx) * (ys[i] - my)
    }
    const b = sxx > 0 ? sxy / sxx : 0
    const a = my - b * mx
    const wr2 = ys.reduce((s, y, i) => s + w[i] * (y - (a + b * xs[i])) ** 2, 0)
    const nEff = wSum ** 2 / w.reduce((s, v) => s + v * v, 0)
    const se = Math.sqrt(wr2 / wSum) * Math.sqrt(nEff / Math.max(1, nEff - 2))
    const etaNow = a + b * todayX
    const halfAt = (d: number): number =>
      TREND_Z * se * Math.sqrt(1 / nEff + (todayX + d - mx) ** 2 / (sxx || 1))
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
      forecast: dampedTrendForecast(today, etaNow, b, halfAt),
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
  const sigma = sd(resid)
  const firstHalf = mean(ys.slice(0, Math.max(1, Math.floor(ys.length / 2))))
  const secondHalf = mean(ys.slice(Math.floor(ys.length / 2)))
  const perPoint = ys.length > 1 ? (secondHalf - firstHalf) / Math.max(1, ys.length / 2) : 0
  const avgGap = spanDays > 0 && n > 1 ? spanDays / (n - 1) : 7
  const slopePerWeek = avgGap > 0 ? round((perPoint / avgGap) * 7, 2) : 0
  const sigmaLevel = sigma * Math.sqrt(alpha / (2 - alpha))
  const halfAt = (d: number): number => TREND_Z * sigmaLevel * Math.sqrt(d / TREND_FORECAST_DAYS)
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
    forecast: dampedTrendForecast(today, level, slopePerWeek / 7, halfAt),
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
const T1_S = 300
const T2_S = 300
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

const TREND_PROJ_CLAMP = 0.08
const TREND_BAND_CLAMP = 0.12

function trendVelocityRatios(tr: SportTrend | undefined): {
  mid: number
  fast: number
  slow: number
  usable: boolean
} {
  const none = { mid: 1, fast: 1, slow: 1, usable: false }
  if (!tr || tr.stale || tr.level == null || tr.level <= 0) return none
  const last = tr.forecast[tr.forecast.length - 1]
  if (!last || last.value <= 0 || last.lo <= 0 || last.hi <= 0) return none
  const mid = clamp(
    tr.invert ? tr.level / last.value : last.value / tr.level,
    1 - TREND_PROJ_CLAMP,
    1 + TREND_PROJ_CLAMP,
  )
  const halfFrac = clamp((last.hi - last.lo) / (2 * last.value), 0, TREND_BAND_CLAMP)
  return { mid, fast: mid * (1 + halfFrac), slow: mid * (1 - halfFrac), usable: true }
}

function buildReadiness(
  distance: RaceDistance,
  thresholds: Map<Sport, ThresholdEstimate>,
  bests: Map<Sport, SportBest>,
  ctlNow: number,
  trends: Map<Sport, SportTrend>,
): RaceReadiness {
  const legKm = RACE_LEGS[distance]
  const ratios = new Map(SPORT_ORDER.map(s => [s, trendVelocityRatios(trends.get(s))]))
  const splitAt = (sport: Sport, vThr: number): number => {
    const raw = legSplitSeconds(legKm[sport], vThr, RIEGEL_K[sport])
    return sport === 'run' ? raw * RUN_BRICK_FADE[distance] : raw
  }
  const legs: RaceLeg[] = SPORT_ORDER.map(sport => {
    const th = thresholds.get(sport)!
    const longestKm = bests.get(sport)?.longestKm ?? 0
    const coverage = clamp(longestKm / legKm[sport], 0, 1)
    const gate = recencyGate(th.staleDays)
    return {
      sport,
      legKm: legKm[sport],
      longestKm,
      coverage,
      recencyGate: gate,
      splitS: round(splitAt(sport, th.vThr * ratios.get(sport)!.mid), 0),
    }
  })

  const trans = T1_S + T2_S
  const total = legs.reduce((s, l) => s + l.splitS, 0) + trans
  let currentTotal = trans
  let fastTotal = trans
  let slowTotal = trans
  for (const sport of SPORT_ORDER) {
    const th = thresholds.get(sport)!
    const r = ratios.get(sport)!
    currentTotal += splitAt(sport, th.vThr)
    fastTotal += splitAt(sport, th.vThr * r.fast)
    slowTotal += splitAt(sport, th.vThr * r.slow)
  }
  const projected = SPORT_ORDER.some(s => ratios.get(s)!.usable && ratios.get(s)!.mid !== 1)

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
    currentTotalS: round(currentTotal, 0),
    predictedFastS: round(fastTotal, 0),
    predictedSlowS: round(slowTotal, 0),
    projected,
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
  return round(average * (end / trend.level), 1)
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

function buildCalibration(
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
    const mine = acts.filter(act => act.sport === sport)
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
    const projected = projectedHuman(average, trends.get(sport), CALIBRATION_PROJECTION_DAYS)
    const delta = average != null && previous != null ? round(average - previous, 1) : null
    const deltaPct = fasterPct(sport, average, previous)
    const projectedDelta =
      projected != null && average != null ? round(projected - average, 1) : null
    const projectedDeltaPct = fasterPct(sport, projected, average)
    return {
      sport,
      unit: th?.unit ?? (sport === 'bike' ? 'km/h' : sport === 'swim' ? 's/100m' : 's/km'),
      average,
      projected,
      previous,
      delta,
      deltaPct,
      projectedDelta,
      projectedDeltaPct,
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

const emptyCalibration = (today: string): CalibrationBlock => ({
  asOf: today,
  windowDays: CALIBRATION_WINDOW_DAYS,
  projectionDays: CALIBRATION_PROJECTION_DAYS,
  paces: [],
  volume: volumeBlock(emptyVolumeBucket(), emptyVolumeBucket()),
})

const emptyBody = (): BodyBlock => ({
  latestKg: null,
  latestLbs: null,
  trendKgPerWeek: null,
  goalKg: null,
  goalLbs: null,
  goalDeltaKg: null,
  goalEtaWeeks: null,
  goalBmr: null,
  goalLeanBmr: null,
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

const katchMcArdleBmrFromLeanKg = (leanKg: number): number => Math.round(370 + 21.6 * leanKg)

const katchMcArdleBmr = (weightKg: number, bodyFatPct: number): number =>
  katchMcArdleBmrFromLeanKg(weightKg * (1 - bodyFatPct / 100))

const mifflinStJeorBmr = (
  weightKg: number,
  heightCm: number,
  ageYears: number,
  sex: typeof ATHLETE.sex,
): number => Math.round(10 * weightKg + 6.25 * heightCm - 5 * ageYears + (sex === 'M' ? 5 : -161))

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
  sleepTargetS: 25200,
  sleepFloorS: 21600,
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
      detail: `${(sleepDebtS / 3600).toFixed(1)} h short of the 7 h target over 14 nights`,
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
  sex: 'M' as 'M' | 'F',
  born: '2001-03',
  bornAnchor: '2001-03-01',
  hrMax: 184 as number | null,
  vo2max: 47.8 as number | null,
  ftp: 230 as number | null,
  goalWeightLb: 160 as number | null,
  goalFTP: 290 as number | null,
  heightCm: 188,
}

const goalWeightKg = ATHLETE.goalWeightLb != null ? ATHLETE.goalWeightLb * 0.45359237 : null

const isoOf = (v: unknown): string | undefined => {
  if (typeof v === 'string') return v
  if (v instanceof Date) return v.toISOString().slice(0, 10)
  return undefined
}

const numAt = (r: Record<string, unknown>, k: string): number | null => {
  const v = numberValue(r[k])
  return v === undefined ? null : v
}

const numArr = (raw: unknown): number[] => {
  if (!Array.isArray(raw)) return []
  const out: number[] = []
  for (const v of raw) {
    const n = numberValue(v)
    if (n !== undefined) out.push(n)
  }
  return out
}

const nullableNum = (raw: unknown): number | null => {
  const n = numberValue(raw)
  return n === undefined ? null : n
}

const vo2StatsOf = (raw: unknown): Vo2LabProfileStats | null => {
  let min: number | undefined
  let max: number | undefined
  let avg: number | undefined
  if (Array.isArray(raw)) {
    min = numberValue(raw[0])
    max = numberValue(raw[1])
    avg = numberValue(raw[2])
  } else if (isRecord(raw)) {
    min = numberValue(raw.min)
    max = numberValue(raw.max)
    avg = numberValue(raw.avg)
  }
  if (min === undefined || max === undefined || avg === undefined) return null
  return { min, max, avg }
}

const vo2StatsMapOf = (raw: unknown): Vo2LabProfileStatsMap => ({
  vo2: isRecord(raw) ? vo2StatsOf(raw.vo2) : null,
  hr: isRecord(raw) ? vo2StatsOf(raw.hr) : null,
  ve: isRecord(raw) ? vo2StatsOf(raw.ve) : null,
  rf: isRecord(raw) ? vo2StatsOf(raw.rf) : null,
  tv: isRecord(raw) ? vo2StatsOf(raw.tv) : null,
})

const vo2TargetStepsOf = (raw: unknown): Vo2LabTargetStep[] => {
  if (!Array.isArray(raw)) return []
  const out: Vo2LabTargetStep[] = []
  for (const item of raw) {
    let t: number | undefined
    let kmh: number | undefined
    if (Array.isArray(item)) {
      t = numberValue(item[0])
      kmh = numberValue(item[1])
    } else if (isRecord(item)) {
      t = numberValue(item.t)
      kmh = numberValue(item.kmh)
    }
    if (t !== undefined && kmh !== undefined && t >= 0 && kmh >= 0) out.push({ t, kmh })
  }
  out.sort((a, b) => a.t - b.t)
  return out
}

const vo2ProfileSamplesOf = (raw: unknown): Vo2LabProfileSample[] => {
  if (!Array.isArray(raw)) return []
  const out: Vo2LabProfileSample[] = []
  for (const item of raw) {
    let sample: Vo2LabProfileSample | null = null
    if (Array.isArray(item)) {
      const t = numberValue(item[0])
      if (t !== undefined)
        sample = {
          t,
          vo2: nullableNum(item[1]),
          hr: nullableNum(item[2]),
          ve: nullableNum(item[3]),
          rf: nullableNum(item[4]),
          tv: nullableNum(item[5]),
        }
    } else if (isRecord(item)) {
      const t = numberValue(item.t)
      if (t !== undefined)
        sample = {
          t,
          vo2: nullableNum(item.vo2),
          hr: nullableNum(item.hr),
          ve: nullableNum(item.ve),
          rf: nullableNum(item.rf),
          tv: nullableNum(item.tv),
        }
    }
    if (sample && sample.t >= 0) out.push(sample)
  }
  out.sort((a, b) => a.t - b.t)
  return out
}

const vo2ProfileOf = (raw: unknown): Vo2LabProfile | null => {
  if (!isRecord(raw)) return null
  const samples = vo2ProfileSamplesOf(raw.samples)
  const targetKmh = vo2TargetStepsOf(raw.targetKmh)
  const duration = numberValue(raw.durationSec) ?? samples[samples.length - 1]?.t
  if (duration === undefined || duration <= 0 || samples.length < 2 || targetKmh.length < 2)
    return null
  return {
    durationSec: duration,
    warmupEndSec: numAt(raw, 'warmupEndSec'),
    cooldownStartSec: numAt(raw, 'cooldownStartSec'),
    vt1Sec: numAt(raw, 'vt1Sec'),
    vo2maxSec: numAt(raw, 'vo2maxSec'),
    stats: vo2StatsMapOf(raw.stats),
    targetKmh,
    samples,
  }
}

const dexaRegionOf = (raw: unknown): DexaRegion | null => {
  if (!isRecord(raw)) return null
  const fat = numberValue(raw.fat)
  const lean = numberValue(raw.lean)
  const bmc = numberValue(raw.bmc)
  if (fat === undefined || lean === undefined || bmc === undefined) return null
  return { fat, lean, bmc }
}

const parseDexa = (raw: unknown): DexaRecord[] => {
  if (!Array.isArray(raw)) return []
  const out: DexaRecord[] = []
  for (const item of raw) {
    if (!isRecord(item)) continue
    const date = isoOf(item.date)
    const totalLbs = numberValue(item.totalLbs)
    const fatLbs = numberValue(item.fatLbs)
    const leanLbs = numberValue(item.leanLbs)
    const bmcLbs = numberValue(item.bmcLbs)
    const bodyFat = numberValue(item.bodyFat)
    if (
      date === undefined ||
      totalLbs === undefined ||
      fatLbs === undefined ||
      leanLbs === undefined ||
      bmcLbs === undefined ||
      bodyFat === undefined
    )
      continue
    out.push({
      date,
      totalLbs,
      fatLbs,
      leanLbs,
      bmcLbs,
      ffmLbs: numAt(item, 'ffmLbs'),
      bodyFat,
      vatLbs: numAt(item, 'vatLbs'),
      bmd: numAt(item, 'bmd'),
      bmdT: numAt(item, 'bmdT'),
      rmr: numAt(item, 'rmr'),
      rsmi: numAt(item, 'rsmi'),
      ag: numAt(item, 'ag'),
      arms: dexaRegionOf(item.arms),
      legs: dexaRegionOf(item.legs),
      trunk: dexaRegionOf(item.trunk),
    })
  }
  out.sort((a, b) => a.date.localeCompare(b.date))
  return out
}

export const hrZoneUppers = (rec: Vo2LabRecord): number[] | null => {
  const e = rec.zonesHr
  if (e.length < 2) return null
  const max = rec.hrMax ?? e[e.length - 1]
  return e.map((_, i) => (i + 1 < e.length ? e[i + 1] : max) - 1)
}

export const parseVo2Lab = (raw: unknown): Vo2LabRecord[] => {
  if (!Array.isArray(raw)) return []
  const out: Vo2LabRecord[] = []
  for (const item of raw) {
    if (!isRecord(item)) continue
    const date = isoOf(item.date)
    const value = numberValue(item.value)
    if (date === undefined || value === undefined) continue
    out.push({
      date,
      value,
      massKg: numAt(item, 'massKg'),
      hrMax: numAt(item, 'hrMax'),
      hrAtVo2max: numAt(item, 'hrAtVo2max'),
      vt1Hr: numAt(item, 'vt1Hr'),
      vt1Kmh: numAt(item, 'vt1Kmh'),
      vt2Hr: numAt(item, 'vt2Hr'),
      vt2Kmh: numAt(item, 'vt2Kmh'),
      caloriesAtVt1: numAt(item, 'caloriesAtVt1'),
      maxKmh: numAt(item, 'maxKmh'),
      ve: numAt(item, 've'),
      percentile: numAt(item, 'percentile'),
      zonesHr: numArr(item.zonesHr),
      zonesKmh: numArr(item.zonesKmh),
      zonesKcal: numArr(item.zonesKcal),
      profile: vo2ProfileOf(item.profile),
    })
  }
  out.sort((a, b) => a.date.localeCompare(b.date))
  return out
}

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
const FTP_VO2_KJ_PER_L = 20.9
const FTP_ACSM_KGM_PER_WATT = 6.12
const FTP_ACSM_VO2_PER_KGM = 1.8
export const FTP_HYPOTHESIS_DEFAULTS: FtpHypothesisParams = {
  crossModalDiscountPct: 8,
  thresholdPct: 85,
  grossEfficiencyPct: 21,
}
const FTP_HYPOTHESIS_ERROR_W = 25
const DANIELS_A = -4.6
const DANIELS_B = 0.182258
const DANIELS_C = 0.000104
const UTH_K = 15.3
const TANAKA_A = 208
const TANAKA_B = 0.7
const COGGAN_SPRINT_WKG: [number, number] = [7, 24]
const COGGAN_FTP_WKG: [number, number] = [1.5, 6.4]
const VAM_ANCHOR: [number, number] = [300, 1500]
const RUN_VAM_ANCHOR: [number, number] = [100, 1200]
const CTL_ANCHOR: [number, number] = [0, 100]
const RUN_SPM_TARGET = 180
const BIKE_RPM_TARGET = 90
const SWIM_STROKE_TARGET = 30
const RUN_SPRINT_MS: [number, number] = [3, 8]
const SWIM_SPRINT_MS: [number, number] = [0.8, 2.2]
const RUN_THR_MS: [number, number] = [2.5, 5.5]
const SWIM_THR_MS: [number, number] = [0.7, 1.7]
const BIKE_SPRINT_WIN_S = 5
const RUN_SPRINT_WIN_S = 30
const SPRINT_CAP_MS: Record<'swim' | 'run', number> = { swim: 3, run: 12 }
const PROJ_WINDOW_D = 28
const PROJ_HORIZON_D = 28
const PROJ_MIN_POINTS = 8
const CAD_TARGET: Record<Sport, number> = {
  swim: SWIM_STROKE_TARGET,
  bike: BIKE_RPM_TARGET,
  run: RUN_SPM_TARGET,
}
const CAD_UNIT: Record<Sport, string> = { swim: 'str/min', bike: 'rpm', run: 'spm' }
const SPEED_SPRINT_MS: Record<'swim' | 'run', [number, number]> = {
  swim: SWIM_SPRINT_MS,
  run: RUN_SPRINT_MS,
}
const SPEED_THR_MS: Record<'swim' | 'run', [number, number]> = {
  swim: SWIM_THR_MS,
  run: RUN_THR_MS,
}
const VAM_ANCHOR_OF: Record<'bike' | 'run', [number, number]> = {
  bike: VAM_ANCHOR,
  run: RUN_VAM_ANCHOR,
}
const ONE_HZ_TOL = 0.15
const DECOUPLE_MIN_S = 1200
const PEAK_WINDOWS = [30, 60, 300, 1200] as const

const ageOn = (iso: string): number =>
  Math.floor((dayMs(iso) - dayMs(ATHLETE.bornAnchor)) / (365.25 * DAY_MS))

const norm01 = (v: number, lo: number, hi: number): number => clamp((v - lo) / (hi - lo), 0, 1)
const round5 = (v: number): number => Math.round(v / 5) * 5
const round10 = (v: number): number => Math.round(v / 10) * 10

export function computeFtpHypothesisFromVo2(
  date: string,
  runningVo2max: number,
  massKg: number,
  params: FtpHypothesisParams = FTP_HYPOTHESIS_DEFAULTS,
): FtpHypothesis | null {
  if (!(runningVo2max > 0) || !(massKg > 0)) return null
  const discount = params.crossModalDiscountPct / 100
  const threshold = params.thresholdPct / 100
  const efficiency = params.grossEfficiencyPct / 100
  const absoluteRunningVo2 = (runningVo2max * massKg) / 1000
  const cyclingVo2max = absoluteRunningVo2 * (1 - discount)
  const thresholdVo2 = cyclingVo2max * threshold
  const metabolicWatts = (thresholdVo2 * FTP_VO2_KJ_PER_L * 1000) / 60
  const efficiencyFtp = metabolicWatts * efficiency
  const cyclingVo2Rel = runningVo2max * (1 - discount)
  const acsmMapWatts =
    (Math.max(0, cyclingVo2Rel - ACSM_BASE) * massKg) / FTP_ACSM_VO2_PER_KGM / FTP_ACSM_KGM_PER_WATT
  const acsmFtp = acsmMapWatts * MAP_FTP_RATIO
  const ftpMean = (efficiencyFtp + acsmFtp) / 2
  const ftp = round10(ftpMean)
  return {
    date,
    conf: 'low',
    massKg: round(massKg, 1),
    runningVo2max: round(runningVo2max, 1),
    crossModalDiscountPct: params.crossModalDiscountPct,
    thresholdPct: params.thresholdPct,
    grossEfficiencyPct: params.grossEfficiencyPct,
    absoluteRunningVo2: round(absoluteRunningVo2, 2),
    cyclingVo2max: round(cyclingVo2max, 2),
    thresholdVo2: round(thresholdVo2, 2),
    metabolicWatts: round(metabolicWatts, 0),
    efficiencyFtp: round(efficiencyFtp, 0),
    acsmMapWatts: round(acsmMapWatts, 0),
    acsmFtp: round(acsmFtp, 0),
    ftp,
    low: round5(ftpMean - FTP_HYPOTHESIS_ERROR_W),
    high: round5(ftpMean + FTP_HYPOTHESIS_ERROR_W),
    wattsPerKg: round(ftp / massKg, 2),
    note: 'running vo2max to cycling ftp',
  }
}

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
    abilities: { sports: [] },
    cardio: { metrics: [], rhrSeries: [], hrvSeries: [], efSeries: [], decouplingSeries: [] },
    ftpHypothesis: null,
  }
}

function buildEngine(
  cache: StravaRawCache,
  acts: Act[],
  daily: DailyPoint[],
  body: BodyBlock,
  thresholds: Map<Sport, ThresholdEstimate>,
  today: string,
  garminVo2: { date: string; v: number }[],
  appleVo2: { date: string; v: number }[],
  heartRateById: Map<number, ActivityHeartRate>,
  vo2Lab: Vo2LabRecord | null,
  ftpOverride: number | null,
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
  if (vo2Lab)
    estimates.push({
      method: 'lab',
      vo2max: round(clamp(vo2Lab.value, VO2_FLOOR, VO2_CEIL), 1),
      conf: 'firm',
    })
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
  const declaredFtp = ftpOverride ?? cache.zones?.ftp ?? null
  let ftp = declaredFtp
  let ftpSrc: 'athlete' | 'strava' | 'derived' | null =
    ftpOverride != null ? 'athlete' : declaredFtp != null ? 'strava' : null
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
    if (m === 'bike') {
      const src = ftpSrc === 'athlete' ? ' (athlete)' : ftpSrc === 'strava' ? ' (strava)' : ''
      return `ftp ${ftp}w${src} · map ${ftp != null ? Math.round(ftp / MAP_FTP_RATIO) : '—'}w · ${bikeKg != null ? round(bikeKg, 1) : '—'}kg`
    }
    if (m === 'run') return 'run hr–speed extrapolation'
    if (m === 'hrratio') return 'upper-bound proxy from sleeping rhr'
    if (m === 'lab') return 'graded exercise test'
    return 'no power or hr data'
  }

  const weekOf = (iso: string): string => {
    const ms = dayMs(iso)
    const dow = new Date(ms).getUTCDay()
    return new Date(ms - ((dow + 6) % 7) * DAY_MS).toISOString().slice(0, 10)
  }
  type Vo2Week = {
    p20: number | null
    kg: number | null
    garmin: number | null
    apple: number | null
  }
  const weeks = new Map<string, Vo2Week>()
  const weekAt = (day: string): Vo2Week => {
    const ws = weekOf(day)
    let w = weeks.get(ws)
    if (!w) weeks.set(ws, (w = { p20: null, kg: null, garmin: null, apple: null }))
    return w
  }
  for (const d of daily) {
    const w = weekAt(d.date)
    if (d.weightKg != null) w.kg = d.weightKg
  }
  for (const b of bikes) {
    const v = peakMean(wattsOf(b.a.id), 1200)
    if (v == null) continue
    const w = weekAt(b.day)
    if (w.p20 == null || v > w.p20) w.p20 = v
  }
  for (const g of garminVo2) weekAt(g.date).garmin = g.v
  for (const p of appleVo2) weekAt(p.date).apple = p.v
  const trend: Vo2Point[] = []
  let last: Vo2Point | null = null
  let kgCarry: number | null = null
  for (const ws of [...weeks.keys()].sort()) {
    const w = weeks.get(ws)!
    if (w.kg != null) kgCarry = w.kg
    const measured = w.garmin ?? w.apple
    if (measured != null)
      last = {
        weekStart: ws,
        vo2max: round(clamp(measured, VO2_FLOOR, VO2_CEIL), 1),
        method: w.garmin != null ? 'garmin' : 'apple',
      }
    else if (w.p20 != null && kgCarry != null && kgCarry > 0)
      last = {
        weekStart: ws,
        vo2max: round(
          clamp(
            (ACSM_WATT_K * ((w.p20 * FTP_FROM_P20) / MAP_FTP_RATIO)) / kgCarry + ACSM_BASE,
            VO2_FLOOR,
            VO2_CEIL,
          ),
          1,
        ),
        method: 'bike',
      }
    if (last) trend.push(last.weekStart === ws ? last : { ...last, weekStart: ws })
  }
  if (vo2Lab) {
    const lw = weekOf(vo2Lab.date)
    const lv = round(clamp(vo2Lab.value, VO2_FLOOR, VO2_CEIL), 1)
    const at = trend.findIndex(p => p.weekStart === lw)
    if (at >= 0) trend[at] = { weekStart: lw, vo2max: lv, method: 'lab' }
    else {
      trend.push({ weekStart: lw, vo2max: lv, method: 'lab' })
      trend.sort((a, b) => a.weekStart.localeCompare(b.weekStart))
    }
  }

  const vo2 = primary?.vo2max ?? null
  const fitnessAge = vo2 != null ? Math.round(clamp(invLerp(FRIEND_MED_M, vo2), 20, 80)) : null
  const ageDeltaYears = fitnessAge != null ? fitnessAge - age : null
  const percentileForAge = vo2 != null ? pctForAge(vo2, age) : null
  const ftpHypothesis = vo2Lab
    ? computeFtpHypothesisFromVo2(vo2Lab.date, vo2Lab.value, vo2Lab.massKg ?? body.latestKg ?? 0)
    : null

  const kgNow = body.latestKg
  let p5: number | null = null
  for (const b of bikes) {
    const v = peakMean(wattsOf(b.a.id), BIKE_SPRINT_WIN_S)
    if (v != null && (p5 == null || v > p5)) p5 = v
  }
  if (p5 == null)
    for (const b of bikes)
      if (b.a.maxWatts != null && (p5 == null || b.a.maxWatts > p5)) p5 = b.a.maxWatts
  const sprintWkg = p5 != null && kgNow ? p5 / kgNow : null
  const ftpWkg = ftp != null && kgNow ? ftp / kgNow : null
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

  const distOf = (id: number): number[] => cache.streams?.[String(id)]?.distance ?? []
  const peakRunSpeedOf = (x: Act): number | null => {
    const d = distOf(x.a.id)
    if (!oneHz(d.length, x.a.movingTime)) return null
    const v = peakDistRate(d, RUN_SPRINT_WIN_S)
    return v != null && v <= SPRINT_CAP_MS.run ? v : null
  }
  const plausibleVgap = (sport: 'swim' | 'run', x: Act): number | null =>
    x.vGap > 0 && x.vGap <= SPRINT_CAP_MS[sport] ? x.vGap : null
  const vamOf = (x: Act): number | null =>
    x.a.movingTime > 0 && x.a.totalElevationGain > 0
      ? (x.a.totalElevationGain * 3600) / x.a.movingTime
      : null
  const cadValOf = (sport: Sport, a: RawStravaActivity): number | null =>
    a.averageCadence == null ? null : sport === 'run' ? a.averageCadence * 2 : a.averageCadence
  const cadScoreOf = (sport: Sport, v: number): number =>
    Math.round(clamp(100 - (Math.abs(v - CAD_TARGET[sport]) / CAD_TARGET[sport]) * 200, 0, 100))
  const sportCtlOf = (d: DailyPoint, sport: Sport): number =>
    sport === 'swim' ? d.swimCtl : sport === 'bike' ? d.bikeCtl : d.runCtl

  const buildSportAbilities = (sport: Sport): SportAbilities => {
    const mine = acts.filter(x => x.sport === sport)
    const ctlHi = round(CTL_ANCHOR[1] * LOAD_SHARE_TARGET[sport], 0)
    const ctlSport = daily.length ? sportCtlOf(daily[daily.length - 1], sport) : 0
    const th = thresholds.get(sport)
    const vThrSport = th && th.conf !== 'prior' ? th.vThr : null

    let sprint: RadarAxis
    let threshold: RadarAxis
    if (sport === 'bike') {
      sprint = {
        key: 'sprint',
        label: 'sprint',
        proj: null,
        score: sprintWkg != null ? Math.round(norm01(sprintWkg, ...COGGAN_SPRINT_WKG) * 100) : null,
        rawValue: sprintWkg != null ? round(sprintWkg, 1) : null,
        rawUnit: 'w/kg',
        lo: COGGAN_SPRINT_WKG[0],
        hi: COGGAN_SPRINT_WKG[1],
      }
      threshold = {
        key: 'threshold',
        label: 'threshold',
        proj: null,
        score: ftpWkg != null ? Math.round(norm01(ftpWkg, ...COGGAN_FTP_WKG) * 100) : null,
        rawValue: ftpWkg != null ? round(ftpWkg, 2) : null,
        rawUnit: 'w/kg',
        lo: COGGAN_FTP_WKG[0],
        hi: COGGAN_FTP_WKG[1],
      }
    } else {
      let vs: number | null = null
      if (sport === 'run')
        for (const x of mine) {
          const v = peakRunSpeedOf(x)
          if (v != null && (vs == null || v > vs)) vs = v
        }
      if (vs == null)
        for (const x of mine) {
          const v = plausibleVgap(sport, x)
          if (v != null && (vs == null || v > vs)) vs = v
        }
      const sAnchor = SPEED_SPRINT_MS[sport]
      sprint = {
        key: 'sprint',
        label: 'sprint',
        proj: null,
        score: vs != null ? Math.round(norm01(vs, ...sAnchor) * 100) : null,
        rawValue: vs != null ? round(vs, 2) : null,
        rawUnit: 'm/s',
        lo: sAnchor[0],
        hi: sAnchor[1],
      }
      const tAnchor = SPEED_THR_MS[sport]
      threshold = {
        key: 'threshold',
        label: 'threshold',
        proj: null,
        score: vThrSport != null ? Math.round(norm01(vThrSport, ...tAnchor) * 100) : null,
        rawValue: vThrSport != null ? round(vThrSport, 2) : null,
        rawUnit: 'm/s',
        lo: tAnchor[0],
        hi: tAnchor[1],
      }
    }

    let climb: RadarAxis
    if (sport === 'swim') {
      climb = {
        key: 'climb',
        label: 'climb',
        proj: null,
        score: null,
        rawValue: null,
        rawUnit: 'm/h',
        lo: 0,
        hi: 0,
      }
    } else {
      let vam: number | null = null
      for (const x of mine) {
        const v = vamOf(x)
        if (v != null && (vam == null || v > vam)) vam = v
      }
      const anchor = VAM_ANCHOR_OF[sport]
      climb = {
        key: 'climb',
        label: 'climb',
        proj: null,
        score: vam != null ? Math.round(norm01(vam, ...anchor) * 100) : null,
        rawValue: vam != null ? round(vam, 0) : null,
        rawUnit: 'm/h',
        lo: anchor[0],
        hi: anchor[1],
      }
    }

    const cads = mine.map(x => cadValOf(sport, x.a)).filter((v): v is number => v != null)
    const axes: RadarAxis[] = [
      sprint,
      threshold,
      {
        key: 'endurance',
        label: 'endurance',
        proj: null,
        score: Math.round(norm01(ctlSport, CTL_ANCHOR[0], ctlHi) * 100),
        rawValue: round(ctlSport, 0),
        rawUnit: 'ctl',
        lo: CTL_ANCHOR[0],
        hi: ctlHi,
      },
      climb,
      {
        key: 'cadence',
        label: 'cadence',
        proj: null,
        score: cads.length ? cadScoreOf(sport, mean(cads)) : null,
        rawValue: cads.length ? round(mean(cads), 0) : null,
        rawUnit: CAD_UNIT[sport],
        lo: 0,
        hi: 100,
      },
      {
        key: 'recovery',
        label: 'recovery',
        proj: null,
        score: recScore,
        rawValue: rdy14.length
          ? round(mean(rdy14), 0)
          : hrv14.length
            ? round(mean(hrv14), 0)
            : null,
        rawUnit: rdy14.length ? 'readiness' : 'ms',
        lo: 0,
        hi: 100,
      },
    ]
    const scored = axes.filter(a => a.score != null).map(a => a.score as number)
    const area = scored.length ? Math.round(mean(scored)) : null

    const stats = mine
      .map(x => ({
        day: x.day,
        sprint:
          sport === 'bike'
            ? (peakMean(wattsOf(x.a.id), BIKE_SPRINT_WIN_S) ?? x.a.maxWatts ?? null)
            : sport === 'run'
              ? (peakRunSpeedOf(x) ?? plausibleVgap(sport, x))
              : plausibleVgap(sport, x),
        vam: sport === 'swim' ? null : vamOf(x),
        cad: cadValOf(sport, x.a),
      }))
      .sort((p, q) => p.day.localeCompare(q.day))
    const sprintScoreOf = (v: number): number | null =>
      sport === 'bike'
        ? kgNow
          ? Math.round(norm01(v / kgNow, ...COGGAN_SPRINT_WKG) * 100)
          : null
        : Math.round(norm01(v, ...SPEED_SPRINT_MS[sport]) * 100)
    const rdyArr = daily.map(d => d.readiness)
    const history: AbilityTrendPoint[] = []
    let si = 0
    let cumSprint: number | null = null
    let cumVam: number | null = null
    const cadAcc: number[] = []
    for (let i = 0; i < daily.length; i++) {
      const cutoff = daily[i].date
      while (si < stats.length && stats[si].day <= cutoff) {
        const st = stats[si]
        if (st.sprint != null && (cumSprint == null || st.sprint > cumSprint)) cumSprint = st.sprint
        if (st.vam != null && (cumVam == null || st.vam > cumVam)) cumVam = st.vam
        if (st.cad != null) cadAcc.push(st.cad)
        si++
      }
      const rdyH = winValues(rdyArr, i, 14)
      history.push({
        date: cutoff,
        sprint: cumSprint != null ? sprintScoreOf(cumSprint) : null,
        threshold: threshold.score,
        endurance: Math.round(norm01(sportCtlOf(daily[i], sport), CTL_ANCHOR[0], ctlHi) * 100),
        climb:
          cumVam != null && sport !== 'swim'
            ? Math.round(norm01(cumVam, ...VAM_ANCHOR_OF[sport]) * 100)
            : null,
        cadence: cadAcc.length ? cadScoreOf(sport, mean(cadAcc)) : null,
        recovery: rdyH.length ? Math.round(mean(rdyH)) : null,
      })
    }
    const recent = history.slice(-PROJ_WINDOW_D)
    for (const a of axes) {
      if (a.score == null) continue
      if (a.key === 'sprint' || a.key === 'climb') {
        a.proj = a.score
        continue
      }
      const xs: number[] = []
      const ys: number[] = []
      recent.forEach((h, i) => {
        const v = h[a.key]
        if (v != null) {
          xs.push(i)
          ys.push(v)
        }
      })
      if (ys.length < PROJ_MIN_POINTS) {
        a.proj = a.score
        continue
      }
      const slope = olsSlope(xs, ys) ?? 0
      a.proj = Math.round(clamp(a.score + slope * PROJ_HORIZON_D, 0, 100))
    }
    return { sport, axes, area, history }
  }

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
    abilities: { sports: SPORT_ORDER.map(buildSportAbilities) },
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
    ftpHypothesis,
  }
}

function emptyAnalytics(athleteId: number, today: string): Analytics {
  return {
    meta: emptyMeta(athleteId, today),
    calibration: emptyCalibration(today),
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
    tests: { dexa: [], vo2max: [] },
  }
}

export function buildAnalytics(
  cache: StravaRawCache | null,
  inputs: AnalyticsInputs = {},
): Analytics {
  const todayFromSync = cache?.lastSync ? localIsoDay(cache.lastSync, inputs.timeZone) : null
  const dexaTests = parseDexa(inputs.dexa)
  const vo2Tests = parseVo2Lab(inputs.vo2labs)
  const latestDexa = dexaTests.length ? dexaTests[dexaTests.length - 1] : null
  const latestVo2Lab = vo2Tests.length ? vo2Tests[vo2Tests.length - 1] : null

  if (!cache) return emptyAnalytics(0, todayFromSync ?? '1970-01-01')

  const sinceDay = inputs.since && /^\d{4}-\d{2}-\d{2}$/.test(inputs.since) ? inputs.since : null
  const raw = Object.values(cache.activities)
    .map(a => ({ a, sport: normalizeSport(a.sportType) }))
    .filter(
      (x): x is { a: RawStravaActivity; sport: Sport } =>
        x.sport !== null && !isTreatment(x.a.sportType, x.a.name),
    )
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
  const latestFfmLbs =
    latestDexa?.ffmLbs ?? (latestDexa ? latestDexa.leanLbs + latestDexa.bmcLbs : null)
  const goalBmr =
    goalKg != null ? mifflinStJeorBmr(goalKg, ATHLETE.heightCm, ageOn(today), ATHLETE.sex) : null
  const goalLeanBmr =
    latestFfmLbs != null ? katchMcArdleBmrFromLeanKg(latestFfmLbs * 0.45359237) : null
  const body: BodyBlock = {
    latestKg,
    latestLbs: latestKg != null ? round(latestKg / 0.45359237, 1) : null,
    trendKgPerWeek,
    goalKg,
    goalLbs: goalKg != null ? round(goalKg / 0.45359237, 1) : null,
    goalDeltaKg,
    goalEtaWeeks,
    goalBmr,
    goalLeanBmr,
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
  if (latestDexa) {
    body.bodyFatPct = latestDexa.bodyFat
    body.boneMassKg = round(latestDexa.bmcLbs * 0.45359237, 2)
  }
  const weekly = buildWeekly(acts, loadById)
  const trends = SPORT_ORDER.map(sport => buildTrend(acts, thresholds.get(sport)!, sport, todayMs))
  const trendMap = new Map<Sport, SportTrend>(trends.map(t => [t.sport, t]))
  const calibration = buildCalibration(acts, thresholds, trendMap, loadById, today, todayMs)
  const bestList = SPORT_ORDER.map(sport => buildBest(acts, sport))
  const bests = new Map<Sport, SportBest>(bestList.map(b => [b.sport, b]))
  const risk = buildRisk(daily, weekly)
  const recovery = buildRecovery(daily, risk)

  const ctlNow = daily.length ? daily[daily.length - 1].ctl : 0
  const races = (['sprint', 'olympic', '70.3', 'ironman'] as RaceDistance[]).map(distance =>
    buildReadiness(distance, thresholds, bests, ctlNow, trendMap),
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
    thresholds,
    today,
    garminVo2,
    appleVo2,
    heartRateById,
    latestVo2Lab,
    inputs.ftp ?? null,
  )

  const walkSummaries: ActivitySummary[] = Object.values(cache.activities)
    .filter(a => normalizeKind(a.sportType) === 'walk' && !isTreatment(a.sportType, a.name))
    .filter(a => !sinceDay || a.startDateLocal.slice(0, 10) >= sinceDay)
    .map(a => ({
      id: a.id,
      date: a.startDateLocal.slice(0, 10),
      sport: 'walk' as const,
      name: a.name ?? '',
      distanceKm: round(a.distance / 1000, 1),
      movingTimeS: a.movingTime,
      load: 0,
      effort: a.sufferScore ?? null,
      cadence: a.averageCadence ?? null,
      strokes: null,
      windKph: inputs.weather?.activities[String(a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(a.id)]?.windGustKph ?? null,
    }))
  const strengthSummaries: ActivitySummary[] = Object.values(cache.activities)
    .filter(a => normalizeKind(a.sportType) === 'strength' && !isTreatment(a.sportType, a.name))
    .filter(a => !sinceDay || a.startDateLocal.slice(0, 10) >= sinceDay)
    .map(a => ({
      id: a.id,
      date: a.startDateLocal.slice(0, 10),
      sport: 'strength' as const,
      name: a.name ?? '',
      distanceKm: 0,
      movingTimeS: a.movingTime,
      load: 0,
      effort: a.sufferScore ?? null,
      cadence: null,
      strokes: null,
      windKph: inputs.weather?.activities[String(a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(a.id)]?.windGustKph ?? null,
    }))
  const treatmentSummaries: ActivitySummary[] = Object.values(cache.activities)
    .filter(a => isTreatment(a.sportType, a.name))
    .filter(a => !sinceDay || a.startDateLocal.slice(0, 10) >= sinceDay)
    .map(a => ({
      id: a.id,
      date: a.startDateLocal.slice(0, 10),
      sport: 'treatment' as const,
      name: a.name ?? '',
      distanceKm: 0,
      movingTimeS: a.movingTime,
      load: 0,
      effort: a.sufferScore ?? null,
      cadence: null,
      strokes: null,
      windKph: inputs.weather?.activities[String(a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(a.id)]?.windGustKph ?? null,
    }))
  const yogaSummaries: ActivitySummary[] = Object.values(cache.activities)
    .filter(a => normalizeKind(a.sportType) === 'yoga' && !isTreatment(a.sportType, a.name))
    .filter(a => !sinceDay || a.startDateLocal.slice(0, 10) >= sinceDay)
    .map(a => ({
      id: a.id,
      date: a.startDateLocal.slice(0, 10),
      sport: 'yoga' as const,
      name: a.name ?? '',
      distanceKm: 0,
      movingTimeS: a.movingTime,
      load: 0,
      effort: a.sufferScore ?? null,
      cadence: null,
      strokes: null,
      windKph: inputs.weather?.activities[String(a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(a.id)]?.windGustKph ?? null,
    }))
  const swimStrokes = swimStrokesByActivityId(acts, inputs.apple)
  const activities: ActivitySummary[] = acts
    .map(act => ({
      id: act.a.id,
      date: act.day,
      sport: act.sport as ActivityKind,
      name: act.a.name ?? '',
      distanceKm: act.distanceKm,
      movingTimeS: act.a.movingTime,
      load: round(loadById.get(act.a.id) ?? 0, 1),
      effort: act.a.sufferScore ?? null,
      cadence: act.a.averageCadence ?? null,
      strokes: swimStrokes.get(act.a.id) ?? null,
      windKph: inputs.weather?.activities[String(act.a.id)]?.windKph ?? null,
      windDir: inputs.weather?.activities[String(act.a.id)]?.windDir ?? null,
      windGustKph: inputs.weather?.activities[String(act.a.id)]?.windGustKph ?? null,
    }))
    .concat(walkSummaries, strengthSummaries, treatmentSummaries, yogaSummaries)
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
    calibration,
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
    tests: { dexa: dexaTests, vo2max: vo2Tests },
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
  'strokes',
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
  'sessions',
  'load',
  'km',
  'hours',
  'swimKm',
  'bikeKm',
  'runKm',
  'swimHours',
  'bikeHours',
  'runHours',
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
  strokes: Record<string, number> | null
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
          strokes: s.strokes,
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
    v: 2,
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
