import type { GarminCache } from '../garmin'
import type { Analytics, EngineBlock } from './schema'
import { emptyBestEfforts } from '../../../util/best-efforts'
import { FTP_FROM_P20 } from '../../../util/cycling-ftp'
import {
  calculateFtpHypothesis,
  FTP_HYPOTHESIS_DEFAULTS,
  type FtpHypothesisParams,
} from '../../../util/ftp-hypothesis'
import { buildSwimPowerCurveBlock } from '../../../util/swim-power'
import { round, type ActivityHeartRate, type Sport, type StravaRawCache } from '../strava'
import { buildAbilities } from './abilities'
import { emptyBody, type BodyBlock, type Vo2LabRecord } from './body-and-labs'
import {
  buildLactateThreshold,
  declaredLactateThresholdHeartRate,
  emptyCalibration,
} from './calibration-and-lactate'
import { buildCardio } from './cardio'
import { emptyDistributions } from './distributions'
import { buildVo2TrendSummary, hrMaxOf, preferObservedVo2, runVo2Of } from './engine-vo2'
import { emptyHeat } from './heat'
import { emptyPowerCurve, type PowerCurveBlock } from './power-to-weight'
import {
  ageOn,
  ATHLETE,
  clamp,
  DAY_MS,
  dayMs,
  HRV_BASE_SPAN,
  peakMean,
  winMedian,
  type Act,
  type DailyPoint,
  type RunningDynamicsMetrics,
  type SwimActivityMetrics,
  type ThresholdEstimate,
} from './primitives'
import { emptyMeta, neutralRisk } from './readiness'
import { emptyRecovery } from './recovery'
import {
  ACSM_BASE,
  ACSM_WATT_K,
  buildFtpPedalingEvidence,
  computeFtpHypothesisFromVo2,
  FRIEND_MED_M,
  ftpPowerEvidence,
  invLerp,
  MAP_FTP_RATIO,
  pctForAge,
  UTH_K,
  VO2_CEIL,
  VO2_FLOOR,
  type FtpHypothesisMassSource,
  type FtpHypothesisVo2maxSource,
  type FtpHypothesisVo2maxSport,
  type GarminVo2Point,
  type ObservedVo2Point,
  type Vo2BikeSource,
  type Vo2Estimate,
  type Vo2Point,
} from './vo2-and-ftp'

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
      trendSummary: {
        direction: null,
        slopePerWeek: null,
        change28d: null,
        sampleSize: 0,
        spanDays: 0,
        method: 'none',
      },
      bikeSource: null,
    },
    lactateThreshold: {
      heartRate: declaredLactateThresholdHeartRate(),
      heartRateSeries: [],
      sports: [],
    },
    abilities: { sports: [] },
    cardio: { metrics: [], rhrSeries: [], hrvSeries: [], efSeries: [], decouplingSeries: [] },
    ftpHypothesis: null,
  }
}

export function buildEngine(
  cache: StravaRawCache,
  acts: Act[],
  daily: DailyPoint[],
  body: BodyBlock,
  thresholds: Map<Sport, ThresholdEstimate>,
  today: string,
  garminVo2: GarminVo2Point[],
  appleVo2: { date: string; v: number }[],
  heartRateById: Map<number, ActivityHeartRate>,
  swimMetrics: ReadonlyMap<number, SwimActivityMetrics>,
  runningDynamics: ReadonlyMap<number, RunningDynamicsMetrics>,
  vo2Lab: Vo2LabRecord | null,
  ftpOverride: number | null,
  garmin: GarminCache | null | undefined,
  powerCurve: PowerCurveBlock,
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

  const weekOf = (iso: string): string => {
    const ms = dayMs(iso)
    const dow = new Date(ms).getUTCDay()
    return new Date(ms - ((dow + 6) % 7) * DAY_MS).toISOString().slice(0, 10)
  }
  const rawBikeVo2 = (ftpW: number, weightKg: number): number =>
    (ACSM_WATT_K * (ftpW / MAP_FTP_RATIO)) / weightKg + ACSM_BASE
  const labFtp =
    vo2Lab?.massKg != null
      ? (calculateFtpHypothesis(vo2Lab.value, vo2Lab.massKg)?.ftp ?? null)
      : null
  const labBaseline =
    vo2Lab != null && vo2Lab.massKg != null && labFtp != null
      ? {
          date: vo2Lab.date,
          vo2max: round(vo2Lab.value, 1),
          ftpW: labFtp,
          weightKg: round(vo2Lab.massKg, 1),
        }
      : null
  const bikeVo2Offset =
    labBaseline == null
      ? 0
      : labBaseline.vo2max - rawBikeVo2(labBaseline.ftpW, labBaseline.weightKg)
  const bikeVo2 = (ftpW: number, weightKg: number): number =>
    round(clamp(rawBikeVo2(ftpW, weightKg) + bikeVo2Offset, VO2_FLOOR, VO2_CEIL), 1)

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
      estimates.push({
        method: 'bike',
        vo2max: bikeVo2(ftp, bikeKg),
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

  type Vo2Week = { p20: number | null; kg: number | null; observed: ObservedVo2Point | null }
  const weeks = new Map<string, Vo2Week>()
  const weekAt = (day: string): Vo2Week => {
    const ws = weekOf(day)
    let w = weeks.get(ws)
    if (!w) weeks.set(ws, (w = { p20: null, kg: null, observed: null }))
    return w
  }
  const observedVo2: ObservedVo2Point[] = garminVo2.map(point => ({
    date: point.date,
    v: point.v,
    method: 'garmin',
    sport: point.generic == null && point.cycling != null ? 'cycling' : 'unknown',
  }))
  for (const point of appleVo2)
    observedVo2.push({ date: point.date, v: point.v, method: 'apple', sport: 'unknown' })
  if (vo2Lab)
    observedVo2.push({ date: vo2Lab.date, v: vo2Lab.value, method: 'lab', sport: 'running' })
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
  for (const point of observedVo2) {
    const week = weekAt(point.date)
    week.observed = preferObservedVo2(week.observed, point)
  }
  weekAt(today)
  const currentWeekStart = weekOf(today)
  const trend: Vo2Point[] = []
  let kgCarry: number | null = null
  for (const ws of [...weeks.keys()].sort()) {
    const w = weeks.get(ws)!
    if (w.kg != null) kgCarry = w.kg
    let point: Vo2Point | null = null
    if (w.observed)
      point = {
        weekStart: ws,
        vo2max: round(clamp(w.observed.v, VO2_FLOOR, VO2_CEIL), 1),
        method: w.observed.method,
      }
    else if (ws === currentWeekStart && ftp != null && bikeKg != null && bikes.length > 0)
      point = { weekStart: ws, vo2max: bikeVo2(ftp, bikeKg), method: 'bike' }
    else if (w.p20 != null && kgCarry != null && kgCarry > 0)
      point = { weekStart: ws, vo2max: bikeVo2(w.p20 * FTP_FROM_P20, kgCarry), method: 'bike' }
    if (point) trend.push(point)
  }

  const latest = trend[trend.length - 1]
  const current = latest
    ? {
        method: latest.method,
        vo2max: latest.vo2max,
        conf: estimates.find(estimate => estimate.method === latest.method)?.conf ?? 'prior',
      }
    : (estimates[0] ?? null)
  const vo2 = current?.vo2max ?? null
  const fitnessAge = vo2 != null ? Math.round(clamp(invLerp(FRIEND_MED_M, vo2), 20, 80)) : null
  const ageDeltaYears = fitnessAge != null ? fitnessAge - age : null
  const percentileForAge = vo2 != null ? pctForAge(vo2, age) : null
  let latestObservedVo2: ObservedVo2Point | null = null
  for (const point of observedVo2) latestObservedVo2 = preferObservedVo2(latestObservedVo2, point)
  const vo2max = latestObservedVo2?.v ?? ATHLETE.vo2max
  const vo2maxDate = latestObservedVo2?.date ?? today
  const vo2maxSource: FtpHypothesisVo2maxSource = latestObservedVo2?.method ?? 'default'
  const vo2maxSport: FtpHypothesisVo2maxSport = latestObservedVo2?.sport ?? 'running'
  const massKg = body.latestKg ?? vo2Lab?.massKg ?? null
  const massDate =
    body.latestKg != null ? (body.series.at(-1)?.date ?? today) : (vo2Lab?.date ?? today)
  const massSource: FtpHypothesisMassSource = body.latestKg != null ? 'daily' : 'lab'
  const ftpParams: FtpHypothesisParams = {
    ...FTP_HYPOTHESIS_DEFAULTS,
    crossModalDiscountPct:
      vo2maxSport === 'cycling' ? 0 : FTP_HYPOTHESIS_DEFAULTS.crossModalDiscountPct,
  }
  const ftpHypothesis =
    vo2max != null && massKg != null
      ? computeFtpHypothesisFromVo2(
          massDate > vo2maxDate ? massDate : vo2maxDate,
          vo2max,
          massKg,
          ftpParams,
          {
            massDate,
            massSource,
            vo2maxDate,
            vo2maxSource,
            vo2maxSport,
            defaultVo2max: ATHLETE.vo2max,
            efficiency: {
              valuePct: ftpParams.grossEfficiencyPct,
              source: 'literature-prior',
              conf: 'prior',
            },
            pedaling: buildFtpPedalingEvidence(garmin, today),
            power: ftpPowerEvidence(powerCurve),
          },
        )
      : null
  const bikeSource: Vo2BikeSource | null =
    current?.method === 'bike' && ftp != null && ftpSrc != null && bikeKg != null
      ? {
          ftpW: ftp,
          ftpSource: ftpSrc,
          mapW: Math.round(ftp / MAP_FTP_RATIO),
          weightKg: round(bikeKg, 1),
          labBaseline,
        }
      : null

  return {
    vo2max: {
      value: vo2,
      method: current?.method ?? 'none',
      conf: current?.conf ?? 'prior',
      hrMax,
      hrMaxSource,
      hrRest,
      chronoAge: age,
      fitnessAge,
      ageDeltaYears,
      percentileForAge,
      estimates,
      trend,
      trendSummary: buildVo2TrendSummary(trend),
      bikeSource,
    },
    lactateThreshold: buildLactateThreshold(thresholds, today, garmin, powerCurve),
    abilities: buildAbilities({
      cache,
      acts,
      daily,
      body,
      thresholds,
      today,
      swimMetrics,
      runningDynamics,
      powerCurve,
      ftp,
    }),
    cardio: buildCardio({ cache, acts, daily, heartRateById, rhrArr }),
    ftpHypothesis,
  }
}

export function emptyAnalytics(
  athleteId: number,
  today: string,
  garmin?: GarminCache | null,
): Analytics {
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
    powerCurve: emptyPowerCurve(today),
    swimPowerCurve: buildSwimPowerCurveBlock([], today),
    bestEfforts: emptyBestEfforts(),
    heat: emptyHeat(),
    distributions: emptyDistributions(),
    engine: {
      ...emptyEngine(),
      lactateThreshold: buildLactateThreshold(new Map(), today, garmin, null),
    },
    events: [],
    activities: [],
    weakestSport: 'run',
    actions: [],
    tests: { dexa: [], vo2max: [] },
    recent: { from: today, trends: [], races: [], projections: [] },
  }
}
