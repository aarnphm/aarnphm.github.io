import { swimPaceSeconds } from '../../../util/swim-metrics'
import { round, SPORT_ORDER, type Sport } from '../strava'
import {
  ANALYTICS_LOAD_NOTE,
  clamp,
  IF_CAP,
  K42,
  K7,
  LOAD_SHARE_TARGET,
  mean,
  sd,
  type Act,
  type AcwrState,
  type AnalyticsMeta,
  type Conf,
  type DailyPoint,
  type RaceDistance,
  type RiskBlock,
  type SportBest,
  type ThresholdEstimate,
  type TrainingAction,
  type TsbZone,
  type WeeklyPoint,
} from './primitives'
import { trendSpeedRatio, type SportTrend } from './trends'

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

export function buildBest(acts: Act[], sport: Sport): SportBest {
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
  const qualifyingRate = (act: Act): number | null => {
    if (act.a.distance < minDistM || act.a.movingTime < minTimeS) return null
    if (sport === 'swim') return swimPaceSeconds(act.a.distance, act.a.movingTime)
    return toRate(act.vGap)
  }

  let fastest: number | null = null
  for (const x of mine) {
    const rate = qualifyingRate(x)
    if (rate == null) continue
    if (fastest === null || better(rate, fastest)) fastest = rate
  }

  const longestKm = mine.reduce((m, x) => Math.max(m, x.distanceKm), 0)
  const biggestClimbM = mine.reduce((m, x) => Math.max(m, x.a.totalElevationGain), 0)
  const totalKm = mine.reduce((s, x) => s + x.distanceKm, 0)
  const totalTimeS = mine.reduce((s, x) => s + x.a.movingTime, 0)

  const bestToDate: { date: string; rate: number }[] = []
  let running: number | null = null
  for (const x of mine) {
    const rate = qualifyingRate(x)
    if (rate == null) continue
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

export function buildRisk(daily: DailyPoint[], weekly: WeeklyPoint[]): RiskBlock {
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
export const RACE_DISTANCES = Object.keys(RACE_LEGS) as RaceDistance[]
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
    trendSpeedRatio(tr, tr.level, last.value),
    1 - TREND_PROJ_CLAMP,
    1 + TREND_PROJ_CLAMP,
  )
  const halfFrac = clamp(
    Math.abs(trendSpeedRatio(tr, last.lo, last.hi) - 1) / 2,
    0,
    TREND_BAND_CLAMP,
  )
  return { mid, fast: mid * (1 + halfFrac), slow: mid * (1 - halfFrac), usable: true }
}

export function buildReadiness(
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

export function buildActions(
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
        value: `${round(ctlBySport[item.sport], 1)} ctl`,
      })
    }
  }

  return { weakest, actions }
}

export function emptyMeta(athleteId: number, today: string): AnalyticsMeta {
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
      note: ANALYTICS_LOAD_NOTE,
    },
  }
}

export function neutralRisk(): RiskBlock {
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
