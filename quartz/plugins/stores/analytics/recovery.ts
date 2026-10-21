import { round } from '../strava'
import {
  ACUTE_MIN,
  ACUTE_SPAN,
  clamp,
  HRV_BASE_MIN,
  HRV_BASE_SPAN,
  RAMP_HOT,
  SLEEP_DEBT_SPAN,
  winMedian,
  winStats,
  winValues,
  type DailyPoint,
  type FlagSeverity,
  type RecoveryFlag,
  type RecoveryStatus,
  type RecoveryThresholds,
  type RiskBlock,
} from './primitives'

export interface RecoveryDay {
  date: string
  status: RecoveryStatus
  baselineDays: number
  hrv: number | null
  hrvBaseline: number | null
  hrvZ: number | null
  rhr: number | null
  rhrBaseline: number | null
  rhrZ: number | null
  sleepS: number | null
  sleepBaselineS: number | null
  sleepTargetS: number
  sleepScore: number | null
  sleepDebtS: number | null
  readiness: number | null
  readinessBaseline: number | null
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

export function emptyRecovery(): RecoveryBlock {
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

export function buildRecovery(daily: DailyPoint[], risk: RiskBlock): RecoveryBlock {
  const firstSignal = daily.findIndex(
    d =>
      d.hrv != null ||
      d.rhr != null ||
      d.sleepDurationS != null ||
      d.sleepMetrics != null ||
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
  const rhrBaseArr: (number | null)[] = []
  const readinessBaseArr: (number | null)[] = []
  const sleepBaseArr: (number | null)[] = []
  const baselineDaysArr: number[] = []
  const debtArr: (number | null)[] = []
  for (let i = 0; i < daily.length; i++) {
    const hb = winStats(lnHrv, i, HRV_BASE_SPAN, HRV_BASE_MIN)
    const ha = winStats(lnHrv, i, ACUTE_SPAN, ACUTE_MIN)
    hrvZArr.push(hb && ha && hb.sd > 0 ? round((ha.mean - hb.mean) / hb.sd, 2) : null)
    hrvBaseArr.push(hb ? round(Math.exp(hb.mean), 1) : null)
    const rb = winStats(rhrArr, i, HRV_BASE_SPAN, HRV_BASE_MIN)
    const ra = winStats(rhrArr, i, ACUTE_SPAN, ACUTE_MIN)
    rhrZArr.push(rb && ra && rb.sd > 0 ? round((ra.mean - rb.mean) / rb.sd, 2) : null)
    rhrBaseArr.push(rb ? round(rb.mean, 1) : null)
    const readinessBase = winStats(rdyArr, i, HRV_BASE_SPAN, 7)
    readinessBaseArr.push(readinessBase ? round(readinessBase.mean, 0) : null)
    sleepBaseArr.push(winMedian(slpArr, i, HRV_BASE_SPAN, 7))
    baselineDaysArr.push(winStats(lnHrv, i, HRV_BASE_SPAN, 1)?.n ?? 0)
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
  const rhrBaseline = ri >= 0 ? rhrBaseArr[ri] : null
  const readinessLatest = yl >= 0 ? rdyArr[yl] : null
  const readinessBaseline = yl >= 0 ? readinessBaseArr[yl] : null
  const tempDevLatest = tl >= 0 ? tmpArr[tl] : null
  const sleepLatestS = sl >= 0 ? slpArr[sl] : null
  const sleepBaselineS = sl >= 0 ? sleepBaseArr[sl] : null
  const sleepDebtS = dl >= 0 ? (debtArr[dl] ?? 0) : 0
  const shortSleepStreak = sl >= 0 ? streakBack(slpArr, sl, v => v < t.sleepFloorS) : 0
  const lowReadinessStreak = yl >= 0 ? streakBack(rdyArr, yl, v => v < t.readinessFloor) : 0

  const baselineDays = baselineDaysArr[last]
  const status: RecoveryStatus =
    baselineDays >= 14 ? 'firm' : baselineDays >= 7 ? 'low' : 'building'

  const series: RecoveryDay[] = daily.slice(firstSignal).map((d, k) => {
    const i = firstSignal + k
    const dayBaselineDays = baselineDaysArr[i]
    return {
      date: d.date,
      status: dayBaselineDays >= 14 ? 'firm' : dayBaselineDays >= 7 ? 'low' : 'building',
      baselineDays: dayBaselineDays,
      hrv: d.hrv,
      hrvBaseline: hrvBaseArr[i],
      hrvZ: hrvZArr[i],
      rhr: d.rhr,
      rhrBaseline: rhrBaseArr[i],
      rhrZ: rhrZArr[i],
      sleepS: d.sleepDurationS,
      sleepBaselineS: sleepBaseArr[i],
      sleepTargetS: t.sleepTargetS,
      sleepScore: d.sleepScore,
      sleepDebtS: debtArr[i],
      readiness: d.readiness,
      readinessBaseline: readinessBaseArr[i],
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
