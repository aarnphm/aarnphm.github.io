import type { CardioBlock, CardioMetric } from './schema'
import { round, type ActivityHeartRate, type StravaRawCache } from '../strava'
import {
  ACUTE_MIN,
  ACUTE_SPAN,
  DAY_MS,
  dayMs,
  decouplingOf,
  efOf,
  HRV_BASE_SPAN,
  mean,
  olsSlope,
  winMedian,
  winStats,
  type Act,
  type DailyPoint,
} from './primitives'

/** Daily recovery series and activities that the cardio trend metrics read. */
interface CardioScope {
  cache: StravaRawCache
  acts: Act[]
  daily: DailyPoint[]
  heartRateById: Map<number, ActivityHeartRate>
  rhrArr: DailyPoint['rhr'][]
}

export function buildCardio(scope: CardioScope): CardioBlock {
  const { cache, acts, daily, heartRateById, rhrArr } = scope
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
      note: 'A lower resting heart rate is better.',
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
      note: 'The 7 day average is compared with the 28 day baseline.',
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
      note: 'This is pace or power per heartbeat.',
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
          ? 'This needs at least 20 minutes with heart rate and pace or power data.'
          : decLatest < 5
            ? 'Under 5% means steady output.'
            : decLatest <= 10
              ? 'From 5% to 10% means some late fade.'
              : 'Over 10% means high late fade.',
    },
  ]

  return {
    metrics,
    rhrSeries: daily.filter(d => d.rhr != null).map(d => ({ date: d.date, rhr: d.rhr as number })),
    hrvSeries: daily.filter(d => d.hrv != null).map(d => ({ date: d.date, hrv: d.hrv as number })),
    // Watts and m/min per beat do not share a scale, so the series carries the headline's pool.
    efSeries: efPool.map(e => ({ date: e.x.day, ef: e.ef, sport: e.x.sport })),
    decouplingSeries: decActs.map(e => ({ date: e.x.day, pct: e.d })),
  }
}
