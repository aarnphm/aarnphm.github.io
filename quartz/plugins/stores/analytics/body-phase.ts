import type { PowerRankMass } from '../power-rank'
import type { AnalyticsInputs } from './schema'
import { resolveSleepMetrics } from '../../../util/sleep-metrics'
import { AppleCache } from '../apple'
import { round } from '../strava'
import {
  fatFreeMassIndex,
  fatFreeMassKg,
  katchMcArdleBmrFromLeanKg,
  mifflinStJeorBmr,
  weightTrendPerWeek,
  type BodyBlock,
  type DexaRecord,
} from './body-and-labs'
import { historicalPowerToWeightMass } from './power-to-weight'
import { ageOn, ATHLETE, dayMs, goalWeightKg, KG_PER_LB, type DailyPoint } from './primitives'

/** Provider inputs that the body block reads. It also fills each daily point's recovery, sleep and weight fields, which later phases read. */
interface BodyScope {
  inputs: AnalyticsInputs
  daily: DailyPoint[]
  latestDexa: DexaRecord | null
  today: string
}

interface BodyPhase {
  appleDays: AppleCache['days']
  latestWeight: PowerRankMass | null
  powerToWeightMassByDate: Map<string, PowerRankMass>
  body: BodyBlock
}

export function buildBody(scope: BodyScope): BodyPhase {
  const { inputs, daily, latestDexa, today } = scope
  const ouraDays = inputs.oura?.days ?? {}
  const appleDays = inputs.apple?.days ?? {}
  const garminWeightRaw = inputs.garmin?.weight
  const garminSamples = Array.isArray(garminWeightRaw) ? garminWeightRaw : []
  const weightByDate = new Map<string, PowerRankMass>()
  for (const w of inputs.weights ?? [])
    if (w.weightKg != null)
      weightByDate.set(w.date, { kg: w.weightKg, date: w.date, source: 'tracking' })
  for (const s of garminSamples)
    if (s.weightKg != null)
      weightByDate.set(s.date, { kg: s.weightKg, date: s.date, source: 'garmin' })
  let carriedWeight: PowerRankMass | null = null
  let latestWeight: PowerRankMass | null = null
  for (const d of daily) {
    const o = ouraDays[d.date]
    const naps = inputs.oura?.details?.[d.date]?.naps
    d.napDurationS = naps ? naps.reduce((sum, nap) => sum + (nap.totalSleepS ?? 0), 0) : null
    d.napCount = naps?.length ?? null
    const garminHealth = inputs.garmin?.health?.[d.date]
    d.garminHealth = garminHealth?.date === d.date ? garminHealth : null
    d.ouraHealth = inputs.oura?.details?.[d.date]?.health ?? null
    const garminSleep = inputs.garmin?.sleep?.[d.date]
    d.sleepMetrics = resolveSleepMetrics(
      inputs.oura?.details?.[d.date],
      garminSleep?.date === d.date ? garminSleep : null,
    )
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
    if (w != null) carriedWeight = w
    d.weightKg = carriedWeight?.kg ?? null
    const ap = appleDays[d.date]
    if (ap) {
      if (d.totalCalories == null && ap.burnKcal != null) d.totalCalories = ap.burnKcal
      d.intakeKcal = ap.intakeKcal
      if (ap.weightKg != null && d.weightKg == null) {
        d.weightKg = ap.weightKg
        latestWeight = { kg: ap.weightKg, date: d.date, source: 'apple' }
      }
    }
    if (carriedWeight != null) latestWeight = carriedWeight
  }
  const latestKg = latestWeight?.kg ?? null
  const powerToWeightMassByDate = historicalPowerToWeightMass(daily, weightByDate, appleDays)
  const dailySeries = [...weightByDate.entries()]
    .map(([date, measurement]) => ({ date, kg: measurement.kg }))
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
  const leanMassSeries = garminSamples
    .flatMap(s => {
      const { weightKg, bodyFatPct } = s
      if (weightKg == null || bodyFatPct == null) return []
      const leanKg = fatFreeMassKg(weightKg, bodyFatPct)
      return [
        {
          date: s.date,
          ts: s.ts,
          bmr: katchMcArdleBmrFromLeanKg(leanKg),
          ffmi: fatFreeMassIndex(leanKg, ATHLETE.heightCm),
        },
      ]
    })
    .sort((p, q) => p.ts - q.ts)
  const bmrSeries = leanMassSeries.map(({ date, ts, bmr }) => ({ date, ts, bmr }))
  const ffmiSeries = leanMassSeries.map(({ date, ts, ffmi }) => ({ date, ts, ffmi }))
  const latestBmr = bmrSeries.length ? bmrSeries[bmrSeries.length - 1].bmr : null
  const composition = garminSamples.map(s => ({
    date: s.date,
    kg: s.weightKg,
    bmi: s.bmi,
    ffmi:
      s.weightKg != null && s.bodyFatPct != null
        ? fatFreeMassIndex(fatFreeMassKg(s.weightKg, s.bodyFatPct), ATHLETE.heightCm)
        : null,
    bodyFatPct: s.bodyFatPct,
    bodyWaterPct: s.bodyWaterPct,
    muscleMassKg: s.muscleMassKg,
    boneMassKg: s.boneMassKg,
  }))
  const lastComp = (
    key: 'bmi' | 'ffmi' | 'bodyFatPct' | 'bodyWaterPct' | 'muscleMassKg' | 'boneMassKg',
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
    latestFfmLbs != null ? katchMcArdleBmrFromLeanKg(latestFfmLbs * KG_PER_LB) : null
  const body: BodyBlock = {
    latestKg,
    latestLbs: latestKg != null ? round(latestKg / KG_PER_LB, 1) : null,
    trendKgPerWeek,
    goalKg,
    goalLbs: goalKg != null ? round(goalKg / KG_PER_LB, 1) : null,
    goalDeltaKg,
    goalEtaWeeks,
    goalBmr,
    goalLeanBmr,
    bmi: lastComp('bmi'),
    ffmi: lastComp('ffmi'),
    bodyFatPct: lastComp('bodyFatPct'),
    bodyWaterPct: lastComp('bodyWaterPct'),
    muscleMassKg: lastComp('muscleMassKg'),
    boneMassKg: lastComp('boneMassKg'),
    series: weightSeries,
    bmrSeries,
    ffmiSeries,
    latestBmr,
    composition,
  }
  if (latestDexa) {
    body.bodyFatPct = latestDexa.bodyFat
    body.ffmi = latestDexa.ffmi
    body.boneMassKg = round(latestDexa.bmcLbs * KG_PER_LB, 2)
  }
  return { appleDays, latestWeight, powerToWeightMassByDate, body }
}
