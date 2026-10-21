import { isRecord, numberValue } from '../../../util/type-guards'
import { round } from '../strava'
import { ATHLETE, DAY_MS, dayMs, KG_PER_LB, mean } from './primitives'

export interface BodyCompositionDay {
  date: string
  kg: number | null
  bmi: number | null
  ffmi: number | null
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
  ffmi: number
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
  ffmi: number | null
  bodyFatPct: number | null
  bodyWaterPct: number | null
  muscleMassKg: number | null
  boneMassKg: number | null
  series: { date: string; ts: number; kg: number }[]
  bmrSeries: { date: string; ts: number; bmr: number }[]
  ffmiSeries: { date: string; ts: number; ffmi: number }[]
  latestBmr: number | null
  composition: BodyCompositionDay[]
}

export const emptyBody = (): BodyBlock => ({
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
  ffmi: null,
  bodyFatPct: null,
  bodyWaterPct: null,
  muscleMassKg: null,
  boneMassKg: null,
  series: [],
  bmrSeries: [],
  ffmiSeries: [],
  latestBmr: null,
  composition: [],
})

export const fatFreeMassKg = (weightKg: number, bodyFatPct: number): number =>
  weightKg * (1 - bodyFatPct / 100)

export const fatFreeMassIndex = (fatFreeMassKgValue: number, heightCm: number): number =>
  round(fatFreeMassKgValue / (heightCm / 100) ** 2, 2)

export const katchMcArdleBmrFromLeanKg = (leanKg: number): number => Math.round(370 + 21.6 * leanKg)

export const katchMcArdleBmr = (weightKg: number, bodyFatPct: number): number =>
  katchMcArdleBmrFromLeanKg(fatFreeMassKg(weightKg, bodyFatPct))

export const mifflinStJeorBmr = (
  weightKg: number,
  heightCm: number,
  ageYears: number,
  sex: typeof ATHLETE.sex,
): number => Math.round(10 * weightKg + 6.25 * heightCm - 5 * ageYears + (sex === 'M' ? 5 : -161))

export function weightTrendPerWeek(series: { date: string; kg: number }[]): number | null {
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

export const parseDexa = (raw: unknown): DexaRecord[] => {
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
    const ffmLbs = numAt(item, 'ffmLbs')
    out.push({
      date,
      totalLbs,
      fatLbs,
      leanLbs,
      bmcLbs,
      ffmLbs,
      ffmi: fatFreeMassIndex((ffmLbs ?? leanLbs + bmcLbs) * KG_PER_LB, ATHLETE.heightCm),
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
