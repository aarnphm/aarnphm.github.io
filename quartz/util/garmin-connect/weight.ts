import type { GarminWeightSample } from '../../plugins/stores/garmin'
import { isRecord, type UnknownRecord } from '../type-guards'
import { finite, firstString, normalizeDate } from './scalars'

const WEIGHT_KEYS = [
  'weight',
  'weightKg',
  'weightInKg',
  'weightInKilograms',
  'weightValue',
  'weightValueInKg',
]
const WEIGHT_GRAM_KEYS = ['weightGram', 'weightGrams', 'weightInGrams', 'weightValueInGrams']
const WEIGHT_LB_KEYS = [
  'weightLb',
  'weightLbs',
  'weightPounds',
  'weightInPounds',
  'weightValueInPounds',
]
const WEIGHT_UNIT_KEYS = ['unit', 'unitKey', 'weightUnit']
const BMI_KEYS = ['bmi', 'bodyMassIndex']
const BODY_FAT_KEYS = ['bodyFat', 'bodyFatPct', 'bodyFatPercent', 'bodyFatPercentage']
const BODY_WATER_KEYS = ['bodyWater', 'bodyWaterPct', 'bodyWaterPercent', 'bodyWaterPercentage']
const MUSCLE_MASS_KEYS = [
  'muscleMass',
  'muscleMassKg',
  'muscleMassInKg',
  'muscleMassInKilograms',
  'skeletalMuscleMass',
  'skeletalMuscleMassKg',
]
const MUSCLE_MASS_GRAM_KEYS = [
  'muscleMassGram',
  'muscleMassGrams',
  'muscleMassInGrams',
  'skeletalMuscleMassInGrams',
]
const BONE_MASS_KEYS = ['boneMass', 'boneMassKg', 'boneMassInKg', 'boneMassInKilograms']
const BONE_MASS_GRAM_KEYS = ['boneMassGram', 'boneMassGrams', 'boneMassInGrams']
const WEIGHT_RECORD_KEYS = ['weight', 'latestWeight', 'latestWeightMetric', 'measurement']
const WEIGHT_LIST_KEYS = [
  'allWeightMetrics',
  'dateWeightList',
  'measurements',
  'weightList',
  'weightMetrics',
  'weights',
]
const WEIGHT_SUMMARY_RECORD_KEYS = ['latestWeight', 'latestWeightMetric']
const WEIGHT_TIMESTAMP_KEYS = [
  'date',
  'measurementTimestampGMT',
  'measurementTimestampLocal',
  'samplePk',
  'timestampGMT',
  'timestampLocal',
  'weighInTimestampGMT',
  'weighInTimestampLocal',
]
const LB_PER_KG = 2.2046226218

const kgOf = (value: unknown): number | null => {
  const n = finite(value)
  if (n == null || n <= 0) return null
  return Math.round((n > 400 ? n / 1000 : n) * 100) / 100
}

const kgOfPounds = (value: unknown): number | null => {
  const n = finite(value)
  return n != null && n > 0 ? Math.round((n / LB_PER_KG) * 100) / 100 : null
}

const pctOf = (value: unknown): number | null => {
  const n = finite(value)
  return n != null && n > 0 && n <= 100 ? Math.round(n * 10) / 10 : null
}

function firstFiniteRecord(record: UnknownRecord, keys: readonly string[]): number | null {
  for (const key of keys) {
    const value = finite(record[key])
    if (value != null) return value
  }
  return null
}

function firstRecordArray(record: UnknownRecord, keys: readonly string[]): UnknownRecord[] | null {
  for (const key of keys) {
    const value = record[key]
    if (Array.isArray(value)) return value.filter(isRecord)
  }
  return null
}

function msOf(value: unknown): number | null {
  const n = finite(value)
  if (n != null) {
    if (n > 1_000_000_000_000) return n
    if (n > 1_000_000_000) return n * 1000
    return null
  }
  if (typeof value !== 'string') return null
  const normalized = normalizeDate(value)
  if (!normalized) return null
  const ms = Date.parse(normalized)
  return Number.isFinite(ms) ? ms : null
}

const isoDayOf = (record: UnknownRecord, keys: string[]): string | null => {
  for (const key of keys) {
    const v = record[key]
    if (typeof v === 'string' && /^\d{4}-\d{2}-\d{2}/.test(v)) return v.slice(0, 10)
    const ms = msOf(v)
    if (ms != null) return new Date(ms).toISOString().slice(0, 10)
  }
  return null
}

const tsOf = (record: UnknownRecord): number | null => {
  for (const key of WEIGHT_TIMESTAMP_KEYS) {
    const ms = msOf(record[key])
    if (ms != null) return ms
  }
  return null
}

function kgFromRecord(
  record: UnknownRecord,
  kgKeys: readonly string[],
  gramKeys: readonly string[],
  poundKeys: readonly string[],
  unitAware = false,
): number | null {
  const kg = firstFiniteRecord(record, kgKeys)
  if (kg != null) {
    const unit = unitAware ? firstString([record], WEIGHT_UNIT_KEYS)?.toLowerCase() : null
    if (unit?.includes('lb') || unit?.includes('pound')) return kgOfPounds(kg)
    if (unit && (unit.includes('gram') || unit === 'g') && !unit.includes('kg')) return kgOf(kg)
    return kgOf(kg)
  }
  const grams = firstFiniteRecord(record, gramKeys)
  if (grams != null) return kgOf(grams)
  const pounds = firstFiniteRecord(record, poundKeys)
  if (pounds != null) return kgOfPounds(pounds)
  return null
}

function nestedKgFromRecord(record: UnknownRecord): number | null {
  for (const key of WEIGHT_RECORD_KEYS) {
    const value = record[key]
    if (!isRecord(value)) continue
    const kg = kgFromRecord(value, WEIGHT_KEYS, WEIGHT_GRAM_KEYS, WEIGHT_LB_KEYS, true)
    if (kg != null) return kg
  }
  return null
}

function pctFromRecord(record: UnknownRecord, keys: readonly string[]): number | null {
  return pctOf(firstFiniteRecord(record, keys))
}

export function garminConnectWeightSamples(raw: unknown): GarminWeightSample[] {
  const out: GarminWeightSample[] = []
  const push = (m: UnknownRecord, dayHint: string | null): void => {
    const ts = tsOf(m)
    const date =
      dayHint ??
      isoDayOf(m, ['calendarDate', 'summaryDate', 'date', 'weightDate']) ??
      (ts != null ? new Date(ts).toISOString().slice(0, 10) : null)
    if (!date) return
    const sample: GarminWeightSample = {
      ts: ts ?? Date.parse(`${date}T12:00:00.000Z`),
      date,
      weightKg:
        kgFromRecord(m, WEIGHT_KEYS, WEIGHT_GRAM_KEYS, WEIGHT_LB_KEYS, true) ??
        nestedKgFromRecord(m),
      bmi: pctFromRecord(m, BMI_KEYS),
      bodyFatPct: pctFromRecord(m, BODY_FAT_KEYS),
      bodyWaterPct: pctFromRecord(m, BODY_WATER_KEYS),
      muscleMassKg: kgFromRecord(m, MUSCLE_MASS_KEYS, MUSCLE_MASS_GRAM_KEYS, []),
      boneMassKg: kgFromRecord(m, BONE_MASS_KEYS, BONE_MASS_GRAM_KEYS, []),
    }
    if (
      sample.weightKg == null &&
      sample.bmi == null &&
      sample.bodyFatPct == null &&
      sample.bodyWaterPct == null &&
      sample.muscleMassKg == null &&
      sample.boneMassKg == null
    )
      return
    out.push(sample)
  }
  const summaries =
    isRecord(raw) && Array.isArray(raw.dailyWeightSummaries) ? raw.dailyWeightSummaries : []
  for (const sum of summaries) {
    if (!isRecord(sum)) continue
    const date = isoDayOf(sum, ['summaryDate', 'calendarDate'])
    const metrics = firstRecordArray(sum, WEIGHT_LIST_KEYS)
    if (metrics && metrics.length) {
      for (const m of metrics) push(m, date ?? isoDayOf(m, ['calendarDate', 'date']))
    } else {
      for (const key of WEIGHT_SUMMARY_RECORD_KEYS) {
        const value = sum[key]
        if (isRecord(value)) push(value, date)
      }
    }
  }
  if (!out.length) {
    const list = isRecord(raw) ? firstRecordArray(raw, WEIGHT_LIST_KEYS) : null
    const records = list ?? (Array.isArray(raw) ? raw.filter(isRecord) : [])
    for (const m of records) push(m, null)
    if (isRecord(raw)) {
      for (const key of WEIGHT_SUMMARY_RECORD_KEYS) {
        const value = raw[key]
        if (isRecord(value)) push(value, null)
      }
    }
  }
  return out.sort((a, b) => a.ts - b.ts)
}
