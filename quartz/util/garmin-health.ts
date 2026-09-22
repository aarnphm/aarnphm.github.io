import type {
  GarminBodyBattery,
  GarminEnduranceScore,
  GarminHealthDay,
  GarminHealthReading,
  GarminHillScore,
  GarminLoadFocus,
  GarminReadinessFactor,
  GarminTrainingReadiness,
  GarminTrainingStatus,
} from '../plugins/stores/garmin-health'
import { isRecord, readNumber, readString, type UnknownRecord } from './type-guards'

const dayPattern = /^\d{4}-\d{2}-\d{2}$/
const day = (value: unknown): value is string => typeof value === 'string' && dayPattern.test(value)
const nonnegative = (row: UnknownRecord, key: string, max = Infinity): number | null => {
  const value = readNumber(row, key)
  return value != null && Number.isFinite(value) && value >= 0 && value <= max ? value : null
}
const string = (row: UnknownRecord, key: string): string | null => readString(row, key) ?? null
const dateOf = (row: UnknownRecord): string | null =>
  day(row.calendarDate) ? row.calendarDate : null
const rows = (raw: unknown): UnknownRecord[] => {
  if (raw == null) return []
  if (Array.isArray(raw)) {
    if (!raw.every(isRecord)) throw new Error('Garmin health response contains a malformed row')
    return raw
  }
  if (isRecord(raw)) return Object.keys(raw).length ? [raw] : []
  throw new Error('Garmin health response has an invalid shape')
}

const timestamp = (value: unknown): number | null => {
  if (typeof value === 'number') return Number.isFinite(value) && value > 0 ? value : null
  if (typeof value !== 'string') return null
  const result = Date.parse(/[zZ]|[+-]\d{2}:?\d{2}$/.test(value) ? value : `${value}Z`)
  return Number.isFinite(result) ? result : null
}

export function garminBodyBattery(raw: unknown, date: string): GarminBodyBattery | null {
  const row = rows(raw).find(row => row.date === date)
  if (!row) return null
  const descriptors = Array.isArray(row.bodyBatteryValueDescriptorDTOList)
    ? row.bodyBatteryValueDescriptorDTOList.filter(isRecord)
    : []
  const column = (key: string, fallback: number): number | null => {
    if (!descriptors.length) return fallback
    const descriptor = descriptors.find(row => row.bodyBatteryValueDescriptorKey === key)
    const index = descriptor ? nonnegative(descriptor, 'bodyBatteryValueDescriptorIndex') : null
    return index != null && Number.isInteger(index) ? index : null
  }
  const timeIndex = column('timestamp', 0)
  const valueIndex = column('bodyBatteryLevel', 1)
  if (timeIndex == null || valueIndex == null)
    throw new Error('Garmin Body Battery columns are missing')
  const samples = new Map<number, number | null>()
  const values = row.bodyBatteryValuesArray
  if (values != null && !Array.isArray(values))
    throw new Error('Garmin Body Battery samples are malformed')
  for (const point of Array.isArray(values) ? values : []) {
    if (!Array.isArray(point)) continue
    const ts = timestamp(point[timeIndex])
    const value: unknown = point[valueIndex]
    if (ts != null)
      samples.set(
        ts,
        typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 100
          ? value
          : null,
      )
  }
  const charged = nonnegative(row, 'charged')
  const drained = nonnegative(row, 'drained')
  if (!samples.size && charged == null && drained == null) return null
  const startUtc = timestamp(row.startTimestampGMT)
  const startLocal = timestamp(row.startTimestampLocal)
  return {
    date,
    charged,
    drained,
    utcOffsetMinutes:
      startUtc != null && startLocal != null ? (startLocal - startUtc) / 60000 : null,
    samples: [...samples]
      .sort(([a], [b]) => a - b)
      .map(([timestamp, value]) => ({ timestamp, value })),
  }
}

const readinessFactors: [string, GarminReadinessFactor['name']][] = [
  ['sleepScore', 'sleep score'],
  ['recoveryTime', 'recovery time'],
  ['acwr', 'acute load'],
  ['stressHistory', 'stress history'],
  ['hrv', 'HRV status'],
  ['sleepHistory', 'sleep history'],
]

export function garminTrainingReadiness(
  raw: unknown,
  date: string,
): GarminTrainingReadiness[] | null {
  const candidates = rows(raw).filter(row => dateOf(row) === date)
  const primary = candidates.filter(row => row.primaryActivityTracker === true)
  const points = new Map<string, GarminTrainingReadiness>()
  for (const row of primary.length ? primary : candidates) {
    const score = nonnegative(row, 'score', 100)
    const recoveryTimeMinutes = nonnegative(row, 'recoveryTime')
    if (score == null && recoveryTimeMinutes == null) continue
    const ts = timestamp(row.timestamp)
    const inputContext = string(row, 'inputContext')
    points.set(`${ts}:${inputContext}`, {
      date,
      timestamp: ts,
      timestampLocal: string(row, 'timestampLocal'),
      score,
      level: string(row, 'level'),
      feedback: string(row, 'feedbackShort'),
      inputContext,
      recoveryTimeMinutes,
      recoveryTimeChange: string(row, 'recoveryTimeChangePhrase'),
      acuteLoad: nonnegative(row, 'acuteLoad'),
      hrvWeeklyAverage: nonnegative(row, 'hrvWeeklyAverage'),
      factors: readinessFactors.map(([key, name]) => ({
        name,
        percent: nonnegative(row, `${key}FactorPercent`, 100),
        feedback: string(row, `${key}FactorFeedback`),
      })),
    })
  }
  const result = [...points.values()].sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
  return result.length ? result : null
}

const nested = (row: UnknownRecord | null, key: string): UnknownRecord | null =>
  row && isRecord(row[key]) ? row[key] : null
const deviceRows = (raw: UnknownRecord | null, requestedDate: string): [string, UnknownRecord][] =>
  Object.entries(raw ?? {})
    .filter(
      (entry): entry is [string, UnknownRecord] =>
        isRecord(entry[1]) &&
        dateOf(entry[1]) != null &&
        String(entry[1].calendarDate) <= requestedDate,
    )
    .sort(
      ([a, left], [b, right]) =>
        Number(right.primaryTrainingDevice === true) -
          Number(left.primaryTrainingDevice === true) ||
        String(right.calendarDate).localeCompare(String(left.calendarDate)) ||
        a.localeCompare(b),
    )

function loadFocus(row: UnknownRecord | null): GarminLoadFocus | null {
  const date = row && dateOf(row)
  if (!row || !date) return null
  const categories: [string, GarminLoadFocus['categories'][number]['name']][] = [
    ['AerobicLow', 'low aerobic'],
    ['AerobicHigh', 'high aerobic'],
    ['Anaerobic', 'anaerobic'],
  ]
  return {
    date,
    feedback: string(row, 'trainingBalanceFeedbackPhrase'),
    categories: categories.map(([key, name]) => ({
      name,
      load: nonnegative(row, `monthlyLoad${key}`),
      targetMin: nonnegative(row, `monthlyLoad${key}TargetMin`),
      targetMax: nonnegative(row, `monthlyLoad${key}TargetMax`),
    })),
  }
}

export function garminTrainingStatus(
  raw: unknown,
  requestedDate: string,
): GarminTrainingStatus | null {
  const row = rows(raw)[0]
  if (!row) return null
  const statuses = deviceRows(
    nested(nested(row, 'mostRecentTrainingStatus'), 'latestTrainingStatusData'),
    requestedDate,
  )
  const balances = deviceRows(
    nested(nested(row, 'mostRecentTrainingLoadBalance'), 'metricsTrainingLoadBalanceDTOMap'),
    requestedDate,
  )
  const selected = statuses[0] ?? balances[0]
  if (!selected) return null
  const [key, status] = selected
  const load = nested(status, 'acuteTrainingLoadDTO') ?? {}
  const balance = balances.find(([device]) => device === key)?.[1] ?? null
  return {
    date: String(status.calendarDate),
    timestamp: timestamp(status.timestamp),
    status:
      typeof status.trainingStatus === 'string' || typeof status.trainingStatus === 'number'
        ? status.trainingStatus
        : null,
    feedback: string(status, 'trainingStatusFeedbackPhrase'),
    sinceDate: day(status.sinceDate) ? status.sinceDate : null,
    acuteLoad: nonnegative(load, 'dailyTrainingLoadAcute'),
    chronicLoad: nonnegative(load, 'dailyTrainingLoadChronic'),
    loadRatio: nonnegative(load, 'dailyAcuteChronicWorkloadRatio'),
    loadStatus: string(load, 'acwrStatus'),
    targetMin: nonnegative(load, 'minTrainingLoadChronic'),
    targetMax: nonnegative(load, 'maxTrainingLoadChronic'),
    loadFocus: loadFocus(balance),
  }
}

const enduranceThresholds = [
  'Intermediate',
  'Trained',
  'WellTrained',
  'Expert',
  'Superior',
  'Elite',
]
export function garminEnduranceScore(raw: unknown, date: string): GarminEnduranceScore | null {
  const row = rows(raw).find(row => dateOf(row) === date)
  const score = row ? nonnegative(row, 'overallScore') : null
  if (!row || score == null) return null
  return {
    date,
    score,
    classification: nonnegative(row, 'classification'),
    thresholds: enduranceThresholds.flatMap(label => {
      const lower = nonnegative(row, `classificationLowerLimit${label}`)
      return lower == null
        ? []
        : [{ label: label === 'WellTrained' ? 'well trained' : label.toLowerCase(), lower }]
    }),
    contributors: (Array.isArray(row.contributors)
      ? row.contributors.filter(isRecord)
      : []
    ).flatMap(row => {
      const contribution = nonnegative(row, 'contribution', 100)
      return contribution == null
        ? []
        : [
            {
              activityTypeId: nonnegative(row, 'activityTypeId'),
              group: nonnegative(row, 'group'),
              contribution,
            },
          ]
    }),
  }
}

export function garminHillScore(raw: unknown, date: string): GarminHillScore | null {
  const row = rows(raw).find(row => dateOf(row) === date)
  const score = row ? nonnegative(row, 'overallScore', 100) : null
  if (!row || score == null) return null
  return {
    date,
    score,
    strength: nonnegative(row, 'strengthScore', 100),
    endurance: nonnegative(row, 'enduranceScore', 100),
    classification: nonnegative(row, 'hillScoreClassificationId'),
  }
}

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const nullableNumber = (v: unknown): boolean => v === null || finite(v)
const nullableString = (v: unknown): boolean => v === null || typeof v === 'string'
const numericFields = (row: UnknownRecord, keys: string[]): boolean =>
  keys.every(key => nullableNumber(row[key]))
const stringFields = (row: UnknownRecord, keys: string[]): boolean =>
  keys.every(key => nullableString(row[key]))
const records = (v: unknown, check: (row: UnknownRecord) => boolean): boolean =>
  Array.isArray(v) && v.every(row => isRecord(row) && check(row))
const reading = (v: unknown, check: (value: unknown) => boolean): boolean =>
  isRecord(v) &&
  ['available', 'unavailable', 'error'].includes(String(v.status)) &&
  nullableNumber(v.fetchedAt) &&
  finite(v.attemptedAt) &&
  (v.value === null ? v.status !== 'available' : v.status !== 'unavailable' && check(v.value))

export function isGarminHealthDay(value: unknown, date: string): value is GarminHealthDay {
  return (
    isRecord(value) &&
    value.source === 'garmin' &&
    value.date === date &&
    reading(
      value.bodyBattery,
      v =>
        isRecord(v) &&
        v.date === date &&
        numericFields(v, ['charged', 'drained', 'utcOffsetMinutes']) &&
        records(v.samples, r => finite(r.timestamp) && nullableNumber(r.value)),
    ) &&
    reading(value.trainingReadiness, v =>
      records(
        v,
        r =>
          r.date === date &&
          numericFields(r, [
            'timestamp',
            'score',
            'recoveryTimeMinutes',
            'acuteLoad',
            'hrvWeeklyAverage',
          ]) &&
          stringFields(r, [
            'timestampLocal',
            'level',
            'feedback',
            'inputContext',
            'recoveryTimeChange',
          ]) &&
          records(
            r.factors,
            f =>
              readinessFactors.some(([, name]) => name === f.name) &&
              nullableNumber(f.percent) &&
              nullableString(f.feedback),
          ),
      ),
    ) &&
    reading(
      value.trainingStatus,
      v =>
        isRecord(v) &&
        day(v.date) &&
        v.date <= date &&
        (v.status === null || finite(v.status) || typeof v.status === 'string') &&
        numericFields(v, [
          'timestamp',
          'acuteLoad',
          'chronicLoad',
          'loadRatio',
          'targetMin',
          'targetMax',
        ]) &&
        stringFields(v, ['feedback', 'sinceDate', 'loadStatus']) &&
        (v.loadFocus === null ||
          (isRecord(v.loadFocus) &&
            day(v.loadFocus.date) &&
            v.loadFocus.date <= date &&
            nullableString(v.loadFocus.feedback) &&
            records(
              v.loadFocus.categories,
              r =>
                ['low aerobic', 'high aerobic', 'anaerobic'].includes(String(r.name)) &&
                numericFields(r, ['load', 'targetMin', 'targetMax']),
            ))),
    ) &&
    reading(
      value.enduranceScore,
      v =>
        isRecord(v) &&
        v.date === date &&
        finite(v.score) &&
        nullableNumber(v.classification) &&
        records(v.thresholds, r => typeof r.label === 'string' && finite(r.lower)) &&
        records(
          v.contributors,
          r =>
            nullableNumber(r.activityTypeId) && nullableNumber(r.group) && finite(r.contribution),
        ),
    ) &&
    reading(
      value.hillScore,
      v =>
        isRecord(v) &&
        v.date === date &&
        finite(v.score) &&
        numericFields(v, ['strength', 'endurance', 'classification']),
    )
  )
}

export function garminHealthFetchResult<T>(value: T | null, now: number): GarminHealthReading<T> {
  return {
    status: value === null ? 'unavailable' : 'available',
    fetchedAt: now,
    attemptedAt: now,
    value,
  }
}

export function garminHealthFetchError<T>(
  previous: GarminHealthReading<T> | undefined,
  now: number,
): GarminHealthReading<T> {
  return {
    status: 'error',
    fetchedAt: previous?.fetchedAt ?? null,
    attemptedAt: now,
    value: previous?.value ?? null,
  }
}
