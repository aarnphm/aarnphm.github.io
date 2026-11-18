import type { AppleSleepDay } from '../plugins/stores/apple-sleep'
import type { GarminSleepSummary } from '../plugins/stores/garmin'
import type { OuraDayDetail } from '../plugins/stores/oura'
import { isRecord } from './type-guards'

export interface SleepMetrics {
  apple?: AppleSleepDay | null
  averageBreathsPerMinute: number | null
  respirationSource: 'oura' | 'garmin' | null
  garmin: GarminSleepSummary | null
  oxygenSaturation?: { averagePct: number; source: 'oura' | 'garmin' }
  breathingDisturbanceIndex?: number
}

const positive = (value: number | null | undefined): number | null =>
  value != null && Number.isFinite(value) && value > 0 ? value : null

export function resolveSleepMetrics(
  oura: Pick<OuraDayDetail, 'avgBreath' | 'health'> | null | undefined,
  garmin: GarminSleepSummary | null | undefined,
  apple?: AppleSleepDay | null,
): SleepMetrics | null {
  const ouraBreath = positive(oura?.avgBreath)
  const garminBreath = positive(garmin?.averageBreathsPerMinute)
  const saturation = (value: number | null | undefined): number | null => {
    const n = positive(value)
    return n != null && n <= 100 ? n : null
  }
  const garminOxygen = saturation(garmin?.averageSpO2)
  const ouraOxygen = saturation(oura?.health?.spo2?.averagePct)
  const averagePct = ouraOxygen ?? garminOxygen
  const bdi = oura?.health?.spo2?.breathingDisturbanceIndex
  const breathingDisturbanceIndex =
    bdi != null && Number.isFinite(bdi) && bdi >= 0 && bdi <= 100 ? bdi : null
  if (
    ouraBreath == null &&
    !garmin &&
    !apple &&
    averagePct == null &&
    breathingDisturbanceIndex == null
  )
    return null
  return {
    averageBreathsPerMinute: ouraBreath ?? garminBreath,
    respirationSource: ouraBreath != null ? 'oura' : garminBreath != null ? 'garmin' : null,
    garmin: garmin ?? null,
    ...(apple ? { apple } : {}),
    ...(averagePct == null
      ? {}
      : { oxygenSaturation: { averagePct, source: ouraOxygen != null ? 'oura' : 'garmin' } }),
    ...(breathingDisturbanceIndex == null ? {} : { breathingDisturbanceIndex }),
  }
}

const nullableNumber = (value: unknown): boolean =>
  value === null || (typeof value === 'number' && Number.isFinite(value))

const timestampIsValid = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 && value <= 8.64e15

const samplesAreValid = (value: unknown, key: string): boolean =>
  value === undefined ||
  (Array.isArray(value) &&
    value.every(
      (sample, index) =>
        isRecord(sample) &&
        timestampIsValid(sample.timestamp) &&
        (index === 0 || sample.timestamp > value[index - 1].timestamp) &&
        (sample[key] === null ||
          (typeof sample[key] === 'number' && positive(sample[key]) != null)),
    ))

const stageNames: ReadonlySet<unknown> = new Set(['deep', 'light', 'rem', 'awake', 'unmeasurable'])

export const sleepStagesAreValid = (value: unknown): boolean =>
  value === undefined ||
  (Array.isArray(value) &&
    value.every(
      (stage, index) =>
        isRecord(stage) &&
        timestampIsValid(stage.startTime) &&
        timestampIsValid(stage.endTime) &&
        stage.endTime > stage.startTime &&
        (index === 0 || stage.startTime >= value[index - 1].endTime) &&
        stageNames.has(stage.stage),
    ))

const scoreFactorNames = new Set([
  'totalDuration',
  'stress',
  'awakeCount',
  'restlessness',
  'lightPercentage',
  'deepPercentage',
  'remPercentage',
])

const nullableString = (value: unknown): boolean => value === null || typeof value === 'string'

const sleepWindowIsValid = (start: unknown, end: unknown): boolean =>
  typeof start === 'string' && typeof end === 'string' && Date.parse(end) > Date.parse(start)

const utcOffsetIsValid = (value: unknown): boolean =>
  value === undefined ||
  (typeof value === 'number' && Number.isInteger(value) && Math.abs(value) <= 14 * 60)

const garminSleepNightIsValid = (value: unknown): boolean =>
  value === undefined ||
  (isRecord(value) &&
    typeof value.totalSleepS === 'number' &&
    positive(value.totalSleepS) != null &&
    nullableString(value.sleepScoreQualifier) &&
    [
      value.deepS,
      value.lightS,
      value.remS,
      value.awakeS,
      value.unmeasurableS,
      value.awakeCount,
      value.sleepScore,
      value.restingHeartRateBpm,
      value.averageHeartRateBpm,
      value.averageHrvMs,
      value.skinTempDeviationC,
    ].every(nullableNumber) &&
    sleepStagesAreValid(value.stages) &&
    samplesAreValid(value.heartRate, 'bpm') &&
    samplesAreValid(value.hrv, 'ms') &&
    (value.scoreFactors === undefined ||
      (isRecord(value.scoreFactors) &&
        Object.entries(value.scoreFactors).every(
          ([name, factor]) =>
            scoreFactorNames.has(name) &&
            isRecord(factor) &&
            nullableString(factor.qualifier) &&
            nullableNumber(factor.value),
        ))))

const garminNapsAreValid = (value: unknown): boolean =>
  value === undefined ||
  (Array.isArray(value) &&
    value.every(
      nap =>
        isRecord(nap) &&
        typeof nap.id === 'string' &&
        timestampIsValid(nap.startTime) &&
        timestampIsValid(nap.endTime) &&
        nap.endTime > nap.startTime &&
        typeof nap.durationS === 'number' &&
        positive(nap.durationS) != null &&
        utcOffsetIsValid(nap.startUtcOffsetMinutes) &&
        utcOffsetIsValid(nap.endUtcOffsetMinutes) &&
        nullableNumber(nap.providerSource) &&
        nullableString(nap.feedback),
    ))

const garminSleepIsValid = (value: unknown, date: string): boolean =>
  value === null ||
  (isRecord(value) &&
    value.source === 'garmin' &&
    value.date === date &&
    (value.startTime === null || typeof value.startTime === 'string') &&
    (value.endTime === null || typeof value.endTime === 'string') &&
    utcOffsetIsValid(value.utcOffsetMinutes) &&
    samplesAreValid(value.respiration, 'breathsPerMinute') &&
    (value.sleep === undefined ||
      (sleepWindowIsValid(value.startTime, value.endTime) &&
        garminSleepNightIsValid(value.sleep))) &&
    garminNapsAreValid(value.naps) &&
    [
      value.averageBreathsPerMinute,
      value.lowestBreathsPerMinute,
      value.highestBreathsPerMinute,
      value.averageSpO2,
      value.lowestSpO2,
      value.bodyBatteryStart,
      value.bodyBatteryEnd,
      value.bodyBatteryChange,
      value.averageStress,
      value.restlessMoments,
    ].every(nullableNumber))

const appleSessionIsValid = (value: unknown): boolean =>
  isRecord(value) &&
  typeof value.id === 'string' &&
  sleepWindowIsValid(value.start, value.end) &&
  typeof value.totalSleepS === 'number' &&
  value.totalSleepS > 0 &&
  value.totalSleepS <= (Date.parse(String(value.end)) - Date.parse(String(value.start))) / 1000 &&
  [value.deepS, value.lightS, value.remS, value.awakeS].every(
    n => nullableNumber(n) && (n === null || Number(n) >= 0),
  ) &&
  typeof value.sourceName === 'string' &&
  typeof value.sourceBundleIdentifier === 'string' &&
  sleepStagesAreValid(value.stages)

const appleSleepIsValid = (value: unknown, date: string): boolean =>
  value == null ||
  (isRecord(value) &&
    value.date === date &&
    (value.sleep === null || appleSessionIsValid(value.sleep)) &&
    Array.isArray(value.naps) &&
    value.naps.every(appleSessionIsValid))

export const isSleepMetrics = (value: unknown, date: string): value is SleepMetrics | null =>
  value === null ||
  (isRecord(value) &&
    nullableNumber(value.averageBreathsPerMinute) &&
    (value.averageBreathsPerMinute === null
      ? value.respirationSource === null
      : value.respirationSource === 'oura' || value.respirationSource === 'garmin') &&
    garminSleepIsValid(value.garmin, date) &&
    appleSleepIsValid(value.apple, date) &&
    (value.oxygenSaturation === undefined ||
      (isRecord(value.oxygenSaturation) &&
        typeof value.oxygenSaturation.averagePct === 'number' &&
        value.oxygenSaturation.averagePct > 0 &&
        value.oxygenSaturation.averagePct <= 100 &&
        (value.oxygenSaturation.source === 'oura' ||
          (value.oxygenSaturation.source === 'garmin' &&
            isRecord(value.garmin) &&
            value.garmin.averageSpO2 === value.oxygenSaturation.averagePct)))) &&
    (value.breathingDisturbanceIndex === undefined ||
      (typeof value.breathingDisturbanceIndex === 'number' &&
        value.breathingDisturbanceIndex >= 0 &&
        value.breathingDisturbanceIndex <= 100)) &&
    (value.respirationSource !== 'garmin' ||
      (isRecord(value.garmin) &&
        value.averageBreathsPerMinute === value.garmin.averageBreathsPerMinute)))
