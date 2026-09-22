import type { GarminSleepSummary } from '../plugins/stores/garmin'
import type { OuraDayDetail } from '../plugins/stores/oura'
import { isRecord } from './type-guards'

export interface SleepMetrics {
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
): SleepMetrics | null {
  const ouraBreath = positive(oura?.avgBreath)
  const garminBreath = positive(garmin?.averageBreathsPerMinute)
  const saturation = (value: number | null | undefined): number | null => {
    const n = positive(value)
    return n != null && n <= 100 ? n : null
  }
  const garminOxygen = saturation(garmin?.averageSpO2)
  const ouraOxygen = saturation(oura?.health?.spo2?.averagePct)
  const averagePct = garminOxygen ?? ouraOxygen
  const bdi = oura?.health?.spo2?.breathingDisturbanceIndex
  const breathingDisturbanceIndex =
    bdi != null && Number.isFinite(bdi) && bdi >= 0 && bdi <= 100 ? bdi : null
  if (ouraBreath == null && !garmin && averagePct == null && breathingDisturbanceIndex == null)
    return null
  return {
    averageBreathsPerMinute: ouraBreath ?? garminBreath,
    respirationSource: ouraBreath != null ? 'oura' : garminBreath != null ? 'garmin' : null,
    garmin: garmin ?? null,
    ...(averagePct == null
      ? {}
      : { oxygenSaturation: { averagePct, source: garminOxygen != null ? 'garmin' : 'oura' } }),
    ...(breathingDisturbanceIndex == null ? {} : { breathingDisturbanceIndex }),
  }
}

const nullableNumber = (value: unknown): boolean =>
  value === null || (typeof value === 'number' && Number.isFinite(value))

const respirationIsValid = (value: unknown): boolean =>
  value === undefined ||
  (Array.isArray(value) &&
    value.every(
      (sample, index) =>
        isRecord(sample) &&
        typeof sample.timestamp === 'number' &&
        Number.isFinite(sample.timestamp) &&
        sample.timestamp > 0 &&
        (index === 0 || sample.timestamp > value[index - 1].timestamp) &&
        nullableNumber(sample.breathsPerMinute) &&
        (sample.breathsPerMinute === null ||
          (typeof sample.breathsPerMinute === 'number' && sample.breathsPerMinute > 0)),
    ))

const garminSleepIsValid = (value: unknown, date: string): boolean =>
  value === null ||
  (isRecord(value) &&
    value.source === 'garmin' &&
    value.date === date &&
    (value.startTime === null || typeof value.startTime === 'string') &&
    (value.endTime === null || typeof value.endTime === 'string') &&
    (value.utcOffsetMinutes === undefined ||
      (typeof value.utcOffsetMinutes === 'number' &&
        Number.isInteger(value.utcOffsetMinutes) &&
        Math.abs(value.utcOffsetMinutes) <= 14 * 60)) &&
    respirationIsValid(value.respiration) &&
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

export const isSleepMetrics = (value: unknown, date: string): value is SleepMetrics | null =>
  value === null ||
  (isRecord(value) &&
    nullableNumber(value.averageBreathsPerMinute) &&
    (value.averageBreathsPerMinute === null
      ? value.respirationSource === null
      : value.respirationSource === 'oura' || value.respirationSource === 'garmin') &&
    garminSleepIsValid(value.garmin, date) &&
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
