import type { GarminSleepSummary } from '../plugins/stores/garmin'
import type { OuraDayDetail } from '../plugins/stores/oura'
import { isRecord } from './type-guards'

export interface SleepMetrics {
  averageBreathsPerMinute: number | null
  respirationSource: 'oura' | 'garmin' | null
  garmin: GarminSleepSummary | null
}

const positive = (value: number | null | undefined): number | null =>
  value != null && Number.isFinite(value) && value > 0 ? value : null

export function resolveSleepMetrics(
  oura: Pick<OuraDayDetail, 'avgBreath'> | null | undefined,
  garmin: GarminSleepSummary | null | undefined,
): SleepMetrics | null {
  const ouraBreath = positive(oura?.avgBreath)
  const garminBreath = positive(garmin?.averageBreathsPerMinute)
  if (ouraBreath == null && !garmin) return null
  return {
    averageBreathsPerMinute: ouraBreath ?? garminBreath,
    respirationSource: ouraBreath != null ? 'oura' : garminBreath != null ? 'garmin' : null,
    garmin: garmin ?? null,
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
    (value.respirationSource !== 'garmin' ||
      (isRecord(value.garmin) &&
        value.averageBreathsPerMinute === value.garmin.averageBreathsPerMinute)))
