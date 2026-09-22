import type { OuraDayDetail, OuraHealthDay } from '../plugins/stores/oura'
import { isRecord } from './type-guards'

export const emptyOuraHealth = (date: string): OuraHealthDay => ({
  date,
  stress: null,
  resilience: null,
  spo2: null,
  activity: null,
  cardiovascular: null,
  vo2Max: null,
  sleepTime: null,
  temperatureTrendC: null,
})

export const ouraHealthCollections = [
  'daily_stress',
  'daily_resilience',
  'daily_spo2',
  'daily_cardiovascular_age',
  'vO2_max',
  'sleep_time',
] as const
export type OuraHealthCollection =
  | (typeof ouraHealthCollections)[number]
  | 'daily_activity'
  | 'daily_readiness'

const number = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null
const nonnegative = (value: unknown): number | null => {
  const n = number(value)
  return n != null && n >= 0 ? n : null
}
const positive = (value: unknown): number | null => {
  const n = number(value)
  return n != null && n > 0 ? n : null
}
const score = (value: unknown): number | null => {
  const n = nonnegative(value)
  return n != null && n <= 100 ? n : null
}
const text = (value: unknown): string | null => (typeof value === 'string' ? value : null)

export function applyOuraHealthRow(
  day: OuraHealthDay,
  collection: OuraHealthCollection,
  row: Record<string, unknown>,
): void {
  if (collection === 'daily_stress')
    day.stress = {
      stressS: nonnegative(row.stress_high),
      restoredS: nonnegative(row.recovery_high),
      summary: text(row.day_summary),
    }
  else if (collection === 'daily_resilience') {
    const contributors = isRecord(row.contributors) ? row.contributors : {}
    day.resilience = {
      level: text(row.level),
      sleepRecovery: score(contributors.sleep_recovery),
      daytimeRecovery: score(contributors.daytime_recovery),
      stress: score(contributors.stress),
    }
  } else if (collection === 'daily_spo2') {
    const average = isRecord(row.spo2_percentage) ? score(row.spo2_percentage.average) : null
    day.spo2 = {
      averagePct: average != null && average > 0 ? average : null,
      breathingDisturbanceIndex: score(row.breathing_disturbance_index),
    }
  } else if (collection === 'daily_cardiovascular_age')
    day.cardiovascular = {
      vascularAge: positive(row.vascular_age),
      pulseWaveVelocity: positive(row.pulse_wave_velocity),
    }
  else if (collection === 'vO2_max') day.vo2Max = positive(row.vo2_max)
  else if (collection === 'daily_readiness')
    day.temperatureTrendC = number(row.temperature_trend_deviation)
  else if (collection === 'sleep_time') {
    const bedtime = isRecord(row.optimal_bedtime) ? row.optimal_bedtime : {}
    day.sleepTime = {
      startOffsetS: number(bedtime.start_offset),
      endOffsetS: number(bedtime.end_offset),
      utcOffsetS: number(bedtime.day_tz),
      recommendation: text(row.recommendation),
      status: text(row.status),
    }
  } else if (collection === 'daily_activity') {
    day.activity = {
      score: score(row.score),
      steps: nonnegative(row.steps),
      activeCalories: nonnegative(row.active_calories),
      targetCalories: positive(row.target_calories),
      equivalentWalkingDistanceM: nonnegative(row.equivalent_walking_distance),
      targetDistanceM: positive(row.target_meters),
      inactivityAlerts: nonnegative(row.inactivity_alerts),
      averageMet: nonnegative(row.average_met_minutes),
      restS: nonnegative(row.resting_time),
      sedentaryS: nonnegative(row.sedentary_time),
      lowS: nonnegative(row.low_activity_time),
      mediumS: nonnegative(row.medium_activity_time),
      highS: nonnegative(row.high_activity_time),
      nonWearS: nonnegative(row.non_wear_time),
      contributors: isRecord(row.contributors)
        ? Object.fromEntries(
            Object.entries(row.contributors).map(([key, value]) => [key, score(value)]),
          )
        : null,
    }
  }
}

export interface OuraRestorationBaseline {
  seconds: number
  days: number
}

export function ouraRestorationBaseline(
  date: string,
  details: Readonly<Record<string, OuraDayDetail>>,
): OuraRestorationBaseline | null {
  const end = Date.parse(`${date}T00:00:00Z`)
  const values = Object.entries(details).flatMap(([day, detail]) => {
    const age = (end - Date.parse(`${day}T00:00:00Z`)) / 86_400_000
    const value = detail.health?.stress?.restoredS
    return age >= 1 && age <= 14 && value != null && Number.isFinite(value) && value >= 0
      ? [value]
      : []
  })
  return values.length >= 3
    ? {
        seconds: values.reduce((sum, value) => sum + value, 0) / values.length,
        days: values.length,
      }
    : null
}

const numericFields = (value: unknown, keys: readonly string[]): boolean =>
  isRecord(value) && keys.every(key => value[key] === null || number(value[key]) != null)
const stringFields = (value: unknown, keys: readonly string[]): boolean =>
  value === null ||
  (isRecord(value) && keys.every(key => value[key] === null || typeof value[key] === 'string'))
const optionalGroup = (value: unknown, keys: readonly string[]): boolean =>
  value === null || numericFields(value, keys)

export function isOuraHealthDay(value: unknown, date: string): value is OuraHealthDay {
  if (!isRecord(value) || value.date !== date) return false
  if (
    value.failedCollections !== undefined &&
    (!Array.isArray(value.failedCollections) ||
      !value.failedCollections.every(key => typeof key === 'string'))
  )
    return false
  return (
    optionalGroup(value.stress, ['stressS', 'restoredS']) &&
    (value.stress === null ||
      (isRecord(value.stress) &&
        (value.stress.summary === null || typeof value.stress.summary === 'string'))) &&
    optionalGroup(value.resilience, ['sleepRecovery', 'daytimeRecovery', 'stress']) &&
    (value.resilience === null ||
      (isRecord(value.resilience) &&
        (value.resilience.level === null || typeof value.resilience.level === 'string'))) &&
    optionalGroup(value.spo2, ['averagePct', 'breathingDisturbanceIndex']) &&
    optionalGroup(value.cardiovascular, ['vascularAge', 'pulseWaveVelocity']) &&
    (value.vo2Max === null || positive(value.vo2Max) != null) &&
    (value.temperatureTrendC === null || number(value.temperatureTrendC) != null) &&
    optionalGroup(value.sleepTime, ['startOffsetS', 'endOffsetS', 'utcOffsetS']) &&
    stringFields(value.sleepTime, ['recommendation', 'status']) &&
    optionalGroup(value.activity, [
      'score',
      'steps',
      'activeCalories',
      'targetCalories',
      'equivalentWalkingDistanceM',
      'targetDistanceM',
      'inactivityAlerts',
      'averageMet',
      'restS',
      'sedentaryS',
      'lowS',
      'mediumS',
      'highS',
      'nonWearS',
    ]) &&
    (value.activity === null ||
      (isRecord(value.activity) &&
        (value.activity.contributors === null ||
          (isRecord(value.activity.contributors) &&
            Object.values(value.activity.contributors).every(
              v => v === null || score(v) != null,
            )))))
  )
}
