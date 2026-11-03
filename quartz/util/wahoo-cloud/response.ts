import { isIP } from 'node:net'
import { isRecord, readNumber, readString, type UnknownRecord } from '../type-guards'

export interface WahooTokenResponse {
  accessToken: string
  refreshToken: string
  expiresInS: number | null
  scope: string | null
}

export interface WahooWorkoutSummaryDto {
  id: number
  name: string | null
  ascentM: number | null
  cadenceAvg: number | null
  calories: number | null
  distanceM: number | null
  durationActiveS: number | null
  durationPausedS: number | null
  durationTotalS: number | null
  heartRateAvg: number | null
  normalizedPowerW: number | null
  trainingStressScore: number | null
  powerAvgW: number | null
  speedAvgMps: number | null
  workJ: number | null
  timeZone: string | null
  manual: boolean | null
  edited: boolean | null
  fitnessAppId: number | null
  fileUrl: string | null
  createdAt: string | null
  updatedAt: string | null
}

export interface WahooWorkoutDto {
  id: number
  starts: string
  minutes: number
  name: string | null
  workoutToken: string | null
  workoutTypeId: number
  summary: WahooWorkoutSummaryDto | null
  createdAt: string | null
  updatedAt: string | null
}

export interface WahooWorkoutPage {
  workouts: WahooWorkoutDto[]
  total: number
  page: number
  perPage: number
}

export type WahooWorkoutFileUploadStatus =
  | 'pending'
  | 'in_progress'
  | 'complete'
  | 'error'
  | 'duplicate'

export interface WahooWorkoutFileUpload {
  id: number
  token: string
  status: WahooWorkoutFileUploadStatus
  timeZone: string | null
  workoutId: number | null
  workoutSummaryId: number | null
  workoutFileId: number | null
  workoutName: string | null
  error: string | null
  targetWorkoutId: number | null
  createdAt: string | null
  updatedAt: string | null
}

export interface WahooWorkoutFileUploadInput {
  bytes: Uint8Array
  filename?: string
  timeZone?: string
  workoutName?: string
  targetWorkoutId?: number
}

export interface WahooWorkoutFileUploadPollOptions {
  intervalMs?: number
  maxAttempts?: number
}

function requiredString(record: UnknownRecord, key: string, label: string): string {
  const value = readString(record, key)?.trim()
  if (!value) throw new Error(`${label} missing ${key}`)
  return value
}

function optionalString(record: UnknownRecord, key: string, label: string): string | null {
  const value = record[key]
  if (value == null) return null
  if (typeof value !== 'string') throw new Error(`${label}.${key} must be a string or null`)
  return value.trim() || null
}

function finiteNumber(value: unknown, label: string): number | null {
  if (value == null || value === '') return null
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
  if (!Number.isFinite(parsed)) throw new Error(`${label} must be a finite number or null`)
  return parsed
}

function optionalNumber(record: UnknownRecord, key: string, label: string): number | null {
  return finiteNumber(record[key], `${label}.${key}`)
}

function requiredNonnegativeInteger(record: UnknownRecord, key: string, label: string): number {
  const value = finiteNumber(record[key], `${label}.${key}`)
  if (value == null || !Number.isInteger(value) || value < 0)
    throw new Error(`${label}.${key} must be a nonnegative integer`)
  return value
}

function optionalBoolean(record: UnknownRecord, key: string, label: string): boolean | null {
  const value = record[key]
  if (value == null) return null
  if (typeof value !== 'boolean') throw new Error(`${label}.${key} must be a boolean or null`)
  return value
}

function isoTimestamp(value: string, label: string): string {
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp)) throw new Error(`${label} must be an ISO timestamp`)
  return new Date(timestamp).toISOString()
}

function optionalTimestamp(record: UnknownRecord, key: string, label: string): string | null {
  const value = optionalString(record, key, label)
  return value == null ? null : isoTimestamp(value, `${label}.${key}`)
}

function ipv4Octets(hostname: string): number[] | null {
  if (isIP(hostname) !== 4) return null
  const octets = hostname.split('.').map(Number)
  return octets.length === 4 ? octets : null
}

function isPrivateHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  if (
    normalized === 'localhost' ||
    normalized.endsWith('.localhost') ||
    normalized.endsWith('.local') ||
    normalized.endsWith('.internal')
  )
    return true
  const octets = ipv4Octets(normalized)
  if (octets) {
    const [first, second] = octets
    return (
      first === 0 ||
      first === 10 ||
      first === 127 ||
      (first === 100 && second >= 64 && second <= 127) ||
      (first === 169 && second === 254) ||
      (first === 172 && second >= 16 && second <= 31) ||
      (first === 192 && second === 168) ||
      first >= 224
    )
  }
  if (isIP(normalized) === 6) {
    return (
      normalized === '::' ||
      normalized === '::1' ||
      /^f[cd]/.test(normalized) ||
      /^fe[89ab]/.test(normalized) ||
      normalized.startsWith('::ffff:')
    )
  }
  return false
}

export function safeWahooFileUrl(value: unknown): string | null {
  if (value == null) return null
  if (typeof value !== 'string' || !value.trim()) throw new Error('Wahoo FIT URL must be a string')
  let url: URL
  try {
    url = new URL(value)
  } catch {
    throw new Error('Wahoo FIT URL is invalid')
  }
  if (
    url.protocol !== 'https:' ||
    url.username ||
    url.password ||
    !url.hostname ||
    isPrivateHostname(url.hostname)
  )
    throw new Error('Wahoo FIT URL must be a public HTTPS URL without credentials')
  return url.toString()
}

function summaryFileUrl(record: UnknownRecord, label: string): string | null {
  const file = record.file
  if (file == null) return null
  if (!isRecord(file)) throw new Error(`${label}.file must be an object or null`)
  return safeWahooFileUrl(file.url)
}

export function parseWahooWorkoutSummary(
  value: unknown,
  label = 'Wahoo workout summary',
): WahooWorkoutSummaryDto {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return {
    id: requiredNonnegativeInteger(value, 'id', label),
    name: optionalString(value, 'name', label),
    ascentM: optionalNumber(value, 'ascent_accum', label),
    cadenceAvg: optionalNumber(value, 'cadence_avg', label),
    calories: optionalNumber(value, 'calories_accum', label),
    distanceM: optionalNumber(value, 'distance_accum', label),
    durationActiveS: optionalNumber(value, 'duration_active_accum', label),
    durationPausedS: optionalNumber(value, 'duration_paused_accum', label),
    durationTotalS: optionalNumber(value, 'duration_total_accum', label),
    heartRateAvg: optionalNumber(value, 'heart_rate_avg', label),
    normalizedPowerW: optionalNumber(value, 'power_bike_np_last', label),
    trainingStressScore: optionalNumber(value, 'power_bike_tss_last', label),
    powerAvgW: optionalNumber(value, 'power_avg', label),
    speedAvgMps: optionalNumber(value, 'speed_avg', label),
    workJ: optionalNumber(value, 'work_accum', label),
    timeZone: optionalString(value, 'time_zone', label),
    manual: optionalBoolean(value, 'manual', label),
    edited: optionalBoolean(value, 'edited', label),
    fitnessAppId: optionalNumber(value, 'fitness_app_id', label),
    fileUrl: summaryFileUrl(value, label),
    createdAt: optionalTimestamp(value, 'created_at', label),
    updatedAt: optionalTimestamp(value, 'updated_at', label),
  }
}

export function parseWahooWorkout(value: unknown, label = 'Wahoo workout'): WahooWorkoutDto {
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  const starts = isoTimestamp(requiredString(value, 'starts', label), `${label}.starts`)
  const minutes = finiteNumber(value.minutes, `${label}.minutes`)
  if (minutes == null || minutes < 0) throw new Error(`${label}.minutes must be nonnegative`)
  const summaryValue = value.workout_summary
  return {
    id: requiredNonnegativeInteger(value, 'id', label),
    starts,
    minutes,
    name: optionalString(value, 'name', label),
    workoutToken: optionalString(value, 'workout_token', label),
    workoutTypeId: requiredNonnegativeInteger(value, 'workout_type_id', label),
    summary:
      summaryValue == null
        ? null
        : parseWahooWorkoutSummary(summaryValue, `${label}.workout_summary`),
    createdAt: optionalTimestamp(value, 'created_at', label),
    updatedAt: optionalTimestamp(value, 'updated_at', label),
  }
}

export function parseWahooWorkoutPage(value: unknown): WahooWorkoutPage {
  if (!isRecord(value)) throw new Error('Wahoo workout page must be an object')
  if (!Array.isArray(value.workouts)) throw new Error('Wahoo workout page missing workouts array')
  const page = requiredNonnegativeInteger(value, 'page', 'Wahoo workout page')
  const perPage = requiredNonnegativeInteger(value, 'per_page', 'Wahoo workout page')
  const total = requiredNonnegativeInteger(value, 'total', 'Wahoo workout page')
  if (page < 1 || perPage < 1) throw new Error('Wahoo workout page and per_page must be positive')
  return {
    workouts: value.workouts.map((workout, index) =>
      parseWahooWorkout(workout, `Wahoo workout page.workouts[${index}]`),
    ),
    total,
    page,
    perPage,
  }
}

export function isWahooOriginatedSummary(summary: WahooWorkoutSummaryDto): boolean {
  return (
    summary.fitnessAppId != null &&
    Number.isInteger(summary.fitnessAppId) &&
    summary.fitnessAppId >= 0 &&
    summary.fitnessAppId < 1000
  )
}

export function parseWahooTokenResponse(value: unknown): WahooTokenResponse {
  if (!isRecord(value)) throw new Error('Wahoo token response must be an object')
  const expiresIn = readNumber(value, 'expires_in')
  if (expiresIn != null && (!Number.isFinite(expiresIn) || expiresIn <= 0))
    throw new Error('Wahoo token response expires_in must be positive')
  return {
    accessToken: requiredString(value, 'access_token', 'Wahoo token response'),
    refreshToken: requiredString(value, 'refresh_token', 'Wahoo token response'),
    expiresInS: expiresIn ?? null,
    scope: optionalString(value, 'scope', 'Wahoo token response'),
  }
}

function uploadStatus(value: unknown): WahooWorkoutFileUploadStatus {
  switch (value) {
    case 'pending':
    case 'in_progress':
    case 'complete':
    case 'error':
    case 'duplicate':
      return value
    default:
      throw new Error(`Wahoo workout file upload has invalid status ${String(value)}`)
  }
}

function nullableNonnegativeInteger(
  record: UnknownRecord,
  key: string,
  label: string,
): number | null {
  if (record[key] == null) return null
  return requiredNonnegativeInteger(record, key, label)
}

export function parseWahooWorkoutFileUpload(value: unknown): WahooWorkoutFileUpload {
  const label = 'Wahoo workout file upload'
  if (!isRecord(value)) throw new Error(`${label} must be an object`)
  return {
    id: requiredNonnegativeInteger(value, 'id', label),
    token: requiredString(value, 'token', label),
    status: uploadStatus(value.status),
    timeZone: optionalString(value, 'time_zone', label),
    workoutId: nullableNonnegativeInteger(value, 'workout_id', label),
    workoutSummaryId: nullableNonnegativeInteger(value, 'workout_summary_id', label),
    workoutFileId: nullableNonnegativeInteger(value, 'workout_file_id', label),
    workoutName: optionalString(value, 'workout_name', label),
    error: optionalString(value, 'error', label),
    targetWorkoutId: nullableNonnegativeInteger(value, 'target_workout_id', label),
    createdAt: optionalTimestamp(value, 'created_at', label),
    updatedAt: optionalTimestamp(value, 'updated_at', label),
  }
}
