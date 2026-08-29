import fs from 'node:fs/promises'
import { isIP } from 'node:net'
import { dirname } from 'node:path'
import { joinSegments, QUARTZ } from './path'
import { isRecord, readNumber, readString, type UnknownRecord } from './type-guards'

export const WAHOO_OAUTH_SCOPES: readonly string[] = [
  'user_read',
  'workouts_read',
  'workouts_write',
  'offline_data',
]
export const DEFAULT_WAHOO_API_BASE_URL = 'https://api.wahooligan.com'
export const DEFAULT_WAHOO_AUTHORIZE_URL = 'https://api.wahooligan.com/oauth/authorize'
export const DEFAULT_WAHOO_TOKEN_URL = 'https://api.wahooligan.com/oauth/token'
export const WAHOO_REFRESH_TOKEN_FILE = joinSegments(QUARTZ, '.quartz-cache', 'wahoo-refresh-token')

const DEFAULT_PER_PAGE = 100
const MAX_REDIRECTS = 5
const MAX_FIT_BYTES = 64 * 1024 * 1024
const DEFAULT_ACCESS_TOKEN_LIFETIME_S = 2 * 60 * 60
const ACCESS_TOKEN_EXPIRY_SKEW_MS = 60_000
const DEFAULT_RATE_LIMIT_DELAY_MS = 60_000
const MAX_RATE_LIMIT_DELAY_MS = 5 * 60_000
const MAX_RATE_LIMIT_RETRIES = 8

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

export interface WahooCloudCredentials {
  clientId: string
  clientSecret: string
  refreshToken: string
}

export class WahooApiError extends Error {
  constructor(
    readonly status: number,
    readonly path: string,
    readonly detail: string,
  ) {
    super(`Wahoo API ${path} failed: ${status} ${detail}`)
    this.name = 'WahooApiError'
  }
}

export interface WahooCloudClientOptions {
  apiBaseUrl?: string
  tokenUrl?: string
  refreshTokenFile?: string
  request?: typeof fetch
  now?: () => number
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

export function isWahooRestrictedWorkoutSummaryError(error: unknown): boolean {
  if (
    !(error instanceof WahooApiError) ||
    error.status !== 401 ||
    !/^\/v1\/workouts\/\d+\/workout_summary$/.test(error.path)
  )
    return false
  try {
    const value: unknown = JSON.parse(error.detail)
    return (
      isRecord(value) &&
      readString(value, 'error') === 'You are not authorized to view this workout summary'
    )
  } catch {
    return false
  }
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

function cleanBaseUrl(value: string, label: string): string {
  const url = new URL(value)
  const localhost = url.hostname === 'localhost' || url.hostname === '127.0.0.1'
  if (url.protocol !== 'https:' && !(localhost && url.protocol === 'http:'))
    throw new Error(`${label} must use HTTPS`)
  if (url.username || url.password) throw new Error(`${label} must not contain credentials`)
  return url.toString().replace(/\/+$/, '')
}

function boundedRateLimitDelay(value: number): number {
  return Math.min(MAX_RATE_LIMIT_DELAY_MS, Math.max(0, Math.ceil(value)))
}

export function wahooRateLimitDelay(headers: Headers, nowMs = Date.now()): number {
  const retryAfter = headers.get('retry-after')?.trim()
  if (retryAfter) {
    const seconds = Number(retryAfter)
    if (Number.isFinite(seconds) && seconds >= 0) return boundedRateLimitDelay(seconds * 1000)
    const timestamp = Date.parse(retryAfter)
    if (Number.isFinite(timestamp)) return boundedRateLimitDelay(timestamp - nowMs)
  }
  const resetHeader = headers.get('x-ratelimit-reset')
  const reset = resetHeader == null ? Number.NaN : Number(resetHeader)
  if (Number.isFinite(reset) && reset >= 0) {
    const delay = reset > nowMs / 1000 - 60 ? reset * 1000 - nowMs : reset * 1000
    return boundedRateLimitDelay(delay)
  }
  return DEFAULT_RATE_LIMIT_DELAY_MS
}

async function responseJson(response: Response, label: string): Promise<unknown> {
  const text = await response.text()
  if (!response.ok) throw new Error(`${label} failed: ${response.status} ${text.slice(0, 500)}`)
  try {
    return JSON.parse(text)
  } catch {
    throw new Error(`${label} returned invalid JSON`)
  }
}

export async function exchangeWahooAuthorizationCode(
  clientId: string,
  clientSecret: string,
  code: string,
  redirectUri: string,
  tokenUrl = DEFAULT_WAHOO_TOKEN_URL,
  request: typeof fetch = fetch,
): Promise<WahooTokenResponse> {
  const body = new URLSearchParams({
    client_id: clientId,
    client_secret: clientSecret,
    code,
    redirect_uri: redirectUri,
    grant_type: 'authorization_code',
  })
  const response = await request(cleanBaseUrl(tokenUrl, 'Wahoo token URL'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  return parseWahooTokenResponse(await responseJson(response, 'Wahoo token exchange'))
}

export class WahooCloudClient {
  private readonly apiBaseUrl: string
  private readonly tokenUrl: string
  private readonly refreshTokenFile: string
  private readonly request: typeof fetch
  private readonly now: () => number
  private refreshToken: string
  private accessToken: string | null = null
  private accessTokenExpiresAt = 0
  private refreshPromise: Promise<string> | null = null

  constructor(
    private readonly credentials: WahooCloudCredentials,
    options: WahooCloudClientOptions = {},
  ) {
    this.apiBaseUrl = cleanBaseUrl(
      options.apiBaseUrl ?? DEFAULT_WAHOO_API_BASE_URL,
      'Wahoo API base URL',
    )
    this.tokenUrl = cleanBaseUrl(options.tokenUrl ?? DEFAULT_WAHOO_TOKEN_URL, 'Wahoo token URL')
    this.refreshTokenFile = options.refreshTokenFile ?? WAHOO_REFRESH_TOKEN_FILE
    this.request = options.request ?? fetch
    this.now = options.now ?? Date.now
    this.refreshToken = credentials.refreshToken
  }

  private async rotateAccessToken(): Promise<string> {
    const body = new URLSearchParams({
      client_id: this.credentials.clientId,
      client_secret: this.credentials.clientSecret,
      grant_type: 'refresh_token',
      refresh_token: this.refreshToken,
    })
    const response = await this.request(this.tokenUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body,
    })
    const token = parseWahooTokenResponse(await responseJson(response, 'Wahoo token refresh'))
    this.refreshToken = token.refreshToken
    this.accessToken = token.accessToken
    this.accessTokenExpiresAt =
      this.now() + (token.expiresInS ?? DEFAULT_ACCESS_TOKEN_LIFETIME_S) * 1000
    await writeWahooRefreshToken(token.refreshToken, this.refreshTokenFile)
    return token.accessToken
  }

  private async accessTokenForRequest(): Promise<string> {
    if (this.accessToken && this.accessTokenExpiresAt - ACCESS_TOKEN_EXPIRY_SKEW_MS > this.now())
      return this.accessToken
    if (!this.refreshPromise) {
      this.refreshPromise = this.rotateAccessToken().finally(() => {
        this.refreshPromise = null
      })
    }
    return this.refreshPromise
  }

  private async apiRequest(path: string, init: RequestInit = {}): Promise<Response> {
    if (!path.startsWith('/')) throw new Error('Wahoo API path must start with /')
    const accessToken = await this.accessTokenForRequest()
    const headers = new Headers(init.headers)
    headers.set('Authorization', `Bearer ${accessToken}`)
    for (let attempt = 0; attempt <= MAX_RATE_LIMIT_RETRIES; attempt++) {
      const response = await this.request(`${this.apiBaseUrl}${path}`, { ...init, headers })
      if (response.ok) return response
      const text = await response.text()
      if (response.status !== 429 || attempt === MAX_RATE_LIMIT_RETRIES)
        throw new WahooApiError(response.status, path, text.slice(0, 500))
      await new Promise(resolveDelay =>
        setTimeout(resolveDelay, wahooRateLimitDelay(response.headers, this.now())),
      )
    }
    throw new Error(`Wahoo API ${path} exhausted rate-limit retries`)
  }

  private async apiJson(path: string, init: RequestInit = {}): Promise<unknown> {
    const response = await this.apiRequest(path, init)
    const text = await response.text()
    try {
      return JSON.parse(text)
    } catch {
      throw new Error(`Wahoo API ${path} returned invalid JSON`)
    }
  }

  async listWorkouts(perPage = DEFAULT_PER_PAGE): Promise<WahooWorkoutDto[]> {
    if (!Number.isInteger(perPage) || perPage <= 0) throw new Error('perPage must be positive')
    const workouts: WahooWorkoutDto[] = []
    const ids = new Set<number>()
    let total = Number.POSITIVE_INFINITY
    for (let pageNumber = 1; workouts.length < total; pageNumber++) {
      const query = new URLSearchParams({ page: String(pageNumber), per_page: String(perPage) })
      const page = parseWahooWorkoutPage(await this.apiJson(`/v1/workouts?${query.toString()}`))
      if (page.page !== pageNumber)
        throw new Error(`Wahoo returned page ${page.page} for ${pageNumber}`)
      total = page.total
      for (const workout of page.workouts) {
        if (ids.has(workout.id)) throw new Error(`Wahoo returned duplicate workout ${workout.id}`)
        ids.add(workout.id)
        workouts.push(workout)
      }
      if (page.workouts.length === 0 || workouts.length >= total) break
      if (page.workouts.length > page.perPage)
        throw new Error(`Wahoo page ${pageNumber} exceeded per_page`)
    }
    if (workouts.length < total)
      throw new Error(`Wahoo pagination ended after ${workouts.length} of ${total} workouts`)
    return workouts
  }

  async getWorkoutSummary(workoutId: number): Promise<WahooWorkoutSummaryDto | null> {
    if (!Number.isInteger(workoutId) || workoutId < 0)
      throw new Error('workoutId must be nonnegative')
    const value = await this.apiJson(`/v1/workouts/${workoutId}/workout_summary`)
    if (isRecord(value) && Object.keys(value).length === 0) return null
    return parseWahooWorkoutSummary(value)
  }

  async updateWorkoutName(workoutId: number, name: string): Promise<WahooWorkoutDto> {
    if (!Number.isInteger(workoutId) || workoutId < 0)
      throw new Error('workoutId must be nonnegative')
    const normalized = name.trim().replace(/\s+/g, ' ')
    if (!normalized) throw new Error('Wahoo workout name must not be empty')
    const body = new URLSearchParams({ 'workout[name]': normalized })
    return parseWahooWorkout(
      await this.apiJson(`/v1/workouts/${workoutId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body,
      }),
    )
  }

  async createWorkoutFileUpload(
    input: WahooWorkoutFileUploadInput,
  ): Promise<WahooWorkoutFileUpload> {
    if (input.bytes.byteLength === 0) throw new Error('Wahoo workout FIT must not be empty')
    if (input.bytes.byteLength > MAX_FIT_BYTES)
      throw new Error('Wahoo workout FIT exceeds size limit')
    if (
      input.targetWorkoutId != null &&
      (!Number.isInteger(input.targetWorkoutId) || input.targetWorkoutId < 0)
    )
      throw new Error('Wahoo targetWorkoutId must be a nonnegative integer')
    const filename = input.filename?.trim() || 'workout.fit'
    const body = new URLSearchParams({
      'workout_file_upload[file]': `data:application/vnd.fit;base64,${Buffer.from(input.bytes).toString('base64')}`,
      'workout_file_upload[filename]': filename,
    })
    if (input.timeZone?.trim()) body.set('workout_file_upload[time_zone]', input.timeZone.trim())
    if (input.workoutName?.trim())
      body.set('workout_file_upload[workout_name]', input.workoutName.trim())
    if (input.targetWorkoutId != null)
      body.set('workout_file_upload[target_workout_id]', String(input.targetWorkoutId))
    return parseWahooWorkoutFileUpload(
      await this.apiJson('/v1/workout_file_uploads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body,
      }),
    )
  }

  async getWorkoutFileUpload(token: string): Promise<WahooWorkoutFileUpload> {
    const normalized = token.trim()
    if (!normalized) throw new Error('Wahoo workout file upload token must not be empty')
    return parseWahooWorkoutFileUpload(
      await this.apiJson(`/v1/workout_file_uploads/${encodeURIComponent(normalized)}`),
    )
  }

  async pollWorkoutFileUpload(
    token: string,
    options: WahooWorkoutFileUploadPollOptions = {},
  ): Promise<WahooWorkoutFileUpload> {
    const intervalMs = options.intervalMs ?? 1000
    const maxAttempts = options.maxAttempts ?? 60
    if (!Number.isInteger(intervalMs) || intervalMs < 0)
      throw new Error('Wahoo upload intervalMs must be a nonnegative integer')
    if (!Number.isInteger(maxAttempts) || maxAttempts <= 0)
      throw new Error('Wahoo upload maxAttempts must be a positive integer')
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const upload = await this.getWorkoutFileUpload(token)
      if (upload.status === 'complete' || upload.status === 'duplicate') return upload
      if (upload.status === 'error')
        throw new Error(`Wahoo workout file upload failed: ${upload.error ?? 'unknown error'}`)
      if (intervalMs > 0 && attempt + 1 < maxAttempts)
        await new Promise(resolveDelay => setTimeout(resolveDelay, intervalMs))
    }
    throw new Error(`Wahoo workout file upload exceeded ${maxAttempts} poll attempts`)
  }

  async downloadFit(sourceUrl: string): Promise<Uint8Array> {
    let url = safeWahooFileUrl(sourceUrl)
    if (!url) throw new Error('Wahoo FIT URL is missing')
    for (let redirects = 0; redirects <= MAX_REDIRECTS; redirects++) {
      const response = await this.request(url, { redirect: 'manual' })
      if (response.status >= 300 && response.status < 400) {
        const location = response.headers.get('location')
        if (!location) throw new Error('Wahoo FIT redirect omitted Location')
        url = safeWahooFileUrl(new URL(location, url).toString())
        if (!url) throw new Error('Wahoo FIT redirect URL is missing')
        continue
      }
      if (!response.ok) throw new Error(`Wahoo FIT download failed: ${response.status}`)
      const contentLength = Number(response.headers.get('content-length'))
      if (Number.isFinite(contentLength) && contentLength > MAX_FIT_BYTES)
        throw new Error('Wahoo FIT file exceeds size limit')
      const bytes = new Uint8Array(await response.arrayBuffer())
      if (bytes.byteLength === 0) throw new Error('Wahoo FIT download returned an empty file')
      if (bytes.byteLength > MAX_FIT_BYTES) throw new Error('Wahoo FIT file exceeds size limit')
      return bytes
    }
    throw new Error(`Wahoo FIT download exceeded ${MAX_REDIRECTS} redirects`)
  }
}

function missingFile(error: unknown): boolean {
  return error instanceof Error && 'code' in error && error.code === 'ENOENT'
}

export async function readWahooRefreshToken(
  path = WAHOO_REFRESH_TOKEN_FILE,
): Promise<string | null> {
  try {
    const refreshToken = (await fs.readFile(path, 'utf8')).trim()
    if (!refreshToken) throw new Error(`Wahoo refresh token file ${path} is empty`)
    return refreshToken
  } catch (error) {
    if (missingFile(error)) return null
    throw error
  }
}

export async function writeWahooRefreshToken(
  refreshToken: string,
  path = WAHOO_REFRESH_TOKEN_FILE,
): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`
  await fs.mkdir(dirname(path), { recursive: true })
  await fs.writeFile(temporary, `${refreshToken}\n`, { mode: 0o600 })
  await fs.rename(temporary, path)
}

export async function readWahooCloudCredentials(
  refreshTokenFile = WAHOO_REFRESH_TOKEN_FILE,
  env: NodeJS.ProcessEnv = process.env,
): Promise<WahooCloudCredentials> {
  const clientId = env.WAHOO_CLIENT_ID?.trim()
  const clientSecret = env.WAHOO_CLIENT_SECRET?.trim()
  const refreshToken =
    (await readWahooRefreshToken(refreshTokenFile)) ?? env.WAHOO_REFRESH_TOKEN?.trim()
  if (!clientId || !clientSecret || !refreshToken)
    throw new Error('set WAHOO_CLIENT_ID, WAHOO_CLIENT_SECRET, and WAHOO_REFRESH_TOKEN in .env')
  return { clientId, clientSecret, refreshToken }
}

export async function wahooCloudClientFromEnv(): Promise<WahooCloudClient> {
  return new WahooCloudClient(await readWahooCloudCredentials(), {
    apiBaseUrl: process.env.WAHOO_API_BASE_URL,
    tokenUrl: process.env.WAHOO_TOKEN_URL,
    refreshTokenFile: WAHOO_REFRESH_TOKEN_FILE,
  })
}
