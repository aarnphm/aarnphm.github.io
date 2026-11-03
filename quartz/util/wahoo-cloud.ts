import fs from 'node:fs/promises'
import { dirname } from 'node:path'
import { joinSegments, QUARTZ } from './path'
import { isRecord, readString } from './type-guards'
import {
  type WahooTokenResponse,
  type WahooWorkoutSummaryDto,
  type WahooWorkoutDto,
  type WahooWorkoutFileUpload,
  type WahooWorkoutFileUploadInput,
  type WahooWorkoutFileUploadPollOptions,
  safeWahooFileUrl,
  parseWahooWorkoutSummary,
  parseWahooWorkout,
  parseWahooWorkoutPage,
  parseWahooTokenResponse,
  parseWahooWorkoutFileUpload,
} from './wahoo-cloud/response'
export * from './wahoo-cloud/response'

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

export function resolveWahooRefreshTokenFile(env: NodeJS.ProcessEnv = process.env): string {
  return env.WAHOO_REFRESH_TOKEN_FILE?.trim() || WAHOO_REFRESH_TOKEN_FILE
}

const DEFAULT_PER_PAGE = 100
const MAX_REDIRECTS = 5
const MAX_FIT_BYTES = 64 * 1024 * 1024
const DEFAULT_ACCESS_TOKEN_LIFETIME_S = 2 * 60 * 60
const ACCESS_TOKEN_EXPIRY_SKEW_MS = 60_000
const DEFAULT_RATE_LIMIT_DELAY_MS = 60_000
const MAX_AUTO_RATE_LIMIT_WAIT_MS = 5 * 60_000
const MAX_RATE_LIMIT_RETRIES = 8

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

function cleanBaseUrl(value: string, label: string): string {
  const url = new URL(value)
  const localhost = url.hostname === 'localhost' || url.hostname === '127.0.0.1'
  if (url.protocol !== 'https:' && !(localhost && url.protocol === 'http:'))
    throw new Error(`${label} must use HTTPS`)
  if (url.username || url.password) throw new Error(`${label} must not contain credentials`)
  return url.toString().replace(/\/+$/, '')
}

function rateLimitDelay(value: number): number {
  return Math.max(0, Math.ceil(value))
}

export function wahooRateLimitDelay(headers: Headers, nowMs = Date.now()): number {
  const retryAfter = headers.get('retry-after')?.trim()
  if (retryAfter) {
    const seconds = Number(retryAfter)
    if (Number.isFinite(seconds) && seconds >= 0) return rateLimitDelay(seconds * 1000)
    const timestamp = Date.parse(retryAfter)
    if (Number.isFinite(timestamp)) return rateLimitDelay(timestamp - nowMs)
  }
  const resetHeader = headers.get('x-ratelimit-reset')
  const reset = resetHeader == null ? Number.NaN : Number(resetHeader)
  if (Number.isFinite(reset) && reset > 0) return rateLimitDelay(reset * 1000)
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
    let waitedMs = 0
    for (let attempt = 0; attempt <= MAX_RATE_LIMIT_RETRIES; attempt++) {
      const response = await this.request(`${this.apiBaseUrl}${path}`, { ...init, headers })
      if (response.ok) return response
      const detail = (await response.text()).slice(0, 500)
      if (response.status !== 429) throw new WahooApiError(response.status, path, detail)
      const delayMs = wahooRateLimitDelay(response.headers, this.now())
      const quota = `rate limit reset in ${Math.ceil(delayMs / 1000)} seconds (remaining=${response.headers.get('x-ratelimit-remaining') ?? 'unknown'}, limits=${response.headers.get('x-ratelimit-limit') ?? 'unknown'})`
      if (attempt === MAX_RATE_LIMIT_RETRIES || delayMs > MAX_AUTO_RATE_LIMIT_WAIT_MS - waitedMs)
        throw new WahooApiError(response.status, path, `${detail}; ${quota}`)
      console.warn(`[wahoo] ${path}: 429, ${quota}; retrying`)
      await new Promise(resolveDelay => setTimeout(resolveDelay, delayMs))
      waitedMs += delayMs
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
    let found = 0
    let total = 0
    for (let attempt = 0; attempt < 3; attempt++) {
      const workouts = new Map<number, WahooWorkoutDto>()
      let expectedTotal: number | undefined
      let pageSize: number | undefined
      for (let pageNumber = 1; ; pageNumber++) {
        const query = new URLSearchParams({ page: String(pageNumber), per_page: String(perPage) })
        const page = parseWahooWorkoutPage(await this.apiJson(`/v1/workouts?${query.toString()}`))
        if (page.page !== pageNumber)
          throw new Error(`Wahoo returned page ${page.page} for ${pageNumber}`)
        if (page.workouts.length > page.perPage)
          throw new Error(`Wahoo page ${pageNumber} exceeded per_page`)
        total = page.total
        if (expectedTotal !== undefined && (total !== expectedTotal || page.perPage !== pageSize))
          break
        expectedTotal = total
        pageSize = page.perPage
        for (const workout of page.workouts) workouts.set(workout.id, workout)
        if (page.workouts.length === 0 || pageNumber * page.perPage >= total) {
          if (workouts.size === total) return [...workouts.values()]
          break
        }
      }
      found = workouts.size
      // Equal start times can overlap page boundaries. Retry from scratch with wider pages;
      // never publish a partial inventory just because duplicate rows were removed.
      if (attempt < 2) {
        perPage = Math.max(perPage, Math.min(1000, Math.max(total, perPage * 2)))
        console.warn(
          `[wahoo] pagination found ${found} of ${total} unique workouts; retrying with per_page=${perPage}`,
        )
      }
    }
    throw new Error(`Wahoo pagination ended after ${found} of ${total} workouts after 3 attempts`)
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
  const target = await fs.realpath(path).catch(() => path)
  const temporary = `${target}.tmp-${process.pid}`
  await fs.mkdir(dirname(target), { recursive: true })
  await fs.writeFile(temporary, `${refreshToken}\n`, { mode: 0o600 })
  await fs.rename(temporary, target)
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
  const refreshTokenFile = resolveWahooRefreshTokenFile()
  return new WahooCloudClient(await readWahooCloudCredentials(refreshTokenFile), {
    apiBaseUrl: process.env.WAHOO_API_BASE_URL,
    tokenUrl: process.env.WAHOO_TOKEN_URL,
    refreshTokenFile,
  })
}
