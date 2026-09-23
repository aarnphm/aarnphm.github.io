import { isDeepStrictEqual } from 'node:util'

const API = 'https://tpapi.trainingpeaks.com'

export interface TrainingPeaksCredentials {
  accessToken?: string
  authCookie?: string
}

export interface TrainingPeaksWorkout extends Record<string, unknown> {
  workoutId: number
  athleteId: number
  title: string
  description: string | null
  workoutTypeValueId: number
  workoutDay: string
  startTime: string | null
  startTimePlanned: string | null
  totalTime: number | null
  distance: number | null
  isLocked: boolean | null
}

export const TRAININGPEAKS_PLANNED_FIELDS = [
  'totalTimePlanned',
  'distancePlanned',
  'tssPlanned',
  'ifPlanned',
  'caloriesPlanned',
  'velocityPlanned',
  'energyPlanned',
  'elevationGainPlanned',
]

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function positiveId(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
}

function nullableNumber(value: unknown): boolean {
  return value === null || (typeof value === 'number' && Number.isFinite(value))
}

function workout(value: unknown): value is TrainingPeaksWorkout {
  return (
    record(value) &&
    positiveId(value.workoutId) &&
    positiveId(value.athleteId) &&
    typeof value.title === 'string' &&
    (value.description === null || typeof value.description === 'string') &&
    positiveId(value.workoutTypeValueId) &&
    typeof value.workoutDay === 'string' &&
    /^\d{4}-\d{2}-\d{2}T/.test(value.workoutDay) &&
    (value.startTime === null || typeof value.startTime === 'string') &&
    (value.startTimePlanned === null || typeof value.startTimePlanned === 'string') &&
    nullableNumber(value.totalTime) &&
    nullableNumber(value.distance) &&
    (value.isLocked === null || typeof value.isLocked === 'boolean') &&
    TRAININGPEAKS_PLANNED_FIELDS.every(field => nullableNumber(value[field]))
  )
}

export function parseTrainingPeaksWorkout(value: unknown, athleteId: number): TrainingPeaksWorkout {
  if (!workout(value)) throw new Error('TrainingPeaks returned an unexpected workout schema')
  if (value.athleteId !== athleteId)
    throw new Error(`TrainingPeaks workout ${value.workoutId} belongs to another athlete`)
  return value
}

export function trainingPeaksMetadataPayload(
  current: TrainingPeaksWorkout,
  title: string,
  description: string,
): TrainingPeaksWorkout {
  if (!title.trim()) throw new Error('TrainingPeaks title must not be empty')
  if (current.isLocked) throw new Error(`TrainingPeaks workout ${current.workoutId} is locked`)
  // The web API uses full-record PUTs. Preserve metrics, planning, comments, and unknown fields.
  return { ...current, title, description }
}

export function verifyTrainingPeaksMetadata(
  expected: TrainingPeaksWorkout,
  actual: TrainingPeaksWorkout,
): void {
  if (actual.title !== expected.title || actual.description !== expected.description)
    throw new Error(`TrainingPeaks metadata readback failed for workout ${expected.workoutId}`)
  const changed = Object.keys(expected).filter(
    // TrainingPeaks changes these server-generated values on every save.
    key =>
      key !== 'lastModifiedDate' &&
      key !== 'sharedWorkoutInformationExpireKey' &&
      !isDeepStrictEqual(expected[key], actual[key]),
  )
  if (changed.length)
    throw new Error(
      `TrainingPeaks changed additional fields on workout ${expected.workoutId}: ${changed.join(', ')}`,
    )
}

export class TrainingPeaksApi {
  private accessToken: string | undefined
  private readonly authCookie: string | undefined

  constructor(credentials: TrainingPeaksCredentials) {
    this.accessToken = credentials.accessToken?.trim().replace(/^Bearer\s+/i, '') || undefined
    this.authCookie = credentials.authCookie?.trim() || undefined
    if (!this.accessToken && !this.authCookie)
      throw new Error(
        'Set TRAININGPEAKS_AUTH_COOKIE or TRAININGPEAKS_ACCESS_TOKEN in .env. Use the Production_tpAuth cookie value or a web API bearer token from your own signed-in TrainingPeaks account.',
      )
    if (this.accessToken && /[^\x21-\x7e]/.test(this.accessToken))
      throw new Error('TRAININGPEAKS_ACCESS_TOKEN must contain a single bearer token')
    if (this.authCookie && /[^\x21-\x7e]|;/.test(this.authCookie))
      throw new Error(
        'TRAININGPEAKS_AUTH_COOKIE must contain only the Production_tpAuth cookie value',
      )
  }

  private async send(
    path: string,
    method: 'GET' | 'PUT',
    headers: Record<string, string>,
    body?: unknown,
  ): Promise<Response> {
    return fetch(`${API}${path}`, {
      ...(method === 'PUT' ? { method, body: JSON.stringify(body) } : { method }),
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        ...headers,
      },
      redirect: 'error',
      signal: AbortSignal.timeout(20_000),
    })
  }

  private async json(response: Response): Promise<unknown> {
    if (response.status === 401 || response.status === 403)
      throw new Error(
        `TrainingPeaks authentication failed (HTTP ${response.status}); refresh the session cookie or access token in .env`,
      )
    if (!response.ok) throw new Error(`TrainingPeaks API returned HTTP ${response.status}`)
    if (!response.headers.get('content-type')?.includes('application/json'))
      throw new Error('TrainingPeaks returned a non-JSON response')
    return response.json()
  }

  private async refreshToken(): Promise<void> {
    if (!this.authCookie)
      throw new Error(
        'The TrainingPeaks access token expired; update TRAININGPEAKS_ACCESS_TOKEN in .env',
      )
    const data = await this.json(
      await this.send('/users/v3/token', 'GET', { Cookie: `Production_tpAuth=${this.authCookie}` }),
    )
    if (
      !record(data) ||
      data.success !== true ||
      !record(data.token) ||
      typeof data.token.access_token !== 'string' ||
      !data.token.access_token
    )
      throw new Error('TrainingPeaks session expired; refresh TRAININGPEAKS_AUTH_COOKIE in .env')
    this.accessToken = data.token.access_token
  }

  private async request(
    path: string,
    method: 'GET' | 'PUT' = 'GET',
    body?: unknown,
  ): Promise<unknown> {
    if (!this.accessToken) await this.refreshToken()
    let response = await this.send(
      path,
      method,
      { Authorization: `Bearer ${this.accessToken}` },
      body,
    )
    if (response.status === 401 && this.authCookie) {
      await response.body?.cancel()
      await this.refreshToken()
      response = await this.send(
        path,
        method,
        { Authorization: `Bearer ${this.accessToken}` },
        body,
      )
    }
    return this.json(response)
  }

  async athleteId(): Promise<number> {
    const data = await this.request('/users/v3/user')
    if (
      !record(data) ||
      !record(data.user) ||
      data.user.isAthlete !== true ||
      !positiveId(data.user.userId)
    )
      throw new Error('TrainingPeaks title sync requires the signed-in athlete account')
    return data.user.userId
  }

  async workouts(athleteId: number, since: string, until: string): Promise<TrainingPeaksWorkout[]> {
    const data = await this.request(`/fitness/v7/athletes/${athleteId}/workouts/${since}/${until}`)
    if (!Array.isArray(data)) throw new Error('TrainingPeaks returned an unexpected workout list')
    return data.map(value => parseTrainingPeaksWorkout(value, athleteId))
  }

  async workout(athleteId: number, workoutId: number): Promise<TrainingPeaksWorkout> {
    const data = await this.request(`/fitness/v6/athletes/${athleteId}/workouts/${workoutId}`)
    const current = parseTrainingPeaksWorkout(data, athleteId)
    if (current.workoutId !== workoutId) throw new Error('TrainingPeaks returned another workout')
    return current
  }

  async updateMetadata(
    current: TrainingPeaksWorkout,
    title: string,
    description: string,
  ): Promise<void> {
    const payload = trainingPeaksMetadataPayload(current, title, description)
    await this.request(
      `/fitness/v6/athletes/${current.athleteId}/workouts/${current.workoutId}`,
      'PUT',
      payload,
    )
    verifyTrainingPeaksMetadata(payload, await this.workout(current.athleteId, current.workoutId))
  }
}
