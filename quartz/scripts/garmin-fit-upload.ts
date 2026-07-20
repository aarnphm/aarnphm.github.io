import {
  applyGarminSetCookies,
  garminConnectRequestHeaders,
  garminResponseSummary,
  garminUrlFor,
  type GarminConnectSession,
} from '../util/garmin-session'
import { isRecord, readNumber, type UnknownRecord } from '../util/type-guards'

const DEFAULT_POLL_DELAY_MS = 1000
const MAX_POLL_DELAY_MS = 10_000
const MAX_POLL_ATTEMPTS = 60

function detailedImportResult(raw: UnknownRecord): UnknownRecord | null {
  if (isRecord(raw.detailedImportResult)) return raw.detailedImportResult
  if (isRecord(raw.data) && isRecord(raw.data.detailedImportResult))
    return raw.data.detailedImportResult
  return null
}

export function fitImportActivityId(raw: unknown): string {
  if (!isRecord(raw)) throw new Error('Garmin FIT import returned a non-object response')
  const result = detailedImportResult(raw)
  if (!result) throw new Error('Garmin FIT import omitted detailedImportResult')
  const failures = result.failures
  if (Array.isArray(failures) && failures.length > 0)
    throw new Error('Garmin FIT import reported a failure')
  const successes = result.successes
  if (!Array.isArray(successes) || successes.length !== 1 || !isRecord(successes[0]))
    throw new Error('Garmin FIT import did not report exactly one success')
  const id = readNumber(successes[0], 'internalId')
  if (id == null || !Number.isSafeInteger(id) || id <= 0)
    throw new Error('Garmin FIT import success omitted a valid internalId')
  return String(id)
}

export function fitImportPollLocation(requestUrl: string, location: string | null): string {
  if (!location) throw new Error('Garmin FIT import 202 response omitted Location')
  const url = new URL(location, requestUrl)
  if (url.protocol !== 'https:')
    throw new Error(`Garmin FIT import returned an unsafe poll location: ${url.toString()}`)
  if (url.hostname === 'connect.garmin.com' && url.pathname.startsWith('/gc-api/'))
    return url.toString()
  if (
    url.hostname === 'connectapi.garmin.com' &&
    url.pathname.startsWith('/activity-service/activity/status/')
  )
    return new URL(`/gc-api${url.pathname}${url.search}`, 'https://connect.garmin.com').toString()
  throw new Error(`Garmin FIT import returned an unsafe poll location: ${url.toString()}`)
}

export function fitImportPollDelay(value: string | null): number {
  if (!value) return DEFAULT_POLL_DELAY_MS
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_POLL_DELAY_MS
  return Math.min(MAX_POLL_DELAY_MS, Math.round(parsed))
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function responseJson(res: Response, label: string): Promise<unknown> {
  const text = await res.text()
  const type = res.headers.get('content-type') ?? ''
  if (type.includes('text/html') || text.trimStart().startsWith('<'))
    throw new Error(`${label} returned HTML: ${garminResponseSummary(res, text)}`)
  if (!type.includes('json'))
    throw new Error(`${label} returned non-JSON: ${garminResponseSummary(res, text)}`)
  try {
    const parsed: unknown = JSON.parse(text)
    return parsed
  } catch {
    throw new Error(`${label} returned malformed JSON`)
  }
}

async function pollFitImport(
  session: GarminConnectSession,
  requestUrl: string,
  location: string,
  initialDelay: number,
): Promise<string> {
  const pollUrl = fitImportPollLocation(requestUrl, location)
  let delayMs = initialDelay
  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
    await sleep(delayMs)
    const res = await fetch(pollUrl, {
      headers: garminConnectRequestHeaders(session),
      redirect: 'manual',
    })
    applyGarminSetCookies(session, res.headers)
    if (res.status === 202) {
      delayMs = fitImportPollDelay(res.headers.get('Location-In-Milliseconds'))
      await res.arrayBuffer()
      continue
    }
    if (res.status !== 200 && res.status !== 201) {
      const text = await res.text()
      throw new Error(
        `Garmin FIT import poll failed: ${res.status} ${garminResponseSummary(res, text)}`,
      )
    }
    return fitImportActivityId(await responseJson(res, 'Garmin FIT import poll'))
  }
  throw new Error(`Garmin FIT import exceeded ${MAX_POLL_ATTEMPTS} poll attempts`)
}

export async function uploadGarminFit(
  session: GarminConnectSession,
  base: string,
  filename: string,
  bytes: Uint8Array,
): Promise<string> {
  const requestUrl = garminUrlFor(base, '/upload-service/upload/.fit')
  const form = new FormData()
  const buffer = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(buffer).set(bytes)
  form.set('userfile', new Blob([buffer], { type: 'application/octet-stream' }), filename)
  const headers = new Headers(garminConnectRequestHeaders(session))
  headers.set('X-Requested-With', 'XMLHttpRequest')
  headers.delete('Content-Type')
  const res = await fetch(requestUrl, { method: 'POST', headers, body: form, redirect: 'manual' })
  applyGarminSetCookies(session, res.headers)
  if (res.status === 202) {
    await res.arrayBuffer()
    return pollFitImport(
      session,
      requestUrl,
      fitImportPollLocation(requestUrl, res.headers.get('Location')),
      fitImportPollDelay(res.headers.get('Location-In-Milliseconds')),
    )
  }
  if (res.status !== 200 && res.status !== 201) {
    const text = await res.text()
    throw new Error(`Garmin FIT import failed: ${res.status} ${garminResponseSummary(res, text)}`)
  }
  return fitImportActivityId(await responseJson(res, 'Garmin FIT import'))
}
