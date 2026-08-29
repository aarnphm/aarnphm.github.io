import { execFile } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { isRecord } from './type-guards'

const execFileAsync = promisify(execFile)
const GARMIN_AUTH_SCRIPT = fileURLToPath(new URL('../scripts/garmin-auth.py', import.meta.url))
const PROJECT_ROOT = fileURLToPath(new URL('../../', import.meta.url))
const FORBIDDEN_AUTH_HEADERS = ['connect-csrf-token', 'cookie']
const LEGACY_AUTH_ENV = [
  'GARMIN_CONNECT_COOKIE',
  'GARMIN_CONNECT_COOKIE_DB',
  'GARMIN_CONNECT_COOKIE_FILE',
  'GARMIN_CONNECT_CSRF_TOKEN',
  'GARMIN_PASSWORD',
  'GARMIN_USERNAME',
]

export const DEFAULT_GARMIN_CONNECT_BASE = 'https://connectapi.garmin.com'
export const DEFAULT_GARMIN_IMPORT_BASE = DEFAULT_GARMIN_CONNECT_BASE

export interface GarminConnectSession {
  headers: Readonly<Record<string, string>>
}

export function cleanGarminConnectBaseUrl(value: string): string {
  return value.replace(/\/+$/, '')
}

export function garminErrorMessage(value: unknown): string {
  return value instanceof Error ? value.message : String(value)
}

function garminAuthEnvironment(): NodeJS.ProcessEnv {
  const env = { ...process.env }
  for (const name of LEGACY_AUTH_ENV) delete env[name]
  return env
}

export function parseGarminConnectSession(value: unknown): GarminConnectSession {
  if (!isRecord(value) || !isRecord(value.headers))
    throw new Error('Garmin auth helper returned an invalid session')
  const normalized = new Headers()
  for (const [name, header] of Object.entries(value.headers)) {
    if (!name.trim() || typeof header !== 'string' || !header.trim())
      throw new Error('Garmin auth helper returned an invalid header')
    normalized.set(name, header)
  }
  for (const name of FORBIDDEN_AUTH_HEADERS) {
    if (normalized.has(name))
      throw new Error(`Garmin auth helper returned forbidden ${name} authentication`)
  }
  const authorization = normalized.get('authorization')
  if (!authorization?.startsWith('Bearer ') || authorization.length <= 'Bearer '.length)
    throw new Error('Garmin auth helper did not return a bearer token')
  const headers: Record<string, string> = {}
  normalized.forEach((header, name) => {
    headers[name] = header
  })
  return { headers }
}

export async function readGarminConnectSession(): Promise<GarminConnectSession> {
  let stdout: string
  try {
    const result = await execFileAsync(
      'uv',
      ['run', '--locked', 'python', GARMIN_AUTH_SCRIPT, 'session'],
      { cwd: PROJECT_ROOT, encoding: 'utf8', env: garminAuthEnvironment(), maxBuffer: 256 * 1024 },
    )
    stdout = result.stdout.toString()
  } catch (error) {
    throw new Error(
      `Garmin bearer authentication failed; run pnpm garmin:auth: ${garminErrorMessage(error)}`,
    )
  }
  let value: unknown
  try {
    value = JSON.parse(stdout)
  } catch {
    throw new Error('Garmin auth helper returned malformed JSON')
  }
  return parseGarminConnectSession(value)
}

export function garminConnectRequestHeaders(
  session: GarminConnectSession,
  contentType?: string,
): HeadersInit {
  const headers = new Headers(session.headers)
  if (contentType) headers.set('Content-Type', contentType)
  return headers
}

export function garminUrlFor(base: string, path: string, params?: URLSearchParams): string {
  const url = new URL(`${base}${path}`)
  if (params) for (const [key, value] of params) url.searchParams.set(key, value)
  return url.toString()
}

export function garminResponseSummary(res: Response, text: string): string {
  const type = res.headers.get('content-type') ?? 'unknown content-type'
  if (type.includes('text/html') || text.trimStart().startsWith('<'))
    return `${type} (${text.length} bytes HTML)`
  return `${type} ${text.trim().slice(0, 300)}`
}

export function assertGarminResponseAuthorized(res: Response): void {
  if (res.status === 401 || res.status === 403)
    throw new Error(`Garmin bearer session rejected (${res.status}); run pnpm garmin:auth`)
}

export async function fetchGarminJson(
  session: GarminConnectSession,
  base: string,
  path: string,
  params?: URLSearchParams,
  init?: RequestInit,
): Promise<unknown> {
  const res = await fetch(garminUrlFor(base, path, params), {
    ...init,
    headers: garminConnectRequestHeaders(session, init?.body ? 'application/json' : undefined),
  })
  const text = await res.text()
  assertGarminResponseAuthorized(res)
  if (!res.ok)
    throw new Error(
      `Garmin Connect request failed: ${res.status} ${garminResponseSummary(res, text)}`,
    )
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('application/json'))
    throw new Error(`Garmin Connect returned non-JSON: ${garminResponseSummary(res, text)}`)
  const value: unknown = JSON.parse(text)
  return value
}

export async function fetchGarminBytes(
  session: GarminConnectSession,
  base: string,
  path: string,
  params?: URLSearchParams,
): Promise<Uint8Array> {
  const res = await fetch(garminUrlFor(base, path, params), {
    headers: garminConnectRequestHeaders(session),
  })
  assertGarminResponseAuthorized(res)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(
      `Garmin Connect request failed: ${res.status} ${garminResponseSummary(res, text)}`,
    )
  }
  return new Uint8Array(await res.arrayBuffer())
}
