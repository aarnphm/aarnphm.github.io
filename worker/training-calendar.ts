import { DurableObject } from 'cloudflare:workers'
import { createHash, randomBytes } from 'node:crypto'
import {
  TRAINING_CALENDAR_API,
  TRAINING_CALENDAR_ASSET,
  TRAINING_CALENDAR_SESSION_API,
} from '../quartz/util/training-calendar-access'
import {
  openTrainingCalendar,
  validTrainingDataKey,
  validTrainingPasswordHash,
  verifyTrainingPassword,
} from '../quartz/util/training-calendar-crypto'
import { isArenaReaderMutationAllowed } from './arena-reader-auth'
import { isRecord } from './type-guards'

interface TrainingCalendarSecrets {
  TRAINING_CALENDAR_PASSWORD_HASH?: string
  TRAINING_CALENDAR_DATA_KEY?: string
}

export interface TrainingCalendarEnv extends TrainingCalendarSecrets {
  ASSETS: Fetcher
  TRAINING_CALENDAR_ACCESS?: DurableObjectNamespace<TrainingCalendarAccess>
}

const cookieName = '__Host-TRAINING_CALENDAR'
const sessionSeconds = 7 * 24 * 60 * 60
const rateWindow = 15 * 60 * 1000
const privateHeaders = {
  'Content-Type': 'application/json; charset=utf-8',
  'Cache-Control': 'private, no-store',
  'CDN-Cache-Control': 'no-store',
  'Cloudflare-CDN-Cache-Control': 'no-store',
  'X-Content-Type-Options': 'nosniff',
  'Cross-Origin-Resource-Policy': 'same-origin',
  'Referrer-Policy': 'no-referrer',
  Vary: 'Cookie',
}

function json(value: unknown, status = 200, extra: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(value), { status, headers: { ...privateHeaders, ...extra } })
}

function sessionCookie(token: string, age = sessionSeconds): string {
  return `${cookieName}=${token}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=${age}`
}

function sessionToken(request: Request): string | null {
  const values = (request.headers.get('Cookie') ?? '')
    .split(';')
    .map(value => value.trim())
    .filter(value => value.startsWith(`${cookieName}=`))
  if (values.length !== 1) return null
  const token = values[0].slice(cookieName.length + 1)
  return /^[a-f0-9]{64}$/.test(token) ? token : null
}

const digest = (value: string): string => createHash('sha256').update(value).digest('hex')

function configured(env: TrainingCalendarSecrets): env is Required<TrainingCalendarSecrets> {
  return (
    validTrainingPasswordHash(env.TRAINING_CALENDAR_PASSWORD_HASH) &&
    validTrainingDataKey(env.TRAINING_CALENDAR_DATA_KEY)
  )
}

async function readPassword(request: Request): Promise<string | Response> {
  if (request.headers.get('Content-Type')?.split(';')[0].trim() !== 'application/json')
    return json({ error: 'Use application/json.' }, 415)
  if (!request.body) return json({ error: 'Password required.' }, 400)
  const reader = request.body.getReader()
  const chunks: Uint8Array[] = []
  let size = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    size += value.byteLength
    if (size > 4096) {
      await reader.cancel()
      return json({ error: 'Request too large.' }, 413)
    }
    chunks.push(value)
  }
  const bytes = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  const value: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes))
  if (!isRecord(value) || typeof value.password !== 'string' || value.password.length > 256)
    return json({ error: 'Invalid password.' }, 400)
  return value.password
}

/** One named object serializes attempt counters and makes session revocation immediate. */
export class TrainingCalendarAccess extends DurableObject<TrainingCalendarSecrets> {
  constructor(ctx: DurableObjectState, env: TrainingCalendarSecrets) {
    super(ctx, env)
    ctx.storage.sql.exec(`CREATE TABLE IF NOT EXISTS sessions (
      token_hash TEXT PRIMARY KEY, origin TEXT NOT NULL, credential TEXT NOT NULL,
      expires_at INTEGER NOT NULL)`)
    const columns = ctx.storage.sql.exec<{ name: string }>('PRAGMA table_info(sessions)').toArray()
    if (columns.some(column => column.name === 'idle_expires_at'))
      ctx.storage.sql.exec('ALTER TABLE sessions DROP COLUMN idle_expires_at')
    ctx.storage.sql.exec(`CREATE TABLE IF NOT EXISTS attempts (
      key TEXT PRIMARY KEY, count INTEGER NOT NULL, resets_at INTEGER NOT NULL)`)
  }

  async fetch(request: Request): Promise<Response> {
    if (!configured(this.env)) return json({ error: 'Training calendar unavailable.' }, 503)
    let password: string | undefined
    if (request.method === 'POST') {
      try {
        const value = await readPassword(request)
        if (value instanceof Response) return value
        password = value
      } catch {
        return json({ error: 'Invalid request.' }, 400)
      }
    }
    const hash = this.env.TRAINING_CALENDAR_PASSWORD_HASH
    const credential = digest(`${hash}:${this.env.TRAINING_CALENDAR_DATA_KEY}`)
    const sql = this.ctx.storage.sql
    const now = Date.now()
    sql.exec('DELETE FROM sessions WHERE expires_at <= ? OR credential != ?', now, credential)
    sql.exec('DELETE FROM attempts WHERE resets_at <= ?', now)
    const origin = new URL(request.url).origin
    const token = sessionToken(request)
    if (request.method === 'DELETE') {
      if (token)
        sql.exec('DELETE FROM sessions WHERE token_hash = ? AND origin = ?', digest(token), origin)
      return json({ authenticated: false }, 200, { 'Set-Cookie': sessionCookie('', 0) })
    }
    if (request.method === 'GET') {
      const sessions = token
        ? sql
            .exec<{ expires_at: number }>(
              'SELECT expires_at FROM sessions WHERE token_hash = ? AND origin = ?',
              digest(token),
              origin,
            )
            .toArray()
        : []
      if (!sessions[0] || !token) return json({ error: 'Authentication required.' }, 401)
      return json({ authenticated: true, expiresAt: sessions[0].expires_at })
    }
    if (password === undefined) return json({ error: 'Method not allowed.' }, 405)
    // Only the edge's client IP is trusted. Missing IPs share a single throttle bucket.
    const ipKey = `ip:${digest(request.headers.get('CF-Connecting-IP') ?? 'unknown')}`
    const limits = [
      { key: 'global', limit: 30 },
      { key: ipKey, limit: 5 },
    ]
    // No awaits between reading and incrementing the counters.
    for (const { key, limit } of limits) {
      const row = sql
        .exec<{ count: number; resets_at: number }>(
          'SELECT count, resets_at FROM attempts WHERE key = ?',
          key,
        )
        .toArray()[0]
      if (row && row.count >= limit)
        return json({ error: 'Too many attempts. Try again later.' }, 429, {
          'Retry-After': String(Math.ceil((row.resets_at - now) / 1000)),
        })
    }
    for (const { key } of limits)
      sql.exec(
        'INSERT INTO attempts (key, count, resets_at) VALUES (?, 1, ?) ON CONFLICT(key) DO UPDATE SET count = count + 1',
        key,
        now + rateWindow,
      )
    if (!verifyTrainingPassword(password, hash)) return json({ error: 'Incorrect password.' }, 401)
    const nextToken = randomBytes(32).toString('hex')
    if (token)
      sql.exec('DELETE FROM sessions WHERE token_hash = ? AND origin = ?', digest(token), origin)
    const expiresAt = now + sessionSeconds * 1000
    sql.exec(
      'INSERT INTO sessions (token_hash, origin, credential, expires_at) VALUES (?, ?, ?, ?)',
      digest(nextToken),
      origin,
      credential,
      expiresAt,
    )
    return json({ authenticated: true, expiresAt }, 200, { 'Set-Cookie': sessionCookie(nextToken) })
  }
}

/** Must run before host rewrites, static assets, redirects, and generic CORS handling. */
export async function handleTrainingCalendarRequest(
  request: Request,
  env: TrainingCalendarEnv,
): Promise<Response | null> {
  const url = new URL(request.url)
  let pathname: string
  try {
    pathname = decodeURIComponent(url.pathname).replace(/\/{2,}/g, '/')
  } catch {
    return json({ error: 'Invalid path.' }, 400)
  }
  if (pathname === TRAINING_CALENDAR_ASSET || pathname.startsWith(`${TRAINING_CALENDAR_ASSET}/`))
    return json({ error: 'Not found.' }, 404)
  if (pathname !== TRAINING_CALENDAR_API && pathname !== TRAINING_CALENDAR_SESSION_API) return null
  if (url.protocol !== 'https:' && !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname))
    return json({ error: 'HTTPS required.' }, 403)
  const isSession = pathname === TRAINING_CALENDAR_SESSION_API
  if (
    isSession &&
    ['POST', 'DELETE'].includes(request.method) &&
    !isArenaReaderMutationAllowed(request)
  )
    return json({ error: 'Same-origin request required.' }, 403)
  const methods = isSession ? ['GET', 'POST', 'DELETE'] : ['GET', 'HEAD']
  if (!methods.includes(request.method))
    return json({ error: 'Method not allowed.' }, 405, { Allow: methods.join(', ') })
  if (!configured(env) || !env.TRAINING_CALENDAR_ACCESS)
    return json({ error: 'Training calendar unavailable.' }, 503)
  try {
    const access = env.TRAINING_CALENDAR_ACCESS.getByName('calendar')
    if (isSession) return await access.fetch(request)
    const session = await access.fetch(
      new Request(new URL(TRAINING_CALENDAR_SESSION_API, url), { headers: request.headers }),
    )
    if (!session.ok) return session
    const asset = await env.ASSETS.fetch(new Request(new URL(TRAINING_CALENDAR_ASSET, url)))
    if (!asset.ok) return json({ error: 'Training calendar unavailable.' }, 503)
    const data: unknown = await asset.json()
    const plaintext = openTrainingCalendar(data, env.TRAINING_CALENDAR_DATA_KEY)
    return new Response(request.method === 'HEAD' ? null : plaintext, { headers: privateHeaders })
  } catch {
    return json({ error: 'Training calendar unavailable.' }, 503)
  }
}
