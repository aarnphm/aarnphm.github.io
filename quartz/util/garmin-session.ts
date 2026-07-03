import fs from 'node:fs/promises'
import { browserCookieHeaders } from './browser-cookie'

export const GARMIN_CONNECT_ORIGIN = 'https://connect.garmin.com'
export const DEFAULT_GARMIN_CONNECT_BASE = `${GARMIN_CONNECT_ORIGIN}/gc-api`
export const DEFAULT_GARMIN_IMPORT_BASE = DEFAULT_GARMIN_CONNECT_BASE

const DEFAULT_USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.7827.53 Safari/537.36'
const MAX_DOCUMENT_REDIRECTS = 8

export interface GarminConnectSession {
  cookie: string
  csrf: string
}

interface CookieCandidate {
  cookie: string
  source: string
  type: 'env' | 'file' | 'browser'
}

export function cleanGarminConnectBaseUrl(value: string): string {
  return value.replace(/\/+$/, '')
}

export function garminErrorMessage(value: unknown): string {
  return value instanceof Error ? value.message : String(value)
}

async function readCookieCandidates(): Promise<CookieCandidate[]> {
  const inline = process.env.GARMIN_CONNECT_COOKIE?.trim()
  if (inline) return [{ cookie: inline, source: 'env', type: 'env' }]
  const file = process.env.GARMIN_CONNECT_COOKIE_FILE?.trim()
  if (file) {
    const cookie = (await fs.readFile(file, 'utf8')).trim()
    if (cookie) return [{ cookie, source: file, type: 'file' }]
  }
  const candidates = await browserCookieHeaders()
  if (candidates.length > 0)
    return candidates.map(candidate => ({
      cookie: candidate.cookie,
      source: `${candidate.browser}:${candidate.db}`,
      type: 'browser',
    }))
  throw new Error(
    'set GARMIN_CONNECT_COOKIE/GARMIN_CONNECT_COOKIE_FILE, or log in to Garmin Connect in a supported browser profile',
  )
}

export async function readGarminConnectSession(): Promise<GarminConnectSession> {
  const candidates = await readCookieCandidates()
  const csrf = process.env.GARMIN_CONNECT_CSRF_TOKEN?.trim()
  let last: unknown = null
  for (const candidate of candidates) {
    const session: GarminConnectSession = { cookie: candidate.cookie, csrf: '' }
    try {
      session.csrf = csrf || (await fetchGarminConnectCsrf(session))
      if (candidate.type === 'browser')
        console.log('[garmin] using Garmin Connect cookies from browser profile')
      return session
    } catch (err) {
      last = err
      if (candidates.length === 1) break
      console.warn(
        `[garmin] skipped Garmin Connect cookie jar ${candidate.source}: ${garminErrorMessage(err)}`,
      )
    }
  }
  throw last instanceof Error ? last : new Error(garminErrorMessage(last))
}

export function garminConnectRequestHeaders(
  session: GarminConnectSession,
  contentType?: string,
): HeadersInit {
  return {
    Accept: 'application/json, text/plain, */*',
    Cookie: session.cookie,
    Origin: GARMIN_CONNECT_ORIGIN,
    Referer: `${GARMIN_CONNECT_ORIGIN}/app/home`,
    'User-Agent': process.env.GARMIN_CONNECT_USER_AGENT?.trim() || DEFAULT_USER_AGENT,
    'Accept-Language': process.env.GARMIN_CONNECT_ACCEPT_LANGUAGE?.trim() || 'en-GB,en;q=0.9',
    'Connect-Csrf-Token': session.csrf,
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    ...(contentType ? { 'Content-Type': contentType } : {}),
  }
}

function garminConnectDocumentHeaders(session: GarminConnectSession): HeadersInit {
  return {
    Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    Cookie: session.cookie,
    'User-Agent': process.env.GARMIN_CONNECT_USER_AGENT?.trim() || DEFAULT_USER_AGENT,
    'Accept-Language': process.env.GARMIN_CONNECT_ACCEPT_LANGUAGE?.trim() || 'en-GB,en;q=0.9',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-User': '?1',
    'Sec-Fetch-Dest': 'document',
    'Upgrade-Insecure-Requests': '1',
  }
}

function cookiePairs(header: string): Map<string, string> {
  const pairs = new Map<string, string>()
  for (const part of header.split('; ')) {
    const index = part.indexOf('=')
    if (index <= 0) continue
    pairs.set(part.slice(0, index), part.slice(index + 1))
  }
  return pairs
}

function splitSetCookie(header: string): string[] {
  return header.split(/,(?=\s*[^;,=\s]+=)/g).map(cookie => cookie.trim())
}

function setCookieHeaders(headers: Headers): string[] {
  const withSetCookie = headers as Headers & { getSetCookie?: () => string[] }
  const values = withSetCookie.getSetCookie?.()
  if (values?.length) return values
  const header = headers.get('set-cookie')
  return header ? splitSetCookie(header) : []
}

export function applyGarminSetCookies(session: GarminConnectSession, headers: Headers): void {
  const updates = setCookieHeaders(headers)
  if (updates.length === 0) return
  const pairs = cookiePairs(session.cookie)
  for (const header of updates) {
    const first = header.split(';', 1)[0]
    const index = first.indexOf('=')
    if (index <= 0) continue
    const name = first.slice(0, index)
    const value = first.slice(index + 1)
    if (!value || /;\s*Max-Age=0(?:;|$)/i.test(header)) pairs.delete(name)
    else pairs.set(name, value)
  }
  session.cookie = [...pairs].map(([name, value]) => `${name}=${value}`).join('; ')
}

async function fetchGarminConnectCsrf(session: GarminConnectSession): Promise<string> {
  let last: string | null = null
  for (const path of ['/app/home', '/app/activities']) {
    const shell = await fetchGarminDocument(session, `${GARMIN_CONNECT_ORIGIN}${path}`)
    const csrf = /<meta\s+name=["']csrf-token["']\s+content=["']([^"']+)["']/i.exec(shell.text)?.[1]
    if (csrf) return csrf
    if (!shell.res.ok) last = `Garmin Connect app shell failed: ${shell.res.status}`
    else if (isGarminSignInUrl(shell.url))
      last = 'Garmin Connect app shell landed on sign-in; refresh the browser session'
    else last = 'Garmin Connect app shell did not expose a CSRF token'
  }
  throw new Error(last ?? 'Garmin Connect app shell did not expose a CSRF token')
}

function isRedirect(status: number): boolean {
  return status === 301 || status === 302 || status === 303 || status === 307 || status === 308
}

function redirectTarget(current: string, location: string | null): string | null {
  if (!location) return null
  let target: URL
  try {
    target = new URL(location, current)
  } catch {
    return null
  }
  if (target.protocol !== 'https:') return null
  if (target.hostname !== 'connect.garmin.com' && !target.hostname.endsWith('.garmin.com'))
    return null
  return target.toString()
}

function isGarminSignInUrl(value: string): boolean {
  const url = new URL(value)
  return url.hostname === 'sso.garmin.com' || /(?:sign-?in|login)/i.test(url.pathname)
}

export async function fetchGarminDocument(
  session: GarminConnectSession,
  start: string,
): Promise<{ res: Response; text: string; url: string }> {
  let url = start
  for (let redirects = 0; redirects <= MAX_DOCUMENT_REDIRECTS; redirects++) {
    const res = await fetch(url, {
      headers: garminConnectDocumentHeaders(session),
      redirect: 'manual',
    })
    applyGarminSetCookies(session, res.headers)
    if (!isRedirect(res.status)) return { res, text: await res.text(), url }
    const next = redirectTarget(url, res.headers.get('location'))
    if (!next) return { res, text: await res.text(), url }
    url = next
  }
  throw new Error('Garmin Connect app shell exceeded redirect limit')
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
  applyGarminSetCookies(session, res.headers)
  if (res.status === 401 || res.status === 403)
    throw new Error(`Garmin Connect session rejected (${res.status}); refresh the cookie`)
  if (!res.ok)
    throw new Error(
      `Garmin Connect request failed: ${res.status} ${garminResponseSummary(res, text)}`,
    )
  const type = res.headers.get('content-type') ?? ''
  if (!type.includes('application/json'))
    throw new Error(`Garmin Connect returned non-JSON: ${garminResponseSummary(res, text)}`)
  return JSON.parse(text) as unknown
}
