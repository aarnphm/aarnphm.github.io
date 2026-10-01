import { isRecord } from './type-guards'

export interface ArenaReaderAuthEnv {
  SESSION_SECRET?: string
  ARENA_READER_DEV?: string
  ARENA_OWNER_LOGIN?: string
}

export interface ArenaReaderIdentity {
  subject: string
}

export interface ArenaResourceScope {
  articleId: string
  snapshotId: string
  resourceId: string
}

export const ARENA_READER_SESSION_COOKIE = '__Host-ARENA_READER'
const sessionLifetimeSeconds = 30 * 24 * 60 * 60
const resourceLifetimeSeconds = 60 * 60
const encoder = new TextEncoder()
const decoder = new TextDecoder('utf-8', { fatal: true })

function ownerLogin(env: ArenaReaderAuthEnv): string {
  return (env.ARENA_OWNER_LOGIN ?? 'aarnphm').trim().toLowerCase()
}

function isOwner(login: string, env: ArenaReaderAuthEnv): boolean {
  return login.length > 0 && login.toLowerCase() === ownerLogin(env)
}

function encodeBase64Url(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
    .replaceAll('+', '-')
    .replaceAll('/', '_')
    .replace(/=+$/, '')
}

function decodeBase64Url(value: string): Uint8Array<ArrayBuffer> | null {
  if (!/^[A-Za-z0-9_-]+$/.test(value) || value.length % 4 === 1) return null
  const encoded = value.replaceAll('-', '+').replaceAll('_', '/')
  const padded = encoded.padEnd(encoded.length + ((4 - (encoded.length % 4)) % 4), '=')
  try {
    const decoded = Uint8Array.from(atob(padded), char => char.charCodeAt(0))
    return encodeBase64Url(decoded) === value ? decoded : null
  } catch {
    return null
  }
}

function signingKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign', 'verify'],
  )
}

async function signClaims(
  secret: string,
  purpose: 'session' | 'resource',
  claims: Record<string, unknown>,
): Promise<string> {
  const payload = encodeBase64Url(encoder.encode(JSON.stringify(claims)))
  const signature = await crypto.subtle.sign(
    'HMAC',
    await signingKey(secret),
    encoder.encode(`arena-reader:${purpose}:v1:${payload}`),
  )
  return `${payload}.${encodeBase64Url(new Uint8Array(signature))}`
}

async function verifyClaims(
  secret: string | undefined,
  purpose: 'session' | 'resource',
  token: string,
): Promise<Record<string, unknown> | null> {
  if (!secret?.trim() || token.length > 4096) return null
  const parts = token.split('.')
  if (parts.length !== 2) return null
  const [payload, encodedSignature] = parts
  const bytes = decodeBase64Url(payload)
  const signature = decodeBase64Url(encodedSignature)
  if (!bytes || signature?.byteLength !== 32) return null
  const valid = await crypto.subtle.verify(
    'HMAC',
    await signingKey(secret),
    signature,
    encoder.encode(`arena-reader:${purpose}:v1:${payload}`),
  )
  if (!valid) return null
  try {
    const claims: unknown = JSON.parse(decoder.decode(bytes))
    return isRecord(claims) ? claims : null
  } catch {
    return null
  }
}

function hasValidLifetime(
  claims: Record<string, unknown>,
  maxAgeSeconds: number,
  now: number,
): boolean {
  const { issuedAt, expiresAt } = claims
  const nowSeconds = Math.floor(now / 1000)
  return (
    typeof issuedAt === 'number' &&
    Number.isSafeInteger(issuedAt) &&
    typeof expiresAt === 'number' &&
    Number.isSafeInteger(expiresAt) &&
    issuedAt <= nowSeconds &&
    expiresAt > nowSeconds &&
    expiresAt > issuedAt &&
    expiresAt - issuedAt <= maxAgeSeconds
  )
}

/** Call only with the user returned by the verified GitHub OAuth callback. */
export async function createArenaReaderSessionCookie(
  request: Request,
  env: ArenaReaderAuthEnv,
  user: { id: number; login: string },
  now = Date.now(),
): Promise<string | null> {
  if (
    !env.SESSION_SECRET?.trim() ||
    !Number.isSafeInteger(user.id) ||
    user.id <= 0 ||
    !isOwner(user.login, env) ||
    new URL(request.url).protocol !== 'https:'
  ) {
    return null
  }
  const issuedAt = Math.floor(now / 1000)
  const token = await signClaims(env.SESSION_SECRET, 'session', {
    id: user.id,
    login: user.login,
    origin: new URL(request.url).origin,
    issuedAt,
    expiresAt: issuedAt + sessionLifetimeSeconds,
  })
  return `${ARENA_READER_SESSION_COOKIE}=${token}; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=${sessionLifetimeSeconds}`
}

export async function getArenaReaderIdentity(
  request: Request,
  env: ArenaReaderAuthEnv,
  now = Date.now(),
): Promise<ArenaReaderIdentity | null> {
  const url = new URL(request.url)
  if (
    env.ARENA_READER_DEV === 'true' &&
    ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
  ) {
    return { subject: `dev:${ownerLogin(env)}` }
  }
  if (url.protocol !== 'https:') return null
  const cookies = (request.headers.get('Cookie') ?? '')
    .split(';')
    .map(value => value.trim())
    .filter(value => value.startsWith(`${ARENA_READER_SESSION_COOKIE}=`))
  if (cookies.length !== 1) return null
  const token = cookies[0].slice(ARENA_READER_SESSION_COOKIE.length + 1)
  const claims = await verifyClaims(env.SESSION_SECRET, 'session', token)
  if (
    !claims ||
    !hasValidLifetime(claims, sessionLifetimeSeconds, now) ||
    typeof claims.id !== 'number' ||
    !Number.isSafeInteger(claims.id) ||
    claims.id <= 0 ||
    typeof claims.login !== 'string' ||
    !isOwner(claims.login, env) ||
    claims.origin !== url.origin
  ) {
    return null
  }
  return { subject: `github:${claims.id}` }
}

/** Flashcards share the owner session; rows stay keyed by the lowercased owner login. */
export async function getOwnerSessionLogin(
  request: Request,
  env: ArenaReaderAuthEnv,
  now = Date.now(),
): Promise<string | null> {
  return (await getArenaReaderIdentity(request, env, now)) ? ownerLogin(env) : null
}

export function getArenaReaderLoginUrl(request: Request): string {
  const source = new URL(request.url)
  const returnTo =
    source.pathname === '/arena/feed' || source.pathname === '/arena/feed/'
      ? `/arena/feed${source.search}${source.hash}`
      : '/arena/feed'
  const login = new URL('/comments/github/login', source.origin)
  login.searchParams.set('returnTo', returnTo)
  return login.href
}

export function isArenaReaderMutationAllowed(request: Request): boolean {
  const fetchSite = request.headers.get('Sec-Fetch-Site')
  return (
    request.headers.get('Origin') === new URL(request.url).origin &&
    (fetchSite === null || fetchSite === 'same-origin')
  )
}

function validResourceScope(scope: ArenaResourceScope): boolean {
  return [scope.articleId, scope.snapshotId, scope.resourceId].every(
    value => typeof value === 'string' && /^[A-Za-z0-9_-]{1,160}$/.test(value),
  )
}

export async function createArenaResourceCapability(
  env: ArenaReaderAuthEnv,
  scope: ArenaResourceScope,
  expiresInSeconds = resourceLifetimeSeconds,
  now = Date.now(),
): Promise<string> {
  if (!env.SESSION_SECRET?.trim()) throw new Error('Arena reader SESSION_SECRET is not configured')
  if (!validResourceScope(scope)) throw new Error('Invalid Arena resource scope')
  if (
    !Number.isSafeInteger(expiresInSeconds) ||
    expiresInSeconds < 1 ||
    expiresInSeconds > resourceLifetimeSeconds
  ) {
    throw new Error('Arena resource capability lifetime must be between 1 and 3600 seconds')
  }
  const issuedAt = Math.floor(now / 1000)
  return signClaims(env.SESSION_SECRET, 'resource', {
    articleId: scope.articleId,
    snapshotId: scope.snapshotId,
    resourceId: scope.resourceId,
    issuedAt,
    expiresAt: issuedAt + expiresInSeconds,
  })
}

export async function verifyArenaResourceCapability(
  env: ArenaReaderAuthEnv,
  token: string,
  scope: ArenaResourceScope,
  now = Date.now(),
): Promise<boolean> {
  if (!validResourceScope(scope)) return false
  const claims = await verifyClaims(env.SESSION_SECRET, 'resource', token)
  return (
    claims !== null &&
    hasValidLifetime(claims, resourceLifetimeSeconds, now) &&
    claims.articleId === scope.articleId &&
    claims.snapshotId === scope.snapshotId &&
    claims.resourceId === scope.resourceId
  )
}
