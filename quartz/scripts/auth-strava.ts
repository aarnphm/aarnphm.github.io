import { spawn } from 'node:child_process'
import fs from 'node:fs/promises'
import http from 'node:http'
import { upsertEnvLine, removeEnvKeys } from '../util/env-file'
import { joinSegments, QUARTZ } from '../util/path'
import { isRecord, readString } from '../util/type-guards'

const PORT = 8721
const REDIRECT = `http://localhost:${PORT}`
const READ_SCOPE = 'activity:read_all,profile:read_all'
const WRITE_SCOPE = `${READ_SCOPE},activity:write`
const AUTHORIZE_URL = 'https://www.strava.com/oauth/authorize'
const TOKEN_URL = 'https://www.strava.com/oauth/token'
const REVOKE_URL = 'https://www.strava.com/oauth/revoke'
const ENV_FILE = '.env'
const cacheFile = joinSegments(QUARTZ, '.quartz-cache', 'strava.json')
const STRAVA_TOKEN_KEYS = ['STRAVA_REFRESH_TOKEN', 'STRAVA_ACCESS_TOKEN'] as const

type TokenTypeHint = 'access_token' | 'refresh_token'

interface TokenResponse {
  access_token: string
  refresh_token: string
  expires_at: number
}

interface RevokeToken {
  token: string
  hint: TokenTypeHint
}

interface AuthorizationGrant {
  code: string
  scope: string
}

function authorizeUrl(clientId: string, scope: string): string {
  const params = new URLSearchParams({
    client_id: clientId,
    response_type: 'code',
    redirect_uri: REDIRECT,
    approval_prompt: 'force',
    scope,
  })
  return `${AUTHORIZE_URL}?${params.toString()}`
}

function openBrowser(url: string): void {
  const child = spawn('open', [url], { detached: true, stdio: 'ignore' })
  child.on('error', () => {})
  child.unref()
}

function waitForCode(): Promise<AuthorizationGrant> {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url ?? '', REDIRECT)
      const code = url.searchParams.get('code')
      const scope = url.searchParams.get('scope') ?? ''
      const error = url.searchParams.get('error')
      res.end(
        error
          ? `Strava authorization failed: ${error}`
          : 'Strava authorized. Close this tab and return to the terminal.',
      )
      server.close()
      if (error) reject(new Error(error))
      else if (code) resolve({ code, scope })
      else reject(new Error('no authorization code in redirect'))
    })
    server.listen(PORT, () => console.log(`[strava] waiting for redirect on ${REDIRECT} ...`))
  })
}

function assertGrantedScope(requested: string, granted: string): void {
  if (!granted) return
  const grantedScopes = new Set(granted.split(',').filter(Boolean))
  const missing = requested.split(',').filter(scope => !grantedScopes.has(scope))
  if (missing.length > 0) throw new Error(`authorization missing scope(s): ${missing.join(',')}`)
}

async function exchange(
  clientId: string,
  clientSecret: string,
  code: string,
): Promise<TokenResponse> {
  const body = new URLSearchParams({
    client_id: clientId,
    client_secret: clientSecret,
    code,
    grant_type: 'authorization_code',
  })
  const res = await fetch(TOKEN_URL, { method: 'POST', body })
  if (!res.ok) throw new Error(`token exchange failed: ${res.status} ${await res.text()}`)
  return (await res.json()) as TokenResponse
}

async function writeRefreshToken(refreshToken: string): Promise<void> {
  await upsertEnvLine(ENV_FILE, 'STRAVA_REFRESH_TOKEN', refreshToken)
}

async function readCachedRefreshToken(): Promise<string | undefined> {
  try {
    const parsed: unknown = JSON.parse(await fs.readFile(cacheFile, 'utf8'))
    if (!isRecord(parsed) || !isRecord(parsed.auth)) return undefined
    return readString(parsed.auth, 'refreshToken')
  } catch {
    return undefined
  }
}

async function resolveRevokeToken(): Promise<RevokeToken> {
  const cachedRefreshToken = await readCachedRefreshToken()
  if (cachedRefreshToken) return { token: cachedRefreshToken, hint: 'refresh_token' }
  const refreshToken = process.env.STRAVA_REFRESH_TOKEN
  if (refreshToken) return { token: refreshToken, hint: 'refresh_token' }
  const accessToken = process.env.STRAVA_ACCESS_TOKEN
  if (accessToken) return { token: accessToken, hint: 'access_token' }
  throw new Error('set STRAVA_REFRESH_TOKEN or STRAVA_ACCESS_TOKEN before revoking')
}

async function revoke(
  clientId: string,
  clientSecret: string,
  { token, hint }: RevokeToken,
): Promise<void> {
  const body = new URLSearchParams({ token, token_type_hint: hint })
  const res = await fetch(REVOKE_URL, {
    method: 'POST',
    headers: {
      Authorization: `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body,
  })
  if (!res.ok) throw new Error(`token revoke failed: ${res.status} ${await res.text()}`)
}

async function main(): Promise<void> {
  const clientId = process.env.STRAVA_CLIENT_ID
  const clientSecret = process.env.STRAVA_CLIENT_SECRET
  if (!clientId || !clientSecret) {
    throw new Error(
      'set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in .env first (strava.com/settings/api)',
    )
  }

  if (process.argv.includes('--revoke')) {
    await revoke(clientId, clientSecret, await resolveRevokeToken())
    await removeEnvKeys(ENV_FILE, STRAVA_TOKEN_KEYS)
    await fs.rm(cacheFile, { force: true })
    console.log('\n[strava] revoked token and removed local Strava auth/cache.')
    return
  }

  const scope = process.argv.includes('--write') ? WRITE_SCOPE : READ_SCOPE
  const url = authorizeUrl(clientId, scope)
  console.log(`\n[strava] opening browser to authorize (scope: ${scope}).`)
  console.log(`if it does not open, visit:\n${url}\n`)
  openBrowser(url)

  const grant = await waitForCode()
  assertGrantedScope(scope, grant.scope)
  const token = await exchange(clientId, clientSecret, grant.code)
  await writeRefreshToken(token.refresh_token)
  console.log('\n[strava] authorized. STRAVA_REFRESH_TOKEN written to .env.')
  console.log('now run:  pnpm strava:sync\n')
}

main().catch(err => {
  console.error(`[strava] auth failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
