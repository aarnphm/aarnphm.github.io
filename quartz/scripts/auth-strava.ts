import { exec } from 'node:child_process'
import fs from 'node:fs/promises'
import http from 'node:http'

const PORT = 8721
const REDIRECT = `http://localhost:${PORT}`
const SCOPE = 'activity:read_all,profile:read_all'

interface TokenResponse {
  access_token: string
  refresh_token: string
  expires_at: number
}

function authorizeUrl(clientId: string): string {
  const params = new URLSearchParams({
    client_id: clientId,
    response_type: 'code',
    redirect_uri: REDIRECT,
    approval_prompt: 'force',
    scope: SCOPE,
  })
  return `https://www.strava.com/oauth/authorize?${params.toString()}`
}

function waitForCode(): Promise<string> {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url ?? '', REDIRECT)
      const code = url.searchParams.get('code')
      const error = url.searchParams.get('error')
      res.end(
        error
          ? `Strava authorization failed: ${error}`
          : 'Strava authorized. Close this tab and return to the terminal.',
      )
      server.close()
      if (error) reject(new Error(error))
      else if (code) resolve(code)
      else reject(new Error('no authorization code in redirect'))
    })
    server.listen(PORT, () => console.log(`[strava] waiting for redirect on ${REDIRECT} ...`))
  })
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
  const res = await fetch('https://www.strava.com/oauth/token', { method: 'POST', body })
  if (!res.ok) throw new Error(`token exchange failed: ${res.status} ${await res.text()}`)
  return (await res.json()) as TokenResponse
}

async function writeRefreshToken(refreshToken: string): Promise<void> {
  let content = ''
  try {
    content = await fs.readFile('.env', 'utf8')
  } catch {
    content = ''
  }
  const line = `STRAVA_REFRESH_TOKEN=${refreshToken}`
  content = /^STRAVA_REFRESH_TOKEN=.*$/m.test(content)
    ? content.replace(/^STRAVA_REFRESH_TOKEN=.*$/m, line)
    : `${content.trimEnd()}\n${line}\n`
  await fs.writeFile('.env', content)
}

async function main(): Promise<void> {
  const clientId = process.env.STRAVA_CLIENT_ID
  const clientSecret = process.env.STRAVA_CLIENT_SECRET
  if (!clientId || !clientSecret) {
    throw new Error(
      'set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in .env first (strava.com/settings/api)',
    )
  }

  const url = authorizeUrl(clientId)
  console.log(`\n[strava] opening browser to authorize (scope: ${SCOPE}).`)
  console.log(`if it does not open, visit:\n${url}\n`)
  exec(`open "${url}"`, () => {})

  const code = await waitForCode()
  const token = await exchange(clientId, clientSecret, code)
  await writeRefreshToken(token.refresh_token)
  console.log('\n[strava] authorized. STRAVA_REFRESH_TOKEN written to .env.')
  console.log('now run:  pnpm strava:sync\n')
}

main().catch(err => {
  console.error(`[strava] auth failed: ${err instanceof Error ? err.message : err}`)
  process.exit(1)
})
