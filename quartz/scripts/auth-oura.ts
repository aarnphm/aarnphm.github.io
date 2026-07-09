import { spawn } from 'node:child_process'
import http from 'node:http'
import path from 'node:path'
import readline from 'node:readline/promises'
import { pathToFileURL } from 'node:url'
import { upsertEnvLine } from '../util/env-file'

const REDIRECT = process.env.OURA_REDIRECT_URI ?? 'https://aarnphm.xyz'
const SCOPE =
  process.env.OURA_SCOPE ??
  'email personal daily heartrate tag workout session spo2 ring_configuration stress heart_health'
const LOCAL_PORT = 8722
const ENV_FILE = '.env'

interface TokenResponse {
  access_token: string
  refresh_token: string
  expires_in: number
}

function authorizeUrl(clientId: string): string {
  const params = new URLSearchParams({
    client_id: clientId,
    response_type: 'code',
    redirect_uri: REDIRECT,
    scope: SCOPE,
  })
  return `https://cloud.ouraring.com/oauth/authorize?${params.toString()}`
}

export function openBrowserArgs(url: string): string[] {
  return [url]
}

function openBrowser(url: string): void {
  const child = spawn('open', openBrowserArgs(url), { detached: true, stdio: 'ignore' })
  child.on('error', () => {})
  child.unref()
}

const isLocal = (uri: string): boolean => uri.startsWith('http://localhost')

function catchCodeOnLocalhost(): Promise<string> {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url ?? '', REDIRECT)
      const code = url.searchParams.get('code')
      const error = url.searchParams.get('error')
      res.end(
        error
          ? `Oura authorization failed: ${error}`
          : 'Oura authorized. Close this tab and return to the terminal.',
      )
      server.close()
      if (error) reject(new Error(error))
      else if (code) resolve(code)
      else reject(new Error('no authorization code in redirect'))
    })
    const port = Number(new URL(REDIRECT).port) || LOCAL_PORT
    server.listen(port, () => console.log(`[oura] waiting for redirect on ${REDIRECT} ...`))
  })
}

function extractCode(input: string): string {
  const trimmed = input.trim()
  if (trimmed.includes('code=')) {
    const code = new URLSearchParams(trimmed.slice(trimmed.indexOf('?') + 1)).get('code')
    if (code) return code
  }
  return trimmed
}

async function promptForCode(): Promise<string> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout })
  const answer = await rl.question(
    `\n[oura] after approving you land on ${REDIRECT}/?code=…\n[oura] paste that full URL (or just the code): `,
  )
  rl.close()
  const code = extractCode(answer)
  if (!code) throw new Error('no authorization code provided')
  return code
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
    redirect_uri: REDIRECT,
  })
  const res = await fetch('https://api.ouraring.com/oauth/token', { method: 'POST', body })
  if (!res.ok) throw new Error(`token exchange failed: ${res.status} ${await res.text()}`)
  return (await res.json()) as TokenResponse
}

export async function writeRefreshToken(refreshToken: string, envFile = ENV_FILE): Promise<void> {
  await upsertEnvLine(envFile, 'OURA_REFRESH_TOKEN', refreshToken)
}

async function main(): Promise<void> {
  const clientId = process.env.OURA_CLIENT_ID
  const clientSecret = process.env.OURA_CLIENT_SECRET
  if (!clientId || !clientSecret) {
    throw new Error(
      'set OURA_CLIENT_ID and OURA_CLIENT_SECRET in .env first (cloud.ouraring.com → OAuth Applications)',
    )
  }

  const url = authorizeUrl(clientId)
  console.log(
    `\n[oura] redirect_uri=${REDIRECT} (must match a Redirect URI registered on the Oura app).`,
  )
  console.log(`[oura] opening browser to authorize (scope: ${SCOPE}).`)
  console.log(`if it does not open, visit:\n${url}\n`)
  openBrowser(url)

  const code = isLocal(REDIRECT) ? await catchCodeOnLocalhost() : await promptForCode()
  const token = await exchange(clientId, clientSecret, code)
  if (!token.refresh_token)
    throw new Error(`no refresh_token in Oura response: ${JSON.stringify(token)}`)
  await writeRefreshToken(token.refresh_token)
  console.log('\n[oura] authorized. OURA_REFRESH_TOKEN written to .env.')
  console.log('now run:  pnpm oura:sync\n')
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch(err => {
    console.error(`[oura] auth failed: ${err instanceof Error ? err.message : err}`)
    process.exit(1)
  })
}
