import { spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import http from 'node:http'
import { upsertEnvLine } from '../util/env-file'
import {
  DEFAULT_WAHOO_AUTHORIZE_URL,
  DEFAULT_WAHOO_TOKEN_URL,
  exchangeWahooAuthorizationCode,
  WAHOO_OAUTH_SCOPES,
} from '../util/wahoo-cloud'
import {
  parseWahooAuthorizationCallback,
  resolveWahooRedirectUri,
  WAHOO_LOCAL_CALLBACK_URI,
} from '../util/wahoo-oauth'

interface AuthorizationGrant {
  code: string
  state: string
}

function authorizationUrl(clientId: string, redirect: string, state: string): string {
  const url = new URL(process.env.WAHOO_AUTHORIZE_URL?.trim() || DEFAULT_WAHOO_AUTHORIZE_URL)
  url.search = new URLSearchParams({
    client_id: clientId,
    redirect_uri: redirect,
    scope: WAHOO_OAUTH_SCOPES.join(' '),
    response_type: 'code',
    state,
  }).toString()
  return url.toString()
}

function openBrowser(url: string): void {
  const child = spawn('open', [url], { detached: true, stdio: 'ignore' })
  child.on('error', () => {})
  child.unref()
}

function waitForGrant(expectedState: string): Promise<AuthorizationGrant> {
  const localUrl = new URL(WAHOO_LOCAL_CALLBACK_URI)
  const port = Number(localUrl.port)
  return new Promise((resolveGrant, reject) => {
    const server = http.createServer((request, response) => {
      const url = new URL(request.url ?? '/', localUrl)
      if (request.method !== 'GET' || url.pathname !== '/') {
        response.writeHead(404, { 'Cache-Control': 'no-store' })
        response.end('Not found')
        return
      }

      try {
        const callback = parseWahooAuthorizationCallback(url.searchParams)
        if (callback.state !== expectedState) throw new Error('Wahoo OAuth state mismatch')
        if (callback.kind === 'error') {
          const detail = callback.errorDescription
            ? `${callback.error}: ${callback.errorDescription}`
            : callback.error
          throw new Error(`Wahoo authorization failed: ${detail}`)
        }
        response.writeHead(200, {
          'Cache-Control': 'no-store',
          'Content-Type': 'text/plain; charset=utf-8',
        })
        response.end('Wahoo authorized. Close this tab and return to the terminal.')
        server.close()
        resolveGrant({ code: callback.code, state: callback.state })
      } catch (error) {
        response.writeHead(400, {
          'Cache-Control': 'no-store',
          'Content-Type': 'text/plain; charset=utf-8',
        })
        response.end('Wahoo authorization failed. Return to the terminal.')
        server.close()
        reject(error)
      }
    })
    server.on('error', reject)
    server.listen(port, localUrl.hostname, () =>
      console.log(`[wahoo] waiting for redirect on ${WAHOO_LOCAL_CALLBACK_URI} ...`),
    )
  })
}

function assertScopes(scope: string | null): void {
  if (!scope) return
  const granted = new Set(scope.split(/[ ,]+/).filter(Boolean))
  const missing = WAHOO_OAUTH_SCOPES.filter(value => !granted.has(value))
  if (missing.length > 0)
    throw new Error(`Wahoo authorization missing scopes: ${missing.join(', ')}`)
}

async function main(): Promise<void> {
  const clientId = process.env.WAHOO_CLIENT_ID?.trim()
  const clientSecret = process.env.WAHOO_CLIENT_SECRET?.trim()
  if (!clientId || !clientSecret)
    throw new Error('set WAHOO_CLIENT_ID and WAHOO_CLIENT_SECRET in .env first')
  const redirect = resolveWahooRedirectUri(process.env.WAHOO_REDIRECT_URI)
  const state = randomBytes(32).toString('hex')
  const url = authorizationUrl(clientId, redirect, state)
  const grantPromise = waitForGrant(state)
  console.log(`\n[wahoo] opening browser with scopes: ${WAHOO_OAUTH_SCOPES.join(' ')}`)
  console.log(`if it does not open, visit:\n${url}\n`)
  openBrowser(url)
  const grant = await grantPromise
  const token = await exchangeWahooAuthorizationCode(
    clientId,
    clientSecret,
    grant.code,
    redirect,
    process.env.WAHOO_TOKEN_URL?.trim() || DEFAULT_WAHOO_TOKEN_URL,
  )
  assertScopes(token.scope)
  await upsertEnvLine('.env', 'WAHOO_REFRESH_TOKEN', token.refreshToken)
  console.log('\n[wahoo] authorized. WAHOO_REFRESH_TOKEN written to .env.')
  console.log('now run: pnpm health:wahoo\n')
}

main().catch(error => {
  console.error(`[wahoo] auth failed: ${error instanceof Error ? error.message : error}`)
  process.exit(1)
})
