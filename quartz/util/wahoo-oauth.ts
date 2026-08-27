export const WAHOO_OAUTH_CALLBACK_PATH = '/oauth/wahoo/callback'
export const DEFAULT_WAHOO_REDIRECT_URI = `https://aarnphm.xyz${WAHOO_OAUTH_CALLBACK_PATH}`
export const WAHOO_LOCAL_CALLBACK_URI = 'http://127.0.0.1:8722/'

export type WahooAuthorizationCallback =
  | { kind: 'grant'; code: string; state: string }
  | { kind: 'error'; error: string; errorDescription: string | null; state: string }

export interface WahooAuthorizationGrant {
  code: string
  state: string
}

function optionalSingleValue(searchParams: URLSearchParams, key: string): string | null {
  const values = searchParams.getAll(key)
  if (values.length > 1) throw new Error(`Wahoo callback repeated ${key}`)
  return values[0] ?? null
}

function requiredSingleValue(searchParams: URLSearchParams, key: string): string {
  const value = optionalSingleValue(searchParams, key)
  if (!value) throw new Error(`Wahoo callback omitted ${key}`)
  return value
}

export function parseWahooAuthorizationCallback(
  searchParams: URLSearchParams,
): WahooAuthorizationCallback {
  const state = requiredSingleValue(searchParams, 'state')
  const code = optionalSingleValue(searchParams, 'code')
  const error = optionalSingleValue(searchParams, 'error')
  const errorDescription = optionalSingleValue(searchParams, 'error_description')

  if (code && error) throw new Error('Wahoo callback included both code and error')
  if (code) {
    if (errorDescription !== null)
      throw new Error('Wahoo callback included error_description without error')
    return { kind: 'grant', code, state }
  }
  if (!error) throw new Error('Wahoo callback omitted code or error')
  return { kind: 'error', error, errorDescription, state }
}

export function requireWahooAuthorizationGrant(
  searchParams: URLSearchParams,
  expectedState: string | null,
): WahooAuthorizationGrant {
  const callback = parseWahooAuthorizationCallback(searchParams)
  if (expectedState !== null && callback.state !== expectedState)
    throw new Error('Wahoo OAuth state mismatch')
  if (callback.kind === 'error') {
    const detail = callback.errorDescription
      ? `${callback.error}: ${callback.errorDescription}`
      : callback.error
    throw new Error(`Wahoo authorization failed: ${detail}`)
  }
  return { code: callback.code, state: callback.state }
}

export function parsePastedWahooAuthorizationGrant(
  value: string,
  registeredRedirectUri: string,
  expectedState: string | null,
): WahooAuthorizationGrant {
  let callbackUrl: URL
  try {
    callbackUrl = new URL(value.trim())
  } catch {
    throw new Error('Pasted Wahoo callback must be a full URL')
  }

  const registeredUrl = new URL(registeredRedirectUri)
  if (
    callbackUrl.origin !== registeredUrl.origin ||
    callbackUrl.pathname !== registeredUrl.pathname ||
    callbackUrl.username ||
    callbackUrl.password ||
    callbackUrl.hash
  ) {
    throw new Error(
      `Pasted Wahoo callback must use ${registeredUrl.origin}${registeredUrl.pathname}`,
    )
  }

  return requireWahooAuthorizationGrant(callbackUrl.searchParams, expectedState)
}

export function resolveWahooRedirectUri(configured: string | undefined): string {
  const url = new URL(configured?.trim() || DEFAULT_WAHOO_REDIRECT_URI)
  if (url.protocol !== 'https:') throw new Error('WAHOO_REDIRECT_URI must use HTTPS')
  if (url.username || url.password || url.search || url.hash)
    throw new Error('WAHOO_REDIRECT_URI cannot include credentials, a query, or a fragment')
  return url.toString()
}
