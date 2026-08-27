import {
  parseWahooAuthorizationCallback,
  type WahooAuthorizationCallback,
  WAHOO_LOCAL_CALLBACK_URI,
  WAHOO_OAUTH_CALLBACK_PATH,
} from '../quartz/util/wahoo-oauth'

const PRIVATE_RESPONSE_HEADERS = {
  'Cache-Control': 'no-store',
  Pragma: 'no-cache',
  'Referrer-Policy': 'no-referrer',
  'X-Content-Type-Options': 'nosniff',
}

function textResponse(body: string, status: number, headers: HeadersInit = {}): Response {
  return new Response(body, {
    status,
    headers: {
      ...PRIVATE_RESPONSE_HEADERS,
      'Content-Type': 'text/plain; charset=utf-8',
      ...headers,
    },
  })
}

export function handleWahooOAuthCallback(request: Request): Response | null {
  const url = new URL(request.url)
  if (url.pathname !== WAHOO_OAUTH_CALLBACK_PATH) return null
  if (request.method !== 'GET') return textResponse('Method not allowed', 405, { Allow: 'GET' })

  let callback: WahooAuthorizationCallback
  try {
    callback = parseWahooAuthorizationCallback(url.searchParams)
  } catch {
    return textResponse('Invalid Wahoo authorization callback', 400)
  }

  const localUrl = new URL(WAHOO_LOCAL_CALLBACK_URI)
  if (callback.kind === 'grant') localUrl.searchParams.set('code', callback.code)
  else {
    localUrl.searchParams.set('error', callback.error)
    if (callback.errorDescription !== null)
      localUrl.searchParams.set('error_description', callback.errorDescription)
  }
  localUrl.searchParams.set('state', callback.state)

  return new Response(null, {
    status: 302,
    headers: { ...PRIVATE_RESPONSE_HEADERS, Location: localUrl.toString() },
  })
}
