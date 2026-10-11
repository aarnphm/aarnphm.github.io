const canonicalPattern = /<link rel="canonical" href="([^"]*)">/

// Callers want the rendered page. Without an explicit preference the worker's Accept
// negotiation treats agent user agents (incl. the Claude app browser) as markdown clients.
const htmlAccept = 'text/html, */*;q=0.8'

export async function fetchCanonical(url: URL, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers)
  if (!headers.has('Accept')) headers.set('Accept', htmlAccept)
  const response = await fetch(url, { ...init, headers })
  if (!response.headers.get('content-type')?.startsWith('text/html')) return response

  const html = await response.clone().text()
  const canonical = html.match(canonicalPattern)?.[1]
  if (!canonical) return response
  const signal = init?.signal
  return fetch(new URL(canonical, url), { headers, ...(signal ? { signal } : {}) })
}
