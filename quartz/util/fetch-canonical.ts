const canonicalPattern = /<link rel="canonical" href="([^"]*)">/

export async function fetchCanonical(url: URL, init?: RequestInit): Promise<Response> {
  const response = await fetch(url, init)
  if (!response.headers.get('content-type')?.startsWith('text/html')) return response

  const html = await response.clone().text()
  const canonical = html.match(canonicalPattern)?.[1]
  if (!canonical) return response
  const signal = init?.signal
  return fetch(new URL(canonical, url), signal ? { signal } : undefined)
}
