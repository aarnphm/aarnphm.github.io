const ABSOLUTE_URL_SCHEME = /^[A-Za-z][A-Za-z0-9+.-]*:/

function normalizeHostname(hostname: string): string {
  return hostname.trim().toLowerCase().replace(/\.$/, '')
}

export function parseExternalUrl(value: string): URL | undefined {
  const trimmed = value.trim()
  if (trimmed.length === 0) return undefined

  try {
    if (ABSOLUTE_URL_SCHEME.test(trimmed)) return new URL(trimmed)
    if (trimmed.startsWith('//')) return new URL(`https:${trimmed}`)
  } catch {
    return undefined
  }

  return undefined
}

export function hostnameMatches(url: URL | undefined, hostname: string): boolean {
  if (!url) return false
  const actual = normalizeHostname(url.hostname)
  const expected = normalizeHostname(hostname)
  if (expected.length === 0) return false
  return actual === expected || actual.endsWith(`.${expected}`)
}
