const IMMUTABLE_ASSET_CACHE_CONTROL = 'public, max-age=31536000, immutable'
const MANIFEST_CACHE_CONTROL = 'no-store, no-cache, must-revalidate'
const HASHED_STATIC_ASSET_PATTERN =
  /(?:^|\/)[^/]+-[0-9a-f]{8}\.(?:css|js|json)$|\/static\/scripts\/chunks\/[^/]+-[a-z0-9_-]{8,}\.js$/i
const FIRST_PARTY_SCRIPT_PATTERN = /^\/(?:pre|post)script\.js$|^\/static\/scripts\/.+\.js$/i
const WORKER_SCRIPT_PATTERN = /^\/static\/scripts\/(?:.+\.)?worker(?:-[^/]+)?\.js$/i

export function cacheHeadersForStaticAsset(
  pathname: string,
  status: number,
): Record<string, string> {
  if (pathname === '/static/scripts/asset-manifest.json') {
    return { 'Cache-Control': MANIFEST_CACHE_CONTROL }
  }
  if (status >= 200 && status < 400 && HASHED_STATIC_ASSET_PATTERN.test(pathname)) {
    return { 'Cache-Control': IMMUTABLE_ASSET_CACHE_CONTROL }
  }
  return {}
}

export function isolationHeadersForStaticAsset(
  pathname: string,
  status: number,
): Record<string, string> {
  if (status < 200 || status >= 400 || !FIRST_PARTY_SCRIPT_PATTERN.test(pathname)) return {}
  return {
    'Cross-Origin-Resource-Policy': 'same-origin',
    ...(WORKER_SCRIPT_PATTERN.test(pathname)
      ? { 'Cross-Origin-Embedder-Policy': 'require-corp' }
      : {}),
  }
}
