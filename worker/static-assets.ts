const IMMUTABLE_ASSET_CACHE_CONTROL = 'public, max-age=31536000, immutable'
const MANIFEST_CACHE_CONTROL = 'no-store, no-cache, must-revalidate'
const HASHED_STATIC_ASSET_PATTERN =
  /(?:^|\/)[^/]+-[0-9a-f]{8}\.(?:css|js|json)$|\/static\/scripts\/chunks\/[^/]+-[a-z0-9_-]{8,}\.js$/i

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
