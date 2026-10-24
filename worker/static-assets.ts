const IMMUTABLE_ASSET_CACHE_CONTROL = 'public, max-age=31536000, immutable'
const MANIFEST_CACHE_CONTROL = 'no-store, no-cache, must-revalidate'
const VOLATILE_ASSET_CACHE_CONTROL = 'no-store, no-cache, must-revalidate'
const HASHED_STATIC_ASSET_PATTERN =
  /(?:^|\/)[^/]+-[0-9a-f]{8}\.(?:css|js|mjs|json|wasm|bin|gz|zst|tar)$|\/static\/scripts\/chunks\/[^/]+-[a-z0-9_-]{8,}\.js$/i
const NOTEBOOK_RUNTIME_BINARY_ASSET_PATTERN =
  /^\/static\/scripts\/notebook-runtimes\/.+\.(?:wasm|bin|gz|zst|tar|rlib\.gz|so\.gz)$/i
const FIRST_PARTY_STYLESHEET_PATTERN =
  /^\/index\.css$|^\/static\/(?:component|resource-style(?:-\d+)?)\.css$/i
const FIRST_PARTY_SCRIPT_PATTERN =
  /^\/(?:pre|post)script\.js$|^\/static\/resource-after(?:-\d+)?\.js$|^\/static\/scripts\/.+\.(?:js|mjs)$/i
const WORKER_SCRIPT_PATTERN = /^\/static\/scripts\/(?:.+[./-])?worker(?:-[^/]+)?\.(?:js|mjs)$/i
// Embedding generations live under their content digest; only the root manifest moves.
const EMBEDDING_GENERATION_PATTERN = /^\/embeddings\/(?:pdf|images)\/[0-9a-f]{64}\/[^/]+$/i

function isHashNamedStaticAsset(pathname: string): boolean {
  return HASHED_STATIC_ASSET_PATTERN.test(pathname) || EMBEDDING_GENERATION_PATTERN.test(pathname)
}

function isVolatileGeneratedStaticAsset(pathname: string): boolean {
  return (
    !isHashNamedStaticAsset(pathname) &&
    !NOTEBOOK_RUNTIME_BINARY_ASSET_PATTERN.test(pathname) &&
    (FIRST_PARTY_STYLESHEET_PATTERN.test(pathname) || FIRST_PARTY_SCRIPT_PATTERN.test(pathname))
  )
}

export function cacheHeadersForStaticAsset(
  pathname: string,
  status: number,
): Record<string, string> {
  if (pathname === '/static/scripts/asset-manifest.json') {
    return { 'Cache-Control': MANIFEST_CACHE_CONTROL }
  }
  if (status >= 200 && status < 400 && isHashNamedStaticAsset(pathname)) {
    return { 'Cache-Control': IMMUTABLE_ASSET_CACHE_CONTROL }
  }
  if (status >= 200 && status < 400 && isVolatileGeneratedStaticAsset(pathname)) {
    return { 'Cache-Control': VOLATILE_ASSET_CACHE_CONTROL }
  }
  return {}
}

export function shouldBypassStaticAssetCache(pathname: string): boolean {
  return (
    pathname === '/static/scripts/asset-manifest.json' || isVolatileGeneratedStaticAsset(pathname)
  )
}

export function requestWithoutCache(request: Request): Request {
  const headers = new Headers(request.headers)
  headers.delete('If-None-Match')
  headers.delete('If-Modified-Since')
  return new Request(request, { headers })
}

export function requestWithoutStaticAssetCache(request: Request, pathname: string): Request {
  return shouldBypassStaticAssetCache(pathname) ? requestWithoutCache(request) : request
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
