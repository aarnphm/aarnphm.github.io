import LFS_CONFIG from '../.lfsconfig.txt'

const MIME = 'application/vnd.git-lfs+json'
export const KEEP_HEADERS = 'Cache-Control'

export type CfCacheStorage = CacheStorage & { readonly default: Cache }

function splitFirst(str: string, delim: string): [string, string?] {
  const idx = str.indexOf(delim)
  return idx === -1 ? [str] : [str.slice(0, idx), str.slice(idx + 1)]
}

function getLfsUrl(config: string): URL | null {
  // TODO: better parser, accept remote.<remote>.lfsurl...
  let section: string | undefined
  for (const raw of config.split('\n')) {
    const line = splitFirst(raw, ';')[0].trim()
    if (line.startsWith('[') && line.endsWith(']')) {
      section = line.slice(1, -1)
    } else if (section === 'lfs') {
      const [key, val] = splitFirst(line, '=')
      if (val === undefined) return null
      if (key.trimEnd() === 'url') return new URL(val.trimStart())
    }
  }
  return null
}

function extendPath(url: URL | string, path: string): URL {
  const u = typeof url === 'string' ? new URL(url) : new URL(url.toString())
  u.pathname = u.pathname.replace(/\/?$/, `/${path}`)
  return u
}

function headersToObject(headers: Headers): Record<string, string> {
  const o: Record<string, string> = {}
  headers.forEach((value, key) => {
    o[key] = value
  })
  return o
}

async function getObjectAction(
  lfsUrl: URL,
  info: { hash_algo: string; oid: string; size: number },
): Promise<{ href: string; header?: Record<string, string> } | null> {
  const url = extendPath(lfsUrl, 'objects/batch')
  const headers: Record<string, string> = { Accept: MIME, 'Content-Type': MIME }
  if (url.username || url.password) {
    headers['Authorization'] = `Basic ${btoa(`${url.username}:${url.password}`)}`
    url.username = url.password = ''
  }
  const body = JSON.stringify({
    operation: 'download',
    transfers: ['basic'],
    objects: [{ oid: info.oid, size: info.size }],
    hash_algo: info.hash_algo,
  })
  const res = await fetch(url.toString(), { method: 'POST', headers, body })
  if (res.ok && res.headers.get('Content-Type')?.startsWith(MIME)) {
    const batch: any = await res.json()
    const obj = batch.objects?.[0]
    if ((!batch.transfer || batch.transfer === 'basic') && obj?.authenticated)
      return obj.actions.download
  }
  return null
}

export async function getModelObject(
  ctx: ExecutionContext,
  bucket: R2Bucket,
  key: string,
  request: Request,
  cacheControl: string,
): Promise<Response> {
  const head = request.method === 'HEAD'
  const cache = (caches as CfCacheStorage).default
  const cacheKey = new Request(request.url)
  if (!head) {
    const cached = await cache.match(cacheKey)
    if (cached) return cached
  }
  const object = head ? await bucket.head(key) : await bucket.get(key)
  if (!object)
    return new Response('not found', {
      status: 404,
      headers: { 'Access-Control-Allow-Origin': '*' },
    })
  const headers = new Headers()
  object.writeHttpMetadata(headers)
  if (object.httpEtag) headers.set('ETag', object.httpEtag)
  headers.set('Cache-Control', cacheControl)
  headers.set('Access-Control-Allow-Origin', '*')
  const resp = new Response(head ? null : (object as R2ObjectBody).body, { headers })
  if (!head) ctx.waitUntil(cache.put(cacheKey, resp.clone()))
  return resp
}

export async function getObjectFromLFS(
  info: { hash_algo: string; oid: string; size: number },
  request: Request,
): Promise<Response> {
  const lfsUrl = getLfsUrl(LFS_CONFIG)
  if (!lfsUrl) return new Response(null, { status: 500 })
  const action = await getObjectAction(lfsUrl, info)
  if (!action) return new Response(null, { status: 500 })
  const headers = action.header
    ? { ...action.header, ...headersToObject(request.headers) }
    : headersToObject(request.headers)
  return fetch(action.href, { method: request.method, headers, cf: { cacheTtl: 31536000 } })
}
