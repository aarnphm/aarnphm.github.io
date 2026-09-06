const MEMO_CONTENT_TYPES: Record<string, string> = {
  m4a: 'audio/mp4',
  mp3: 'audio/mpeg',
  wav: 'audio/wav',
  ogg: 'audio/ogg',
  flac: 'audio/flac',
  aac: 'audio/aac',
}

export function lfsAssetContentType(pathname: string): string | undefined {
  const extension = pathname.split('.').at(-1)?.toLowerCase()
  if (extension === 'pdf') return 'application/pdf'
  if (!pathname.startsWith('/triathlon/memos/') || !extension) return undefined
  return Object.hasOwn(MEMO_CONTENT_TYPES, extension) ? MEMO_CONTENT_TYPES[extension] : undefined
}

export function lfsPointerRequest(request: Request): Request {
  return new Request(request.url, { headers: { 'Accept-Encoding': 'identity' } })
}

export async function getObjectInfo(
  response: Response,
): Promise<{ hash_algo: string; oid: string; size: number } | null> {
  const reader = response.body?.getReader()
  if (!reader) return null
  const bytes = new Uint8Array(1024)
  let length = 0
  try {
    while (length < bytes.length) {
      const { value, done } = await reader.read()
      if (done) break
      if (value.length > bytes.length - length) return null
      bytes.set(value, length)
      length += value.length
    }
    const text = new TextDecoder().decode(bytes.subarray(0, length))
    const match = text.match(
      /^version https:\/\/git-lfs.github.com\/spec\/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n?$/,
    )
    if (!match) return null
    const size = Number(match[2])
    return Number.isSafeInteger(size) ? { hash_algo: 'sha256', oid: match[1], size } : null
  } finally {
    void reader.cancel().catch(() => {})
  }
}

function byteRange(
  header: string | null,
  size: number,
): { offset: number; length: number } | 'unsatisfiable' | undefined {
  if (!header) return undefined
  const match = header.match(/^bytes=(\d*)-(\d*)$/)
  if (!match || (!match[1] && !match[2])) return undefined
  const start = match[1] ? Number(match[1]) : undefined
  const end = match[2] ? Number(match[2]) : undefined
  if (
    (start !== undefined && !Number.isSafeInteger(start)) ||
    (end !== undefined && !Number.isSafeInteger(end))
  )
    return undefined
  if (start === undefined) {
    if (!end || !size) return 'unsatisfiable'
    const length = Math.min(end, size)
    return { offset: size - length, length }
  }
  if (end !== undefined && end < start) return undefined
  if (start >= size) return 'unsatisfiable'
  return { offset: start, length: Math.min(end ?? size - 1, size - 1) - start + 1 }
}

export async function getObjectFromBucket(
  bucket: R2Bucket,
  key: string,
  request: Request,
  contentType: string,
): Promise<Response> {
  const metadata = await bucket.head(key)
  if (!metadata) return new Response(null, { status: 404 })
  const headers = new Headers()
  headers.set('Content-Type', contentType)
  headers.set('Content-Length', String(metadata.size))
  headers.set('Accept-Ranges', 'bytes')
  headers.set('ETag', metadata.httpEtag)
  headers.set('Last-Modified', metadata.uploaded.toUTCString())
  headers.set('Cache-Control', 'public, max-age=300')
  headers.set('Access-Control-Allow-Origin', '*')
  headers.set('X-Content-Type-Options', 'nosniff')
  const etags = request.headers.get('If-None-Match')
  const modifiedSince = request.headers.get('If-Modified-Since')
  const uploaded = Math.floor(metadata.uploaded.getTime() / 1000) * 1000
  if (
    etags
      ? etags
          .split(',')
          .some(
            value => value.trim() === '*' || value.trim().replace(/^W\//, '') === metadata.httpEtag,
          )
      : modifiedSince && uploaded <= Date.parse(modifiedSince)
  ) {
    return new Response(null, { status: 304, headers })
  }
  if (request.method === 'HEAD') return new Response(null, { headers })
  const ifRange = request.headers.get('If-Range')
  const matchesRange =
    !ifRange ||
    ifRange === metadata.httpEtag ||
    (!ifRange.startsWith('W/') && uploaded <= Date.parse(ifRange))
  const range = byteRange(matchesRange ? request.headers.get('Range') : null, metadata.size)
  if (range === 'unsatisfiable') {
    headers.set('Content-Range', `bytes */${metadata.size}`)
    headers.set('Content-Length', '0')
    return new Response(null, { status: 416, headers })
  }
  const object = await bucket.get(key, { range })
  if (!object) return new Response(null, { status: 404 })
  if (range) {
    headers.set(
      'Content-Range',
      `bytes ${range.offset}-${range.offset + range.length - 1}/${metadata.size}`,
    )
    headers.set('Content-Length', String(range.length))
  }
  return new Response(object.body, { status: range ? 206 : 200, headers })
}
