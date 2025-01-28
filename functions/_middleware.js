// raw URL: https://raw.githubusercontent.com/aarnphm/aarnphm.github.io/refs/heads/main/content/thoughts/university/twenty-four-twenty-five/sfwr-2fa3/Automata%20and%20Computability.pdf

import LFS_CONFIG from "./.lfsconfig.txt"

const VERSION = "version https://git-lfs.github.com/spec/v1\n"
const MIME = "application/vnd.git-lfs+json"

// the bucket URL is immutable, but the served URL is not necessarily.
// therefore set Cache-Control as it is for the underlying static page.
const KEEP_HEADERS = "Cache-Control"

String.prototype.splitFirst = function (delim) {
  const index = this.indexOf(delim)
  if (index !== -1) {
    return [this.substring(0, index), this.substring(index + 1)]
  } else {
    return [this]
  }
}

function strictDecode(bytes) {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes)
  } catch {
    return null
  }
}

function getLfsUrl(config) {
  // TODO: better parser, accept remote.<remote>.lfsurl...
  let section
  for (const _line of config.split("\n")) {
    const line = _line.splitFirst(";")[0].trim()
    if (line.startsWith("[") && line.endsWith("]")) {
      section = line.substring(1, line.length - 1)
    } else if (section === "lfs") {
      const [key, val] = line.splitFirst("=")
      if (val === undefined) {
        return null
      }
      if (key.trimEnd() === "url") {
        return val.trimStart()
      }
    }
  }

  return null
}

function extendPath(url, path) {
  let urlobj = new URL(url)
  urlobj.pathname = urlobj.pathname.replace(/\/?$/, `/${path}`)
  return urlobj
}

function withHeaders(response, newHeaders) {
  if (Object.keys(newHeaders).length === 0) {
    return response
  }

  const headers = new Headers(response.headers)
  for (const [key, val] of headers) {
    if (val === null) {
      headers.delete(key)
    } else {
      headers.set(key, val)
    }
  }
  const { status, statusText } = response
  return new Response(response.body, { headers, status, statusText })
}

function withHeadersFromSource(response, source, headers) {
  const newHeaders = headers.map((h) => [h, source.headers.get(h)])
  return withHeaders(response, newHeaders)
}

async function getObjectInfo(response) {
  // TODO: theoretically an LFS pointer could be >256 bytes.
  // however, even the LFS client spec seems to only read 100:
  // https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md
  const buffer = new Uint8Array(256)
  const reader = response.body.getReader({ mode: "byob" })
  const chunk = await reader.readAtLeast(buffer.length, buffer)

  const text = strictDecode(chunk.value)
  if (text === null) {
    return null
  }

  if (!text.startsWith(VERSION)) {
    return null
  }
  const rest = text.substring(VERSION.length)

  let hash_algo, oid, size
  for (const line of rest.split("\n")) {
    if (line === "") {
      continue
    }

    const [key, val] = line.splitFirst(" ")
    if (val === undefined) {
      return null
    }

    switch (key) {
      case "oid":
        ;[hash_algo, oid] = val.splitFirst(":")
        if (oid === undefined) {
          return null
        }
        break
      case "size":
        size = parseInt(val)
        if (Number.isNaN(size)) {
          return null
        }
        break
    }
  }

  if (hash_algo && oid && size) {
    return { hash_algo, oid, size }
  }

  return null
}

async function getObjectAction(lfsUrl, objectInfo) {
  const url = extendPath(lfsUrl, "objects/batch")

  const headers = { Accept: MIME, "Content-Type": MIME }
  if (url.username !== "" || url.password !== "") {
    const encoded = btoa(`${url.username}:${url.password}`)
    headers["Authorization"] = `Basic ${encoded}`
    url.username = url.password = ""
  }

  const { hash_algo, oid, size } = objectInfo
  const body = JSON.stringify({
    operation: "download",
    transfers: ["basic"],
    objects: [{ oid, size }],
    hash_algo,
  })

  const response = await fetch(url, { method: "POST", headers, body })

  // TODO: better error handling
  if (response.ok && response.headers.get("Content-Type").startsWith(MIME)) {
    const batch = await response.json()
    if (
      (batch.transfer === undefined || batch.transfer === "basic") &&
      batch.objects[0] &&
      batch.objects[0].authenticated === true
    ) {
      return batch.objects[0].actions.download
    }
  }

  return null
}

async function getObjectFromBucket(context, bucket, bucketUrl, path, request) {
  const cacheKey = new Request(extendPath(bucketUrl, path), request)
  const cached = await caches.default.match(cacheKey)
  if (cached) {
    return cached
  }

  const { method } = request
  const object = await bucket[method.toLowerCase()](path)

  const headers = new Headers()
  object.writeHttpMetadata(headers)
  if (object.httpEtag) {
    headers.set("ETag", object.httpEtag)
  }

  const objectResponse = new Response(object.body, { method, headers })

  const cacheResponse = objectResponse.clone()
  context.waitUntil(
    caches.default.put(
      cacheKey,
      withHeaders(cacheResponse, {
        // Git LFS clients really should set this on upload, but they don't
        "Cache-Control": "immutable, max-age=31536000",
      }),
    ),
  )

  return objectResponse
}

async function getObjectFromLFS(objectInfo, request) {
  const lfsUrl = getLfsUrl(LFS_CONFIG)
  const action = await getObjectAction(lfsUrl, objectInfo)

  return await fetch(action.href, {
    method: request.method,
    headers: !action.header
      ? request.headers
      : {
          ...Object.fromEntries(action.header),
          ...Object.fromEntries(request.headers),
        },
    cf: { cacheTtl: 31536000 },
  })
}

/**
 * @type {import("@cloudflare/workers-types").PagesFunction<{
 *   LFS_BUCKET?: string,
 *   LFS_BUCKET_URL?: string,
 *   KEEP_HEADERS?: string
 * }>}
 */
export async function onRequest(context) {
  const { env, request } = context

  const url = new URL(request.url)
  if (url.pathname === "/.lfsconfig") {
    return new Response(null, { status: 404 })
  }

  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response(null, {
      status: request.method === "OPTIONS" ? 200 : 405,
      headers: { Allow: "GET, HEAD, OPTIONS" },
    })
  }

  if (url.pathname.endsWith(".pdf")) {
    const rawUrl = `https://raw.githubusercontent.com/aarnphm/aarnphm.github.io/refs/heads/main/content${url.pathname}`

    const response = await fetch(new Request(rawUrl, { method: "GET", headers: request.headers }))

    if (response.body) {
      const objectInfo = await getObjectInfo(response.clone())
      if (objectInfo) {
        const objectResponse =
          env.LFS_BUCKET && env.LFS_BUCKET_URL
            ? await getObjectFromBucket(
                context,
                env.LFS_BUCKET,
                env.LFS_BUCKET_URL,
                objectInfo.oid,
                request,
              )
            : await getObjectFromLFS(objectInfo, request)

        // TODO: copy more headers from source `response`? rn just Cache-Control
        const keepHeaders = (env.KEEP_HEADERS || KEEP_HEADERS).split(",")
        return withHeadersFromSource(objectResponse, response, keepHeaders)
      }
    }
  }

  return await context.next()
}
