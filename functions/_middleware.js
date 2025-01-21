// https://media.githubusercontent.com/media/aarnphm/aarnphm.github.io/refs/heads/main/content{url}
// url: thoughts/university/twenty-four-twenty-five/sfwr-2fa3/Automata%20and%20Computability.pdf

/**
 * @type {import("@cloudflare/workers-types").PagesFunction<{
 *   LFS_BUCKET?: R2Bucket,
 *   LFS_BUCKET_URL?: string,
 *   KEEP_HEADERS?: string
 * }>}
 */
export async function onRequest(context) {
  const { request } = context

  const url = new URL(request.url)

  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response(null, {
      status: request.method === "OPTIONS" ? 200 : 405,
      headers: { Allow: "GET, HEAD, OPTIONS" },
    })
  }

  const response =
    request.method === "GET"
      ? await context.next()
      : // if we request the HEAD of an LFS pointer, we want to GET the underlying
        // object's info (including URL) and then return the object's HEAD instead
        // so that Content-Length, etc. are correct
        await context.next(context.request, { method: "GET" })

  if (url.pathname.endsWith(".pdf")) {
    const githubUrl = `https://media.githubusercontent.com/media/aarnphm/aarnphm.github.io/refs/heads/main/content${url.pathname}`
    return await fetch(new Request(githubUrl, { method: "GET", headers: request.headers })).then(
      async (resp) => {
        const headers = new Headers(response.headers)
        // Force Content-Type to 'application/pdf'
        headers.set("Content-Type", "application/pdf")

        return new Response(resp.body, {
          status: resp.status,
          statusText: resp.statusText,
          headers,
        })
      },
    )
  }

  return response
}
