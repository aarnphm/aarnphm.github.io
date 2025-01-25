// https://media.githubusercontent.com/media/aarnphm/aarnphm.github.io/refs/heads/main/content/{url}
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

  // Only apply logic for notes.aarnphm.xyz and root path
  if (url.hostname === "notes.aarnphm.xyz" && url.pathname === "/") {
    const slug = "notes"
    const newUrl = new URL(`/${slug}`, request.url)
    newUrl.searchParams.set("stackedNotes", btoa(slug).replace(/=+$/, ""))
    // Rewrite the request with the new URL
    const newRequest = new Request(newUrl, request)
    return context.next(newRequest)
  } else if (url.pathname.endsWith(".pdf")) {
    const githubUrl = `https://media.githubusercontent.com/media/aarnphm/aarnphm.github.io/refs/heads/main/content${url.pathname}`
    return await fetch(new Request(githubUrl, { method: "GET", headers: request.headers })).then(
      async (resp) => {
        const headers = new Headers(resp.headers)
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

  return await context.next()
}
