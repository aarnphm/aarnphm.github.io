import { rewrite, next } from "@vercel/edge"

export const config = {
  matcher: ["/((?!api|static).*)"],
}

export function middleware(request: Request) {
  const url = new URL(request.url)
  const hasStackedNotes = url.searchParams.has("stackedNotes")

  // Only proceed if we're on notes.aarnphm.xyz
  if (!url.hostname.startsWith("notes.aarnphm.xyz")) {
    return next()
  }

  // If no stacked notes parameter exists, add the current path or index
  if (!hasStackedNotes) {
    const path = url.pathname === "/" ? "index" : url.pathname.substring(1)
    url.searchParams.append("stackedNotes", btoa(path))

    return rewrite(url)
  }
  return next()
}
