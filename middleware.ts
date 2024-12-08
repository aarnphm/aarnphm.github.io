import { rewrite, next } from "@vercel/edge"

export const config = {
  matcher: ["/:path*"],
}

export function middleware(request: Request) {
  const url = new URL(request.url)
  // Only proceed if we're on notes.aarnphm.xyz
  if (!url.hostname.startsWith("notes.aarnphm.xyz")) {
    return next()
  }
  return rewrite(new URL("/notes?stackedNotes=bm90ZXM", request.url))
}
