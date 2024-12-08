import { rewrite, next } from "@vercel/edge"

export const config = {
  matcher: ["/:path*"],
}

export default function middleware(request: Request) {
  const url = new URL(request.url)
  // Only apply logic for notes.aarnphm.xyz
  if (url.hostname !== "notes.aarnphm.xyz" || url.pathname !== "/") {
    return next()
  }
  url.pathname = "/notes?stackedNotes=bm90ZXM"
  return rewrite(url)
}
