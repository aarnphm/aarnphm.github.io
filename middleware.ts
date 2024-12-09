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
  const slug = "notes"
  url.pathname = `/${slug}`
  url.searchParams.set("stackedNotes", btoa(slug.toString()).replace(/=+$/, ""))
  return rewrite(url)
}
