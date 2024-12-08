import { rewrite } from "@vercel/edge"

export function middleware(request: Request) {
  const url = new URL(request.url)
  // Only proceed if we're on notes.aarnphm.xyz
  console.log(url)
  if (url.hostname.startsWith("notes.aarnphm.xyz")) {
    return rewrite(new URL("/notes?stackedNotes=bm90ZXM", request.url))
  }
}
