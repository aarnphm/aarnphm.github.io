import type { VercelRequest, VercelResponse } from "@vercel/node"

export default async function handler(req: VercelRequest, resp: VercelResponse) {
  const { subdomain, path } = req.query
  const redirectUrl = `https://endpoints.aarnphm.xyz/api/raw?slug=${subdomain}/${path}`
  resp.writeHead(307, { Location: redirectUrl })
  resp.end()
}
