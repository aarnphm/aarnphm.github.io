import type { VercelRequest, VercelResponse } from "@vercel/node"

export default async function handler(req: VercelRequest, resp: VercelResponse) {
  const apiUrl = "https://curius.app/api/users/3584/links?page=0"

  try {
    const response = await fetch(apiUrl, {
      headers: { "Content-Type": "application/json" },
    })
    resp.status(200).json(await response.json())
  } catch (error: any) {
    resp.status(500).json({ error: error.message })
  }
}
