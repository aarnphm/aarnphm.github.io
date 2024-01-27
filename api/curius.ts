import type { VercelResponse } from "@vercel/node"

export const config = {
  runtime: "edge",
}

export default async function handler(resp: VercelResponse) {
  resp.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate=59")
  const baseUrl = "https://curius.app/api/users/3584/links"
  let allLinks: any[] = []
  let page = 0
  let lastPage = false

  try {
    while (!lastPage) {
      const response = await fetch(`${baseUrl}?page=${page}`, {
        headers: { "Content-Type": "application/json" },
      })
      const data = await response.json()
      if (data.userSaved && data.userSaved.length > 0) {
        allLinks = [...allLinks, ...data.userSaved]
        page++
      } else {
        lastPage = true
      }
    }
    resp.status(200).json({ userSaved: allLinks })
  } catch (error: any) {
    resp.status(500).json({ error: error.message || "An error occurred while fetching data." })
  }
}
