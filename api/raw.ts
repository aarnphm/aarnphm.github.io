import path from "path"
import type { VercelRequest, VercelResponse } from "@vercel/node"

export default async function handler(req: VercelRequest, resp: VercelResponse) {
  // Constructing the base URL
  const protocol = req.headers["x-forwarded-proto"] || "http"
  const host = req.headers["x-forwarded-host"] || req.headers.host
  const baseUrl = `${protocol}://${host}`
  const { path: slug } = req.query
  try {
    if (typeof slug !== "string") {
      resp.status(400).send({ error: "given query is not a string" })
      return
    }

    const staticUrl = path.join(baseUrl, typeof slug !== "string" ? [...slug].join("/") : slug)

    const data = await fetch(staticUrl)
      .then((res) => res.text())
      .catch((e) => {
        resp.status(500).send({ error: e.message })
        return
      })

    resp.setHeader("Content-Type", "text/plain")
    resp.status(200).send(data)
  } catch (e: any) {
    resp.status(500).send({ error: e.message })
  }
}
