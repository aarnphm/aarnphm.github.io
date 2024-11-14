import type { VercelRequest, VercelResponse } from "@vercel/node"
import { XMLParser } from "fast-xml-parser"
import { Readable } from "stream"

// arXiv API base URL
const ARXIV_API_BASE = "http://export.arxiv.org/api/query"
const USER_AGENT = "Mozilla/5.0 (compatible; ArxivFetcher/1.0; mailto:contact@aarnphm.xyz)"

interface ArxivResponse {
  feed: {
    entry?: {
      id: string
      title: string
      summary: string
      author: Array<{ name: string }> | { name: string }
      published: string
      "arxiv:primary_category": {
        "@_term": string
      }
      link: Array<{
        "@_href": string
        "@_type": string
        "@_rel": string
      }>
    }
  }
}

async function getArxivMetadata(identifier: string) {
  // Clean the identifier (remove 'arxiv:', 'pdf/', etc.)
  const cleanId = identifier.replace(/^(arxiv:)?(pdf\/)?/, "").replace(".pdf$", "")

  // Construct the API query URL
  const queryUrl = `${ARXIV_API_BASE}?id_list=${cleanId}`

  try {
    const response = await fetch(queryUrl, { headers: { "User-Agent": USER_AGENT } })

    if (!response.ok) {
      throw new Error(`arXiv API error: ${response.statusText}`)
    }

    const xmlData = await response.text()
    const parser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: "@_",
    })

    const result = parser.parse(xmlData) as ArxivResponse

    if (!result.feed.entry) {
      throw new Error("Paper not found")
    }

    // Find the PDF link
    const pdfLink = Array.isArray(result.feed.entry.link)
      ? result.feed.entry.link.find((link) => link["@_type"] === "application/pdf")
      : null

    if (!pdfLink) {
      throw new Error("PDF link not found")
    }

    return {
      id: cleanId,
      title: result.feed.entry.title,
      authors: Array.isArray(result.feed.entry.author)
        ? result.feed.entry.author.map((a) => a.name)
        : [result.feed.entry.author.name],
      summary: result.feed.entry.summary,
      published: result.feed.entry.published,
      category: result.feed.entry["arxiv:primary_category"]["@_term"],
      pdfUrl: pdfLink["@_href"],
    }
  } catch (error) {
    throw new Error(`Failed to fetch arXiv metadata: ${error.message}`)
  }
}

async function streamPDF(url: string, response: VercelResponse) {
  const fetched = await fetch(url, { headers: { "User-Agent": USER_AGENT } })

  if (!fetched.ok) {
    throw new Error(`Failed to fetch PDF: ${fetched.statusText}`)
  }

  if (!fetched.body) {
    throw new Error("No response body received")
  }

  // Convert the ReadableStream to a Node.js Readable stream
  const reader = fetched.body.getReader()
  const stream = new Readable({
    async read() {
      try {
        const { done, value } = await reader.read()
        if (done) {
          this.push(null)
        } else {
          this.push(Buffer.from(value))
        }
      } catch (error) {
        this.destroy(error as Error)
      }
    },
  })

  // Pipe the stream directly to the response
  return new Promise((resolve, reject) => {
    stream.pipe(response)
    stream.on("end", resolve)
    stream.on("error", reject)
  })
}

export default async function handler(request: VercelRequest, response: VercelResponse) {
  if (request.method !== "GET") {
    return response.status(405).json({ error: "Method not allowed" })
  }

  try {
    const { identifier } = request.query

    if (!identifier || typeof identifier !== "string") {
      return response.status(400).json({ error: "arXiv identifier is required" })
    }

    // First get metadata and PDF URL from arXiv API
    const metadata = await getArxivMetadata(identifier)

    if (request.query.metadata === "true") {
      // If metadata is requested, return only metadata
      return response.json(metadata)
    }

    // Set response headers
    response.setHeader("Content-Type", "application/pdf")
    response.setHeader("Content-Disposition", `inline; filename=${identifier}.pdf`)
    response.setHeader("Transfer-Encoding", "chunked")

    // Otherwise fetch and return the PDF
    await streamPDF(metadata.pdfUrl, response)

    return
  } catch (error: any) {
    console.error("Error processing request:", error)
    // Only send error response if headers haven't been sent yet
    if (!response.headersSent) {
      return response.status(500).json({
        error: "Failed to process request",
        details: error.message,
      })
    }

    // If headers were sent, we need to end the response
    response.end()
  }
}
