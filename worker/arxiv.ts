import { XMLParser } from "fast-xml-parser"

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

function cleanIdentifier(identifier: string) {
  return identifier.replace(/^(arxiv:)?(pdf\/)?/, "").replace(".pdf", "")
}

async function getArxivMetadata(identifier: string) {
  const id = cleanIdentifier(identifier)
  const url = `${ARXIV_API_BASE}?id_list=${id}`
  const response = await fetch(url, { headers: { "User-Agent": USER_AGENT } })
  if (!response.ok) throw new Error(`arXiv API error: ${response.statusText}`)
  const xmlData = await response.text()
  const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" })
  const result = parser.parse(xmlData) as ArxivResponse
  if (!result.feed.entry) throw new Error("Paper not found")
  const pdfLink = Array.isArray(result.feed.entry.link)
    ? result.feed.entry.link.find((l) => l["@_type"] === "application/pdf")
    : null
  if (!pdfLink) throw new Error("PDF link not found")
  return {
    id,
    title: result.feed.entry.title,
    authors: Array.isArray(result.feed.entry.author)
      ? result.feed.entry.author.map((a) => a.name)
      : [result.feed.entry.author.name],
    summary: result.feed.entry.summary,
    published: result.feed.entry.published,
    category: result.feed.entry["arxiv:primary_category"]["@_term"],
    pdfUrl: pdfLink["@_href"],
  }
}

export default async function handleArxiv(request: Request): Promise<Response> {
  if (request.method !== "GET")
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    })
  try {
    const url = new URL(request.url)
    const identifier = url.searchParams.get("identifier")
    if (!identifier)
      return new Response(JSON.stringify({ error: "arXiv identifier is required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      })
    const metadata = await getArxivMetadata(identifier)
    if (url.searchParams.get("metadata") === "true")
      return new Response(JSON.stringify(metadata), {
        headers: { "Content-Type": "application/json" },
      })
    const pdfResp = await fetch(metadata.pdfUrl, { headers: { "User-Agent": USER_AGENT } })
    if (!pdfResp.ok || !pdfResp.body)
      return new Response(JSON.stringify({ error: `Failed to fetch PDF: ${pdfResp.statusText}` }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      })
    const headers = new Headers()
    headers.set("Content-Type", "application/pdf")
    headers.set("Content-Disposition", `inline; filename=${identifier}.pdf`)
    return new Response(pdfResp.body, { status: pdfResp.status, headers })
  } catch (error: any) {
    return new Response(
      JSON.stringify({ error: "Failed to process request", details: error?.message || "unknown" }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    )
  }
}
