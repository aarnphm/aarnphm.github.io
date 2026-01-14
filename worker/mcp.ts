import { McpAgent } from "agents/mcp"
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { z } from "zod"
import { semanticSearch } from "./semantic"

type ContentIndexEntry = {
  slug: string
  title: string
  filePath: string
  links: string[]
  aliases: string[]
  tags: string[]
  layout: string
  content: string
  fileName: string
  date?: string
  description?: string
}

type SimplifiedIndex = Record<string, ContentIndexEntry>

const INDEX_PATH = "/static/contentIndex.json"

async function fetchAssetText(path: string): Promise<string> {
  const u = new URL(path.startsWith("/") ? path : `/${path}`, "https://aarnphm.xyz")
  const res = await fetch(u.toString(), { method: "GET" })
  if (!res.ok) throw new Error(`asset ${u.pathname} ${res.status}`)
  return await res.text()
}

function getBaseUrl(): string {
  return "https://aarnphm.xyz"
}

let cachedIndex: { data: SimplifiedIndex; ts: number } | null = null

async function loadIndex(): Promise<SimplifiedIndex> {
  if (cachedIndex && Date.now() - cachedIndex.ts < 60_000) return cachedIndex.data
  const txt = await fetchAssetText(INDEX_PATH)
  const data = JSON.parse(txt) as SimplifiedIndex
  cachedIndex = { data, ts: Date.now() }
  return data
}

function ensureMdPath(p: string): string {
  if (p.endsWith(".md") || p.endsWith(".mdx") || p.endsWith(".txt")) return p
  return `${p}.md`
}

const MAX_CONTENT_TOKENS = 512

type WeightProfile = { exact: number; partial: number; fuzzy: number }

function tokenize(input: string): string[] {
  return input
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .map((segment) => segment.trim())
    .filter(Boolean)
}

function expandTokens(tokens: string[]): string[] {
  const expanded = new Set(tokens)
  for (const token of tokens) {
    if (token.length > 3 && token.endsWith("s")) expanded.add(token.slice(0, -1))
    if (token.length > 4 && token.endsWith("es")) expanded.add(token.slice(0, -2))
  }
  return Array.from(expanded)
}

function bigrams(input: string): string[] {
  const normalized = input.replace(/[^a-z0-9]/gi, "")
  const grams: string[] = []
  for (let i = 0; i < normalized.length - 1; i += 1) grams.push(normalized.slice(i, i + 2))
  return grams
}

function diceCoefficient(a: string, b: string): number {
  if (a === b) return 1
  if (a.length < 2 || b.length < 2) return 0
  const bigramsA = bigrams(a)
  const bigramsB = bigrams(b)
  if (bigramsA.length === 0 || bigramsB.length === 0) return 0
  let overlap = 0
  const counts = new Map<string, number>()
  for (const gram of bigramsA) counts.set(gram, (counts.get(gram) ?? 0) + 1)
  for (const gram of bigramsB) {
    const available = counts.get(gram)
    if (available && available > 0) {
      overlap += 1
      counts.set(gram, available - 1)
    }
  }
  return (2 * overlap) / (bigramsA.length + bigramsB.length)
}

function computeFieldScore(
  value: string | undefined,
  tokens: string[],
  weights: WeightProfile,
): number {
  if (!value) return 0
  const lower = value.toLowerCase()
  const phrase = tokens.join(" ")
  let score = 0
  if (phrase && lower === phrase) score += weights.exact * tokens.length
  else if (phrase && lower.includes(phrase)) score += weights.partial * tokens.length
  for (const token of tokens) {
    if (lower.includes(token)) score += weights.partial
    else {
      const similarity = diceCoefficient(lower, token)
      if (similarity > 0.65) score += weights.fuzzy * similarity
    }
  }
  return score
}

function computeListScore(
  values: string[] | undefined,
  tokens: string[],
  weights: WeightProfile,
): number {
  if (!values || values.length === 0) return 0
  let score = 0
  for (const value of values) score += computeFieldScore(value, tokens, weights)
  return score
}

function escapeRegex(token: string): string {
  return token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

function contentTermScore(content: string | undefined, tokens: string[], phrase: string): number {
  if (!content) return 0
  const lower = content.toLowerCase()
  let score = 0
  if (phrase && phrase.length > 3) {
    const phraseRegex = new RegExp(`\\b${escapeRegex(phrase)}\\b`, "g")
    const matches = lower.match(phraseRegex)
    if (matches) score += matches.length * 4
  }
  const contentTokens = tokenize(content).slice(0, MAX_CONTENT_TOKENS)
  const tokenSet = new Set(contentTokens)
  let coverage = 0
  for (const token of tokens) {
    const regex = new RegExp(`\\b${escapeRegex(token)}(?:[a-z0-9]+)?\\b`, "g")
    const matches = lower.match(regex)
    if (matches) score += Math.min(matches.length, 6) * 1.2
    if (tokenSet.has(token)) coverage += 1
  }
  if (coverage) score += coverage * 2
  return score
}

function computeRecencyBoost(dateStr?: string): number {
  if (!dateStr) return 1
  const parsed = Date.parse(dateStr)
  if (Number.isNaN(parsed)) return 1
  const ageDays = (Date.now() - parsed) / (1000 * 60 * 60 * 24)
  if (ageDays <= 0) return 1.1
  if (ageDays < 30) return 1.08
  if (ageDays < 180) return 1.05
  if (ageDays < 365) return 1.02
  if (ageDays > 3650) return 0.95
  return 1
}

function scoreEntry(e: ContentIndexEntry, query: string): number {
  const baseTokens = tokenize(query)
  if (baseTokens.length === 0) return 0
  const tokens = expandTokens(baseTokens)
  const phrase = baseTokens.join(" ")

  let score = 0
  score += computeFieldScore(e.slug, tokens, { exact: 18, partial: 7, fuzzy: 4 })
  score += computeFieldScore(e.fileName, tokens, { exact: 10, partial: 4, fuzzy: 2 })
  score += computeFieldScore(e.title, tokens, { exact: 14, partial: 6, fuzzy: 3 })
  score += computeListScore(e.aliases, tokens, { exact: 12, partial: 5, fuzzy: 3 })
  score += computeListScore(e.tags, tokens, { exact: 9, partial: 4, fuzzy: 2 })
  score += computeListScore(e.links, tokens, { exact: 4, partial: 2, fuzzy: 1 })
  score += computeFieldScore(e.description, tokens, { exact: 6, partial: 2.5, fuzzy: 1 })
  score += contentTermScore(e.content, tokens, phrase)

  return score * computeRecencyBoost(e.date)
}

type Props = {
  login: string
  name: string
  email: string
  accessToken: string
}

export class Garden extends McpAgent<Env, Record<string, never>, Props> {
  server = new McpServer({ name: "aarnphm.xyz", version: "1.0.0" })

  async init() {
    this.server.tool(
      "search",
      `semantic search over aaron's notes. returns ranked results by embedding similarity. use retrieve to read full content of hits.`,
      {
        query: z.string().describe("Search query to find relevant content"),
        limit: z.number().optional().describe("Maximum number of results to return (default: 8)"),
      },
      async (args: { query: string; limit?: number }) => {
        const { query, limit = 8 } = args as { query: string; limit?: number }
        const base = getBaseUrl()

        try {
          const semanticResults = await semanticSearch(this.env, query, limit)

          const idx = await loadIndex()
          const results = semanticResults.map(({ slug, score }) => {
            const entry = idx[slug]
            const mdPath = ensureMdPath(`/${slug}`)
            return {
              slug,
              path: mdPath.replace(/^\//, ""),
              url: `${base}${mdPath}`,
              title: entry?.title || slug,
              score,
            }
          })

          return { content: [{ type: "text", text: JSON.stringify({ results }) }] }
        } catch {
          const idx = await loadIndex()
          const ranked = Object.values(idx)
            .map((e) => ({ e, score: scoreEntry(e, query) }))
            .filter(({ score }) => score > 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, limit)
            .map(({ e, score }) => {
              const mdPath = ensureMdPath(`/${e.slug}`)
              return {
                slug: e.slug,
                path: mdPath.replace(/^\//, ""),
                url: `${base}${mdPath}`,
                title: e.title,
                score,
              }
            })

          return {
            content: [{ type: "text", text: JSON.stringify({ results: ranked, fallback: true }) }],
          }
        }
      },
    )

    this.server.tool(
      "retrieve",
      `fetch full markdown content of a note. use after search/rabbithole to read what you found.`,
      {
        slug: z
          .string()
          .describe("The slug of the content to retrieve (e.g., 'thoughts/attention')"),
      },
      async (args: { slug: string }) => {
        const { slug } = args as { slug: string }
        const mdPath = `/${slug}.md`
        let text: string
        try {
          text = await fetchAssetText(mdPath)
        } catch {
          throw new Error(`not found: ${slug}`)
        }
        return { content: [{ type: "text", text }] }
      },
    )
  }
}

export default Garden
