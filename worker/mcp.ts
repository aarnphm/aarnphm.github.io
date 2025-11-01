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
  if (p.endsWith(".md") || p.endsWith(".txt")) return p
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

export class Garden extends McpAgent {
  server = new McpServer({ name: "aarnphm.xyz", version: "1.0.0" })
  private env: any

  constructor(state: DurableObjectState, env: any) {
    super(state, env)
    this.env = env
  }

  async init() {
    this.server.tool(
      "search",
      "Semantic search across content using embeddings",
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
      "retrieve_content",
      "Retrieve LLM-optimized content for a given slug",
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

    this.server.tool(
      "temporal",
      "Find notes related to a topic within a time window",
      {
        query: z.string().describe("Topic or term to find temporal neighbors for"),
        days: z
          .number()
          .min(1)
          .max(30)
          .optional()
          .describe("Number of days to look within (default: 5)"),
      },
      async (args: { query: string; days?: number }) => {
        const { query, days = 5 } = args
        const idx = await loadIndex()
        const base = getBaseUrl()

        const semanticResults = await semanticSearch(this.env, query, 3)
        if (semanticResults.length === 0) {
          return {
            content: [
              { type: "text", text: JSON.stringify({ results: [], message: "No matches found" }) },
            ],
          }
        }

        const primarySlug = semanticResults[0].slug
        const primaryEntry = idx[primarySlug]

        if (!primaryEntry?.date) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  results: [],
                  message: "Primary note has no date information",
                }),
              },
            ],
          }
        }

        const primaryDate = new Date(primaryEntry.date)
        const windowMs = days * 24 * 60 * 60 * 1000
        const startDate = new Date(primaryDate.getTime() - windowMs)
        const endDate = new Date(primaryDate.getTime() + windowMs)

        const temporalNeighbors = Object.values(idx)
          .filter((e) => {
            if (!e.date || e.slug === primarySlug) return false
            const entryDate = new Date(e.date)
            return entryDate >= startDate && entryDate <= endDate
          })
          .map((e) => {
            const entryDate = new Date(e.date!)
            const daysDiff = Math.abs(
              (entryDate.getTime() - primaryDate.getTime()) / (24 * 60 * 60 * 1000),
            )
            return {
              slug: e.slug,
              title: e.title,
              date: e.date,
              daysDiff: Math.round(daysDiff * 10) / 10,
              url: `${base}/${e.slug}.md`,
              tags: e.tags,
            }
          })
          .sort((a, b) => a.daysDiff - b.daysDiff)
          .slice(0, 20)

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                anchor: {
                  slug: primarySlug,
                  title: primaryEntry.title,
                  date: primaryEntry.date,
                },
                window: {
                  days,
                  startDate: startDate.toISOString(),
                  endDate: endDate.toISOString(),
                },
                neighbors: temporalNeighbors,
              }),
            },
          ],
        }
      },
    )

    this.server.tool(
      "rabbithole",
      "Stateless deep search primitive for agent-orchestrated exploration. Returns contextual results for one query, excluding already-explored notes. Agents call this iteratively to go deep.",
      {
        query: z.string().describe("Research query or topic to explore"),
        exclude_slugs: z
          .array(z.string())
          .optional()
          .describe("List of slugs to exclude (already explored notes)"),
        breadth: z
          .number()
          .min(1)
          .max(10)
          .optional()
          .describe("Number of results to return (default: 5)"),
        include_content: z
          .boolean()
          .optional()
          .describe("Include full content snippets for agent analysis (default: true)"),
      },
      async (args: {
        query: string
        exclude_slugs?: string[]
        breadth?: number
        include_content?: boolean
      }) => {
        const { query, exclude_slugs = [], breadth = 5, include_content = true } = args
        const base = getBaseUrl()
        const excludeSet = new Set(exclude_slugs)

        try {
          const results = await semanticSearch(this.env, query, breadth * 2)
          const idx = await loadIndex()

          const findings = results
            .filter((r) => !excludeSet.has(r.slug))
            .slice(0, breadth)
            .map((r) => {
              const entry = idx[r.slug]
              const content = entry?.content || ""

              const finding: any = {
                slug: r.slug,
                title: entry?.title || r.slug,
                score: r.score,
                url: `${base}/${r.slug}.md`,
                tags: entry?.tags || [],
                date: entry?.date,
              }

              if (include_content) {
                const snippet = content.slice(0, 500).trim()
                finding.snippet = snippet + (content.length > 500 ? "..." : "")
                finding.word_count = content.split(/\s+/).length
              }

              if (entry?.links) {
                finding.outgoing_links = entry.links.slice(0, 10)
              }

              return finding
            })

          const relatedTerms = new Set<string>()
          findings.forEach((f) => {
            f.tags?.forEach((tag: string) => relatedTerms.add(tag))
            f.outgoing_links?.forEach((link: string) => {
              const cleanLink = link.split("/").pop()?.replace(/\.md$/, "")
              if (cleanLink && !excludeSet.has(cleanLink)) {
                relatedTerms.add(cleanLink)
              }
            })
          })

          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  query,
                  findings,
                  explored_count: exclude_slugs.length,
                  new_count: findings.length,
                  related_terms: Array.from(relatedTerms).slice(0, 15),
                  suggestions: {
                    continue_exploration: findings.length > 0,
                    recommended_depth: 7,
                    next_steps: [
                      "Analyze findings and identify interesting threads",
                      "Call rabbithole again with updated exclude_slugs",
                      "Explore related_terms or outgoing_links as new queries",
                      "Use retrieve_content for full note analysis",
                    ],
                  },
                }),
              },
            ],
          }
        } catch (error) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  query,
                  findings: [],
                  error: "Search failed",
                  explored_count: exclude_slugs.length,
                }),
              },
            ],
          }
        }
      },
    )

    this.server.tool(
      "entropy",
      "Find underexplored aspects within a topic area. Given a broad topic with existing coverage, identify specific queries that have low relevance scores, suggesting where to expand and connect new content.",
      {
        topic: z.string().describe("Broad topic area with existing notes (e.g., 'interpretability')"),
        query: z
          .string()
          .describe("Specific aspect to check coverage for (e.g., 'parameter decomposition')"),
      },
      async (args: { topic: string; query: string }) => {
        const { topic, query } = args
        const base = getBaseUrl()

        try {
          const topicResults = await semanticSearch(this.env, topic, 15)
          const queryResults = await semanticSearch(this.env, query, 10)
          const idx = await loadIndex()

          if (topicResults.length === 0) {
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    topic,
                    query,
                    entropy: "undefined",
                    recommendation: `No notes found on "${topic}". Cannot assess gap for "${query}".`,
                  }),
                },
              ],
            }
          }

          const topicSlugs = new Set(topicResults.map((r) => r.slug))
          const queryTopScore = queryResults.length > 0 ? queryResults[0].score : 0
          const queryInTopicCount = queryResults.filter((r) => topicSlugs.has(r.slug)).length

          const hasLowCoverage = queryTopScore < 0.65 || queryInTopicCount < 2
          const entropy = hasLowCoverage ? "high" : queryTopScore < 0.8 ? "medium" : "low"

          const connectionPoints: Array<{
            slug: string
            title: string
            relevance: string
            why: string
            url: string
          }> = []

          topicResults.slice(0, 8).forEach((r) => {
            const entry = idx[r.slug]
            if (!entry) return

            const isDirectMatch = queryResults.some((qr) => qr.slug === r.slug)
            if (isDirectMatch) {
              connectionPoints.push({
                slug: r.slug,
                title: entry.title || r.slug,
                relevance: "existing",
                why: "Already covers this aspect, could be expanded",
                url: `${base}/${r.slug}.md`,
              })
            } else {
              const hasRelatedTags = entry.tags?.some((tag) =>
                query.toLowerCase().includes(tag.toLowerCase()),
              )
              const titleMatch = entry.title?.toLowerCase().includes(query.toLowerCase())

              if (hasRelatedTags || titleMatch) {
                connectionPoints.push({
                  slug: r.slug,
                  title: entry.title || r.slug,
                  relevance: "adjacent",
                  why: "Related topic, good place to add section or link",
                  url: `${base}/${r.slug}.md`,
                })
              } else {
                connectionPoints.push({
                  slug: r.slug,
                  title: entry.title || r.slug,
                  relevance: "potential",
                  why: "Core topic note, could mention this aspect",
                  url: `${base}/${r.slug}.md`,
                })
              }
            }
          })

          const existingFiles = connectionPoints.filter((c) => c.relevance === "existing")
          const adjacentFiles = connectionPoints.filter((c) => c.relevance === "adjacent")
          const potentialFiles = connectionPoints.filter((c) => c.relevance === "potential")

          let recommendation: string
          let suggestedAction: string

          if (entropy === "high") {
            recommendation = `"${query}" is significantly underexplored within "${topic}". High-value enrichment opportunity.`
            if (adjacentFiles.length > 0) {
              suggestedAction = `Create new note or expand ${adjacentFiles[0].slug} to cover "${query}"`
            } else {
              suggestedAction = `Create new note under thoughts/ covering "${query}" and link to ${potentialFiles[0]?.slug || topicResults[0].slug}`
            }
          } else if (entropy === "medium") {
            recommendation = `"${query}" has some coverage but could be deeper or better connected.`
            suggestedAction =
              existingFiles.length > 0
                ? `Expand ${existingFiles[0].slug} with more detail on "${query}"`
                : `Add sections to ${adjacentFiles[0]?.slug || potentialFiles[0]?.slug} covering "${query}"`
          } else {
            recommendation = `"${query}" is well-covered within "${topic}".`
            suggestedAction = "Consider advanced aspects or alternative perspectives"
          }

          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  topic,
                  query,
                  entropy,
                  entropy_details: {
                    query_top_score: Math.round(queryTopScore * 100) / 100,
                    query_in_topic: queryInTopicCount,
                    total_topic_notes: topicResults.length,
                  },
                  recommendation,
                  suggested_action: suggestedAction,
                  expansion_targets: {
                    existing: existingFiles,
                    adjacent: adjacentFiles,
                    potential: potentialFiles.slice(0, 3),
                  },
                  enrichment_instructions: {
                    create_new:
                      existingFiles.length === 0 && adjacentFiles.length === 0
                        ? {
                            suggested_path: `thoughts/${query.toLowerCase().replace(/\s+/g, "-")}.md`,
                            connect_to: potentialFiles.slice(0, 3).map((f) => f.slug),
                            tags_to_add: topicResults[0] ? idx[topicResults[0].slug]?.tags || [] : [],
                          }
                        : null,
                    expand_existing:
                      existingFiles.length > 0
                        ? {
                            files: existingFiles.map((f) => f.slug),
                            action: "Add dedicated section or expand coverage",
                          }
                        : null,
                    add_sections:
                      adjacentFiles.length > 0
                        ? {
                            files: adjacentFiles.map((f) => f.slug),
                            action: "Add section covering this aspect and cross-link",
                          }
                        : null,
                  },
                }),
              },
            ],
          }
        } catch (error) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  topic,
                  query,
                  error: "Analysis failed",
                }),
              },
            ],
          }
        }
      },
    )
  }
}

export default Garden
