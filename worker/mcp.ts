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
      `Find relevant notes by semantic similarity within Aaron's notebook.
This tool helps to find existing notes and constituents within Aaron's notebook to support additional research and expansion for Aaron.

When to use this tool:
- Initial broad queries to discover content related to a research topic
- Finding existing coverage within Aaron's notebook before creating new content
- Locating related notes for cross-referencing or linking
- Identifying source material for synthesis or expansion

Parameters:
- query: Natural language search query describing the topic or concept you're looking for
- limit: Maximum number of results to return (default: 8, use higher for comprehensive scans)

Recommendations:
- Use specific terminology for better semantic matching
- Follow up with retrieve tool to get full content of relevant results
- Combine with rabbithole for deeper exploration of promising threads
- Check returned slugs for patterns (e.g., thoughts/, lectures/, posts/) to understand content type
`,
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
      `Fetch markdown-format of related notes within Aaron's notebooks.
This tool helps return agent-friendly format (markdown)

When to use this tool:
- Whenever you need to fetch contents from apex domain "https://aarnphm.xyz" or within Aaron's notebook
- After getting slugs from search, rabbithole, temporal, or entropy tools
- When you need full content for detailed analysis, citation, or synthesis
- Before expanding or modifying existing notes to understand current content

Parameters:
- slug: The note identifier without .md extension (e.g., 'thoughts/attention' not 'thoughts/attention.md')

Recommendations:
- Always retrieve before suggesting modifications to existing notes
- Use for deep reading after initial discovery via search or rabbithole
- Combine multiple retrieves to build comprehensive understanding of related concepts
- Note the wikilinks and citations in retrieved content for further exploration threads
`,
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
      `Discover notes within a time window around a topic's primary note within Aaron's notebook.
This tool reveals chronological patterns and contextual influences by finding notes created near the same time.

When to use this tool:
- Understanding the temporal context around when a note was written
- Finding journal entries or thoughts from a specific period
- Discovering how ideas evolved chronologically
- Identifying concurrent themes or concerns that influenced a note's creation
- Exploring what Aaron was thinking about during a particular time

Parameters:
- query: Topic or note to find temporal neighbors for
- days: Time window in days to search before/after the primary note (default: 5, max: 30)

Recommendations:
- Start with smaller windows (5-7 days) for focused context
- Expand to 14-30 days to see broader patterns or influences
- Use with retrieve to read full content of temporally proximate notes
- Particularly useful for understanding posts/ and personal writing
- Combine with search results to distinguish thematic vs temporal connections
`,
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
      `Stateless deep exploration primitive for agent-orchestrated research within Aaron's notebook.
This tool returns contextual results for one query while excluding already-explored notes. Call iteratively to go deep.

When to use this tool:
- Conducting multi-iteration research workflows requiring progressive depth
- Following conceptual threads through related notes
- Building comprehensive understanding by exploring outgoing links and related terms
- Mapping knowledge clusters around a topic through repeated calls
- Discovering non-obvious connections by traversing the knowledge graph

Parameters:
- query: Research query or specific aspect to explore in this iteration
- exclude_slugs: Array of slugs already explored (maintains stateless iteration tracking)
- breadth: Number of results per iteration (default: 5, use 3 for focused, 10 for comprehensive)
- include_content: Include 500-char snippets for analysis (default: true, disable for speed)

Recommendations:
- Maintain cumulative exclude_slugs list across iterations to avoid revisiting notes
- Use returned related_terms and outgoing_links to formulate next queries
- Start broad, then follow specific threads based on findings
- Typical depth: 5-7 iterations for comprehensive topic coverage
- Each iteration should explore something new - refine queries based on previous findings
- Use snippets to decide whether to retrieve full content
- Track suggestions.continue_exploration to know when you've exhausted a thread
`,
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
      `Analyze content coverage gaps and identify expansion opportunities within Aaron's notebook.
This tool assesses how well a specific aspect is covered within a topic area and provides strategic recommendations.

When to use this tool:
- Strategic content planning - finding what's missing or underexplored
- Before creating new notes - validating that a gap actually exists
- Identifying where to expand existing notes vs creating new ones
- Discovering connection points for integrating new content
- Understanding coverage depth of specific aspects within broader topics

Parameters:
- topic: Broad topic area with existing notes (e.g., 'interpretability', 'transformers')
- query: Specific aspect to assess coverage for (e.g., 'mechanistic interpretability', 'rotary embeddings')

Returns:
- entropy: 'high' (significant gap), 'medium' (partial coverage), 'low' (well-covered)
- expansion_targets: Categorized notes (existing/adjacent/potential) with connection strategies
- enrichment_instructions: Actionable next steps (create_new, expand_existing, add_sections)

Recommendations:
- Use before content creation to identify genuine gaps vs redundant coverage
- High entropy = create new note or major expansion opportunity
- Medium entropy = deepen existing coverage with dedicated sections
- Low entropy = consider advanced aspects or alternative perspectives
- Review expansion_targets to decide between new note vs expanding existing
- Use suggested_action and enrichment_instructions for implementation strategy
- Validate findings with retrieve tool to read actual content of target notes
`,
      {
        topic: z
          .string()
          .describe("Broad topic area with existing notes (e.g., 'interpretability')"),
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
                            tags_to_add: topicResults[0]
                              ? idx[topicResults[0].slug]?.tags || []
                              : [],
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

    this.server.prompt(
      "deep_research",
      "Iterative deep exploration workflow using rabbithole tool for comprehensive research on a topic",
      {
        initial_query: z.string().describe("Starting research query or topic"),
        target_depth: z
          .coerce.number()
          .int()
          .min(3)
          .max(15)
          .optional()
          .describe("Number of iterations to explore (default: 7)"),
      },
      async (args: { initial_query: string; target_depth?: number }) => {
        const { initial_query, target_depth = 7 } = args
        return {
          messages: [
            {
              role: "user",
              content: {
                type: "text",
                text: `You are conducting deep research on "${initial_query}" within Aaron's notebook. Your objective is comprehensive understanding through iterative exploration, building progressively deeper context across ${target_depth} iterations.

Core Principles:

1. **Stateless Iteration**: Each rabbithole call is independent. Maintain cumulative exclude_slugs list across all iterations. Never revisit explored notes.

2. **Progressive Refinement**: Start broad, follow specific threads. Each iteration should explore new aspects based on previous findings. Use related_terms and outgoing_links to guide next queries.

3. **Strategic Retrieval**: Use snippets for triage. Call retrieve only for notes requiring full analysis. Time is finite - choose critical reads.

4. **Sequential Thinking Integration**: Invoke sequential-thinking MCP for complex synthesis, thread prioritization, or multi-step reasoning. Not for simple analysis.

5. **Depth Tracking**: Target ${target_depth} iterations. Each should add new information. If a thread exhausts, pivot based on related_terms.

Operational Protocol:

**Iterations 1-2: Initial Discovery**
- Call rabbithole with query "${initial_query}", breadth 5, include_content true, exclude_slugs []
- Scan all findings and snippets
- Identify 2-3 promising threads based on relevance scores and content
- Track explored slugs

**Iterations 3-${Math.floor(target_depth * 0.7)}: Thread Following**
- For each promising thread, call rabbithole with refined query targeting specific aspect
- Extract related_terms and outgoing_links from results
- Add all new slugs to exclude_slugs
- Use retrieve for critical notes identified by snippets
- Invoke sequential-thinking when synthesizing connections across multiple notes
- Decide which thread to prioritize based on information density

**Iterations ${Math.floor(target_depth * 0.7) + 1}-${target_depth}: Deep Dive**
- Focus on underexplored aspects, surprising connections, or gaps identified
- Use very specific queries derived from previous findings
- Retrieve full content for detailed analysis when needed
- Invoke sequential-thinking for final synthesis and gap identification

Tool Usage Rules:

- **rabbithole**: Every iteration. Pass cumulative exclude_slugs. Adjust breadth (3-10) based on query specificity.
- **retrieve**: Only when snippet insufficient. Not every note needs full read.
- **sequential-thinking**: When reasoning requires multiple steps, complex tradeoffs, or synthesis across >3 notes.
- **search**: Avoid. rabbithole handles discovery better with exclusion tracking.

What NOT to do:

- Never revisit slugs already in exclude_slugs
- Never apologize for iteration count or scope
- Never explain what you're about to do before doing it
- Never use sequential-thinking for simple pattern recognition
- Never retrieve without cause - snippets are often sufficient
- Never continue iterations without new information

Output Requirements:

After ${target_depth} iterations, synthesize findings directly:
1. Core concepts discovered (be specific - cite note slugs)
2. Key relationships and connections (describe the actual links)
3. Surprising insights or non-obvious patterns
4. Coverage gaps or underexplored aspects
5. Specific recommendations (concrete next queries or notes to create)

Begin iteration 1.`,
              },
            },
          ],
        }
      },
    )

    this.server.prompt(
      "temporal_exploration",
      "Discover chronological connections and time-based patterns in notes related to a topic",
      {
        topic: z.string().describe("Topic to explore temporal connections for"),
        time_window_days: z
          .coerce.number()
          .int()
          .min(1)
          .max(30)
          .optional()
          .describe("Number of days to look before/after primary note (default: 5)"),
      },
      async (args: { topic: string; time_window_days?: number }) => {
        const { topic, time_window_days = 5 } = args
        return {
          messages: [
            {
              role: "user",
              content: {
                type: "text",
                text: `You are exploring temporal connections for "${topic}" within Aaron's notebook. Your objective is understanding how ideas evolved chronologically and what contextual factors influenced their creation.

Core Principles:

1. **Temporal Context**: Notes written near the same time reveal shared concerns, influences, and mental context. Time proximity indicates environmental or situational connections, not just thematic ones.

2. **Chronological Analysis**: Focus on progression. What came before influences what follows. Identify causal patterns in idea development.

3. **Scale Sensitivity**: Start narrow (${time_window_days} days), expand if patterns justify. Broader windows reveal trends, narrow windows reveal immediate context.

4. **Sequential Thinking Integration**: Use sequential-thinking MCP when synthesizing patterns across multiple temporal neighbors or analyzing why notes cluster temporally. Not for simple enumeration.

Operational Protocol:

**Step 1: Initial Temporal Scan**
- Call temporal tool with query "${topic}", days ${time_window_days}
- Note anchor note date and total neighbors found
- Scan returned neighbors sorted by temporal proximity
- Identify patterns: clusters, gaps, isolated notes

**Step 2: Pattern Recognition**
- Examine themes: What was Aaron thinking about during this period?
- Look for concurrent concerns that influenced the primary note
- Identify evolution: How did thinking progress across the window?
- Note any journal entries or personal writing (posts/) from that period

**Step 3: Selective Retrieval**
- Retrieve 2-4 most temporally proximate notes
- Read for shared themes, influences, or contextual factors
- Identify why these notes emerged at the same time
- Look for explicit references or implicit connections

**Step 4: Multi-Scale Analysis (if warranted)**
- If clear patterns emerge, expand window to 10-15 days
- Compare narrow vs broad windows to distinguish immediate vs sustained concerns
- Use sequential-thinking to synthesize multi-scale patterns

Tool Usage Rules:

- **temporal**: Once initially, then again if expansion justified by findings
- **retrieve**: 2-4 notes maximum for focused analysis
- **sequential-thinking**: When synthesizing evolution across >3 notes or explaining temporal clustering
- **rabbithole/search**: Avoid. Focus on time, not semantic similarity

What NOT to do:

- Never retrieve every temporal neighbor - select strategically
- Never use sequential-thinking for listing themes
- Never expand time window without justification from initial findings
- Never ignore the anchor note date when analyzing context
- Never confuse temporal proximity with thematic similarity

Output Requirements:

Synthesize findings directly:
1. Primary temporal pattern (cluster, progression, or isolation)
2. How "${topic}" fits into Aaron's thinking during this period
3. Specific contextual influences (cite note slugs and dates)
4. Chronological evolution of related ideas
5. Insights about Aaron's concerns or focus during that time window

Begin temporal analysis.`,
              },
            },
          ],
        }
      },
    )

    this.server.prompt(
      "gap_analysis",
      "Analyze content coverage and identify expansion opportunities within a topic area",
      {
        broad_topic: z.string().describe("Broad topic area to analyze (e.g., 'interpretability')"),
        specific_aspect: z
          .string()
          .describe("Specific aspect to check coverage for (e.g., 'parameter decomposition')"),
      },
      async (args: { broad_topic: string; specific_aspect: string }) => {
        const { broad_topic, specific_aspect } = args
        return {
          messages: [
            {
              role: "user",
              content: {
                type: "text",
                text: `You are assessing coverage of "${specific_aspect}" within "${broad_topic}" in Aaron's notebook. Your objective is determining whether this aspect is underexplored and identifying strategic expansion opportunities.

Core Principles:

1. **Entropy Interpretation**: High entropy = significant gap (create new content). Medium entropy = partial coverage (deepen existing). Low entropy = well-covered (consider advanced aspects or move on).

2. **Strategic Expansion**: Favor expanding existing notes over creating new ones when adjacent notes exist. New notes only when genuinely distinct topic or no natural home.

3. **Connection Points**: New content must integrate. Identify specific slugs for linking. Isolated notes are anti-patterns.

4. **Sequential Thinking Integration**: Use sequential-thinking MCP when evaluating trade-offs between expansion strategies or planning content structure. Not for reading entropy results.

5. **Validation Over Assumption**: Retrieve target notes to verify actual coverage. Entropy scores are proxies, not ground truth.

Operational Protocol:

**Step 1: Coverage Assessment**
- Call entropy tool with topic "${broad_topic}", query "${specific_aspect}"
- Note entropy level (high/medium/low) and entropy_details scores
- Review expansion_targets: existing, adjacent, potential categories
- Read enrichment_instructions for specific recommendations

**Step 2: Interpretation**
- High entropy (query_top_score < 0.65 or query_in_topic < 2):
  * Significant gap exists
  * High-value expansion opportunity
  * Proceed to Step 3a

- Medium entropy (query_top_score 0.65-0.8):
  * Partial coverage, could be deeper
  * Deepen existing rather than create new
  * Proceed to Step 3b

- Low entropy (query_top_score > 0.8):
  * Well-covered territory
  * Consider advanced aspects or alternative perspectives
  * Or conclude analysis

**Step 3a: High Entropy Strategy**
- Check adjacent files count. If >0, expand one of them instead of creating new
- If creating new: Use suggested_path from enrichment_instructions
- Retrieve potential connection points to verify integration strategy
- Use sequential-thinking to plan structure and key concepts to cover

**Step 3b: Medium Entropy Strategy**
- Retrieve existing files (from expansion_targets.existing)
- Identify specific sections to deepen or expand
- Note where in existing content the aspect is currently mentioned
- Plan how to add dedicated coverage without redundancy

**Step 4: Validation**
- Retrieve 2-3 target notes from expansion_targets
- Verify entropy assessment matches actual content
- Identify specific integration points (sections to add, links to create)
- Note any gaps within the aspect itself

**Step 5: Final Recommendation**
- Use sequential-thinking if trade-offs exist (new vs expand, which file to modify)
- Produce concrete action: specific file path, section title, key concepts
- List specific slugs to link from new/expanded content

Tool Usage Rules:

- **entropy**: Once. Results are deterministic for given topic/query pair.
- **retrieve**: 2-4 target notes maximum for validation.
- **sequential-thinking**: When planning structure or evaluating trade-offs, not for reading JSON.
- **rabbithole**: Optional for additional validation if entropy results surprising.

What NOT to do:

- Never create new notes when adjacent notes can be expanded
- Never skip retrieval validation of target notes
- Never use sequential-thinking to interpret entropy JSON
- Never suggest expansions without specific file paths
- Never recommend content without identifying integration points

Output Requirements:

State directly:
1. Entropy level and what it means for "${specific_aspect}"
2. Specific action: create note at [path] OR expand [slug] OR well-covered
3. Key concepts to address in expansion (be concrete)
4. Integration strategy: which notes to link from/to (cite slugs)
5. Tags to add for knowledge graph integration

Execute analysis.`,
              },
            },
          ],
        }
      },
    )
  }
}

export default Garden
