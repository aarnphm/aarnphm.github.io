interface StackedNoteData {
  slug: string
  title: string
  content: string
  metadata?: string
}

type Env = {
  ASSETS: Fetcher
  STACKED_CACHE?: KVNamespace
  PUBLIC_BASE_URL?: string
}

interface ContentIndexEntry {
  title: string
  links: string[]
  tags: string[]
  content: string
  richContent?: string
  date?: string
  description?: string
  fileData?: {
    dates?: {
      created: string
      modified: string
    }
  }
  readingTime?: {
    text: string
    minutes: number
    time: number
    words: number
  }
}

type ContentIndex = Record<string, ContentIndexEntry>

const STACKED_CACHE_TTL = 300 // 5 minutes
const CONTENT_INDEX_CACHE_KEY = "content-index:json"

// CSS custom property values - must match quartz/components/styles/matuschak.scss
const NOTE_CONTENT_WIDTH = 620 // --note-content-width
const NOTE_TITLE_WIDTH = 40 // --note-title-width

/** Generates URL-safe hash for a slug. Uses base64 with special handling for dots */
function hashSlug(slug: string): string {
  const safePath = slug.toString().replace(/\./g, "___DOT___")
  return btoa(safePath).replace(/=+$/, "")
}

async function decodeStackedHash(hash: string): Promise<string | null> {
  try {
    const decoded = atob(hash)
    const restored = decoded.replace(/___DOT___/g, ".")
    if (restored.match(/^[a-zA-Z0-9/.-]+$/)) return restored
  } catch {}
  return null
}

/**
 * Extract title from HTML. Looks for <title> tag.
 */
function extractTitle(html: string): string {
  const titleMatch = html.match(/<title[^>]*>(.*?)<\/title>/i)
  if (titleMatch) {
    return titleMatch[1].trim().replace(/ \| .*$/, "") // Remove site suffix
  }
  return ""
}

/**
 * Extract all .popover-hint content from main.
 * First extracts <main> content, then parses .popover-hint elements.
 */
async function extractPopoverHintContent(html: string): Promise<string> {
  // Step 1: Extract <main> content using HTMLRewriter (for verification)
  let mainContent = ""
  let captureMain = false

  const mainRewriter = new HTMLRewriter().on("main", {
    element() {
      captureMain = true
    },
    text(text) {
      if (captureMain) {
        mainContent += text.text
      }
    },
  })

  await mainRewriter.transform(new Response(html)).text()

  if (!mainContent) return ""

  // Step 2: Use regex to extract .popover-hint elements from the original HTML's <main> section
  const mainMatch = html.match(/<main[^>]*>([\s\S]*?)<\/main>/i)
  if (!mainMatch) return ""

  const mainHtml = mainMatch[1]

  // Extract all elements with class="...popover-hint..."
  const popoverHintRegex =
    /<([a-z][a-z0-9]*)\s[^>]*class="[^"]*popover-hint[^"]*"[^>]*>[\s\S]*?<\/\1>/gi
  const matches = [...mainHtml.matchAll(popoverHintRegex)]

  if (matches.length === 0) return ""

  // Filter out empty page-footers
  const filtered = matches
    .map((m) => m[0])
    .filter((html) => {
      const isPageFooter = html.includes("page-footer")
      if (!isPageFooter) return true

      // Check if page-footer has content
      const textContent = html.replace(/<[^>]*>/g, "").trim()
      return textContent.length > 0
    })

  return filtered.join("\n")
}

/**
 * Fetch and cache contentIndex.json
 */
async function getContentIndex(env: Env, request: Request): Promise<ContentIndex | null> {
  // Try cache first
  if (env.STACKED_CACHE) {
    const cached = await env.STACKED_CACHE.get(CONTENT_INDEX_CACHE_KEY)
    if (cached) {
      return JSON.parse(cached) as ContentIndex
    }
  }

  // Fetch from ASSETS
  const indexUrl = new URL("/static/contentIndex.json", request.url)
  const indexResp = await env.ASSETS.fetch(indexUrl.toString())

  if (!indexResp.ok) return null

  const index = (await indexResp.json()) as ContentIndex

  // Cache for 1 hour
  if (env.STACKED_CACHE) {
    await env.STACKED_CACHE.put(CONTENT_INDEX_CACHE_KEY, JSON.stringify(index), {
      expirationTtl: 3600,
    })
  }

  return index
}

/**
 * Generate metadata footer HTML for a slug
 */
function buildMetadataFooter(entry: ContentIndexEntry | undefined): string {
  if (!entry) return ""

  const date = entry.fileData?.dates?.modified
    ? new Date(entry.fileData.dates.modified)
    : entry.date
      ? new Date(entry.date)
      : null

  if (!date) return ""

  const readingTime = entry.readingTime?.minutes || 0

  return `<div class="published">
  <span lang="fr" class="metadata" dir="auto">dernière modification par <time datetime="${date.toISOString()}">${formatDate(date)}</time> (${readingTime} min de lecture)</span>
</div>`
}

/**
 * Format date in French locale (matches client's formatDate)
 */
function formatDate(date: Date): string {
  return date.toLocaleDateString("fr-FR", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  })
}

/**
 * Fetch note data for a single slug
 */
async function fetchNoteData(
  slug: string,
  env: Env,
  request: Request,
  contentIndex: ContentIndex | null,
): Promise<StackedNoteData | null> {
  const noteUrl = new URL(`/${slug}`, request.url)
  noteUrl.search = ""
  noteUrl.hash = ""

  const noteResp = await env.ASSETS.fetch(
    new Request(noteUrl.toString(), {
      method: "GET",
      headers: { Accept: "text/html" },
    }),
  )

  if (!noteResp.ok) return null

  const html = await noteResp.text()
  const content = await extractPopoverHintContent(html)

  if (!content) return null

  const title = extractTitle(html)

  // Generate metadata footer
  const entry = contentIndex?.[slug]
  const metadata = buildMetadataFooter(entry)

  return { slug, title, content, metadata }
}

/**
 * Build HTML for a single stacked note with exact positioning
 */
function buildStackedNoteHtml(note: StackedNoteData, index: number, totalCount: number): string {
  // Escape slug for attribute
  const escapedSlug = note.slug.replace(/&/g, "&amp;").replace(/"/g, "&quot;")

  // Escape HTML in title to prevent injection
  const escapedTitle = note.title
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;")

  // Calculate positioning (same logic as client's createNote)
  const left = index * NOTE_TITLE_WIDTH
  const right =
    -(NOTE_CONTENT_WIDTH - NOTE_TITLE_WIDTH) + (totalCount - index - 1) * NOTE_TITLE_WIDTH

  // Generate hashed ID
  const hashedId = hashSlug(note.slug)

  return `<div class="stacked-note" id="${hashedId}" data-slug="${escapedSlug}" style="left: ${left}px; right: ${right}px;">
  <div class="stacked-content">
    ${note.content}
    ${note.metadata || ""}
  </div>
  <div class="stacked-title">${escapedTitle}</div>
</div>`
}

/**
 * Main handler for stacked notes requests.
 * Returns server-rendered HTML with stacked notes injected.
 */
export async function handleStackedNotesRequest(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response | null> {
  try {
    const url = new URL(request.url)
    const stackedParams = url.searchParams.getAll("stackedNotes")

    if (stackedParams.length === 0) return null

    // Decode and validate slugs
    const slugs = await Promise.all(stackedParams.map(decodeStackedHash))
    const validSlugs = slugs.filter(Boolean) as string[]

    if (validSlugs.length === 0) return null

    // Hash slug list to stay under KV key limit (512 bytes)
    const slugsText = validSlugs.join(",")
    const slugsHash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(slugsText))
    const hashHex = Array.from(new Uint8Array(slugsHash))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("")

    // Try cache first
    const cacheKey = `stacked:html:${hashHex}`
    const cached = env.STACKED_CACHE ? await env.STACKED_CACHE.get(cacheKey) : null

    if (cached) {
      return new Response(cached, {
        headers: {
          "Content-Type": "text/html; charset=utf-8",
          "Cache-Control": "s-maxage=300, stale-while-revalidate=60",
          "X-Stacked-Cache": "hit",
          "Content-Security-Policy": "frame-ancestors 'self' *",
        },
      })
    }

    // Fetch the first note's complete HTML as base
    const firstSlug = validSlugs[0]
    const baseUrl = new URL(`/${firstSlug}`, request.url)
    baseUrl.search = "" // Remove query params for base page
    baseUrl.hash = ""

    const baseResp = await env.ASSETS.fetch(
      new Request(baseUrl.toString(), {
        method: "GET",
        headers: { Accept: "text/html" },
      }),
    )

    if (!baseResp.ok) return null

    const baseHtml = await baseResp.text()

    // Fetch contentIndex for metadata
    const contentIndex = await getContentIndex(env, request)

    // Fetch all notes in parallel
    const noteDataPromises = validSlugs.map((slug) =>
      fetchNoteData(slug, env, request, contentIndex),
    )
    const notesData = (await Promise.all(noteDataPromises)).filter(Boolean) as StackedNoteData[]

    if (notesData.length === 0) return null

    // Build stacked notes HTML with positioning
    const totalCount = notesData.length
    const stackedNotesHtml = notesData
      .map((note, index) => buildStackedNoteHtml(note, index, totalCount))
      .join("\n")

    // Use HTMLRewriter to inject stacked notes and activate container
    const rewriter = new HTMLRewriter()
      // Inject stacked notes into the column
      .on(".stacked-notes-column", {
        element(el) {
          el.append(stackedNotesHtml, { html: true })
        },
      })
      // Activate stacked container
      .on("#stacked-notes-container", {
        element(el) {
          el.setAttribute("class", "all-col active")
        },
      })
      // Add stack-mode class to body
      .on("body", {
        element(el) {
          const existingClass = el.getAttribute("class") || ""
          el.setAttribute("class", existingClass + " stack-mode")
        },
      })
      // Activate toggle button
      .on("#stacked-note-toggle", {
        element(el) {
          el.setAttribute("aria-checked", "true")
        },
      })
      // Add grid classes to header
      .on(".header", {
        element(el) {
          const existingClass = el.getAttribute("class") || ""
          el.setAttribute("class", existingClass + " grid all-col")
        },
      })

    const response = rewriter.transform(new Response(baseHtml))
    const finalHtml = await response.text()

    // Cache the result
    if (env.STACKED_CACHE) {
      ctx.waitUntil(
        env.STACKED_CACHE.put(cacheKey, finalHtml, { expirationTtl: STACKED_CACHE_TTL }),
      )
    }

    return new Response(finalHtml, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "s-maxage=300, stale-while-revalidate=60",
        "X-Stacked-Cache": "miss",
        "Content-Security-Policy": "frame-ancestors 'self' *",
      },
    })
  } catch (e) {
    console.error("failed to handle stacked notes request:", e)
    return null
  }
}
