interface StackedNoteData {
  slug: string
  title: string
  content: string
  metadata?: string
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

const NOTE_CONTENT_WIDTH = 620
const NOTE_TITLE_WIDTH = 40

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

function extractTitle(html: string): string {
  const titleMatch = html.match(/<title[^>]*>(.*?)<\/title>/i)
  if (titleMatch) {
    return titleMatch[1].trim().replace(/ \| .*$/, "")
  }
  return ""
}

function extractPopoverHintContent(html: string): string {
  const mainMatch = html.match(/<main[^>]*>([\s\S]*?)<\/main>/i)
  if (!mainMatch) return ""

  const mainHtml = mainMatch[1]

  const popoverHintRegex =
    /<([a-z][a-z0-9]*)\s[^>]*class="[^"]*popover-hint[^"]*"[^>]*>[\s\S]*?<\/\1>/gi
  const matches = [...mainHtml.matchAll(popoverHintRegex)]

  if (matches.length === 0) return ""

  const filtered = matches
    .map((m) => m[0])
    .filter((html) => {
      const isPageFooter = html.includes("page-footer")
      if (!isPageFooter) return true

      const textContent = html.replace(/<[^>]*>/g, "").trim()
      return textContent.length > 0
    })

  return filtered.join("\n")
}

async function getContentIndex(env: Env, request: Request): Promise<ContentIndex | null> {
  const indexUrl = new URL("/static/contentIndex.json", request.url)
  const indexResp = await env.ASSETS.fetch(indexUrl.toString())

  if (!indexResp.ok) return null

  return (await indexResp.json()) as ContentIndex
}

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

function formatDate(date: Date): string {
  return date.toLocaleDateString("fr-FR", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  })
}

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
  const content = extractPopoverHintContent(html)

  if (!content) return null

  const title = extractTitle(html)

  const entry = contentIndex?.[slug]
  const metadata = buildMetadataFooter(entry)

  return { slug, title, content, metadata }
}

function buildStackedNoteHtml(note: StackedNoteData, index: number, totalCount: number): string {
  const escapedSlug = note.slug.replace(/&/g, "&amp;").replace(/"/g, "&quot;")

  const escapedTitle = note.title
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;")

  const left = index * NOTE_TITLE_WIDTH
  const right =
    -(NOTE_CONTENT_WIDTH - NOTE_TITLE_WIDTH) + (totalCount - index - 1) * NOTE_TITLE_WIDTH

  return `<div class="stacked-note" id="${hashSlug(note.slug)}" data-slug="${escapedSlug}" style="left: ${left}px; right: ${right}px;">
  <div class="stacked-content">
    ${note.content}
    ${note.metadata || ""}
  </div>
  <div class="stacked-title">${escapedTitle}</div>
</div>`
}

export async function handleStackedNotesRequest(
  request: Request,
  env: Env,
): Promise<Response | null> {
  try {
    const url = new URL(request.url)
    const stackedParams = url.searchParams.getAll("stackedNotes")

    if (stackedParams.length === 0) return null

    const slugs = await Promise.all(stackedParams.map(decodeStackedHash))
    const validSlugs = slugs.filter(Boolean) as string[]

    if (validSlugs.length === 0) return null

    const firstSlug = validSlugs[0]
    const baseUrl = new URL(`/${firstSlug}`, request.url)
    baseUrl.search = ""
    baseUrl.hash = ""

    const baseResp = await env.ASSETS.fetch(
      new Request(baseUrl.toString(), {
        method: "GET",
        headers: { Accept: "text/html" },
      }),
    )

    if (!baseResp.ok) return null

    const baseHtml = await baseResp.text()

    const contentIndex = await getContentIndex(env, request)

    const notesData = (
      await Promise.all(validSlugs.map((slug) => fetchNoteData(slug, env, request, contentIndex)))
    ).filter(Boolean) as StackedNoteData[]

    if (notesData.length === 0) return null

    const totalCount = notesData.length
    const stackedNotesHtml = notesData
      .map((note, index) => buildStackedNoteHtml(note, index, totalCount))
      .join("\n")

    const rewriter = new HTMLRewriter()
      .on(".stacked-notes-column", {
        element(el) {
          el.append(stackedNotesHtml, { html: true })
        },
      })
      .on("#stacked-notes-container", {
        element(el) {
          el.setAttribute("class", "all-col active")
        },
      })
      .on("body", {
        element(el) {
          const existingClass = el.getAttribute("class") || ""
          el.setAttribute("class", existingClass + " stack-mode")
        },
      })
      .on("#stacked-note-toggle", {
        element(el) {
          el.setAttribute("aria-checked", "true")
        },
      })
      .on(".header", {
        element(el) {
          const existingClass = el.getAttribute("class") || ""
          el.setAttribute("class", existingClass + " grid all-col")
        },
      })

    const finalHtml = await rewriter.transform(new Response(baseHtml)).text()

    return new Response(finalHtml, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Content-Security-Policy": "frame-ancestors 'self' *",
      },
    })
  } catch (e) {
    console.error("failed to handle stacked notes request:", e)
    return null
  }
}
