export interface StackedNoteData {
  slug: string
  title: string
  content: string
  metadata?: string
  state: 'pending' | 'ready' | 'failed'
}

interface ContentIndexEntry {
  title: string
  links: string[]
  tags: string[]
  content: string
  richContent?: string
  date?: string
  description?: string
  fileData?: { dates?: { created: string; modified: string } }
  readingTime?: { text: string; minutes: number; time: number; words: number }
}

type ContentIndex = Record<string, ContentIndexEntry>

interface StackedEnv {
  ASSETS: { fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> }
}

interface HtmlRewriterElement {
  before(content: string, options?: { html?: boolean }): void
  after(content: string, options?: { html?: boolean }): void
  append(content: string, options?: { html?: boolean }): void
  setAttribute(name: string, value: string): void
  getAttribute(name: string): string | null
}

interface HtmlRewriterHandlers {
  element(el: HtmlRewriterElement): void
}

interface HtmlRewriter {
  on(selector: string, handlers: HtmlRewriterHandlers): HtmlRewriter
  transform(response: Response): Response
}

declare const HTMLRewriter: { new (): HtmlRewriter }

const NOTE_CONTENT_WIDTH = 620
const NOTE_TITLE_WIDTH = 40
const CONTENT_INDEX_TTL_MS = 60_000
const SERVER_BODY_TAIL_COUNT = 1
const STACKED_NOTE_START = '<!--__STACKED_NOTE_START__-->'
const STACKED_NOTE_END = '<!--__STACKED_NOTE_END__-->'

let contentIndexCache: { value: ContentIndex | null; expires: number } | null = null
let contentIndexInflight: Promise<ContentIndex | null> | null = null

export function hashSlug(slug: string): string {
  const safePath = slug.toString().replace(/\./g, '___DOT___')
  return btoa(safePath).replace(/=+$/, '')
}

async function decodeStackedHash(hash: string): Promise<string | null> {
  try {
    const decoded = atob(hash)
    const restored = decoded.replace(/___DOT___/g, '.')
    if (restored.match(/^[a-zA-Z0-9/.-]+$/)) return restored
  } catch {}
  return null
}

function extractTitle(html: string): string {
  const titleMatch = html.match(/<title[^>]*>(.*?)<\/title>/i)
  if (titleMatch) {
    return titleMatch[1].trim().replace(/ \| .*$/, '')
  }
  return ''
}

function stripHtmlTags(html: string): string {
  let text = ''
  let inTag = false
  let quote: string | undefined
  for (const char of html) {
    if (inTag) {
      if (quote) {
        if (char === quote) quote = undefined
      } else if (char === '"' || char === "'") {
        quote = char
      } else if (char === '>') {
        inTag = false
      }
      continue
    }
    if (char === '<') {
      inTag = true
      continue
    }
    text += char
  }
  return text
}

async function extractPopoverHintContent(html: string): Promise<string> {
  const markedHtml = await new HTMLRewriter()
    .on('main .popover-hint', {
      element(el) {
        el.before(STACKED_NOTE_START, { html: true })
        el.after(STACKED_NOTE_END, { html: true })
      },
    })
    .transform(new Response(html, { headers: { 'Content-Type': 'text/html; charset=utf-8' } }))
    .text()

  const chunks: string[] = []
  let offset = 0
  while (offset < markedHtml.length) {
    const start = markedHtml.indexOf(STACKED_NOTE_START, offset)
    if (start === -1) break
    const contentStart = start + STACKED_NOTE_START.length
    const end = markedHtml.indexOf(STACKED_NOTE_END, contentStart)
    if (end === -1) break
    const chunk = markedHtml.slice(contentStart, end).trim()
    offset = end + STACKED_NOTE_END.length
    if (!chunk) continue
    if (chunk.includes('page-footer') && stripHtmlTags(chunk).trim().length === 0) continue
    chunks.push(chunk)
  }

  return chunks.join('\n')
}

async function getContentIndex(env: StackedEnv, request: Request): Promise<ContentIndex | null> {
  const now = Date.now()
  if (contentIndexCache && contentIndexCache.expires > now) {
    return contentIndexCache.value
  }
  if (contentIndexInflight) return contentIndexInflight

  const indexUrl = new URL('/static/contentIndex.json', request.url)
  contentIndexInflight = env.ASSETS.fetch(indexUrl.toString())
    .then(indexResp => {
      if (!indexResp.ok) return null
      return indexResp.json() as Promise<ContentIndex>
    })
    .then(value => {
      contentIndexCache = { value, expires: Date.now() + CONTENT_INDEX_TTL_MS }
      contentIndexInflight = null
      return value
    })
    .catch(error => {
      console.error('failed to fetch content index:', error)
      contentIndexInflight = null
      return null
    })

  return contentIndexInflight
}

function buildMetadataFooter(entry: ContentIndexEntry | undefined): string {
  if (!entry) return ''

  const date = entry.fileData?.dates?.modified
    ? new Date(entry.fileData.dates.modified)
    : entry.date
      ? new Date(entry.date)
      : null

  if (!date) return ''

  const readingTime = entry.readingTime?.minutes || 0

  return `<div class="published">
  <span lang="fr" class="metadata" dir="auto">dernière modification par <time datetime="${date.toISOString()}">${formatDate(date)}</time> (${readingTime} min de lecture)</span>
</div>`
}

function formatDate(date: Date): string {
  return date.toLocaleDateString('fr-FR', { year: 'numeric', month: 'short', day: '2-digit' })
}

export function failedNoteData(slug: string, title: string = slug): StackedNoteData {
  return {
    slug,
    title,
    content: `<div class="stacked-note-status stacked-note-status-failed"><p>Impossible de charger cette note.</p></div>`,
    metadata: '',
    state: 'failed',
  }
}

function pendingNoteData(slug: string, contentIndex: ContentIndex | null): StackedNoteData {
  return {
    slug,
    title: contentIndex?.[slug]?.title || slug,
    content: '<div class="stacked-note-status" role="status">chargement...</div>',
    metadata: '',
    state: 'pending',
  }
}

function shouldIncludeServerBody(index: number, totalCount: number): boolean {
  return index >= Math.max(0, totalCount - SERVER_BODY_TAIL_COUNT)
}

async function fetchNoteData(
  slug: string,
  env: StackedEnv,
  request: Request,
  contentIndex: ContentIndex | null,
): Promise<StackedNoteData> {
  const noteUrl = new URL(`/${slug}`, request.url)
  noteUrl.search = ''
  noteUrl.hash = ''

  const noteResp = await env.ASSETS.fetch(
    new Request(noteUrl.toString(), { method: 'GET', headers: { Accept: 'text/html' } }),
  )

  if (!noteResp.ok) return failedNoteData(slug)

  const html = await noteResp.text()
  const content = await extractPopoverHintContent(html)
  const title = extractTitle(html) || slug

  if (!content) return failedNoteData(slug, title)

  const entry = contentIndex?.[slug]
  const metadata = buildMetadataFooter(entry)

  return { slug, title, content, metadata, state: 'ready' }
}

export function buildStackedNoteHtml(
  note: StackedNoteData,
  index: number,
  totalCount: number,
): string {
  const escapedSlug = note.slug.replace(/&/g, '&amp;').replace(/"/g, '&quot;')

  const escapedTitle = note.title
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')

  const left = index * NOTE_TITLE_WIDTH
  const right =
    -(NOTE_CONTENT_WIDTH - NOTE_TITLE_WIDTH) + (totalCount - index - 1) * NOTE_TITLE_WIDTH

  return `<div class="stacked-note ${note.state}" id="${hashSlug(note.slug)}" data-slug="${escapedSlug}" data-state="${note.state}" style="left: ${left}px; right: ${right}px;">
  <div class="stacked-content">
    ${note.content}
    ${note.metadata || ''}
  </div>
  <div class="stacked-title">${escapedTitle}</div>
</div>`
}

export function dedupeSlugs(slugs: string[]): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const slug of slugs) {
    if (seen.has(slug)) continue
    seen.add(slug)
    result.push(slug)
  }
  return result
}

async function timed<T>(name: string, timings: string[], fn: () => Promise<T>): Promise<T> {
  const start = performance.now()
  try {
    return await fn()
  } finally {
    timings.push(`${name};dur=${Math.max(0, performance.now() - start).toFixed(1)}`)
  }
}

export async function handleStackedNotesRequest(
  request: Request,
  env: StackedEnv,
): Promise<Response | null> {
  try {
    const url = new URL(request.url)
    const stackedParams = url.searchParams.getAll('stackedNotes')
    const timings: string[] = []

    if (stackedParams.length === 0) return null

    const slugs = await Promise.all(stackedParams.map(decodeStackedHash))
    const validSlugs = dedupeSlugs(slugs.filter(Boolean) as string[])

    if (validSlugs.length === 0) return null

    const firstSlug = validSlugs[0]
    const baseUrl = new URL(`/${firstSlug}`, request.url)
    baseUrl.search = ''
    baseUrl.hash = ''

    const baseResp = await timed('base', timings, () =>
      env.ASSETS.fetch(
        new Request(baseUrl.toString(), { method: 'GET', headers: { Accept: 'text/html' } }),
      ),
    )

    if (!baseResp.ok) return null

    const baseHtml = await timed('baseText', timings, () => baseResp.text())

    const contentIndex = await timed('contentIndex', timings, () => getContentIndex(env, request))

    const totalCount = validSlugs.length
    const notesData = await timed('notes', timings, () =>
      Promise.all(
        validSlugs.map((slug, index) =>
          shouldIncludeServerBody(index, totalCount)
            ? fetchNoteData(slug, env, request, contentIndex)
            : Promise.resolve(pendingNoteData(slug, contentIndex)),
        ),
      ),
    )

    if (notesData.length === 0) return null

    const stackedNotesHtml = notesData
      .map((note, index) => buildStackedNoteHtml(note, index, totalCount))
      .join('\n')

    const rewriter = new HTMLRewriter()
      .on('.stacked-notes-column', {
        element(el) {
          el.append(stackedNotesHtml, { html: true })
        },
      })
      .on('#stacked-notes-container', {
        element(el) {
          el.setAttribute('class', 'all-col active')
        },
      })
      .on('body', {
        element(el) {
          const existingClass = el.getAttribute('class') || ''
          el.setAttribute('class', existingClass + ' stack-mode')
        },
      })
      .on('#stacked-note-toggle', {
        element(el) {
          el.setAttribute('aria-checked', 'true')
        },
      })
      .on('.header', {
        element(el) {
          const existingClass = el.getAttribute('class') || ''
          el.setAttribute('class', existingClass + ' grid all-col')
        },
      })

    const finalHtml = await rewriter.transform(new Response(baseHtml)).text()

    return new Response(finalHtml, {
      headers: {
        'Content-Type': 'text/html; charset=utf-8',
        'Content-Security-Policy': "frame-ancestors 'self' *",
        'Server-Timing': timings.join(', '),
      },
    })
  } catch (e) {
    console.error('failed to handle stacked notes request:', e)
    return null
  }
}
