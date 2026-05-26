import {
  STACKED_NOTE_METADATA_CLASSES,
  decodeStackedNoteHash,
  hashStackedNoteSlug,
  normalizeStackedNoteSlug,
  stackedNoteMetadataHtml,
} from '../quartz/util/stacked-notes'

export interface StackedNoteData {
  slug: string
  title: string
  content: string
  metadata?: string
  state: 'pending' | 'ready' | 'protected' | 'failed'
}

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
const SERVER_BODY_HEAD_COUNT = 1
const STACKED_NOTE_START = '<!--__STACKED_NOTE_START__-->'
const STACKED_NOTE_END = '<!--__STACKED_NOTE_END__-->'
const STACKED_NOTE_METADATA_SELECTORS = STACKED_NOTE_METADATA_CLASSES.map(
  className => `main .page-header .content-meta > li.${className}`,
)
const STACKED_NOTE_RETRY_BUTTON_HTML = `<button type="button" data-stacked-retry aria-label="Réessayer">
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <path d="M20 6v5h-5" />
    <path d="M4 18v-5h5" />
    <path d="M18.5 9a7 7 0 0 0-12.3-2.4L4 8.7" />
    <path d="M5.5 15a7 7 0 0 0 12.3 2.4L20 15.3" />
  </svg>
</button>`

export function hashSlug(slug: string): string {
  return hashStackedNoteSlug(slug)
}

export { normalizeStackedNoteSlug }

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

async function extractHtmlFragments(html: string, selectors: string[]): Promise<string[]> {
  let rewriter = new HTMLRewriter()
  const handler: HtmlRewriterHandlers = {
    element(el) {
      el.before(STACKED_NOTE_START, { html: true })
      el.after(STACKED_NOTE_END, { html: true })
    },
  }

  for (const selector of selectors) {
    rewriter = rewriter.on(selector, handler)
  }

  const markedHtml = await rewriter
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
    chunks.push(chunk)
  }

  return chunks
}

async function extractPopoverHintContent(html: string): Promise<string> {
  const chunks = await extractHtmlFragments(html, ['main .popover-hint'])
  return chunks
    .filter(chunk => !(chunk.includes('page-footer') && stripHtmlTags(chunk).trim().length === 0))
    .join('\n')
}

async function extractStackedNoteMetadata(html: string): Promise<string> {
  return stackedNoteMetadataHtml(await extractHtmlFragments(html, STACKED_NOTE_METADATA_SELECTORS))
}

export function failedNoteData(slug: string, title: string = slug): StackedNoteData {
  return {
    slug,
    title,
    content: `<div class="stacked-note-status stacked-note-status-failed"><p>Impossible de charger cette note.</p>${STACKED_NOTE_RETRY_BUTTON_HTML}</div>`,
    metadata: '',
    state: 'failed',
  }
}

function protectedNoteData(slug: string, title: string): StackedNoteData {
  return {
    slug,
    title,
    content: `<div class="protected-stacked-note">
  <div class="protected-overlay">
    <div class="protected-message">
      <p>ce contenu est protégé</p>
      <p class="protected-hint">visitez la page principale pour y accéder</p>
    </div>
  </div>
</div>`,
    metadata: '',
    state: 'protected',
  }
}

export function pendingNoteData(slug: string): StackedNoteData {
  return {
    slug,
    title: slug,
    content: '<div class="stacked-note-status" role="status">chargement...</div>',
    metadata: '',
    state: 'pending',
  }
}

export function shouldIncludeServerBody(index: number): boolean {
  return index < SERVER_BODY_HEAD_COUNT
}

async function noteDataFromHtml(slug: string, html: string): Promise<StackedNoteData> {
  const title = extractTitle(html) || slug
  if (/<article\b[^>]*\bdata-protected=["']true["']/i.test(html)) {
    return protectedNoteData(slug, title)
  }

  const [content, metadata] = await Promise.all([
    extractPopoverHintContent(html),
    extractStackedNoteMetadata(html),
  ])

  if (!content) return failedNoteData(slug, title)

  return { slug, title, content, metadata, state: 'ready' }
}

async function fetchNoteData(
  slug: string,
  env: StackedEnv,
  request: Request,
): Promise<StackedNoteData> {
  const noteUrl = new URL(`/${slug}`, request.url)
  noteUrl.search = ''
  noteUrl.hash = ''

  const noteResp = await env.ASSETS.fetch(
    new Request(noteUrl.toString(), { method: 'GET', headers: { Accept: 'text/html' } }),
  )

  if (!noteResp.ok) return failedNoteData(slug)

  const html = await noteResp.text()
  return noteDataFromHtml(slug, html)
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

export async function handleStackedNoteDataRequest(
  request: Request,
  env: StackedEnv,
): Promise<Response> {
  const timings: string[] = []
  const url = new URL(request.url)
  const slug = normalizeStackedNoteSlug(url.searchParams.get('slug'))

  if (!slug) {
    return new Response('missing slug', {
      status: 400,
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    })
  }

  const note = await timed('note', timings, () => fetchNoteData(slug, env, request))

  return new Response(JSON.stringify(note), {
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'public, max-age=60, stale-while-revalidate=300',
      'Server-Timing': timings.join(', '),
    },
  })
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

    const slugs = stackedParams.map(decodeStackedNoteHash)
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

    const firstNoteData = await timed('firstNote', timings, () =>
      noteDataFromHtml(firstSlug, baseHtml),
    )

    const totalCount = validSlugs.length
    const notesData = validSlugs.map((slug, index) =>
      shouldIncludeServerBody(index) ? firstNoteData : pendingNoteData(slug),
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
