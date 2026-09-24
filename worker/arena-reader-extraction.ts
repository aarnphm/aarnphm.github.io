import type DefuddleType from 'defuddle'
import type { DOMPurify as Purifier } from 'dompurify'

declare const Defuddle: typeof DefuddleType
declare const DOMPurify: Purifier
declare const arenaReaderFetch: (
  url: string,
  headers: Record<string, string>,
) => Promise<ArenaExtractionResponse>

export interface ArenaExtractionResponse {
  body: string
  status: number
  contentType: string
  url: string
}

export interface ArenaExtractionInput {
  html: string
  finalUrl: string
  idPrefix: string
}

export interface ArenaExtractedDocument {
  title: string
  readerHtml: string | null
  text: string
  articleLength: number
  hasArticle: boolean
  imageUrls: string[]
}

// This function runs only in the isolated parsing page. Keep it self-contained so
// Puppeteer can serialize it without importing Worker bindings into the browser.
export async function extractArenaReaderDocument(
  input: ArenaExtractionInput,
): Promise<ArenaExtractedDocument> {
  const source = new DOMParser().parseFromString(input.html, 'text/html')
  if (source.querySelectorAll('*').length > 30_000)
    throw new Error('The source document exceeds the reader element limit.')
  for (const node of source.querySelectorAll('base')) node.remove()
  const base = source.createElement('base')
  base.href = input.finalUrl
  source.head.appendChild(base)
  for (const image of source.querySelectorAll('img')) {
    const width = Number(image.getAttribute('width'))
    const height = Number(image.getAttribute('height'))
    if ((width > 0 && width <= 1) || (height > 0 && height <= 1)) {
      image.remove()
      continue
    }
    const lazySource = image.getAttribute('data-src') || image.getAttribute('data-original')
    if (lazySource) image.setAttribute('src', lazySource)
    // RDFa resource identifies Wikipedia's file page, not the image bytes. Defuddle's
    // lazy-image heuristic otherwise promotes this metadata URL into src.
    image.removeAttribute('resource')
  }
  type FigureLayout = { alignment: 'left' | 'right' | 'none'; width: number; group: Element | null }
  const figureLayouts = new Map<string, FigureLayout | null>()
  if (/(^|\.)wikipedia\.org$/.test(new URL(input.finalUrl).hostname)) {
    for (const original of source.querySelectorAll(
      '#mw-content-text figure, #mw-content-text .thumb:not(.tmulti), #mw-content-text .tmulti .tsingle',
    )) {
      const image = original.querySelector('img')
      const caption = original.querySelector('figcaption, .thumbcaption')
      if (!image || !caption || original.querySelectorAll('img').length !== 1) continue
      const group = original.closest('.tmulti')
      const classes = (group ?? original).classList
      const alignment =
        classes.contains('mw-halign-left') ||
        classes.contains('tleft') ||
        classes.contains('floatleft')
          ? 'left'
          : classes.contains('mw-halign-none') || classes.contains('mw-halign-center')
            ? 'none'
            : classes.contains('mw-halign-right') ||
                classes.contains('tright') ||
                classes.contains('floatright') ||
                original.matches('figure[typeof~="mw:File/Thumb"], figure[typeof~="mw:File/Frame"]')
              ? 'right'
              : 'none'
      const width = Number(image.getAttribute('width'))
      figureLayouts.set(
        image.src,
        figureLayouts.has(image.src)
          ? null
          : {
              alignment,
              width: Number.isInteger(width) && width > 1 && width <= 1600 ? width : 250,
              group,
            },
      )
      // Defuddle rebuilds native figures from one image and plain caption text.
      // Keep the caption in its parsing flow so links and footnotes are standardized too.
      const figure = source.createElement('div')
      figure.setAttribute('role', 'figure')
      const legend = source.createElement('figcaption')
      legend.append(...Array.from(caption.childNodes))
      figure.append(image, legend)
      original.replaceWith(figure)
    }
    // The reader displays figures inline, so the preview image must not make
    // Defuddle discard a captioned image as a duplicate cover after normalization.
    if (figureLayouts.size) {
      for (const preview of source.querySelectorAll(
        'meta[property="og:image"], meta[name="twitter:image"]',
      ))
        preview.remove()
    }
  }
  const headingIds = new Map<string, string | null>()
  for (const heading of source.querySelectorAll('h1, h2, h3, h4, h5, h6')) {
    const text = heading.textContent?.replace(/\s+/g, ' ').trim()
    if (!text) continue
    headingIds.set(text, headingIds.has(text) ? null : heading.getAttribute('id'))
  }
  // DOMParser keeps source scripts inert. Defuddle needs their JSON-LD and math data
  // before the sanitizer removes executable content from the extracted article.
  const readable = new DOMParser().parseFromString(source.documentElement.outerHTML, 'text/html')
  // Defuddle strips the fn prefix from endnote definitions when matching references.
  // Give semantic endnotes a neutral ID in the parsing clone so both sides still match.
  const endnoteIds = new Map<string, string>()
  for (const note of readable.querySelectorAll('[role="doc-endnotes"] li[id^="fn"]')) {
    let id = `arena-${note.id}`
    while (readable.getElementById(id)) id = `arena-${id}`
    endnoteIds.set(note.id, id)
    note.id = id
  }
  for (const reference of readable.querySelectorAll('a[href^="#"]')) {
    const id = endnoteIds.get(reference.getAttribute('href')?.slice(1) ?? '')
    if (!id) continue
    reference.setAttribute('href', `#${id}`)
    reference.setAttribute('data-type', 'noteref')
  }
  // Object methods remain self-contained when Wrangler preserves function names.
  const parser = {
    create() {
      const doc = new DOMParser().parseFromString(readable.documentElement.outerHTML, 'text/html')
      Object.defineProperty(doc, 'URL', { value: input.finalUrl, configurable: true })
      return new Defuddle(doc, {
        url: input.finalUrl,
        // Images are not fetched during capture, so intrinsic dimensions are unavailable.
        removeSmallImages: false,
        async fetch(resource, init) {
          const request = new Request(resource, init)
          if (request.method !== 'GET' || request.body || finished)
            throw new TypeError('Article extraction only permits bounded public GET requests.')
          request.signal.throwIfAborted()
          const headers: Record<string, string> = {}
          for (const name of ['Accept', 'Accept-Language']) {
            const value = request.headers.get(name)
            if (value) headers[name] = value
          }
          const result = await arenaReaderFetch(request.url, headers)
          request.signal.throwIfAborted()
          const response = new Response(
            [204, 205, 304].includes(result.status) ? null : result.body,
            { status: result.status, headers: { 'Content-Type': result.contentType } },
          )
          Object.defineProperty(response, 'url', { value: result.url })
          return response
        },
      })
    },
  }
  let finished = false
  let timer: ReturnType<typeof setTimeout> | undefined
  let article: ReturnType<InstanceType<typeof Defuddle>['parse']>
  try {
    article = await Promise.race([
      parser.create().parseAsync(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error('Article extraction timed out.')), 8000)
      }),
    ])
    if (!article.content.trim()) article = parser.create().parse()
  } catch {
    // Async extractors can time out after mutating their document. Retry a fresh clone.
    article = parser.create().parse()
  } finally {
    finished = true
    clearTimeout(timer)
  }
  const imageUrls = new Set<string>()
  const sanitized = DOMPurify.sanitize(article.content, {
    USE_PROFILES: { html: true, mathMl: true },
    FORBID_TAGS: [
      'style',
      'form',
      'input',
      'textarea',
      'select',
      'button',
      'iframe',
      'object',
      'embed',
      'audio',
      'video',
      'source',
      'link',
      'meta',
      'base',
      'svg',
      'template',
    ],
    FORBID_ATTR: ['style', 'srcset', 'sizes', 'name', 'autofocus', 'contenteditable'],
    ALLOW_DATA_ATTR: false,
    ADD_ATTR: ['data-lang', 'data-latex', 'data-callout'],
  })
  const safe = new DOMParser().parseFromString(sanitized, 'text/html').body
  const figures = new Map<HTMLElement, FigureLayout>()
  for (const wrapper of safe.querySelectorAll('div[role="figure"]')) {
    const image = wrapper.querySelector('img')
    const caption = wrapper.querySelector(':scope > figcaption')
    if (!image || !caption || wrapper.querySelectorAll('img').length !== 1) continue
    const figure = safe.ownerDocument.createElement('figure')
    figure.append(image, caption)
    wrapper.replaceWith(figure)
    const src = image.getAttribute('src') ?? ''
    if (URL.canParse(src, input.finalUrl)) {
      const layout = figureLayouts.get(new URL(src, input.finalUrl).href)
      if (layout) figures.set(figure, layout)
    }
  }
  // Defuddle removes heading IDs while retaining links to them. Restore only unique
  // source headings; ambiguous or removed targets link back to the original document.
  const ids = new Set(Array.from(safe.querySelectorAll('[id]'), node => node.id))
  for (const heading of safe.querySelectorAll('h1, h2, h3, h4, h5, h6')) {
    const id = headingIds.get(heading.textContent?.replace(/\s+/g, ' ').trim() ?? '')
    if (!heading.id && id && !ids.has(id)) {
      heading.id = id
      ids.add(id)
    }
  }
  for (const element of safe.querySelectorAll('*')) {
    const classes = Array.from(element.classList).filter(name =>
      /^(callout(?:-title(?:-inner)?|-content)?|footnotes?|footnote-backref)$/.test(name),
    )
    if (element.hasAttribute('data-callout') && !classes.includes('callout'))
      classes.push('callout')
    element.removeAttribute('class')
    if (classes.length)
      element.setAttribute('class', classes.map(name => `arena-${name}`).join(' '))
    for (const attribute of ['data-lang', 'data-callout']) {
      const value = element.getAttribute(attribute)
      if (value && !/^[a-z\d_+-]{1,64}$/i.test(value)) element.removeAttribute(attribute)
    }
    if (element.localName !== 'math') element.removeAttribute('data-latex')
    for (const attribute of ['width', 'height']) {
      const dimension = Number(element.getAttribute(attribute))
      if (
        element.tagName !== 'IMG' ||
        !Number.isInteger(dimension) ||
        dimension < 2 ||
        dimension > 8192
      )
        element.removeAttribute(attribute)
    }
    // Retain only semantic attributes. Every URL-bearing attribute is handled below.
    for (const attribute of Array.from(element.attributes)) {
      if (
        ![
          'id',
          'class',
          'data-lang',
          'data-latex',
          'data-callout',
          'href',
          'src',
          'alt',
          'width',
          'height',
          'title',
          'colspan',
          'rowspan',
          'scope',
          'start',
          'reversed',
          'value',
          'datetime',
          'lang',
          'dir',
          'display',
          'mathvariant',
          'mathsize',
          'stretchy',
          'fence',
          'separator',
          'encoding',
          'accent',
          'accentunder',
          'displaystyle',
          'scriptlevel',
          'columnalign',
          'rowalign',
          'columnspan',
          'rowspan',
        ].includes(attribute.name)
      )
        element.removeAttribute(attribute.name)
    }
    const id = element.getAttribute('id')
    if (id) element.setAttribute('id', `${input.idPrefix}${id}`)
    const href = element.getAttribute('href')
    if (href !== null) {
      element.removeAttribute('href')
      if (element.tagName === 'A') {
        try {
          const target = new URL(href, input.finalUrl)
          const origin = new URL(input.finalUrl)
          const local =
            target.origin === origin.origin &&
            target.pathname === origin.pathname &&
            target.search === origin.search &&
            target.hash.length > 1
          let fragment = target.hash.slice(1)
          if (local) {
            try {
              fragment = decodeURIComponent(fragment)
            } catch {
              // Keep malformed fragments on the original document.
            }
          }
          if (local && ids.has(fragment)) {
            element.setAttribute('href', `#${input.idPrefix}${fragment}`)
          } else if (
            ['https:', 'http:', 'mailto:', 'tel:'].includes(target.protocol) &&
            !target.username &&
            !target.password
          ) {
            element.setAttribute('href', target.href)
            element.setAttribute('target', '_blank')
            element.setAttribute('rel', 'noopener noreferrer')
          }
        } catch {
          // Invalid source links are left as text.
        }
      }
    }
    const src = element.getAttribute('src')
    element.removeAttribute('src')
    if (src && element.tagName === 'IMG') {
      try {
        const target = new URL(src, input.finalUrl)
        if (
          ['https:', 'http:'].includes(target.protocol) &&
          !target.username &&
          !target.password &&
          imageUrls.size < 300
        ) {
          imageUrls.add(target.href)
          element.setAttribute(
            'data-arena-image',
            String(Array.from(imageUrls).indexOf(target.href)),
          )
          element.setAttribute('loading', 'lazy')
          element.setAttribute('decoding', 'async')
          element.setAttribute('referrerpolicy', 'no-referrer')
        }
      } catch {
        // An invalid source image retains its alternative text.
      }
    }
  }
  for (const [figure, layout] of figures) {
    figure.className = `arena-figure arena-figure-${layout.alignment}`
    figure.setAttribute('data-arena-figure-width', String(layout.width))
  }
  for (const [figure, layout] of figures) {
    if (!layout.group || figure.parentElement?.classList.contains('arena-figure-group')) continue
    const siblings = [figure]
    let next = figure.nextElementSibling
    while (next instanceof HTMLElement && figures.get(next)?.group === layout.group) {
      siblings.push(next)
      next = next.nextElementSibling
    }
    if (siblings.length < 2) continue
    const group = safe.ownerDocument.createElement('figure')
    group.className = `arena-figure arena-figure-group arena-figure-${layout.alignment}`
    const width = siblings.reduce((sum, sibling) => sum + (figures.get(sibling)?.width ?? 0), 0)
    group.setAttribute(
      'data-arena-figure-width',
      String(Math.min(1600, width + 16 * (siblings.length - 1))),
    )
    figure.before(group)
    group.append(...siblings)
  }
  const html = safe.innerHTML.trim()
  const text = safe.textContent?.trim() ?? ''
  return {
    title: article.title || source.title || '',
    readerHtml: html || null,
    text: text.slice(0, 500_000),
    articleLength: text.length,
    hasArticle: source.querySelector('article, main, [role="main"]') !== null,
    imageUrls: Array.from(imageUrls),
  }
}
