import type { Readability as ReadabilityType } from '@mozilla/readability'
import type { DOMPurify as Purifier } from 'dompurify'

declare const Readability: typeof ReadabilityType
declare const DOMPurify: Purifier

export interface ArenaExtractionInput {
  html: string
  finalUrl: string
  idPrefix: string
}

export interface ArenaExtractedDocument {
  title: string
  readerHtml: string | null
  documentHtml: string
  text: string
  documentLength: number
  articleLength: number
  hasArticle: boolean
  imageUrls: string[]
}

// This function runs only in the network-disabled parsing page. Keep it self-contained so
// Puppeteer can serialize it without importing Worker bindings into the browser.
export function extractArenaReaderDocument(input: ArenaExtractionInput): ArenaExtractedDocument {
  const source = new DOMParser().parseFromString(input.html, 'text/html')
  for (const node of source.querySelectorAll('base,script,style,noscript,template')) node.remove()
  const base = source.createElement('base')
  base.href = input.finalUrl
  source.head.appendChild(base)
  const readable = new DOMParser().parseFromString(source.documentElement.outerHTML, 'text/html')
  const article = new Readability(readable, {
    keepClasses: false,
    charThreshold: 100,
    maxElemsToParse: 30_000,
  }).parse()
  const modes = [article?.content ?? null, source.body.innerHTML]
  const output: (string | null)[] = []
  const imageUrls = new Set<string>()
  for (const markup of modes) {
    if (markup === null) {
      output.push(null)
      continue
    }
    const sanitized = DOMPurify.sanitize(markup, {
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
      FORBID_ATTR: ['style', 'class', 'srcset', 'sizes', 'name', 'autofocus', 'contenteditable'],
      ALLOW_DATA_ATTR: false,
    })
    const safe = new DOMParser().parseFromString(sanitized, 'text/html').body
    for (const element of safe.querySelectorAll('*')) {
      // Retain only semantic attributes. Every URL-bearing attribute is handled below.
      for (const attribute of Array.from(element.attributes)) {
        if (
          ![
            'id',
            'href',
            'src',
            'alt',
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
            if (local) {
              let fragment = target.hash.slice(1)
              try {
                fragment = decodeURIComponent(fragment)
              } catch {
                // An invalid encoded fragment remains an inert, unmatched local anchor.
              }
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
    output.push(safe.innerHTML)
  }
  const text = article?.textContent?.trim() || source.body.textContent?.trim() || ''
  return {
    title: article?.title || source.title || '',
    readerHtml: output[0] ?? null,
    documentHtml: output[1] ?? '',
    text: text.slice(0, 500_000),
    documentLength: source.body.textContent?.trim().length ?? 0,
    articleLength: article?.textContent?.trim().length ?? 0,
    hasArticle: source.querySelector('article, main, [role="main"]') !== null,
    imageUrls: Array.from(imageUrls),
  }
}
