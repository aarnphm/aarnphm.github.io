import { normalizeRelativeURLs } from '../../../util/path'
import {
  cacheStackedNotePayload,
  getCachedStackedNotePayload,
  MAX_STACKED_NOTE_PAYLOAD_CACHE_ENTRIES,
  readStackedNotePayload,
  stackedNotePayloadUrl,
  type NoteDocument,
  type StackedNotePayload,
} from '../../../util/stacked-notes'
import { parseHtml } from './html-parser'
import { transformHostInternalLinks } from './link-policy'

const STACKED_NOTE_RETRY_BUTTON_HTML = `<button type="button" data-stacked-retry aria-label="Réessayer">
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <path d="M20 6v5h-5" />
    <path d="M4 18v-5h5" />
    <path d="M18.5 9a7 7 0 0 0-12.3-2.4L4 8.7" />
    <path d="M5.5 15a7 7 0 0 0 12.3 2.4L20 15.3" />
  </svg>
</button>`

function normalizeStackedNoteURLs(html: Document, url: URL) {
  normalizeRelativeURLs(html, url)
  for (const attribute of ['data-href', 'poster']) {
    html.querySelectorAll(`[${attribute}]`).forEach(element => {
      const value = element.getAttribute(attribute)
      if (!value || !/^\.{1,2}\//.test(value)) return
      const resolved = new URL(value, url)
      element.setAttribute(attribute, resolved.pathname + resolved.search + resolved.hash)
    })
  }
}

export class StackedDocumentStore {
  private documentCache: Map<string, NoteDocument> = new Map()
  private inflight: Map<string, Promise<NoteDocument>> = new Map()

  constructor(private retained: (slug: string) => boolean) {}

  get(slug: string): NoteDocument | undefined {
    return this.documentCache.get(slug)
  }

  invalidate(slug: string) {
    this.documentCache.delete(slug)
  }

  clear() {
    this.documentCache.clear()
    this.inflight.clear()
  }

  captureMarkup(
    slug: string,
    title: string,
    markup: string,
    state: NoteDocument['state'],
    url: URL,
    hash: string,
  ) {
    const html = parseHtml(markup)
    normalizeStackedNoteURLs(html, new URL(`/${slug}`, url))
    this.cacheDocument(slug, { slug, title, bodyHtml: html.body.innerHTML, state, hash })
  }

  captureCurrent(slug: string, url: URL) {
    this.cacheDocument(slug, this.currentDocument(slug, url))
  }

  private currentDocument(slug: string, url: URL): NoteDocument {
    const contents = Array.from(
      document.querySelectorAll<HTMLElement>('main .popover-hint'),
    ).filter(element => !element.closest('#stacked-notes-container'))
    const title = document.querySelector<HTMLElement>('main h1')?.textContent || document.title
    const html = parseHtml(this.serializeElements(contents))
    normalizeStackedNoteURLs(html, new URL(`/${slug}`, url))
    return { slug, title, hash: url.hash, bodyHtml: html.body.innerHTML, state: 'ready' }
  }

  private payloadDocument(payload: StackedNotePayload, target: URL, slug: string, hash: string) {
    const html = parseHtml(payload.content)
    normalizeStackedNoteURLs(html, target)
    transformHostInternalLinks(html)
    const elements = Array.from(html.body.children).flatMap(el =>
      el instanceof HTMLElement ? [el] : [],
    )

    return {
      slug,
      title: payload.title,
      hash,
      bodyHtml: this.serializeElements(elements),
      state: payload.state,
    }
  }

  private async fetchContent(url: URL, slug: string): Promise<NoteDocument> {
    const target = new URL(`/${slug}`, url)
    const hash = url.hash

    const cachedPayload = getCachedStackedNotePayload(slug)
    if (cachedPayload) return this.payloadDocument(cachedPayload, target, slug, hash)

    const response = await fetch(stackedNotePayloadUrl(slug), {
      headers: { Accept: 'application/json' },
    }).catch(error => {
      console.error(error)
      return null
    })
    if (!response || !response.ok) return this.failedDocument(slug, hash)

    const json = await response.json().catch(error => {
      console.error(error)
      return null
    })
    const payload = readStackedNotePayload(json)
    if (!payload) return this.failedDocument(slug, hash)
    cacheStackedNotePayload(payload)

    return this.payloadDocument(payload, target, slug, hash)
  }

  private serializeElements(elements: HTMLElement[]): string {
    return elements.map(el => el.outerHTML).join('\n')
  }

  private failedDocument(slug: string, hash?: string, title?: string): NoteDocument {
    return {
      slug,
      title: title || slug,
      hash,
      bodyHtml: `<div class="stacked-note-status stacked-note-status-failed"><p>Impossible de charger cette note.</p>${STACKED_NOTE_RETRY_BUTTON_HTML}</div>`,
      state: 'failed',
    }
  }

  pendingDocument(slug: string, href: URL): NoteDocument {
    return {
      slug,
      title: slug,
      hash: href.hash,
      bodyHtml: `<div class="stacked-note-status" role="status">chargement...</div>`,
      state: 'pending',
    }
  }

  private cacheDocument(slug: string, noteDocument: NoteDocument) {
    this.documentCache.delete(slug)
    while (this.documentCache.size >= MAX_STACKED_NOTE_PAYLOAD_CACHE_ENTRIES) {
      const evictSlug = Array.from(this.documentCache.keys()).find(key => !this.retained(key))
      if (!evictSlug) break
      this.documentCache.delete(evictSlug)
    }
    this.documentCache.set(slug, noteDocument)
  }

  async loadDocument(href: URL, slug: string): Promise<NoteDocument> {
    const cached = this.documentCache.get(slug)
    if (cached && cached.state !== 'pending') {
      this.documentCache.delete(slug)
      this.documentCache.set(slug, cached)
      return cached
    }

    const pending = this.inflight.get(slug)
    if (pending) return pending

    const promise = this.fetchContent(href, slug)
      .catch(error => {
        console.error(`Failed to fetch stacked note ${slug}:`, error)
        return this.failedDocument(slug, href.hash)
      })
      .then(noteDocument => {
        this.cacheDocument(slug, noteDocument)
        this.inflight.delete(slug)
        return noteDocument
      })

    this.inflight.set(slug, promise)
    return promise
  }
}
