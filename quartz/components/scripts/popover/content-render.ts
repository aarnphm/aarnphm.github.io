import xmlFormat from 'xml-formatter'
import { documentPreviewUrl, readDocumentPreview } from '../../../util/document-preview'
import { getContentType } from '../../../util/mime'
import { normalizeRelativeURLs } from '../../../util/path'
import {
  cacheStackedNotePayload,
  getCachedStackedNotePayload,
  readStackedNotePayload,
  type StackedNotePayload,
  stackedNotePayloadUrl,
  stackedNoteRootSlug,
} from '../../../util/stacked-notes'
import { renderDocumentPreview } from '../document-preview-card'
import { currentNavSignal } from '../nav-lifecycle'
import { previewHref, previewText, renderPagePreview } from '../page-preview-card'
import { renderCustomPagePreview } from './custom-page-preview'
import { handlePdfContent } from './pdf-preview'
import { findHashTarget, isAbortError } from './placement'

type ContentHandler = (
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
  hash: string,
  signal: AbortSignal,
) => Promise<void>

export const p = new DOMParser()

export function cloneTelescopicPreview(
  container: ParentNode,
  targetUrl: URL,
  hash: string,
  title: string,
): HTMLElement | null {
  if (!hash) return null
  const target = findHashTarget(container, hash)
  const telescope = target?.closest<HTMLElement>('.telescopic-container')
  if (!telescope) return null
  const preview = document.createElement('div')
  preview.className = 'telescopic-preview'
  const clone = telescope.cloneNode(true)
  if (!(clone instanceof HTMLElement)) return null
  clone.querySelectorAll('.details').forEach(detail => detail.classList.replace('open', 'close'))
  clone
    .querySelectorAll('button.summary')
    .forEach(button => button.setAttribute('aria-expanded', 'false'))
  const source = document.createElement('a')
  const destination = new URL(targetUrl)
  destination.hash = hash
  source.href = destination.toString()
  source.className = 'internal transclude-src'
  source.dataset.noPopover = 'true'
  source.textContent = title
  preview.append(clone, source)
  prefixPopoverIds(preview)
  return preview
}

function cleanAbsoluteElement(element: HTMLElement): HTMLElement {
  const refsAndNotes = element.querySelectorAll<HTMLElement>(
    'section[data-references], section[data-footnotes], [data-skip-preview], .telescopic-container, .fr-margin, .fr-group, .fr-leaders',
  )
  refsAndNotes.forEach(section => section.remove())
  return element
}

function prefixPopoverIds(container: ParentNode) {
  container.querySelectorAll<HTMLElement>('[id]').forEach(el => {
    el.id = `popover-${el.id}`
  })
}

function isStreamDocumentPath(pathname: string): boolean {
  const normalized = pathname.replace(/\/+$/, '')
  return normalized === '/stream' || normalized.startsWith('/stream/on/')
}

export function streamPopoverKey(targetUrl: URL): string {
  const entry = targetUrl.searchParams.get('entry')?.trim()
  const hash = decodeURIComponent(targetUrl.hash).replace(/^#/, '').trim()
  if (isStreamDocumentPath(targetUrl.pathname) && (entry || hash)) {
    return `popover-${targetUrl.pathname}-${entry || hash}`
  }
  return `popover-${targetUrl.pathname}${hash ? `-${hash}` : ''}`
}

export function streamPopoverFetchUrl(targetUrl: URL): URL {
  const fetchUrl = new URL(targetUrl.toString())
  fetchUrl.hash = ''
  if (!isStreamDocumentPath(fetchUrl.pathname)) {
    fetchUrl.search = ''
  }
  return fetchUrl
}

function streamEntryFromTarget(
  container: ParentNode,
  targetUrl: URL,
  hash: string,
): HTMLElement | null {
  const entryId = targetUrl.searchParams.get('entry')?.trim()
  if (entryId) {
    const entry = findHashTarget(container, entryId, 'popover-')?.closest<HTMLElement>(
      '.stream-entry',
    )
    if (entry) return entry
  }

  if (!hash) return null
  return findHashTarget(container, hash, 'popover-')?.closest<HTMLElement>('.stream-entry') ?? null
}

function isolateStreamEntryPreview(container: ParentNode, entry: HTMLElement): HTMLElement | null {
  const article = entry.closest<HTMLElement>('article.stream.popover-hint')
  if (!article || !article.contains(entry)) return null

  const list = entry.closest<HTMLElement>('ol.stream-feed')
  if (list && article.contains(list)) {
    const listClone = list.cloneNode(false)
    if (!(listClone instanceof HTMLElement)) return null
    listClone.appendChild(entry)
    article.replaceChildren(listClone)
  } else {
    article.replaceChildren(entry)
  }

  return container instanceof HTMLElement ? container : article
}

function narrowStreamPopover(popoverInner: HTMLElement, targetUrl: URL, hash: string) {
  if (!isStreamDocumentPath(targetUrl.pathname)) return
  const entry = streamEntryFromTarget(popoverInner, targetUrl, hash)
  if (!entry) return
  isolateStreamEntryPreview(popoverInner, entry)
}

export function cloneCurrentStreamEntryPreview(hash: string): HTMLElement | null {
  const article = document.querySelector<HTMLElement>('article.stream.popover-hint')
  if (!article) return null
  const target = findHashTarget(article, hash)
  const entry = target?.closest<HTMLElement>('.stream-entry')
  if (!entry) return null

  const articleClone = article.cloneNode(false)
  if (!(articleClone instanceof HTMLElement)) return null

  const list = entry.closest<HTMLElement>('ol.stream-feed')
  if (list && article.contains(list)) {
    const listClone = list.cloneNode(false)
    if (!(listClone instanceof HTMLElement)) return null
    listClone.appendChild(entry.cloneNode(true))
    articleClone.appendChild(listClone)
  } else {
    articleClone.appendChild(entry.cloneNode(true))
  }

  prefixPopoverIds(articleClone)
  return cleanAbsoluteElement(articleClone)
}

async function handleImageContent(targetUrl: URL, popoverInner: HTMLDivElement) {
  const img = document.createElement('img')
  img.src = targetUrl.toString()
  img.alt = targetUrl.pathname
  popoverInner.appendChild(img)
}

async function handleXmlContent(response: Response, targetUrl: URL, popoverInner: HTMLDivElement) {
  const contents = await response.text()
  const xml = p.parseFromString(contents, 'application/xml')
  const atom =
    xml.documentElement.localName === 'feed' &&
    xml.documentElement.namespaceURI === 'http://www.w3.org/2005/Atom'
  const channel = atom ? xml.documentElement : xml.querySelector('rss > channel')
  if (channel && !xml.querySelector('parsererror')) {
    const text = (node: Element, name: string) =>
      Array.from(node.children)
        .find(child => child.localName === name)
        ?.textContent?.trim() ?? ''
    const items = Array.from(channel.children).filter(
      child => child.localName === (atom ? 'entry' : 'item'),
    )
    renderPagePreview(
      {
        kind: 'rss',
        title: previewText(text(channel, 'title') || 'rss feed', 100),
        href: targetUrl.href,
        description:
          previewText(text(channel, atom ? 'subtitle' : 'description'), 240) ||
          'subscribe to this feed in your rss reader.',
        meta: [
          atom ? 'atom' : 'rss',
          `${items.length} ${items.length === 1 ? 'entry' : 'entries'}`,
        ],
        rows: items.length
          ? items.slice(0, 4).map(item => {
              const date = new Date(
                atom ? text(item, 'published') || text(item, 'updated') : text(item, 'pubDate'),
              )
              const href = atom
                ? (Array.from(item.children)
                    .find(
                      child =>
                        child.localName === 'link' &&
                        (!child.getAttribute('rel') || child.getAttribute('rel') === 'alternate'),
                    )
                    ?.getAttribute('href') ?? '')
                : text(item, 'link')
              return {
                title: previewText(text(item, 'title') || 'untitled', 100),
                href: href ? previewHref(href, targetUrl.href) : undefined,
                detail: Number.isNaN(date.valueOf()) ? undefined : date.toISOString().slice(0, 10),
              }
            })
          : [{ title: 'no entries yet' }],
      },
      popoverInner,
    )
    return
  }
  const rss = document.createElement('pre')
  rss.classList.add('rss-viewer')
  rss.append(xmlFormat(contents, { indentation: '  ', lineSeparator: '\n' }))
  popoverInner.append(rss)
}

/**
 * Links rendered at runtime, or to a base view route, carry no data-document-preview; their page
 * body still names the type, so the popover shows the same summary card instead of a bookcase or
 * vault shell that only its runtime can draw.
 */
async function renderPageDocumentPreview(
  html: Document,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
): Promise<boolean> {
  const { slug, isBase, isCanvas } = html.body.dataset
  const type = isBase === 'true' ? 'base' : isCanvas === 'true' ? 'canvas' : null
  if (!type || !slug) return false
  const response = await fetch(documentPreviewUrl(slug, window.location.href)).catch(() => null)
  const preview =
    response?.ok === true
      ? readDocumentPreview(await response.json().catch(() => null), type, slug)
      : null
  if (!preview) return false
  popoverInner.closest('.popover')?.classList.add('document-popover')
  renderDocumentPreview(preview, targetUrl.toString(), popoverInner)
  return true
}

async function handleDefaultContent(
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
  hash = '',
  signal = currentNavSignal(),
) {
  const contents = await response.text()
  const html = p.parseFromString(contents, 'text/html')
  normalizeRelativeURLs(html, targetUrl)
  const telescope = cloneTelescopicPreview(
    html,
    targetUrl,
    hash,
    html.querySelector('h1')?.textContent ?? html.title,
  )
  if (telescope) {
    popoverInner.append(telescope)
    return
  }
  if (await renderPageDocumentPreview(html, targetUrl, popoverInner)) return
  if (await renderCustomPagePreview(html, targetUrl, popoverInner, signal)) return
  popoverInner.classList.add('grid')
  prefixPopoverIds(html)
  const elts = [
    ...(html.getElementsByClassName('popover-hint') as HTMLCollectionOf<HTMLDivElement>),
  ].map(cleanAbsoluteElement)
  if (elts.length === 0) return
  popoverInner.append(...elts)
  narrowStreamPopover(popoverInner, targetUrl, hash)
}

const contentHandlers: Record<string, ContentHandler> = {
  image: async (_, targetUrl, popoverInner) => handleImageContent(targetUrl, popoverInner),
  'application/pdf': handlePdfContent,
  'application/xml': handleXmlContent,
  'text/xml': handleXmlContent,
  'application/rss+xml': handleXmlContent,
  'application/atom+xml': handleXmlContent,
  default: handleDefaultContent,
}

export async function populatePopoverContent(
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
  hash = '',
  signal = currentNavSignal(),
) {
  delete popoverInner.dataset.state
  popoverInner.replaceChildren()
  const headerContentType = response.headers.get('Content-Type')
  const contentType = headerContentType
    ? headerContentType.split(';')[0]
    : getContentType(targetUrl)
  const [contentTypeCategory] = contentType.split('/')
  popoverInner.dataset.contentType = contentType ?? undefined

  const handler =
    contentHandlers[contentTypeCategory] ??
    contentHandlers[contentType] ??
    contentHandlers['default']

  await handler(response, targetUrl, popoverInner, hash, signal)
}

export function isNoteDocumentUrl(targetUrl: URL): boolean {
  const leaf = targetUrl.pathname.split('/').pop() ?? ''
  return leaf === '' || !leaf.includes('.') || leaf.endsWith('.html') || leaf.endsWith('.htm')
}

export async function fetchStackedNotePayload(
  targetUrl: URL,
  signal: AbortSignal,
): Promise<StackedNotePayload | null> {
  const slug = stackedNoteRootSlug(targetUrl)
  if (!slug) return null
  const cached = getCachedStackedNotePayload(slug)
  if (cached) return cached

  const response = await fetch(stackedNotePayloadUrl(slug), {
    headers: { Accept: 'application/json' },
    signal,
  }).catch(error => {
    if (!isAbortError(error)) console.error(error)
    return null
  })
  if (!response || !response.ok) return null
  const json = await response.json().catch(error => {
    console.error(error)
    return null
  })
  const payload = readStackedNotePayload(json)
  if (payload?.state === 'ready') {
    cacheStackedNotePayload(payload)
  }
  return payload
}

export function populateStackedPayloadContent(
  payload: StackedNotePayload,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
  hash = '',
) {
  popoverInner.classList.add('grid')
  const html = p.parseFromString(payload.content, 'text/html')
  normalizeRelativeURLs(html, new URL(`/${payload.slug}`, targetUrl))
  const telescope = cloneTelescopicPreview(html, targetUrl, hash, payload.title)
  if (telescope) {
    popoverInner.classList.remove('grid')
    popoverInner.append(telescope)
    return
  }
  html.querySelectorAll('.byline, .page-footer').forEach(element => element.remove())
  prefixPopoverIds(html)
  const elements = Array.from(html.body.children).flatMap(el =>
    el instanceof HTMLElement ? [cleanAbsoluteElement(el)] : [],
  )
  popoverInner.append(...elements)
}
