import type { Placement, ReferenceElement, Strategy, VirtualElement } from '@floating-ui/dom'
import { arrow as floatingArrow, computePosition, flip, offset, shift } from '@floating-ui/dom'
import xmlFormat from 'xml-formatter'
import { arenaPdfViewerSource } from '../../util/arena-embed'
import { fetchCanonical } from '../../util/fetch-canonical'
import {
  lessWrongPreviewApiUrl,
  readLessWrongPreview,
  type LessWrongPreview,
  type LessWrongTarget,
} from '../../util/lesswrong'
import { getContentType } from '../../util/mime'
import { getFullSlug, isFullSlug, normalizeRelativeURLs } from '../../util/path'
import { type PreviewTocEntry } from '../../util/preview'
import { readSepPreview, sepPreviewApiUrl, type SepPreview, type SepTarget } from '../../util/sep'
import {
  cacheStackedNotePayload,
  getCachedStackedNotePayload,
  readStackedNotePayload,
  stackedNotePayloadUrl,
  type StackedNotePayload,
} from '../../util/stacked-notes'
import {
  readWikipediaPreviewResponse,
  wikipediaActionApiUrl,
  type WikipediaPreview,
  type WikipediaTarget,
} from '../../util/wikipedia'
import { currentNavSignal } from './nav-lifecycle'
import { beginSidePanelRequest, disposeSidePanel, getOrCreateSidePanel } from './side-panel'

type ContentHandler = (
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
  hash: string,
) => Promise<void>

interface PositioningOptions {
  clientX: number
  clientY: number
  placement?: Placement
  strategy?: Strategy
}

interface ShowPopoverOptions {
  placement?: Placement
  strategy?: Strategy
  hash?: string
  popoverInner?: HTMLElement
}

interface Point {
  x: number
  y: number
}

interface PdfEmbedOptions {
  preload?: boolean
  annotations?: boolean
  rootMargin?: string
  overscan?: number
  maxPixelRatio?: number
  textLayer?: boolean
}

const PDF_LOAD_TIMEOUT = 8 * 1000
const PDF_IDLE_DOCUMENT_PRELOAD_LIMIT = 1
const PDF_PREVIEW_OPTIONS: PdfEmbedOptions = {
  preload: false,
  annotations: false,
  rootMargin: '0px',
  overscan: 0,
  maxPixelRatio: 2,
  textLayer: false,
}
const PDF_SIDE_PANEL_OPTIONS = PDF_PREVIEW_OPTIONS

const p = new DOMParser()
let activeAnchor: HTMLAnchorElement | null = null
let activeDagHeading: HTMLElement | null = null
let activePopoverReq: { abort: () => void; link: HTMLAnchorElement } | null = null
let stackedPopoverEvents: AbortController | null = null
let pdfRuntimePreloadScheduled = false
let delegatedPopoverSignal: AbortSignal | undefined

function cleanAbsoluteElement(element: HTMLElement): HTMLElement {
  const refsAndNotes = element.querySelectorAll<HTMLElement>(
    'section[data-references], section[data-footnotes], [data-skip-preview], .telescopic-container',
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

function streamPopoverKey(targetUrl: URL): string {
  const entry = targetUrl.searchParams.get('entry')?.trim()
  const hash = decodeURIComponent(targetUrl.hash).replace(/^#/, '').trim()
  if (isStreamDocumentPath(targetUrl.pathname) && (entry || hash)) {
    return `popover-${targetUrl.pathname}-${entry || hash}`
  }
  return `popover-${targetUrl.pathname}`
}

function streamPopoverFetchUrl(targetUrl: URL): URL {
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

function cloneCurrentStreamEntryPreview(hash: string): HTMLElement | null {
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

function createPopoverElement(...classes: string[]): {
  popoverElement: HTMLElement
  popoverInner: HTMLDivElement
} {
  const popoverElement = document.createElement('div')
  popoverElement.classList.add('popover', ...classes)
  const popoverArrow = document.createElement('div')
  popoverArrow.classList.add('popover-arrow')
  const popoverInner = document.createElement('div')
  popoverInner.classList.add('popover-inner')
  popoverElement.append(popoverArrow, popoverInner)
  return { popoverElement, popoverInner }
}

function cleanArxivIdentifier(identifier: string): string {
  return identifier.replace(/^(arxiv:)?(pdf\/)?/i, '').replace(/\.pdf$/i, '')
}

function arxivPdfUrl(identifier: string): URL {
  const path = cleanArxivIdentifier(identifier).split('/').map(encodeURIComponent).join('/')
  return new URL(`https://arxiv.org/pdf/${path}`)
}

function isPdfUrl(url: URL): boolean {
  return url.pathname.toLowerCase().endsWith('.pdf')
}

function pdfRuntimeSource(rawSrc: string): string {
  try {
    const url = new URL(rawSrc, window.location.href)
    if (url.protocol === 'http:' || url.protocol === 'https:') {
      if (url.origin !== window.location.origin) return arenaPdfViewerSource(url.toString())
      return `${url.pathname}${url.search}`
    }
  } catch {
    return rawSrc
  }

  return rawSrc
}

function pdfTitleFromUrl(rawSrc: string): string {
  try {
    const url = new URL(rawSrc, window.location.href)
    const filename = url.pathname.split('/').filter(Boolean).pop()
    return filename ? decodeURIComponent(filename) : 'document.pdf'
  } catch {
    return 'document.pdf'
  }
}

function createPdfEmbed(
  src: string,
  title = pdfTitleFromUrl(src),
  options: PdfEmbedOptions = {},
): HTMLDivElement {
  const pdf = document.createElement('div')
  pdf.classList.add('internal-embed', 'pdf-embed', 'pdf-embed-preview')
  pdf.dataset.pdfSrc = pdfRuntimeSource(src)
  pdf.dataset.pdfTitle = title
  pdf.dataset.pdfFit = 'width'
  if (options.preload !== undefined) pdf.dataset.pdfPreload = String(options.preload)
  if (options.annotations !== undefined) {
    pdf.dataset.pdfEnableAnnotations = String(options.annotations)
  }
  if (options.rootMargin !== undefined) pdf.dataset.pdfRootMargin = options.rootMargin
  if (options.overscan !== undefined) pdf.dataset.pdfOverscan = String(options.overscan)
  if (options.maxPixelRatio !== undefined) {
    pdf.dataset.pdfMaxPixelRatio = String(options.maxPixelRatio)
  }
  if (options.textLayer !== undefined) pdf.dataset.pdfTextLayer = String(options.textLayer)
  pdf.tabIndex = 0

  const loading = document.createElement('span')
  loading.className = 'pdf-embed-loading'
  loading.textContent = 'Loading PDF'
  pdf.appendChild(loading)

  return pdf
}

type IdleCallback = (callback: () => void, options?: { timeout: number }) => number

function isIdleCallback(value: unknown): value is IdleCallback {
  return typeof value === 'function'
}

function schedulePdfRuntimePreload(sources: string[]) {
  if (pdfRuntimePreloadScheduled) return
  pdfRuntimePreloadScheduled = true
  const preloadSources = [...new Set(sources)].slice(0, PDF_IDLE_DOCUMENT_PRELOAD_LIMIT)

  const preload = () => {
    const runtime = window.quartzPdfEmbeds
    if (!runtime) {
      pdfRuntimePreloadScheduled = false
      return
    }

    void runtime
      .preload()
      .then(() => {
        for (const source of preloadSources) {
          void runtime.preload(source).catch(console.error)
        }
      })
      .catch(console.error)
  }

  const requestIdle: unknown = Reflect.get(window, 'requestIdleCallback')
  if (isIdleCallback(requestIdle)) {
    requestIdle(preload, { timeout: 1500 })
    return
  }

  window.setTimeout(preload, 250)
}

function pdfPreviewSource(link: HTMLAnchorElement): string | null {
  if (link.dataset.arxivId) return pdfRuntimeSource(arxivPdfUrl(link.dataset.arxivId).toString())

  try {
    const targetUrl = new URL(link.href)
    targetUrl.hash = ''
    targetUrl.search = ''
    return isPdfUrl(targetUrl) ? pdfRuntimeSource(targetUrl.toString()) : null
  } catch {
    return null
  }
}

function clearPopoverLoading(popoverInner: HTMLDivElement) {
  delete popoverInner.dataset.state
  popoverInner.querySelector<HTMLElement>(':scope > .popover-loading')?.remove()
}

function mountPdfPreview(
  pdf: HTMLElement,
  signal: AbortSignal,
  onReady: () => void,
  onFailed: () => void,
) {
  const runtime = window.quartzPdfEmbeds
  if (!runtime) {
    onFailed()
    return
  }
  const pdfRuntime = runtime

  let settled = false
  const timeoutId = window.setTimeout(() => ready(), PDF_LOAD_TIMEOUT)
  const observer = new MutationObserver(() => {
    if (pdf.dataset.pdfStatus === 'loaded') ready()
    if (pdf.dataset.pdfStatus === 'error') failed()
  })

  function cleanup() {
    window.clearTimeout(timeoutId)
    signal.removeEventListener('abort', aborted)
    observer.disconnect()
  }

  function finish(cb: () => void) {
    if (settled) return
    settled = true
    cleanup()
    cb()
  }

  function ready() {
    finish(onReady)
  }

  function failed() {
    finish(onFailed)
  }

  function aborted() {
    pdfRuntime.cleanup(pdf)
    pdf.remove()
    finish(onFailed)
  }

  observer.observe(pdf, { attributes: true, attributeFilter: ['data-pdf-status'] })
  signal.addEventListener('abort', aborted, { once: true })
  pdfRuntime.mount(pdf)

  if (signal.aborted) aborted()
}

function mountPdfPreviews(root: ParentNode) {
  window.quartzPdfEmbeds?.mount(root)
}

function setPopoverLoading(popoverInner: HTMLDivElement, label: string, contentType?: string) {
  popoverInner.replaceChildren()
  popoverInner.dataset.state = 'loading'
  if (contentType) popoverInner.dataset.contentType = contentType

  const loading = document.createElement('div')
  loading.classList.add('popover-loading')
  loading.role = 'status'
  loading.ariaLabel = label
  loading.ariaLive = 'polite'

  const dots = document.createElement('span')
  dots.classList.add('popover-loading-dots')
  dots.ariaHidden = 'true'

  for (let index = 0; index < 3; index++) {
    dots.appendChild(document.createElement('span'))
  }

  loading.appendChild(dots)
  popoverInner.appendChild(loading)
}

function placementSide(placement: Placement): 'top' | 'right' | 'bottom' | 'left' {
  if (placement.startsWith('top')) return 'top'
  if (placement.startsWith('right')) return 'right'
  if (placement.startsWith('bottom')) return 'bottom'
  return 'left'
}

function positionPopoverArrow(
  arrowElement: HTMLElement,
  placement: Placement,
  x: number | undefined,
  y: number | undefined,
) {
  arrowElement.style.left = ''
  arrowElement.style.right = ''
  arrowElement.style.top = ''
  arrowElement.style.bottom = ''

  if (x !== undefined) arrowElement.style.left = `${Math.round(x)}px`
  if (y !== undefined) arrowElement.style.top = `${Math.round(y)}px`

  const offset = 'var(--popover-arrow-inset)'
  switch (placementSide(placement)) {
    case 'top':
      arrowElement.style.bottom = offset
      break
    case 'right':
      arrowElement.style.left = offset
      break
    case 'bottom':
      arrowElement.style.top = offset
      break
    case 'left':
      arrowElement.style.right = offset
      break
  }
}

function isVisibleRect(rect: DOMRectReadOnly): boolean {
  return rect.width > 0 && rect.height > 0
}

function squaredDistanceToRect(point: Point, rect: DOMRectReadOnly): number {
  const dx = point.x < rect.left ? rect.left - point.x : Math.max(point.x - rect.right, 0)
  const dy = point.y < rect.top ? rect.top - point.y : Math.max(point.y - rect.bottom, 0)
  return dx * dx + dy * dy
}

function closestRectToPoint(rects: DOMRect[], point: Point): DOMRect | undefined {
  let closest = rects[0]
  let closestDistance = closest ? squaredDistanceToRect(point, closest) : Number.POSITIVE_INFINITY

  for (let index = 1; index < rects.length; index++) {
    const rect = rects[index]
    const distance = squaredDistanceToRect(point, rect)
    if (distance < closestDistance) {
      closest = rect
      closestDistance = distance
    }
  }

  return closest
}

function textClientRects(element: HTMLElement): DOMRect[] {
  const rects: DOMRect[] = []
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      return node.textContent?.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT
    },
  })
  const range = document.createRange()

  while (walker.nextNode()) {
    range.selectNodeContents(walker.currentNode)
    rects.push(...Array.from(range.getClientRects()).filter(isVisibleRect))
  }

  range.detach()
  return rects
}

function popoverReference(link: HTMLElement, point: Point): ReferenceElement {
  const textRect = closestRectToPoint(textClientRects(link), point)
  if (!textRect) return link

  const rect = new DOMRect(textRect.left, textRect.top, textRect.width, textRect.height)
  const reference: VirtualElement = {
    contextElement: link,
    getBoundingClientRect: () => rect,
    getClientRects: () => [rect],
  }
  return reference
}

function findHashTarget(container: ParentNode, hash: string, prefix = ''): HTMLElement | null {
  const rawId = hash.startsWith('#') ? hash.slice(1) : hash
  if (!rawId) return null
  const id = prefix && !rawId.startsWith(prefix) ? `${prefix}${rawId}` : rawId
  return container.querySelector<HTMLElement>(`#${CSS.escape(id)}`)
}

function scrollPopoverToHash(popoverInner: HTMLElement, hash: string) {
  if (!hash) return
  const heading = findHashTarget(popoverInner, hash, 'popover-')
  if (!heading) return
  const top =
    heading.getBoundingClientRect().top -
    popoverInner.getBoundingClientRect().top +
    popoverInner.scrollTop -
    12
  popoverInner.scroll({ top, behavior: 'instant' })
}

async function handleImageContent(targetUrl: URL, popoverInner: HTMLDivElement) {
  const img = document.createElement('img')
  img.src = targetUrl.toString()
  img.alt = targetUrl.pathname
  popoverInner.appendChild(img)
}

async function handlePdfContent(response: Response, targetUrl: URL, popoverInner: HTMLDivElement) {
  const src = response.url || targetUrl.toString()
  const pdf = createPdfEmbed(src, pdfTitleFromUrl(targetUrl.toString()), PDF_PREVIEW_OPTIONS)
  popoverInner.appendChild(pdf)
}

async function handleXmlContent(response: Response, popoverInner: HTMLDivElement) {
  const contents = await response.text()
  const rss = document.createElement('pre')
  rss.classList.add('rss-viewer')
  rss.append(xmlFormat(contents, { indentation: '  ', lineSeparator: '\n' }))
  popoverInner.append(rss)
}

async function handleDefaultContent(
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
  hash = '',
) {
  popoverInner.classList.add('grid')
  const contents = await response.text()
  const html = p.parseFromString(contents, 'text/html')
  normalizeRelativeURLs(html, targetUrl)
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
  'application/xml': async (response, _targetUrl, popoverInner) =>
    handleXmlContent(response, popoverInner),
  default: handleDefaultContent,
}

async function populatePopoverContent(
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
  hash = '',
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

  await handler(response, targetUrl, popoverInner, hash)
}

function stackedNoteSlugFromUrl(targetUrl: URL): string {
  const slug = targetUrl.pathname.replace(/^\/+|\/+$/g, '')
  return slug === '' ? 'index' : slug
}

function isNoteDocumentUrl(targetUrl: URL): boolean {
  const leaf = targetUrl.pathname.split('/').pop() ?? ''
  return leaf === '' || !leaf.includes('.') || leaf.endsWith('.html') || leaf.endsWith('.htm')
}

async function fetchStackedNotePayload(
  targetUrl: URL,
  signal: AbortSignal,
): Promise<StackedNotePayload | null> {
  const slug = stackedNoteSlugFromUrl(targetUrl)
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

function populateStackedPayloadContent(
  payload: StackedNotePayload,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
) {
  popoverInner.classList.add('grid')
  const html = p.parseFromString(payload.content, 'text/html')
  normalizeRelativeURLs(html, targetUrl)
  prefixPopoverIds(html)
  const elements = Array.from(html.body.children).flatMap(el =>
    el instanceof HTMLElement ? [cleanAbsoluteElement(el)] : [],
  )
  popoverInner.append(...elements)
}

async function setPosition(
  link: HTMLElement,
  popoverElement: HTMLElement,
  { clientX, clientY, placement, strategy = 'fixed' }: PositioningOptions,
) {
  const arrowElement = popoverElement.querySelector<HTMLElement>('.popover-arrow')
  const point = { x: clientX, y: clientY }
  const reference = popoverReference(link, point)
  const middleware = [
    offset(2),
    shift(),
    flip(),
    arrowElement ? floatingArrow({ element: arrowElement, padding: 12 }) : null,
  ]
  const {
    x,
    y,
    placement: finalPlacement,
    middlewareData,
  } = await computePosition(reference, popoverElement, { placement, strategy, middleware })

  popoverElement.style.position = strategy
  popoverElement.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`
  popoverElement.dataset.placement = finalPlacement
  if (arrowElement) {
    positionPopoverArrow(
      arrowElement,
      finalPlacement,
      middlewareData.arrow?.x,
      middlewareData.arrow?.y,
    )
  }
}

async function showPopover(
  link: HTMLAnchorElement,
  popoverElement: HTMLElement,
  pointer: { clientX: number; clientY: number },
  options: ShowPopoverOptions = {},
) {
  clearActivePopover(popoverElement)
  popoverElement.classList.add('active-popover')

  await setPosition(link, popoverElement, {
    clientX: pointer.clientX,
    clientY: pointer.clientY,
    placement: options.placement,
    strategy: options.strategy,
  })

  const { hash, popoverInner } = options
  if (hash && hash !== '' && popoverInner) {
    scrollPopoverToHash(popoverInner, hash)
  }
}

async function showLoadingPopover(
  link: HTMLAnchorElement,
  popoverElement: HTMLElement,
  pointer: { clientX: number; clientY: number },
) {
  clearActivePopover(popoverElement)
  activeAnchor = link
  popoverElement.classList.add('active-popover')
  await setPosition(link, popoverElement, { clientX: pointer.clientX, clientY: pointer.clientY })
}

function removePopoverElement(popoverElement: HTMLElement) {
  window.quartzPdfEmbeds?.cleanup(popoverElement)
  popoverElement.remove()
}

function blurPopoverFocus(popoverElement: HTMLElement) {
  const activeElement = document.activeElement
  if (activeElement instanceof HTMLElement && popoverElement.contains(activeElement)) {
    activeElement.blur()
  }
}

function clearActivePopover(except?: HTMLElement) {
  activeAnchor = null
  activeDagHeading?.classList.remove('dag')
  activeDagHeading = null
  const allPopoverElements = document.querySelectorAll<HTMLElement>('.popover')
  allPopoverElements.forEach(popoverElement => {
    if (popoverElement === except) return
    blurPopoverFocus(popoverElement)
    popoverElement.classList.remove('active-popover')
  })
}

function notifyProtectedContentLoaded(container: ParentNode) {
  document.dispatchEvent(new CustomEvent('protectedcontentloaded', { detail: { container } }))
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function compareUrls(a: URL, b: URL): boolean {
  const u1 = new URL(a.toString())
  const u2 = new URL(b.toString())
  u1.hash = ''
  u1.search = ''
  u2.hash = ''
  u2.search = ''
  return u1.toString() === u2.toString()
}

async function handleBibliography(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  const href = link.getAttribute('href')
  if (!href) return

  const entryId = href.replace('#', '')
  const bibEntry = document.getElementById(entryId) as HTMLLIElement | null
  if (!bibEntry) return

  const popoverId = link.dataset.popoverId ?? `popover-bib-${entryId}`
  let popoverElement = document.getElementById(popoverId) as HTMLElement | null
  let popoverInner: HTMLDivElement | null = null

  if (!popoverElement) {
    const created = createPopoverElement('bib-popover')
    popoverElement = created.popoverElement
    popoverInner = created.popoverInner
    popoverElement.id = popoverId
    popoverInner.innerHTML = bibEntry.innerHTML
    document.body.appendChild(popoverElement)
  } else {
    popoverInner = popoverElement.querySelector('.popover-inner') as HTMLDivElement | null
  }

  if (!popoverInner) return

  link.dataset.popoverId = popoverId
  await showPopover(link, popoverElement, pointer, { placement: 'top' })
}

async function handleFootnote(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  const href = link.getAttribute('href')
  if (!href) return

  const entryId = href.replace('#', '')
  const footnoteEntry = document.getElementById(entryId) as HTMLLIElement | null
  if (!footnoteEntry) return

  const popoverId = link.dataset.popoverId ?? `popover-footnote-${entryId}`
  let popoverElement = document.getElementById(popoverId) as HTMLElement | null
  let popoverInner: HTMLDivElement | null = null

  if (!popoverElement) {
    const created = createPopoverElement('footnote-popover')
    popoverElement = created.popoverElement
    popoverInner = created.popoverInner
    popoverElement.id = popoverId
    popoverInner.innerHTML = footnoteEntry.innerHTML
    popoverInner.querySelectorAll('[data-footnote-backref]').forEach(el => el.remove())
    document.body.appendChild(popoverElement)
  } else {
    popoverInner = popoverElement.querySelector('.popover-inner') as HTMLDivElement | null
  }

  if (!popoverInner) return

  link.dataset.popoverId = popoverId
  await showPopover(link, popoverElement, pointer, { placement: 'top' })
}

function wikipediaTargetFromLink(link: HTMLAnchorElement): WikipediaTarget | undefined {
  const { wikipediaLang, wikipediaTitle } = link.dataset
  if (!wikipediaLang || !wikipediaTitle) return undefined
  return { lang: wikipediaLang, title: wikipediaTitle }
}

function renderWikipediaPreview(preview: WikipediaPreview, popoverInner: HTMLDivElement) {
  popoverInner.dataset.contentType = 'text/x-wikipedia'

  const card = document.createElement('article')
  card.classList.add('wikipedia-popover-card')
  if (preview.thumbnail) card.classList.add('has-thumbnail')

  const header = document.createElement('header')
  header.classList.add('wikipedia-popover-header')

  const heading = document.createElement('ul')
  heading.classList.add('wikipedia-popover-heading')

  const title = document.createElement('li')
  title.classList.add('wikipedia-popover-title')

  const titleLink = document.createElement('a')
  titleLink.href = preview.pageUrl
  titleLink.target = '_blank'
  titleLink.rel = 'noopener noreferrer'
  titleLink.textContent = preview.title
  title.appendChild(titleLink)
  heading.appendChild(title)

  if (preview.description) {
    const description = document.createElement('li')
    description.classList.add('wikipedia-popover-description')
    description.textContent = preview.description
    heading.appendChild(description)
  }

  header.appendChild(heading)
  card.appendChild(header)

  if (preview.thumbnail) {
    const thumbnail = document.createElement('img')
    thumbnail.classList.add('wikipedia-popover-thumbnail')
    thumbnail.src = preview.thumbnail.source
    thumbnail.alt = preview.title
    thumbnail.decoding = 'async'
    if (preview.thumbnail.width) thumbnail.width = preview.thumbnail.width
    if (preview.thumbnail.height) thumbnail.height = preview.thumbnail.height
    card.appendChild(thumbnail)
  }

  const extract = document.createElement('p')
  extract.classList.add('wikipedia-popover-extract')
  extract.textContent = preview.extract
  card.appendChild(extract)

  const source = document.createElement('a')
  source.classList.add('wikipedia-popover-source')
  source.href = preview.pageUrl
  source.target = '_blank'
  source.rel = 'noopener noreferrer'
  source.textContent = 'Wikipedia'
  card.appendChild(source)

  popoverInner.appendChild(card)
}

async function handleWikipedia(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  const target = wikipediaTargetFromLink(link)
  if (!target) return

  const popoverId =
    link.dataset.popoverId ?? `popover-wikipedia-${target.lang}-${encodeURIComponent(target.title)}`
  const existingPopover = document.getElementById(popoverId)
  if (existingPopover) {
    await showPopover(link, existingPopover, pointer)
    return
  }

  if (activePopoverReq && activePopoverReq.link !== link) {
    activePopoverReq.abort()
    activePopoverReq = null
  }

  const controller = new AbortController()
  activePopoverReq = { abort: () => controller.abort(), link }

  const response = await fetch(wikipediaActionApiUrl(target), { signal: controller.signal }).catch(
    error => {
      if (!isAbortError(error)) console.error(error)
      return null
    },
  )
  if (!response || !response.ok) {
    activePopoverReq = null
    return
  }

  const preview = readWikipediaPreviewResponse(await response.json(), target)
  if (!preview || activeAnchor !== link) {
    activePopoverReq = null
    return
  }

  const { popoverElement, popoverInner } = createPopoverElement('wikipedia-popover')
  popoverElement.id = popoverId
  renderWikipediaPreview(preview, popoverInner)

  if (document.getElementById(popoverId)) {
    activePopoverReq = null
    return
  }

  link.dataset.popoverId = popoverId
  document.body.appendChild(popoverElement)
  activePopoverReq = null

  if (activeAnchor !== link) return
  await showPopover(link, popoverElement, pointer, { popoverInner })
}

function lessWrongTargetFromLink(link: HTMLAnchorElement): LessWrongTarget | undefined {
  const { lesswrongPostId, lesswrongSlug } = link.dataset
  if (!lesswrongPostId) return undefined
  return { postId: lesswrongPostId, ...(lesswrongSlug ? { slug: lesswrongSlug } : {}) }
}

function appendTextItem(parent: HTMLElement, text: string, className?: string) {
  const item = document.createElement('li')
  if (className) item.classList.add(className)
  item.textContent = text
  parent.appendChild(item)
}

function appendLessWrongTag(parent: HTMLElement, text: string) {
  const item = document.createElement('li')
  const label = document.createElement('span')
  label.classList.add('internal', 'tag-link', 'lesswrong-popover-tag')
  label.textContent = text
  item.appendChild(label)
  parent.appendChild(item)
}

function pluralized(value: number, singular: string, plural: string): string {
  return `${value} ${Math.abs(value) === 1 ? singular : plural}`
}

function formatLessWrongDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  }).format(date)
}

function popoverTocHref(pageUrl: string, href: string): string {
  try {
    return new URL(href, pageUrl).toString()
  } catch {
    return pageUrl
  }
}

function appendPopoverToc(
  parent: HTMLElement,
  pageUrl: string,
  toc: PreviewTocEntry[] | undefined,
  ariaLabel: string,
) {
  if (!toc || toc.length === 0) return

  const section = document.createElement('section')
  section.classList.add('popover-toc')
  section.ariaLabel = ariaLabel

  const label = document.createElement('div')
  label.classList.add('popover-toc-title')
  label.textContent = 'Contents'
  section.appendChild(label)

  const list = document.createElement('ol')
  list.classList.add('popover-toc-list')

  for (const entry of toc) {
    const item = document.createElement('li')
    item.classList.add('popover-toc-item')
    item.style.setProperty('--popover-toc-indent', `${(entry.depth - 1) * 0.72}rem`)

    const link = document.createElement('a')
    link.classList.add('popover-toc-link')
    link.href = popoverTocHref(pageUrl, entry.href)
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    link.textContent = entry.text
    item.appendChild(link)
    list.appendChild(item)
  }

  section.appendChild(list)
  parent.appendChild(section)
}

function renderLessWrongPreview(preview: LessWrongPreview, popoverInner: HTMLDivElement) {
  popoverInner.dataset.contentType = 'text/x-lesswrong'

  const card = document.createElement('article')
  card.classList.add('lesswrong-popover-card')

  const previewHeader = document.createElement('div')
  previewHeader.classList.add('lesswrong-popover-header')

  const title = document.createElement('h2')
  title.classList.add('lesswrong-popover-title')

  const titleLink = document.createElement('a')
  titleLink.href = preview.pageUrl
  titleLink.target = '_blank'
  titleLink.rel = 'noopener noreferrer'
  titleLink.textContent = preview.title
  title.appendChild(titleLink)
  previewHeader.appendChild(title)

  const meta = document.createElement('ul')
  meta.classList.add('lesswrong-popover-meta')
  meta.ariaLabel = 'LessWrong metadata'
  if (preview.score !== undefined)
    appendTextItem(meta, pluralized(preview.score, 'point', 'points'))
  if (preview.author) appendTextItem(meta, preview.author, 'author')
  if (preview.postedAt) appendTextItem(meta, formatLessWrongDate(preview.postedAt))
  if (preview.commentCount !== undefined) {
    appendTextItem(meta, pluralized(preview.commentCount, 'comment', 'comments'))
  }
  if (preview.readTimeMinutes !== undefined) {
    appendTextItem(meta, pluralized(preview.readTimeMinutes, 'min read', 'min read'))
  }
  if (meta.childElementCount > 0) previewHeader.appendChild(meta)

  if (preview.tags && preview.tags.length > 0) {
    const tags = document.createElement('ul')
    tags.classList.add('lesswrong-popover-tags', 'tags')
    tags.ariaLabel = 'LessWrong tags'
    for (const tag of preview.tags) appendLessWrongTag(tags, tag)
    previewHeader.appendChild(tags)
  }

  card.appendChild(previewHeader)
  appendPopoverToc(card, preview.pageUrl, preview.toc, 'LessWrong contents')

  const extract = document.createElement('p')
  extract.classList.add('lesswrong-popover-extract')
  extract.textContent = preview.extract
  card.appendChild(extract)

  const source = document.createElement('a')
  source.classList.add('lesswrong-popover-source')
  source.href = preview.pageUrl
  source.target = '_blank'
  source.rel = 'noopener noreferrer'
  source.textContent = 'LessWrong'
  card.appendChild(source)

  popoverInner.appendChild(card)
}

async function handleLessWrong(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  const target = lessWrongTargetFromLink(link)
  if (!target) return

  const popoverId = link.dataset.popoverId ?? `popover-lesswrong-${target.postId}`
  const existingPopover = document.getElementById(popoverId)
  if (existingPopover) {
    await showPopover(link, existingPopover, pointer)
    return
  }

  if (activePopoverReq && activePopoverReq.link !== link) {
    activePopoverReq.abort()
    activePopoverReq = null
  }

  const controller = new AbortController()
  activePopoverReq = { abort: () => controller.abort(), link }

  const response = await fetch(lessWrongPreviewApiUrl(target, window.location.toString()), {
    signal: controller.signal,
  }).catch(error => {
    if (!isAbortError(error)) console.error(error)
    return null
  })
  if (!response || !response.ok) {
    activePopoverReq = null
    return
  }

  const preview = readLessWrongPreview(await response.json())
  if (!preview || activeAnchor !== link) {
    activePopoverReq = null
    return
  }

  const { popoverElement, popoverInner } = createPopoverElement('lesswrong-popover')
  popoverElement.id = popoverId
  renderLessWrongPreview(preview, popoverInner)

  if (document.getElementById(popoverId)) {
    activePopoverReq = null
    return
  }

  link.dataset.popoverId = popoverId
  document.body.appendChild(popoverElement)
  activePopoverReq = null

  if (activeAnchor !== link) return
  await showPopover(link, popoverElement, pointer, { popoverInner })
}

function sepTargetFromLink(link: HTMLAnchorElement): SepTarget | undefined {
  const { sepEntry, sepArchive } = link.dataset
  if (!sepEntry) return undefined
  return { entry: sepEntry, ...(sepArchive ? { archive: sepArchive } : {}) }
}

function renderSepPreview(preview: SepPreview, popoverInner: HTMLDivElement) {
  popoverInner.dataset.contentType = 'text/x-sep'

  const card = document.createElement('article')
  card.classList.add('sep-popover-card')

  const previewHeader = document.createElement('div')
  previewHeader.classList.add('sep-popover-header')

  const title = document.createElement('h2')
  title.classList.add('sep-popover-title')

  const titleLink = document.createElement('a')
  titleLink.href = preview.pageUrl
  titleLink.target = '_blank'
  titleLink.rel = 'noopener noreferrer'
  titleLink.textContent = preview.title
  title.appendChild(titleLink)
  previewHeader.appendChild(title)

  const meta = document.createElement('ul')
  meta.classList.add('sep-popover-meta')
  meta.ariaLabel = 'SEP metadata'
  for (const author of preview.authors ?? []) appendTextItem(meta, author, 'author')
  if (preview.pubInfo) appendTextItem(meta, preview.pubInfo)
  if (meta.childElementCount > 0) previewHeader.appendChild(meta)

  card.appendChild(previewHeader)
  appendPopoverToc(card, preview.pageUrl, preview.toc, 'SEP contents')

  const extract = document.createElement('p')
  extract.classList.add('sep-popover-extract')
  extract.textContent = preview.extract
  card.appendChild(extract)

  const source = document.createElement('a')
  source.classList.add('sep-popover-source')
  source.href = preview.pageUrl
  source.target = '_blank'
  source.rel = 'noopener noreferrer'
  source.textContent = 'Stanford Encyclopedia of Philosophy'
  card.appendChild(source)

  popoverInner.appendChild(card)
}

async function handleSep(link: HTMLAnchorElement, pointer: { clientX: number; clientY: number }) {
  const target = sepTargetFromLink(link)
  if (!target) return

  const popoverId =
    link.dataset.popoverId ?? `popover-sep-${target.archive ?? 'current'}-${target.entry}`
  const existingPopover = document.getElementById(popoverId)
  if (existingPopover) {
    await showPopover(link, existingPopover, pointer)
    return
  }

  if (activePopoverReq && activePopoverReq.link !== link) {
    activePopoverReq.abort()
    activePopoverReq = null
  }

  const controller = new AbortController()
  activePopoverReq = { abort: () => controller.abort(), link }

  const response = await fetch(sepPreviewApiUrl(target, window.location.toString()), {
    signal: controller.signal,
  }).catch(error => {
    if (!isAbortError(error)) console.error(error)
    return null
  })
  if (!response || !response.ok) {
    activePopoverReq = null
    return
  }

  const preview = readSepPreview(await response.json())
  if (!preview || activeAnchor !== link) {
    activePopoverReq = null
    return
  }

  const { popoverElement, popoverInner } = createPopoverElement('sep-popover')
  popoverElement.id = popoverId
  renderSepPreview(preview, popoverInner)

  if (document.getElementById(popoverId)) {
    activePopoverReq = null
    return
  }

  link.dataset.popoverId = popoverId
  document.body.appendChild(popoverElement)
  activePopoverReq = null

  if (activeAnchor !== link) return
  await showPopover(link, popoverElement, pointer, { popoverInner })
}

async function handleTriathlon(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  const date = link.dataset.triathlonDate
  if (!date) return

  const popoverId = link.dataset.popoverId ?? `popover-triathlon-${date}`
  const existingPopover = document.getElementById(popoverId)
  if (existingPopover) {
    await showPopover(link, existingPopover, pointer)
    return
  }

  const card = await window.quartzTriathlon
    ?.dayCard(date, new URL('/static/strava-detail.json', link.href).toString(), {
      location: link.dataset.triathlonLoc,
      event: link.dataset.triathlonEvent,
    })
    .catch(() => null)
  if (!card || activeAnchor !== link) return

  const { popoverElement, popoverInner } = createPopoverElement('triathlon-popover')
  popoverElement.id = popoverId
  popoverInner.dataset.contentType = 'text/x-triathlon'
  popoverInner.appendChild(card)

  if (document.getElementById(popoverId)) return

  link.dataset.popoverId = popoverId
  document.body.appendChild(popoverElement)

  if (activeAnchor !== link) return
  await showPopover(link, popoverElement, pointer, { popoverInner })
}

async function handleStackedNotes(
  stacked: HTMLElement | null,
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  if (!stacked) return
  clearActivePopover()
  activeAnchor = link

  if (activePopoverReq && activePopoverReq.link !== link) {
    activePopoverReq.abort()
    activePopoverReq = null
  }

  const column = stacked.querySelector<HTMLDivElement>('.stacked-notes-column')
  if (!column) return

  column
    .querySelectorAll<HTMLDivElement>('div[class~="stacked-popover"]')
    .forEach(removePopoverElement)

  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)
  targetUrl.hash = ''
  targetUrl.search = ''

  const controller = new AbortController()
  activePopoverReq = { abort: () => controller.abort(), link }

  const { popoverElement, popoverInner } = createPopoverElement('stacked-popover')
  if (isNoteDocumentUrl(targetUrl)) {
    const payload = await fetchStackedNotePayload(targetUrl, controller.signal)
    if (!payload || activeAnchor !== link) {
      removePopoverElement(popoverElement)
      activePopoverReq = null
      return
    }
    populateStackedPayloadContent(payload, targetUrl, popoverInner)
  } else {
    const response = await fetchCanonical(new URL(targetUrl.toString()), {
      signal: controller.signal,
    }).catch(error => {
      if (!isAbortError(error)) console.error(error)
      return null
    })

    if (!response || activeAnchor !== link) {
      removePopoverElement(popoverElement)
      activePopoverReq = null
      return
    }
    await populatePopoverContent(response, targetUrl, popoverInner)
  }

  if (popoverInner.childElementCount === 0) {
    removePopoverElement(popoverElement)
    activePopoverReq = null
    return
  }

  column.appendChild(popoverElement)
  hideStackedPopoversOnLeave(stacked, link, popoverElement)
  mountPdfPreviews(popoverInner)
  notifyProtectedContentLoaded(popoverInner)
  await setPosition(link, popoverElement, {
    clientX: pointer.clientX,
    clientY: pointer.clientY,
    placement: 'right',
    strategy: 'absolute',
  })

  if (hash !== '') {
    scrollPopoverToHash(popoverInner, hash)
  }

  requestAnimationFrame(() => {
    if (activeAnchor === link) popoverElement.classList.add('active-popover')
  })

  activePopoverReq = null
}

function closestStackedPopoverLink(
  target: EventTarget | null,
  stacked: HTMLElement,
): HTMLAnchorElement | null {
  if (!target || !(target instanceof Element)) return null
  if (stackedPopoverForTarget(target)) return null
  const link = target.closest('a.internal')
  if (!(link instanceof HTMLAnchorElement)) return null
  if (!stacked.contains(link)) return null
  return link
}

function stackedPopoverForTarget(target: EventTarget | null): HTMLElement | null {
  if (!target || !(target instanceof Element)) return null
  const popover = target.closest('.stacked-popover')
  return popover instanceof HTMLElement ? popover : null
}

function containsRelatedTarget(element: HTMLElement, relatedTarget: EventTarget | null) {
  return relatedTarget instanceof Node && element.contains(relatedTarget)
}

function popoverForLink(link: HTMLAnchorElement): HTMLElement | null {
  const { popoverId } = link.dataset
  if (!popoverId) return null
  const popover = document.getElementById(popoverId)
  return popover instanceof HTMLElement ? popover : null
}

function mouseLeaveHandler(this: HTMLAnchorElement, event: MouseEvent) {
  if (containsRelatedTarget(this, event.relatedTarget)) return

  const popover = popoverForLink(this)
  if (popover && containsRelatedTarget(popover, event.relatedTarget)) {
    popover.addEventListener(
      'mouseleave',
      popoverEvent => {
        if (!containsRelatedTarget(this, popoverEvent.relatedTarget)) {
          clearActivePopover()
        }
      },
      { once: true },
    )
    return
  }

  if (activePopoverReq?.link === this) {
    activePopoverReq.abort()
    activePopoverReq = null
  }
  clearActivePopover()
}

function allowsStackedPopover(link: HTMLAnchorElement): boolean {
  if (link.dataset.wikipediaLang && link.dataset.wikipediaTitle) return false
  if (link.dataset.lesswrongPostId) return false
  if (link.dataset.sepEntry) return false
  if (link.dataset.triathlonDate) return false
  if (link.dataset.noPopover === '' || link.dataset.noPopover === 'true') {
    return link.dataset.backlink !== undefined
  }
  return true
}

function opensSidePanel(event: MouseEvent): boolean {
  return event.altKey || event.metaKey || event.ctrlKey
}

function hideStackedPopovers(stacked: HTMLElement) {
  stacked.querySelectorAll<HTMLElement>('.stacked-popover').forEach(popoverElement => {
    popoverElement.classList.remove('active-popover')
    window.setTimeout(() => {
      if (!popoverElement.classList.contains('active-popover')) {
        removePopoverElement(popoverElement)
      }
    }, 180)
  })
}

function hideStackedPopoversOnLeave(
  stacked: HTMLElement,
  link: HTMLAnchorElement,
  popoverElement: HTMLElement,
) {
  popoverElement.addEventListener('mouseleave', event => {
    if (containsRelatedTarget(link, event.relatedTarget)) return
    if (activeAnchor === link) activeAnchor = null
    hideStackedPopovers(stacked)
  })
}

function setupStackedPopoverLinks() {
  stackedPopoverEvents?.abort()
  stackedPopoverEvents = null

  const stacked = document.getElementById('stacked-notes-container')
  if (!(stacked instanceof HTMLElement)) return

  const events = new AbortController()
  stackedPopoverEvents = events

  stacked.addEventListener(
    'mouseover',
    event => {
      const link = closestStackedPopoverLink(event.target, stacked)
      if (!link || containsRelatedTarget(link, event.relatedTarget)) return
      if (activeAnchor === link && stackedPopoverForTarget(event.relatedTarget)) return
      if (!allowsStackedPopover(link)) return
      activeAnchor = link
      void handleStackedNotes(stacked, link, { clientX: event.clientX, clientY: event.clientY })
    },
    { signal: events.signal },
  )

  stacked.addEventListener(
    'mouseout',
    event => {
      const link = closestStackedPopoverLink(event.target, stacked)
      if (!link || containsRelatedTarget(link, event.relatedTarget)) return
      if (stackedPopoverForTarget(event.relatedTarget)) return
      if (activeAnchor === link) activeAnchor = null
      if (activePopoverReq?.link === link) {
        activePopoverReq.abort()
        activePopoverReq = null
      }
      hideStackedPopovers(stacked)
    },
    { signal: events.signal },
  )
}

async function mouseEnterHandler(
  this: HTMLAnchorElement,
  { clientX, clientY }: { clientX: number; clientY: number },
) {
  // eslint-disable-next-line @typescript-eslint/no-this-alias
  activeAnchor = this
  const link = activeAnchor

  if (link.dataset.bib === '') {
    await handleBibliography(link, { clientX, clientY })
    return
  }

  if (link.dataset.footnoteRef === '') {
    await handleFootnote(link, { clientX, clientY })
    return
  }

  const container = document.getElementById('stacked-notes-container')

  if (link.dataset.noPopover === '' || link.dataset.noPopover === 'true') {
    return
  }

  if (link.dataset.wikipediaLang && link.dataset.wikipediaTitle) {
    await handleWikipedia(link, { clientX, clientY })
    return
  }

  if (link.dataset.lesswrongPostId) {
    await handleLessWrong(link, { clientX, clientY })
    return
  }

  if (link.dataset.sepEntry) {
    await handleSep(link, { clientX, clientY })
    return
  }

  if (link.dataset.triathlonDate) {
    await handleTriathlon(link, { clientX, clientY })
    return
  }

  if (getFullSlug(window) === 'notes' || container?.classList.contains('active')) {
    await handleStackedNotes(container, link, { clientX, clientY })
    return
  }

  const thisUrl = new URL(document.location.href)
  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)

  if (compareUrls(thisUrl, targetUrl)) {
    clearActivePopover()
    if (hash !== '') {
      const currentStreamPreview = cloneCurrentStreamEntryPreview(hash)
      if (currentStreamPreview) {
        const popoverId = streamPopoverKey(targetUrl)
        link.dataset.popoverId = popoverId
        const existingPopover = document.getElementById(popoverId) as HTMLElement | null
        const popoverInner = existingPopover?.querySelector<HTMLDivElement>('.popover-inner')
        if (existingPopover && popoverInner) {
          await showPopover(link, existingPopover, { clientX, clientY }, { hash, popoverInner })
          return
        }

        const created = createPopoverElement()
        const { popoverElement } = created
        popoverElement.id = popoverId
        const { popoverInner: createdInner } = created
        createdInner.classList.add('grid')
        createdInner.dataset.contentType = 'text/html'
        createdInner.appendChild(currentStreamPreview)
        document.body.appendChild(popoverElement)
        notifyProtectedContentLoaded(createdInner)
        await showPopover(
          link,
          popoverElement,
          { clientX, clientY },
          { hash, popoverInner: createdInner },
        )
        return
      }

      const article = document.querySelector('article')
      const heading = article ? findHashTarget(article, hash) : null
      if (heading) {
        activeDagHeading?.classList.remove('dag')
        heading.classList.add('dag')
        activeDagHeading = heading
      }
    }
    return
  }

  const fetchUrl = streamPopoverFetchUrl(targetUrl)

  const popoverId = streamPopoverKey(targetUrl)
  link.dataset.popoverId = popoverId
  const existingPopover = document.getElementById(popoverId) as HTMLElement | null

  if (existingPopover) {
    const popoverInner = existingPopover.querySelector('.popover-inner') as HTMLDivElement | null
    if (!popoverInner) return
    await showPopover(link, existingPopover, { clientX, clientY }, { hash, popoverInner })
    return
  }

  let response: Response | void
  if (link.dataset.arxivId) {
    if (activePopoverReq && activePopoverReq.link !== link) {
      activePopoverReq.abort()
      activePopoverReq = null
    }

    const controller = new AbortController()
    activePopoverReq = { abort: () => controller.abort(), link }

    const { popoverElement, popoverInner } = createPopoverElement()
    popoverElement.id = popoverId
    setPopoverLoading(popoverInner, 'Fetching arXiv PDF', 'application/pdf')
    document.body.appendChild(popoverElement)
    await showLoadingPopover(link, popoverElement, { clientX, clientY })

    const arxivUrl = arxivPdfUrl(link.dataset.arxivId)
    const pdf = createPdfEmbed(
      arxivUrl.toString(),
      `${cleanArxivIdentifier(link.dataset.arxivId)}.pdf`,
      PDF_PREVIEW_OPTIONS,
    )
    popoverInner.appendChild(pdf)
    mountPdfPreview(
      pdf,
      controller.signal,
      () => {
        clearPopoverLoading(popoverInner)
        if (activePopoverReq?.link === link) activePopoverReq = null
      },
      () => {
        removePopoverElement(popoverElement)
        if (activePopoverReq?.link === link) activePopoverReq = null
      },
    )

    if (activeAnchor !== link) {
      controller.abort()
      activePopoverReq = null
      removePopoverElement(popoverElement)
      return
    }

    notifyProtectedContentLoaded(popoverInner)
    await setPosition(link, popoverElement, { clientX, clientY })
    return
  } else if (isPdfUrl(fetchUrl)) {
    const { popoverElement, popoverInner } = createPopoverElement()
    popoverElement.id = popoverId
    popoverInner.dataset.contentType = 'application/pdf'
    popoverInner.appendChild(
      createPdfEmbed(
        fetchUrl.toString(),
        pdfTitleFromUrl(fetchUrl.toString()),
        PDF_PREVIEW_OPTIONS,
      ),
    )

    if (document.getElementById(popoverId)) return

    document.body.appendChild(popoverElement)
    mountPdfPreviews(popoverInner)
    notifyProtectedContentLoaded(popoverInner)
    if (activeAnchor !== link) {
      removePopoverElement(popoverElement)
      return
    }

    await showPopover(link, popoverElement, { clientX, clientY }, { hash, popoverInner })
    return
  } else {
    response = await fetchCanonical(fetchUrl).catch(error => {
      console.error(error)
    })
  }

  if (!response) return
  if (activeAnchor !== link) return

  const { popoverElement, popoverInner } = createPopoverElement()
  popoverElement.id = popoverId

  await populatePopoverContent(response, targetUrl, popoverInner, hash)

  if (document.getElementById(popoverId)) return

  document.body.appendChild(popoverElement)
  mountPdfPreviews(popoverInner)
  notifyProtectedContentLoaded(popoverInner)
  if (activeAnchor !== link) {
    removePopoverElement(popoverElement)
    return
  }

  await showPopover(link, popoverElement, { clientX, clientY }, { hash, popoverInner })
}

async function mouseClickHandler(evt: MouseEvent, link: HTMLAnchorElement) {
  clearActivePopover()

  const thisUrl = new URL(document.location.href)
  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)
  targetUrl.hash = ''
  targetUrl.search = ''

  const container = document.getElementById('stacked-notes-container')

  if (link.dataset.wikipediaLang && link.dataset.wikipediaTitle) {
    return
  }

  if (link.dataset.lesswrongPostId) {
    return
  }

  if (link.dataset.sepEntry) {
    return
  }

  if (opensSidePanel(evt) && !container?.classList.contains('active')) {
    evt.preventDefault()
    evt.stopPropagation()

    // derive slug from href if data-slug not set (e.g. generated pages like /tags/*)
    const slug = (link.dataset.slug || targetUrl.pathname).replace(/^\/+|\/+$/g, '') || 'index'
    if (!isFullSlug(slug)) return

    let request: ReturnType<typeof beginSidePanelRequest> | undefined
    try {
      const asidePanel = getOrCreateSidePanel()
      const activeRequest = beginSidePanelRequest(asidePanel)
      request = activeRequest

      if (link.dataset.arxivId) {
        const arxivUrl = arxivPdfUrl(link.dataset.arxivId)
        const pdf = createPdfEmbed(
          arxivUrl.toString(),
          `${cleanArxivIdentifier(link.dataset.arxivId)}.pdf`,
          PDF_SIDE_PANEL_OPTIONS,
        )
        const sideInner = activeRequest.mount(slug, pdf)
        if (!sideInner) return
        mountPdfPreviews(sideInner)
        window.notifyNav(slug)
        return
      }

      let response: Response | void
      const fetchUrl = new URL(link.href)
      fetchUrl.hash = ''
      fetchUrl.search = ''

      if (isPdfUrl(fetchUrl)) {
        const pdf = createPdfEmbed(
          fetchUrl.toString(),
          pdfTitleFromUrl(fetchUrl.toString()),
          PDF_SIDE_PANEL_OPTIONS,
        )
        const sideInner = activeRequest.mount(slug, pdf)
        if (!sideInner) return
        mountPdfPreviews(sideInner)
        window.notifyNav(slug)
        return
      }

      response = await fetchCanonical(fetchUrl, { signal: activeRequest.signal }).catch(error => {
        if (!activeRequest.signal.aborted) console.error(error)
      })

      if (!response) {
        activeRequest.cancel()
        return
      }

      const headerContentType = response.headers.get('Content-Type')
      const contentType = headerContentType
        ? headerContentType.split(';')[0]
        : getContentType(targetUrl)

      if (contentType === 'application/pdf') {
        const src = response.url || fetchUrl.toString()
        const pdf = createPdfEmbed(
          src,
          pdfTitleFromUrl(fetchUrl.toString()),
          PDF_SIDE_PANEL_OPTIONS,
        )
        const sideInner = activeRequest.mount(slug, pdf)
        if (!sideInner) return
        mountPdfPreviews(sideInner)
      } else {
        const contents = await response.text()
        const html = p.parseFromString(contents, 'text/html')
        normalizeRelativeURLs(html, targetUrl)
        html.querySelectorAll('[id]').forEach(el => {
          const targetID = `popover-${el.id}`
          el.id = targetID
        })
        const elts = [
          ...(html.getElementsByClassName('popover-hint') as HTMLCollectionOf<HTMLElement>),
        ]
        if (elts.length === 0) {
          activeRequest.cancel()
          return
        }

        if (!activeRequest.mount(slug, ...elts)) return
      }

      window.notifyNav(slug)
    } catch (error) {
      const wasAborted = request?.signal.aborted ?? false
      request?.cancel()
      if (!wasAborted) console.error('Failed to create side panel:', error)
    }
    return
  }

  if (compareUrls(thisUrl, targetUrl) && hash !== '') {
    const mainContent = document.querySelector('article')
    const heading = mainContent ? findHashTarget(mainContent, hash) : null
    if (!heading) return
    evt.preventDefault()
    heading.scrollIntoView({ behavior: 'smooth' })
    history.pushState(null, '', hash)
  }
}

function closestPopoverLink(target: EventTarget | null): HTMLAnchorElement | null {
  const element =
    target instanceof Element ? target : target instanceof Node ? target.parentElement : null
  const link = element?.closest<HTMLAnchorElement>('a.internal') ?? null
  return link?.closest('#stacked-notes-container') ? null : link
}

function setupPopoverDelegation(): void {
  const signal = currentNavSignal()
  if (delegatedPopoverSignal === signal) return
  delegatedPopoverSignal = signal

  document.addEventListener(
    'mouseover',
    event => {
      const link = closestPopoverLink(event.target)
      if (!link || containsRelatedTarget(link, event.relatedTarget)) return
      void mouseEnterHandler.call(link, event)
    },
    { signal },
  )
  document.addEventListener(
    'mouseout',
    event => {
      const link = closestPopoverLink(event.target)
      if (!link || containsRelatedTarget(link, event.relatedTarget)) return
      mouseLeaveHandler.call(link, event)
    },
    { signal },
  )
  document.addEventListener(
    'click',
    event => {
      const link = closestPopoverLink(event.target)
      if (link) void mouseClickHandler(event, link)
    },
    { signal },
  )
  signal.addEventListener(
    'abort',
    () => {
      if (delegatedPopoverSignal === signal) delegatedPopoverSignal = undefined
    },
    { once: true },
  )
}

function preloadPopoverLinks(container: Document | HTMLElement = document): void {
  const links = ([...container.getElementsByClassName('internal')] as HTMLAnchorElement[]).filter(
    link => !link.closest('#stacked-notes-container'),
  )

  const pdfSources = links.flatMap(link => {
    const source = pdfPreviewSource(link)
    return source ? [source] : []
  })
  if (pdfSources.length > 0) schedulePdfRuntimePreload(pdfSources)
}

function cleanupPopoverRuntime(): void {
  activePopoverReq?.abort()
  activePopoverReq = null
  stackedPopoverEvents?.abort()
  stackedPopoverEvents = null
  clearActivePopover()
}

document.addEventListener('nav', () => {
  window.addCleanup(disposeSidePanel)
  window.addCleanup(cleanupPopoverRuntime)
  setupPopoverDelegation()
  preloadPopoverLinks()
  setupStackedPopoverLinks()
})

document.addEventListener('contentdecrypted', e => {
  const { content } = e.detail
  if (content) preloadPopoverLinks(content)
})
