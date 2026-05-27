import type { Placement, ReferenceElement, Strategy, VirtualElement } from '@floating-ui/dom'
import { arrow as floatingArrow, computePosition, flip, offset, shift } from '@floating-ui/dom'
import xmlFormat from 'xml-formatter'
import {
  lessWrongPreviewApiUrl,
  readLessWrongPreview,
  type LessWrongPreview,
  type LessWrongTarget,
} from '../../util/lesswrong'
import { getContentType } from '../../util/mime'
import { FullSlug, getFullSlug, normalizeRelativeURLs } from '../../util/path'
import {
  readWikipediaPreviewResponse,
  wikipediaActionApiUrl,
  type WikipediaPreview,
  type WikipediaTarget,
} from '../../util/wikipedia'
import {
  cacheStackedNotePayload,
  createSidePanel,
  fetchCanonical,
  getCachedStackedNotePayload,
  getOrCreateSidePanel,
  readStackedNotePayload,
  stackedNotePayloadUrl,
  StackedNotePayload,
} from './util'

type ContentHandler = (
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
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

const blobCleanupMap = new Map<string, NodeJS.Timeout>()
const DEFAULT_BLOB_TIMEOUT = 30 * 60 * 1000 // 30 minutes

const p = new DOMParser()
let activeAnchor: HTMLAnchorElement | null = null
let activePopoverReq: { abort: () => void; link: HTMLAnchorElement } | null = null
let stackedPopoverEvents: AbortController | null = null

function createManagedBlobUrl(blob: Blob, timeoutMs: number = DEFAULT_BLOB_TIMEOUT): string {
  const blobUrl = URL.createObjectURL(blob)
  const existingTimeout = blobCleanupMap.get(blobUrl)

  if (existingTimeout) {
    clearTimeout(existingTimeout)
  }

  const timeoutId = setTimeout(() => {
    URL.revokeObjectURL(blobUrl)
    blobCleanupMap.delete(blobUrl)
  }, timeoutMs)

  blobCleanupMap.set(blobUrl, timeoutId)
  return blobUrl
}

function cleanupBlobUrl(blobUrl: string): void {
  const timeoutId = blobCleanupMap.get(blobUrl)
  if (timeoutId !== undefined) {
    clearTimeout(timeoutId)
    URL.revokeObjectURL(blobUrl)
    blobCleanupMap.delete(blobUrl)
  }
}

function cleanAbsoluteElement(element: HTMLElement): HTMLElement {
  const refsAndNotes = element.querySelectorAll<HTMLElement>(
    'section[data-references], section[data-footnotes], [data-skip-preview], .telescopic-container',
  )
  refsAndNotes.forEach(section => section.remove())
  return element
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

async function handleImageContent(targetUrl: URL, popoverInner: HTMLDivElement) {
  const img = document.createElement('img')
  img.src = targetUrl.toString()
  img.alt = targetUrl.pathname
  popoverInner.appendChild(img)
}

async function handlePdfContent(response: Response, popoverInner: HTMLDivElement) {
  const pdf = document.createElement('iframe')
  const blob = await response.blob()
  const blobUrl = createManagedBlobUrl(blob)
  pdf.src = blobUrl
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
) {
  popoverInner.classList.add('grid')
  const contents = await response.text()
  const html = p.parseFromString(contents, 'text/html')
  normalizeRelativeURLs(html, targetUrl)
  html.querySelectorAll('[id]').forEach(el => {
    const targetID = `popover-${el.id}`
    el.id = targetID
  })
  const elts = [
    ...(html.getElementsByClassName('popover-hint') as HTMLCollectionOf<HTMLDivElement>),
  ].map(cleanAbsoluteElement)
  if (elts.length === 0) return
  popoverInner.append(...elts)
}

const contentHandlers: Record<string, ContentHandler> = {
  image: async (_, targetUrl, popoverInner) => handleImageContent(targetUrl, popoverInner),
  'application/pdf': async (response, _targetUrl, popoverInner) =>
    handlePdfContent(response, popoverInner),
  'application/xml': async (response, _targetUrl, popoverInner) =>
    handleXmlContent(response, popoverInner),
  default: handleDefaultContent,
}

async function populatePopoverContent(
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
) {
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

  await handler(response, targetUrl, popoverInner)
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
  html.querySelectorAll('[id]').forEach(el => {
    const targetID = `popover-${el.id}`
    el.id = targetID
  })
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
  clearActivePopover()
  popoverElement.classList.add('active-popover')

  await setPosition(link, popoverElement, {
    clientX: pointer.clientX,
    clientY: pointer.clientY,
    placement: options.placement,
    strategy: options.strategy,
  })

  const { hash, popoverInner } = options
  if (hash && hash !== '' && popoverInner) {
    const heading = findHashTarget(popoverInner, hash, 'popover-')
    if (heading) {
      popoverInner.scroll({ top: heading.offsetTop - 12, behavior: 'instant' })
    }
  }
}

function clearActivePopover() {
  activeAnchor = null
  const allPopoverElements = document.querySelectorAll('.popover')
  allPopoverElements.forEach(popoverElement => popoverElement.classList.remove('active-popover'))
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

function lessWrongTocHref(pageUrl: string, href: string): string {
  try {
    return new URL(href, pageUrl).toString()
  } catch {
    return pageUrl
  }
}

function appendLessWrongToc(parent: HTMLElement, preview: LessWrongPreview) {
  if (!preview.toc || preview.toc.length === 0) return

  const section = document.createElement('section')
  section.classList.add('lesswrong-popover-toc')
  section.ariaLabel = 'LessWrong contents'

  const label = document.createElement('div')
  label.classList.add('lesswrong-popover-toc-title')
  label.textContent = 'Contents'
  section.appendChild(label)

  const list = document.createElement('ol')
  list.classList.add('lesswrong-popover-toc-list')

  for (const entry of preview.toc) {
    const item = document.createElement('li')
    item.classList.add('lesswrong-popover-toc-item')
    item.style.setProperty('--lesswrong-popover-toc-indent', `${(entry.depth - 1) * 0.72}rem`)

    const link = document.createElement('a')
    link.classList.add('lesswrong-popover-toc-link')
    link.href = lessWrongTocHref(preview.pageUrl, entry.href)
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
  appendLessWrongToc(card, preview)

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
    .forEach(popover => popover.remove())

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
      popoverElement.remove()
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
      popoverElement.remove()
      activePopoverReq = null
      return
    }
    await populatePopoverContent(response, targetUrl, popoverInner)
  }

  if (popoverInner.childElementCount === 0) {
    popoverElement.remove()
    activePopoverReq = null
    return
  }

  column.appendChild(popoverElement)
  hideStackedPopoversOnLeave(stacked, link, popoverElement)
  notifyProtectedContentLoaded(popoverInner)
  await setPosition(link, popoverElement, {
    clientX: pointer.clientX,
    clientY: pointer.clientY,
    placement: 'right',
    strategy: 'absolute',
  })

  if (hash !== '') {
    const heading = findHashTarget(popoverInner, hash, 'popover-')
    if (heading) {
      popoverInner.scroll({ top: heading.offsetTop - 12, behavior: 'instant' })
    }
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

  clearActivePopover()
}

function allowsStackedPopover(link: HTMLAnchorElement): boolean {
  if (link.dataset.wikipediaLang && link.dataset.wikipediaTitle) return false
  if (link.dataset.lesswrongPostId) return false
  if (link.dataset.noPopover === '' || link.dataset.noPopover === 'true') {
    return link.dataset.backlink !== undefined
  }
  return true
}

function hideStackedPopovers(stacked: HTMLElement) {
  stacked.querySelectorAll<HTMLElement>('.stacked-popover').forEach(popoverElement => {
    popoverElement.classList.remove('active-popover')
    window.setTimeout(() => {
      if (!popoverElement.classList.contains('active-popover')) {
        popoverElement.remove()
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
      const article = document.querySelector('article')
      const heading = article ? findHashTarget(article, hash) : null
      if (heading) {
        heading.classList.add('dag')
        const cleanup = () => {
          heading.classList.remove('dag')
          link.removeEventListener('mouseleave', cleanup)
        }
        link.addEventListener('mouseleave', cleanup)
        window.addCleanup(() => link.removeEventListener('mouseleave', cleanup))
      }
    }
    return
  }

  targetUrl.hash = ''
  targetUrl.search = ''

  const popoverId = `popover-${targetUrl.pathname}`
  const existingPopover = document.getElementById(popoverId) as HTMLElement | null

  if (existingPopover) {
    const popoverInner = existingPopover.querySelector('.popover-inner') as HTMLDivElement | null
    if (!popoverInner) return
    await showPopover(link, existingPopover, { clientX, clientY }, { hash, popoverInner })
    return
  }

  let response: Response | void
  if (link.dataset.arxivId) {
    const url = new URL(`https://aarnphm.xyz/api/arxiv?identifier=${link.dataset.arxivId}`)
    response = await fetchCanonical(url).catch(error => {
      console.error(error)
    })
  } else {
    response = await fetchCanonical(new URL(targetUrl.toString())).catch(error => {
      console.error(error)
    })
  }

  if (!response) return
  if (activeAnchor !== link) return

  const { popoverElement, popoverInner } = createPopoverElement()
  popoverElement.id = popoverId

  await populatePopoverContent(response, targetUrl, popoverInner)

  if (document.getElementById(popoverId)) return

  document.body.appendChild(popoverElement)
  notifyProtectedContentLoaded(popoverInner)
  if (activeAnchor !== link) {
    return
  }

  await showPopover(link, popoverElement, { clientX, clientY }, { hash, popoverInner })
}

async function mouseClickHandler(evt: MouseEvent) {
  const link = evt.currentTarget as HTMLAnchorElement
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

  if (evt.altKey && !container?.classList.contains('active')) {
    evt.preventDefault()
    evt.stopPropagation()

    // derive slug from href if data-slug not set (e.g. generated pages like /tags/*)
    const slug = link.dataset.slug || targetUrl.pathname

    try {
      const asidePanel = getOrCreateSidePanel()
      asidePanel.dataset.slug = slug

      let response: Response | void
      if (link.dataset.arxivId) {
        const url = new URL(`https://aarnphm.xyz/api/arxiv?identifier=${link.dataset.arxivId}`)
        response = await fetchCanonical(url).catch(console.error)
      } else {
        const fetchUrl = new URL(link.href)
        fetchUrl.hash = ''
        fetchUrl.search = ''
        response = await fetchCanonical(fetchUrl).catch(console.error)
      }

      if (!response) return

      const headerContentType = response.headers.get('Content-Type')
      const contentType = headerContentType
        ? headerContentType.split(';')[0]
        : getContentType(targetUrl)

      if (contentType === 'application/pdf') {
        const pdf = document.createElement('iframe')
        const blob = await response.blob()
        const blobUrl = createManagedBlobUrl(blob)
        pdf.src = blobUrl
        createSidePanel(asidePanel, pdf)
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
        if (elts.length === 0) return

        createSidePanel(asidePanel, ...elts)
      }

      window.notifyNav(slug as FullSlug)
    } catch (error) {
      console.error('Failed to create side panel:', error)
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

function setupPopoverLinks(container: Document | HTMLElement = document) {
  const links = ([...container.getElementsByClassName('internal')] as HTMLAnchorElement[]).filter(
    link => !link.closest('#stacked-notes-container'),
  )

  for (const link of links) {
    link.addEventListener('mouseenter', mouseEnterHandler)
    link.addEventListener('mouseleave', mouseLeaveHandler)
    link.addEventListener('click', mouseClickHandler)

    window.addCleanup(() => {
      link.removeEventListener('mouseenter', mouseEnterHandler)
      link.removeEventListener('mouseleave', mouseLeaveHandler)
      link.removeEventListener('click', mouseClickHandler)

      for (const blobUrl of Array.from(blobCleanupMap.keys())) {
        cleanupBlobUrl(blobUrl)
      }

      if (activePopoverReq) {
        activePopoverReq.abort()
        activePopoverReq = null
      }
    })
  }
}

document.addEventListener('nav', () => {
  setupPopoverLinks()
  setupStackedPopoverLinks()
})

document.addEventListener('contentdecrypted', e => {
  const { content } = e.detail
  if (content) setupPopoverLinks(content)
})
