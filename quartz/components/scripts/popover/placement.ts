import {
  arrow as floatingArrow,
  autoPlacement,
  computePosition,
  flip,
  type Placement,
  offset,
  type ReferenceElement,
  shift,
  size,
  type Strategy,
  type VirtualElement,
} from '@floating-ui/dom'
import { getFullSlug } from '../../../util/path'
import { beginSitePerformanceSample, endSitePerformanceSample } from '../performance-sample'
import { blurPopoverFocus, touchPopoverElement, evictInactivePopovers } from './dom'

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

// One object so sibling files can reassign the session; ES import bindings are read-only.
export const popoverState: {
  anchor: HTMLAnchorElement | null
  dagHeading: HTMLElement | null
  request: { abort: () => void; link: HTMLAnchorElement } | null
} = { anchor: null, dagHeading: null, request: null }

export function createPopoverElement(...classes: string[]): {
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

export function clearPopoverLoading(popoverInner: HTMLDivElement) {
  delete popoverInner.dataset.state
  popoverInner.querySelector<HTMLElement>(':scope > .popover-loading')?.remove()
}

export function setPopoverLoading(
  popoverInner: HTMLDivElement,
  label: string,
  contentType?: string,
) {
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
  if (link.dataset.backlink !== undefined)
    return link.closest<HTMLElement>('.backlink-card') ?? link
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

export function findHashTarget(
  container: ParentNode,
  hash: string,
  prefix = '',
): HTMLElement | null {
  const rawId = hash.startsWith('#') ? hash.slice(1) : hash
  if (!rawId) return null
  const id = prefix && !rawId.startsWith(prefix) ? `${prefix}${rawId}` : rawId
  return container.querySelector<HTMLElement>(`#${CSS.escape(id)}`)
}

export function scrollPopoverToHash(popoverInner: HTMLElement, hash: string) {
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

export async function setPosition(
  link: HTMLElement,
  popoverElement: HTMLElement,
  { clientX, clientY, placement, strategy = 'fixed' }: PositioningOptions,
) {
  const arrowElement = popoverElement.querySelector<HTMLElement>('.popover-arrow')
  const point = { x: clientX, y: clientY }
  const reference = popoverReference(link, point)
  const middleware = [
    offset(2),
    placement
      ? flip({ padding: 8, fallbackAxisSideDirection: 'end' })
      : autoPlacement({ padding: 8 }),
    shift({ padding: 8 }),
    // a course card is taller than the room beside a short window; it scrolls instead of clipping
    popoverElement.matches('.course-popover, .page-preview-popover')
      ? size({
          padding: 8,
          apply({ availableHeight, elements }) {
            const inner = elements.floating.querySelector<HTMLElement>('.popover-inner')
            if (!inner) return
            // the popover element wraps the inner box with its arrow gutter
            const gutter = elements.floating.offsetHeight - inner.offsetHeight
            inner.style.maxHeight = `min(${Math.max(0, Math.floor(availableHeight - gutter))}px, 75vh, 30rem)`
          },
        })
      : null,
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

export async function showPopover(
  link: HTMLAnchorElement,
  popoverElement: HTMLElement,
  pointer: { clientX: number; clientY: number },
  options: ShowPopoverOptions = {},
) {
  const startedAt = beginSitePerformanceSample()
  touchPopoverElement(popoverElement)
  clearActivePopover(popoverElement)
  popoverElement.classList.toggle('keyboard-popover', link.matches(':focus-visible'))
  popoverElement.classList.add('active-popover')
  evictInactivePopovers(popoverElement)

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
  endSitePerformanceSample('popover', startedAt)
}

export async function showLoadingPopover(
  link: HTMLAnchorElement,
  popoverElement: HTMLElement,
  pointer: { clientX: number; clientY: number },
) {
  const startedAt = beginSitePerformanceSample()
  touchPopoverElement(popoverElement)
  clearActivePopover(popoverElement)
  popoverState.anchor = link
  popoverElement.classList.add('active-popover')
  evictInactivePopovers(popoverElement)
  await setPosition(link, popoverElement, { clientX: pointer.clientX, clientY: pointer.clientY })
  endSitePerformanceSample('popover', startedAt)
}

export function clearActivePopover(except?: HTMLElement) {
  popoverState.anchor = null
  popoverState.dagHeading?.classList.remove('dag')
  popoverState.dagHeading = null
  const allPopoverElements = document.querySelectorAll<HTMLElement>('.popover')
  allPopoverElements.forEach(popoverElement => {
    if (popoverElement === except) return
    blurPopoverFocus(popoverElement)
    popoverElement.classList.remove('active-popover')
  })
  evictInactivePopovers(except)
}

export function notifyProtectedContentLoaded(container: ParentNode) {
  document.dispatchEvent(new CustomEvent('protectedcontentloaded', { detail: { container } }))
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

export function compareUrls(a: URL, b: URL): boolean {
  const u1 = new URL(a.toString())
  const u2 = new URL(b.toString())
  u1.hash = ''
  u1.search = ''
  u2.hash = ''
  u2.search = ''
  return u1.toString() === u2.toString()
}

function referenceEntry(link: HTMLAnchorElement): { entry: HTMLElement; key: string } | null {
  const hash = new URL(link.href).hash
  const note = link.closest<HTMLElement>('.stacked-note')
  const entry = findHashTarget(note ?? document, decodeURIComponent(hash))
  if (!entry) return null
  return { entry, key: `${note?.dataset.slug ?? getFullSlug(window)}-${entry.id}` }
}

export async function handleBibliography(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  const reference = referenceEntry(link)
  if (!reference) return
  const { entry: bibEntry, key } = reference

  const popoverId = `popover-bib-${key}`
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

export async function handleFootnote(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  const reference = referenceEntry(link)
  if (!reference) return
  const { entry: footnoteEntry, key } = reference

  const popoverId = `popover-footnote-${key}`
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
