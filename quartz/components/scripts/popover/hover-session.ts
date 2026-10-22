import { fetchCanonical } from '../../../util/fetch-canonical'
import { getFullSlug } from '../../../util/path'
import { beginSitePerformanceSample, endSitePerformanceSample } from '../performance-sample'
import {
  cloneCurrentStreamEntryPreview,
  fetchStackedNotePayload,
  isNoteDocumentUrl,
  populatePopoverContent,
  populateStackedPayloadContent,
  streamPopoverFetchUrl,
  streamPopoverKey,
} from './content-render'
import { removePopoverElement } from './dom'
import {
  arxivPdfUrl,
  cleanArxivIdentifier,
  createPdfEmbed,
  PDF_PREVIEW_OPTIONS,
  pdfTitleFromUrl,
  isPdfUrl,
  mountPdfPreview,
  mountPdfPreviews,
} from './pdf-preview'
import {
  clearActivePopover,
  clearPopoverLoading,
  compareUrls,
  createPopoverElement,
  findHashTarget,
  handleBibliography,
  handleFootnote,
  isAbortError,
  notifyProtectedContentLoaded,
  popoverState,
  scrollPopoverToHash,
  setPopoverLoading,
  setPosition,
  showLoadingPopover,
  showPopover,
} from './placement'
import { summaryPreviewSource, type SummaryPreviewSource } from './summary-preview'

export function dismissPopover() {
  popoverState.request?.abort()
  popoverState.request = null
  document
    .querySelectorAll<HTMLElement>('.popover.active-popover, .popover:hover')
    .forEach(removePopoverElement)
  clearActivePopover()
}

async function handleSummaryPreview(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
  source: SummaryPreviewSource,
) {
  const { popoverId } = source
  const existingPopover = document.getElementById(popoverId)
  if (existingPopover) {
    const loading = existingPopover.querySelector('.popover-inner[data-state="loading"]')
    if (loading && popoverState.request?.link !== link) {
      popoverState.request?.abort()
      removePopoverElement(existingPopover)
    } else {
      link.dataset.popoverId = popoverId
      await showPopover(link, existingPopover, pointer)
      return
    }
  }

  if (popoverState.request && popoverState.request.link !== link) {
    popoverState.request.abort()
    popoverState.request = null
  }

  const controller = new AbortController()
  const request = { abort: () => controller.abort(), link }
  popoverState.request = request
  const { popoverElement, popoverInner } = createPopoverElement(...source.className.split(/\s+/))
  popoverElement.id = popoverId
  if (source.loadingLabel) {
    setPopoverLoading(popoverInner, source.loadingLabel)
    const label = document.createElement('span')
    label.textContent = source.loadingLabel
    popoverInner.querySelector('.popover-loading')?.appendChild(label)
    document.body.appendChild(popoverElement)
    await showLoadingPopover(link, popoverElement, pointer)
  }
  const render = await source.load(controller.signal).catch(error => {
    if (!isAbortError(error)) console.error(error)
    return null
  })
  if (popoverState.request === request) popoverState.request = null
  if (!render || controller.signal.aborted || !link.isConnected || popoverState.anchor !== link) {
    removePopoverElement(popoverElement)
    return
  }
  clearPopoverLoading(popoverInner)
  render(popoverInner)
  link.dataset.popoverId = popoverId
  if (!popoverElement.isConnected) document.body.appendChild(popoverElement)
  await showPopover(link, popoverElement, pointer, { popoverInner })
}

async function handleStackedNotes(
  stacked: HTMLElement | null,
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  if (!stacked) return
  clearActivePopover()
  popoverState.anchor = link

  if (popoverState.request && popoverState.request.link !== link) {
    popoverState.request.abort()
    popoverState.request = null
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
  popoverState.request = { abort: () => controller.abort(), link }

  const { popoverElement, popoverInner } = createPopoverElement('stacked-popover')
  popoverElement.id = 'stacked-note-preview'
  link.dataset.popoverId = popoverElement.id
  if (isNoteDocumentUrl(targetUrl)) {
    const payload = await fetchStackedNotePayload(targetUrl, controller.signal)
    if (
      !payload ||
      controller.signal.aborted ||
      !link.isConnected ||
      popoverState.anchor !== link
    ) {
      removePopoverElement(popoverElement)
      if (popoverState.request?.link === link) popoverState.request = null
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

    if (
      !response ||
      controller.signal.aborted ||
      !link.isConnected ||
      popoverState.anchor !== link
    ) {
      removePopoverElement(popoverElement)
      if (popoverState.request?.link === link) popoverState.request = null
      return
    }
    await populatePopoverContent(response, targetUrl, popoverInner, '', controller.signal)
  }

  if (
    popoverInner.childElementCount === 0 ||
    controller.signal.aborted ||
    !link.isConnected ||
    popoverState.anchor !== link
  ) {
    removePopoverElement(popoverElement)
    if (popoverState.request?.link === link) popoverState.request = null
    return
  }

  const startedAt = beginSitePerformanceSample()
  column.appendChild(popoverElement)
  mountPdfPreviews(popoverInner)
  notifyProtectedContentLoaded(popoverInner)
  await setPosition(link, popoverElement, {
    clientX: pointer.clientX,
    clientY: pointer.clientY,
    placement: 'right',
    strategy: 'fixed',
  })

  if (hash !== '') {
    scrollPopoverToHash(popoverInner, hash)
  }

  requestAnimationFrame(() => {
    if (link.isConnected && popoverState.anchor === link)
      popoverElement.classList.add('active-popover')
  })
  endSitePerformanceSample('popover', startedAt)

  if (popoverState.request?.link === link) popoverState.request = null
}

export function containsRelatedTarget(element: HTMLElement, relatedTarget: EventTarget | null) {
  return relatedTarget instanceof Node && element.contains(relatedTarget)
}

function popoverForLink(link: HTMLAnchorElement): HTMLElement | null {
  const { popoverId } = link.dataset
  if (!popoverId) return null
  const popover = document.getElementById(popoverId)
  return popover instanceof HTMLElement ? popover : null
}

export function mouseLeaveHandler(this: HTMLAnchorElement, event: MouseEvent | FocusEvent) {
  if (containsRelatedTarget(this, event.relatedTarget)) return
  const containingPopover = this.closest<HTMLElement>('.popover')
  if (containingPopover && containsRelatedTarget(containingPopover, event.relatedTarget)) return

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

  if (popoverState.request?.link === this) {
    popoverState.request.abort()
    popoverState.request = null
  }
  clearActivePopover()
}

export async function mouseEnterHandler(
  this: HTMLAnchorElement,
  { clientX, clientY }: { clientX: number; clientY: number },
) {
  // eslint-disable-next-line @typescript-eslint/no-this-alias
  popoverState.anchor = this
  const link = popoverState.anchor

  if (link.dataset.bib !== undefined) {
    await handleBibliography(link, { clientX, clientY })
    return
  }

  if (link.dataset.footnoteRef !== undefined) {
    await handleFootnote(link, { clientX, clientY })
    return
  }

  const container = document.getElementById('stacked-notes-container')

  const stackedBacklink = link.dataset.backlink !== undefined && container?.contains(link)
  if ((link.dataset.noPopover === '' || link.dataset.noPopover === 'true') && !stackedBacklink) {
    return
  }

  const summary = summaryPreviewSource(link)
  if (summary) {
    await handleSummaryPreview(link, { clientX, clientY }, summary)
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
        popoverState.dagHeading?.classList.remove('dag')
        heading.classList.add('dag')
        popoverState.dagHeading = heading
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
  let contentRequest: AbortController | undefined
  if (link.dataset.arxivId) {
    if (popoverState.request && popoverState.request.link !== link) {
      popoverState.request.abort()
      popoverState.request = null
    }

    const controller = new AbortController()
    popoverState.request = { abort: () => controller.abort(), link }

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
        if (popoverState.request?.link === link) popoverState.request = null
      },
      () => {
        removePopoverElement(popoverElement)
        if (popoverState.request?.link === link) popoverState.request = null
      },
    )

    if (popoverState.anchor !== link) {
      controller.abort()
      popoverState.request = null
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
    if (popoverState.anchor !== link) {
      removePopoverElement(popoverElement)
      return
    }

    await showPopover(link, popoverElement, { clientX, clientY }, { hash, popoverInner })
    return
  } else {
    popoverState.request?.abort()
    contentRequest = new AbortController()
    const controller = contentRequest
    popoverState.request = { abort: () => controller.abort(), link }
    response = await fetchCanonical(fetchUrl, { signal: controller.signal }).catch(error => {
      if (!isAbortError(error)) console.error(error)
    })
  }

  if (!response) {
    if (popoverState.request?.link === link) popoverState.request = null
    return
  }
  if (popoverState.anchor !== link) return

  const { popoverElement, popoverInner } = createPopoverElement()
  popoverElement.id = popoverId

  try {
    await populatePopoverContent(response, targetUrl, popoverInner, hash, contentRequest?.signal)
  } catch (error) {
    if (!isAbortError(error)) console.error(error)
    return
  } finally {
    if (popoverState.request?.link === link) popoverState.request = null
  }
  if (contentRequest?.signal.aborted || !link.isConnected || popoverState.anchor !== link) return

  if (document.getElementById(popoverId)) return

  document.body.appendChild(popoverElement)
  mountPdfPreviews(popoverInner)
  notifyProtectedContentLoaded(popoverInner)
  if (popoverState.anchor !== link) {
    removePopoverElement(popoverElement)
    return
  }

  await showPopover(link, popoverElement, { clientX, clientY }, { hash, popoverInner })
}
