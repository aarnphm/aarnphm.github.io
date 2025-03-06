import { arrow, computePosition, flip, inline, Placement, shift } from "@floating-ui/dom"
import { FullSlug, getFullSlug, normalizeRelativeURLs } from "../../util/path"
import { getContentType } from "../../util/mime"
import xmlFormat from "xml-formatter"
import { createSidePanel, fetchCanonical } from "./util"

type ContentHandler = (
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
) => Promise<void>

// Helper to manage blob URL cleanup
const blobCleanupMap = new Map<string, NodeJS.Timeout>()

/**
 * Creates a blob URL and schedules it for cleanup
 * @param blob The blob to create a URL for
 * @param timeoutMs Time in milliseconds after which to revoke the blob URL (default: 5 minutes)
 * @returns The created blob URL
 */
function createManagedBlobUrl(blob: Blob, timeoutMs: number = 5 * 60 * 1000): string {
  const blobUrl = URL.createObjectURL(blob)

  // Clear any existing timeout for this URL
  if (blobCleanupMap.has(blobUrl)) {
    clearTimeout(blobCleanupMap.get(blobUrl))
  }

  // Schedule cleanup
  const timeoutId = setTimeout(() => {
    URL.revokeObjectURL(blobUrl)
    blobCleanupMap.delete(blobUrl)
  }, timeoutMs)

  blobCleanupMap.set(blobUrl, timeoutId)

  return blobUrl
}

/**
 * Immediately cleanup a blob URL if it exists
 * @param blobUrl The blob URL to cleanup
 * @param timeoutId The timeout ID associated with the blob URL
 */
function cleanupBlobUrl(blobUrl: string, timeoutId: NodeJS.Timeout): void {
  if (blobCleanupMap.has(blobUrl)) {
    clearTimeout(timeoutId)
    URL.revokeObjectURL(blobUrl)
    blobCleanupMap.delete(blobUrl)
  }
}

// Set a longer default timeout since we're not cleaning up on popover close
const DEFAULT_BLOB_TIMEOUT = 30 * 60 * 1000 // 30 minutes

const p = new DOMParser()
function cleanAbsoluteElement(element: HTMLElement): HTMLElement {
  const refsAndNotes = element.querySelectorAll(
    "section[data-references], section[data-footnotes], [data-skip-preview]",
  )
  refsAndNotes.forEach((section) => section.remove())
  return element
}

function hasPositionChanged(link: HTMLElement): boolean {
  const popover = link.lastChild as HTMLElement
  if (!popover.classList.contains("popover")) return true
  const current = link.getBoundingClientRect()

  // Check if we stored previous position
  const prevTop = popover.dataset.linkTop
  const prevLeft = popover.dataset.linkLeft

  return (
    !prevTop ||
    !prevLeft ||
    Math.abs(parseFloat(prevTop) - current.top) > 1 ||
    Math.abs(parseFloat(prevLeft) - current.left) > 1
  )
}

// Helper functions
function createPopoverElement(className?: string): {
  popoverElement: HTMLElement
  popoverInner: HTMLDivElement
} {
  const popoverElement = document.createElement("div")
  popoverElement.classList.add("popover", ...(className ? [className] : []))
  const popoverInner = document.createElement("div")
  popoverInner.classList.add("popover-inner")
  popoverElement.appendChild(popoverInner)
  const arrowElement = document.createElement("div")
  arrowElement.id = "arrow"
  popoverElement.appendChild(arrowElement)
  return { popoverElement, popoverInner }
}

function compareUrls(a: URL, b: URL): boolean {
  const u1 = new URL(a.toString())
  const u2 = new URL(b.toString())
  u1.hash = ""
  u1.search = ""
  u2.hash = ""
  u2.search = ""
  return u1.toString() === u2.toString()
}

async function handleImageContent(targetUrl: URL, popoverInner: HTMLDivElement) {
  const img = document.createElement("img")
  img.src = targetUrl.toString()
  img.alt = targetUrl.pathname
  popoverInner.appendChild(img)
}

// NOTE: Given that we will run this on cloudflare workers, all PDF will be fetched
// directly from Git LFS server.
async function handlePdfContent(response: Response, popoverInner: HTMLDivElement) {
  const pdf = document.createElement("iframe")
  const blob = await response.blob()
  const blobUrl = createManagedBlobUrl(blob, DEFAULT_BLOB_TIMEOUT)
  pdf.src = blobUrl
  popoverInner.appendChild(pdf)
}

async function handleXmlContent(response: Response, popoverInner: HTMLDivElement) {
  const contents = await response.text()
  const rss = document.createElement("pre")
  rss.classList.add("rss-viewer")
  rss.append(xmlFormat(contents, { indentation: "  ", lineSeparator: "\n" }))
  popoverInner.append(rss)
}

async function handleDefaultContent(
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
) {
  popoverInner.classList.add("grid")
  const contents = await response.text()
  const html = p.parseFromString(contents, "text/html")
  normalizeRelativeURLs(html, targetUrl)
  const elts = [
    ...(html.getElementsByClassName("popover-hint") as HTMLCollectionOf<HTMLElement>),
  ].map(cleanAbsoluteElement)
  if (elts.length === 0) return
  popoverInner.append(...elts)
}

async function setPosition(
  link: HTMLElement,
  popoverElement: HTMLElement,
  placement: Placement,
  clientX: number,
  clientY: number,
) {
  const element = popoverElement.querySelector("#arrow") as HTMLElement
  const ignoreArrow =
    link.closest(".sidepanel-inner") !== null &&
    !popoverElement.classList.contains("bib-popover") &&
    !popoverElement.classList.contains("footnote-popover")
  const middleware = [inline({ x: clientX, y: clientY }), shift(), flip()]

  // Only add arrow middleware if not in sidepanel
  if (!ignoreArrow) middleware.push(arrow({ element, padding: 8 }))
  const {
    x,
    y,
    middlewareData,
    placement: finalPlacement,
  } = await computePosition(link, popoverElement, { placement, middleware })
  Object.assign(popoverElement.style, { left: `${x}px`, top: `${y}px` })
  if (ignoreArrow) {
    element.style.display = "none"
  }

  if (middlewareData.arrow) {
    const { x: arrowX, y: arrowY } = middlewareData.arrow
    const staticSide = {
      top: "bottom",
      right: "left",
      bottom: "top",
      left: "right",
    }[finalPlacement.split("-")[0]] as string

    Object.assign(element.style, {
      left: arrowX != null ? `${arrowX}px` : "",
      top: arrowY != null ? `${arrowY}px` : "",
    })
    element.dataset.placement = staticSide
    if (staticSide === "right") {
      element.style.right = "2px"
    }
  }

  const linkRect = link.getBoundingClientRect()
  popoverElement.dataset.linkTop = linkRect.top.toString()
  popoverElement.dataset.linkLeft = linkRect.left.toString()
}

const hasAlreadyBeenFetched = (link: HTMLAnchorElement, classname?: string) =>
  [...link.children].some((child) => child.classList.contains(classname ?? "popover"))

async function handleBibliography(link: HTMLAnchorElement, clientX: number, clientY: number) {
  const href = link.getAttribute("href")!

  if (hasAlreadyBeenFetched(link, "bib-popover")) {
    if (hasPositionChanged(link)) {
      return setPosition(link, link.lastChild as HTMLElement, "top", clientX, clientY)
    }
    return
  }

  const bibEntry = document.getElementById(href.replace("#", "")) as HTMLLIElement

  const { popoverElement, popoverInner } = createPopoverElement("bib-popover")
  popoverInner.innerHTML = bibEntry.innerHTML

  setPosition(link, popoverElement, "top", clientX, clientY)
  link.appendChild(popoverElement)
  return popoverElement
}

async function handleFootnote(link: HTMLAnchorElement, clientX: number, clientY: number) {
  const href = link.getAttribute("href")!

  if (hasAlreadyBeenFetched(link, "footnote-popover")) {
    if (hasPositionChanged(link)) {
      return setPosition(link, link.lastChild as HTMLElement, "top", clientX, clientY)
    }
    return
  }

  const footnoteEntry = document.getElementById(href.replace("#", "")) as HTMLLIElement
  const { popoverElement, popoverInner } = createPopoverElement("footnote-popover")
  popoverInner.innerHTML = footnoteEntry.innerHTML
  popoverInner.querySelectorAll("[data-footnote-backref]").forEach((el) => el.remove())

  setPosition(link, popoverElement, "top", clientX, clientY)
  link.appendChild(popoverElement)
  return popoverElement
}

// Track current active popover request
let activePopoverReq: { abort: () => void; link: HTMLAnchorElement } | null = null

async function handleStackedNotes(
  stacked: HTMLDivElement,
  link: HTMLAnchorElement,
  { clientX, clientY }: { clientX: number; clientY: number },
) {
  // If there's an active request for a different link, cancel it
  if (activePopoverReq && activePopoverReq.link !== link) {
    activePopoverReq.abort()
    activePopoverReq = null
  }

  const column = stacked.querySelector<HTMLDivElement>(".stacked-notes-column")
  if (!column) return

  // Remove any existing popovers
  const current = column.querySelectorAll<HTMLDivElement>('div[class~="stacked-popover"]')
  current.forEach((popover) => popover.remove())

  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)

  // Create an AbortController for this request
  const controller = new AbortController()
  activePopoverReq = { abort: () => controller.abort(), link }

  const response = await fetchCanonical(new URL(`${targetUrl}`), {
    signal: controller.signal,
  }).catch((error) => {
    if (error.name === "AbortError") return null
    console.error(error)
    return null
  })
  if (!response) return
  const contentType = response.headers.get("Content-Type")
    ? response.headers.get("Content-Type")!.split(";")[0]
    : getContentType(targetUrl)
  const [contentTypeCategory, _] = contentType.split("/")

  const { popoverElement, popoverInner } = createPopoverElement("stacked-popover")
  popoverInner.dataset.contentType = contentType ?? undefined

  popoverInner.dataset.contentType = contentType ?? undefined
  popoverElement.dataset.arrow = (contentType! !== "application/pdf").toString()

  const contentHandlers: Record<string, ContentHandler> = {
    image: async (_, targetUrl, popoverInner) => handleImageContent(targetUrl, popoverInner),
    "application/pdf": async (response, _, popoverInner) =>
      handlePdfContent(response, popoverInner),
    "application/xml": async (response, _, popoverInner) =>
      handleXmlContent(response, popoverInner),
    default: handleDefaultContent,
  }

  const handler =
    contentHandlers[contentTypeCategory] ||
    contentHandlers[contentType] ||
    contentHandlers["default"]

  handler(response, targetUrl, popoverInner)
  setPosition(link, popoverElement, "right", clientX, clientY)
  column!.appendChild(popoverElement)
  popoverElement.style.visibility = "visible"
  popoverElement.style.opacity = "1"

  if (hash !== "") {
    const heading = popoverInner.querySelector<HTMLElement>(
      `h1${hash}, h2${hash}, h3${hash}, h4${hash}, h5${hash}, h6${hash}`,
    )
    if (heading) {
      popoverInner.scroll({ top: heading.offsetTop - 12, behavior: "instant" })
    }
  }

  const onMouseLeave = () => {
    if (activePopoverReq?.link === link) {
      activePopoverReq.abort()
      activePopoverReq = null
    }
    popoverElement.style.visibility = "hidden"
    popoverElement.style.opacity = "0"
    setTimeout(() => {
      if (popoverElement.style.visibility === "hidden") {
        popoverElement.remove()
      }
    }, 100)
  }

  link.addEventListener("mouseleave", onMouseLeave)
  window.addCleanup(() => {
    link.removeEventListener("mouseleave", onMouseLeave)
    if (activePopoverReq?.link === link) {
      activePopoverReq.abort()
      activePopoverReq = null
    }
  })
  return popoverElement
}

async function mouseEnterHandler(
  this: HTMLAnchorElement,
  { clientX, clientY }: { clientX: number; clientY: number },
) {
  const link = this

  if (link.dataset.bib === "") {
    return handleBibliography(link, clientX, clientY)
  }

  if (link.dataset.footnoteRef === "") {
    return handleFootnote(link, clientX, clientY)
  }

  const container = document.getElementById("stacked-notes-container") as HTMLDivElement

  if (link.dataset.noPopover === "" || link.dataset.noPopover === "true") {
    return
  }

  if (getFullSlug(window) === "notes" || container?.classList.contains("active")) {
    return handleStackedNotes(container, link, { clientX, clientY })
  }

  let position: Placement = "right"
  // Check if link is within sidepanel
  const isInSidepanel = link.closest(".sidepanel-inner") !== null
  if (window.document.body.dataset.isFolderTag === "true") {
    position = "left"
  } else if (link.closest(".tag-link") !== null) {
    position = "right"
  } else if (isInSidepanel) {
    position = "bottom"
  }

  if (hasAlreadyBeenFetched(link)) {
    if (hasPositionChanged(link)) {
      return setPosition(link, link.lastChild as HTMLElement, position, clientX, clientY)
    }
    return
  }

  const thisUrl = new URL(document.location.href)
  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)

  // prevent hover of the same page
  if (compareUrls(thisUrl, targetUrl)) {
    // Handle same-page hash links
    if (hash !== "") {
      const article = document.querySelector("article")
      const heading = article?.querySelector(
        `h1${hash}, h2${hash}, h3${hash}, h4${hash}, h5${hash}, h6${hash}`,
      ) as HTMLElement | null
      if (heading) {
        heading.classList.add("dag")
        // Add cleanup for mouseleave
        const cleanup = () => {
          heading.classList.remove("dag")
          link.removeEventListener("mouseleave", cleanup)
        }
        link.addEventListener("mouseleave", cleanup)
        window.addCleanup(() => link.removeEventListener("mouseleave", cleanup))
      }
    }
    return
  }

  let response: Response | void
  if (link.dataset.arxivId) {
    const url = new URL(`https://cdn.aarnphm.xyz/api/arxiv?identifier=${link.dataset.arxivId}`)
    response = await fetchCanonical(url).catch(console.error)
  } else {
    response = await fetchCanonical(new URL(`${targetUrl}`)).catch(console.error)
  }

  if (!response) return
  const contentType = response.headers.get("Content-Type")
    ? response.headers.get("Content-Type")!.split(";")[0]
    : getContentType(targetUrl)
  const [contentTypeCategory, _] = contentType.split("/")

  const { popoverElement, popoverInner } = createPopoverElement()
  popoverInner.dataset.contentType = contentType ?? undefined
  popoverElement.dataset.arrow = (contentType! !== "application/pdf").toString()

  const contentHandlers: Record<string, ContentHandler> = {
    image: async (_, targetUrl, popoverInner) => handleImageContent(targetUrl, popoverInner),
    "application/pdf": async (response, _, popoverInner) =>
      handlePdfContent(response, popoverInner),
    "application/xml": async (response, _, popoverInner) =>
      handleXmlContent(response, popoverInner),
    default: handleDefaultContent,
  }

  const handler =
    contentHandlers[contentTypeCategory] ||
    contentHandlers[contentType] ||
    contentHandlers["default"]

  handler(response, targetUrl, popoverInner)
  setPosition(link, popoverElement, position, clientX, clientY)
  link.appendChild(popoverElement)

  if (hash !== "") {
    const heading = popoverInner.querySelector<HTMLElement>(
      `h1${hash}, h2${hash}, h3${hash}, h4${hash}, h5${hash}, h6${hash}`,
    )
    if (heading) {
      popoverInner.scroll({ top: heading.offsetTop - 12, behavior: "instant" })
    }
  }
  return popoverElement
}

async function mouseClickHandler(evt: MouseEvent) {
  const link = evt.currentTarget as HTMLAnchorElement
  const thisUrl = new URL(document.location.href)
  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)

  const container = document.getElementById("stacked-notes-container") as HTMLDivElement

  if (evt.altKey && !container?.classList.contains("active")) {
    evt.preventDefault()
    evt.stopPropagation()
    const asidePanel = document.querySelector<HTMLDivElement>(
      "main > aside[class~='sidepanel-container']",
    )

    if (!asidePanel) return
    asidePanel.dataset.slug = link.dataset.slug

    let response: Response | void
    if (link.dataset.arxivId) {
      const url = new URL(`https://cdn.aarnphm.xyz/api/arxiv?identifier=${link.dataset.arxivId}`)
      response = await fetchCanonical(url).catch(console.error)
    } else {
      response = await fetchCanonical(new URL(`${targetUrl}`)).catch(console.error)
    }

    if (!response) return
    const contentType = response.headers.get("Content-Type")
      ? response.headers.get("Content-Type")!.split(";")[0]
      : getContentType(targetUrl)

    if (contentType === "application/pdf") {
      const pdf = document.createElement("iframe")
      const blob = await response.blob()
      const blobUrl = createManagedBlobUrl(blob, DEFAULT_BLOB_TIMEOUT)
      pdf.src = blobUrl
      createSidePanel(asidePanel, pdf)
    } else {
      const contents = await response.text()
      const html = p.parseFromString(contents, "text/html")
      normalizeRelativeURLs(html, targetUrl)
      const elts = [
        ...(html.getElementsByClassName("popover-hint") as HTMLCollectionOf<HTMLElement>),
      ]
      if (elts.length === 0) return

      createSidePanel(asidePanel, ...elts)
    }
    window.notifyNav(link.dataset.slug as FullSlug)
    return
  }

  if (compareUrls(thisUrl, targetUrl) && hash !== "") {
    evt.preventDefault()
    const mainContent = document.querySelector("article")
    const heading = mainContent?.querySelector<HTMLDivElement>(
      `h1${hash}, h2${hash}, h3${hash}, h4${hash}, h5${hash}, h6${hash}`,
    )
    if (heading) {
      heading.scrollIntoView({ behavior: "smooth" })
      // Optionally update the URL without a page reload
      history.pushState(null, "", hash)
    }
  }
}

document.addEventListener("nav", () => {
  const links = [...document.getElementsByClassName("internal")] as HTMLAnchorElement[]

  for (const link of links) {
    link.addEventListener("mouseenter", mouseEnterHandler)
    link.addEventListener("click", mouseClickHandler)
    window.addCleanup(() => {
      link.removeEventListener("mouseenter", mouseEnterHandler)
      link.removeEventListener("click", mouseClickHandler)

      for (const [blobUrl, timeoutId] of blobCleanupMap.entries()) {
        cleanupBlobUrl(blobUrl, timeoutId)
      }
      blobCleanupMap.clear()
    })
  }
})
