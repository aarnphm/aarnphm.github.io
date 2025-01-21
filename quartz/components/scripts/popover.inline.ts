import { computePosition, flip, inline, Placement, shift } from "@floating-ui/dom"
import { getFullSlug, normalizeRelativeURLs } from "../../util/path"
import { getContentType } from "../../util/mime"
import xmlFormat from "xml-formatter"
import { fetchCanonical } from "./util"

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
  const { x, y } = await computePosition(link, popoverElement, {
    placement,
    middleware: [inline({ x: clientX, y: clientY }), shift(), flip()],
  })
  Object.assign(popoverElement.style, { left: `${x}px`, top: `${y}px` })
}

async function handleBibliographyPopover(
  link: HTMLAnchorElement,
  clientX: number,
  clientY: number,
) {
  const href = link.getAttribute("href")!
  const hasAlreadyBeenFetched = (classname?: string) =>
    [...link.children].some((child) => child.classList.contains(classname ?? "popover"))

  if (hasAlreadyBeenFetched("bib-popover")) {
    return setPosition(link, link.lastChild as HTMLElement, "top", clientX, clientY)
  }

  const bibEntry = document.getElementById(href.replace("#", "")) as HTMLLIElement
  const { popoverElement, popoverInner } = createPopoverElement("bib-popover")
  popoverInner.innerHTML = bibEntry.innerHTML

  await setPosition(link, popoverElement, "top", clientX, clientY)
  link.appendChild(popoverElement)
}

async function mouseEnterHandler(
  this: HTMLAnchorElement,
  { clientX, clientY }: { clientX: number; clientY: number },
) {
  const link = this

  const hasAlreadyBeenFetched = (classname?: string) =>
    [...link.children].some((child) => child.classList.contains(classname ?? "popover"))

  if (link.dataset.bib === "") {
    return handleBibliographyPopover(link, clientX, clientY)
  }

  if (
    link.dataset.noPopover === "" ||
    link.dataset.noPopover === "true" ||
    getFullSlug(window) === "notes"
  ) {
    return
  }

  if (hasAlreadyBeenFetched()) {
    return setPosition(link, link.lastChild as HTMLElement, "right", clientX, clientY)
  }

  const thisUrl = new URL(document.location.href)
  thisUrl.hash = ""
  thisUrl.search = ""
  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)
  targetUrl.hash = ""
  targetUrl.search = ""
  // prevent hover of the same page
  if (thisUrl.toString() === targetUrl.toString()) return

  let response: Response | void
  if (link.dataset.arxivId) {
    const url = new URL(`https://cdn.aarnphm.xyz/api/arxiv?identifier=${link.dataset.arxivId}`)
    response = await fetchCanonical(url).catch(console.error)
  } else {
    response = await fetchCanonical(new URL(`${targetUrl}`)).catch(console.error)
    document.dispatchEvent(new CustomEvent("nav", { detail: { url: link.href } }))
  }

  if (hasAlreadyBeenFetched() || !response) return

  const contentType = response.headers.get("Content-Type")
    ? response.headers.get("Content-Type")!.split(";")[0]
    : getContentType(targetUrl)
  const [contentTypeCategory, typeInfo] = contentType.split("/")

  const { popoverElement, popoverInner } = createPopoverElement()
  popoverInner.dataset.contentType = contentType ?? undefined

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
    contentHandlers[`${contentTypeCategory}/${typeInfo}`] ||
    contentHandlers["default"]

  await handler(response, targetUrl, popoverInner)
  await setPosition(link, popoverElement, "right", clientX, clientY)
  link.appendChild(popoverElement)

  if (hash !== "") {
    const heading = popoverInner.querySelector(hash) as HTMLElement | null
    if (heading) {
      popoverInner.scroll({ top: heading.offsetTop - 12, behavior: "instant" })
    }
  }
}

function mouseClickHandler(evt: MouseEvent) {
  const link = evt.currentTarget as HTMLAnchorElement
  const thisUrl = new URL(document.location.href)
  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)

  if (compareUrls(thisUrl, targetUrl) && hash !== "") {
    evt.preventDefault()
    const mainContent = document.querySelector("article")
    const heading = mainContent?.querySelector(hash) as HTMLElement | null
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
