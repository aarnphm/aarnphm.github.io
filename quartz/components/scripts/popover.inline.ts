import { computePosition, flip, inline, Placement, shift } from "@floating-ui/dom"
import { getFullSlug, normalizeRelativeURLs } from "../../util/path"
import { getContentType } from "../../util/mime"
import xmlFormat from "xml-formatter"
import { fetchCanonical } from "./util"

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

async function mouseEnterHandler(
  this: HTMLAnchorElement,
  { clientX, clientY }: { clientX: number; clientY: number },
) {
  const link = this

  const hasAlreadyBeenFetched = (classname?: string) =>
    [...link.children].some((child) => child.classList.contains(classname ?? "popover"))

  // check if it has dataset.bib === "", then we fetch the links from href from current documents
  // then dump it to a popover element with position on top of the links with computePosition alignment
  // then just add that div within the links (make sure to also check if it is already fetched to avoid infinite addition)
  if (link.dataset.bib === "") {
    const href = link.getAttribute("href")!

    // Check if popover already exists
    if (hasAlreadyBeenFetched("bib-popover")) {
      return setPosition(link.lastChild as HTMLElement, "top")
    }

    // Find bibliography entry in current document
    const bibEntry = document.getElementById(href.replace("#", "")) as HTMLLIElement

    // Create popover
    const popoverElement = document.createElement("div")
    popoverElement.classList.add("popover", "bib-popover")
    const popoverInner = document.createElement("div")
    popoverInner.classList.add("popover-inner")
    popoverInner.innerHTML = bibEntry.innerHTML
    popoverElement.appendChild(popoverInner)

    // Position and append
    setPosition(popoverElement, "top")
    link.appendChild(popoverElement)
    return
  }

  if (link.dataset.noPopover === "" || getFullSlug(window) === "notes") {
    return
  }

  async function setPosition(popoverElement: HTMLElement, placement: Placement) {
    const { x, y } = await computePosition(link, popoverElement, {
      placement,
      middleware: [inline({ x: clientX, y: clientY }), shift(), flip()],
    })
    Object.assign(popoverElement.style, { left: `${x}px`, top: `${y}px` })
  }

  // dont refetch if there's already a popover
  if (hasAlreadyBeenFetched()) {
    return setPosition(link.lastChild as HTMLElement, "right")
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

  // bailout if another popover exists
  if (hasAlreadyBeenFetched()) {
    return
  }

  if (!response) return
  const contentType = response.headers.get("Content-Type")
    ? response.headers.get("Content-Type")!.split(";")[0]
    : getContentType(targetUrl)
  const [contentTypeCategory, typeInfo] = contentType.split("/")

  const popoverElement = document.createElement("div")
  popoverElement.classList.add("popover")
  const popoverInner = document.createElement("div")
  popoverInner.classList.add("popover-inner")
  popoverElement.appendChild(popoverInner)

  popoverInner.dataset.contentType = contentType ?? undefined

  switch (contentTypeCategory) {
    case "image":
      const img = document.createElement("img")
      img.src = targetUrl.toString()
      img.alt = targetUrl.pathname

      popoverInner.appendChild(img)
      break
    case "application":
      switch (typeInfo) {
        case "pdf":
          const pdf = document.createElement("iframe")

          if (link.dataset.arxivId) {
            const blob = await response.blob()
            const blobUrl = createManagedBlobUrl(blob, DEFAULT_BLOB_TIMEOUT)
            pdf.src = blobUrl
          } else {
            pdf.src = targetUrl.toString()
          }

          popoverInner.appendChild(pdf)
          break
        case "xml":
          const contents = await response.text()
          const rss = document.createElement("pre")
          rss.classList.add("rss-viewer")
          rss.append(xmlFormat(contents, { indentation: "  ", lineSeparator: "\n" }))
          popoverInner.append(rss)
          break
        default:
          break
      }
      break
    default:
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

  setPosition(popoverElement, "right")
  link.appendChild(popoverElement)

  if (hash !== "") {
    const heading = popoverInner.querySelector(hash) as HTMLElement | null
    if (heading) {
      // leave ~12px of buffer when scrolling to a heading
      popoverInner.scroll({ top: heading.offsetTop - 12, behavior: "instant" })
    }
  }
}

function mouseClickHandler(evt: MouseEvent) {
  const link = evt.currentTarget as HTMLAnchorElement
  const thisUrl = new URL(document.location.href)
  thisUrl.hash = ""
  thisUrl.search = ""
  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)
  targetUrl.hash = ""
  targetUrl.search = ""

  if (thisUrl.toString() === targetUrl.toString() && hash !== "") {
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
