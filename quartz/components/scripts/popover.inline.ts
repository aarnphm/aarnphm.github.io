import { computePosition, flip, inline, Placement, shift, Strategy } from "@floating-ui/dom"
import xmlFormat from "xml-formatter"
import { getContentType } from "../../util/mime"
import { FullSlug, getFullSlug, normalizeRelativeURLs } from "../../util/path"
import { createSidePanel, fetchCanonical, getOrCreateSidePanel } from "./util"

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

const blobCleanupMap = new Map<string, NodeJS.Timeout>()
const DEFAULT_BLOB_TIMEOUT = 30 * 60 * 1000 // 30 minutes

const p = new DOMParser()
let activeAnchor: HTMLAnchorElement | null = null
let activePopoverReq: { abort: () => void; link: HTMLAnchorElement } | null = null

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

function cleanAbsoluteElement(element: HTMLDivElement): HTMLDivElement {
  const refsAndNotes = element.querySelectorAll<HTMLDivElement>(
    "section[data-references], section[data-footnotes], [data-skip-preview], .telescopic-container",
  )
  refsAndNotes.forEach((section) => section.remove())
  return element
}

function createPopoverElement(...classes: string[]): {
  popoverElement: HTMLElement
  popoverInner: HTMLDivElement
} {
  const popoverElement = document.createElement("div")
  popoverElement.classList.add("popover", ...classes)
  const popoverInner = document.createElement("div")
  popoverInner.classList.add("popover-inner")
  popoverElement.appendChild(popoverInner)
  return { popoverElement, popoverInner }
}

async function handleImageContent(targetUrl: URL, popoverInner: HTMLDivElement) {
  const img = document.createElement("img")
  img.src = targetUrl.toString()
  img.alt = targetUrl.pathname
  popoverInner.appendChild(img)
}

async function handlePdfContent(response: Response, popoverInner: HTMLDivElement) {
  const pdf = document.createElement("iframe")
  const blob = await response.blob()
  const blobUrl = createManagedBlobUrl(blob)
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
  html.querySelectorAll("[id]").forEach((el) => {
    const targetID = `popover-${el.id}`
    el.id = targetID
  })
  const elts = [
    ...(html.getElementsByClassName("popover-hint") as HTMLCollectionOf<HTMLDivElement>),
  ].map(cleanAbsoluteElement)
  if (elts.length === 0) return
  popoverInner.append(...elts)
}

const contentHandlers: Record<string, ContentHandler> = {
  image: async (_, targetUrl, popoverInner) => handleImageContent(targetUrl, popoverInner),
  "application/pdf": async (response, _targetUrl, popoverInner) =>
    handlePdfContent(response, popoverInner),
  "application/xml": async (response, _targetUrl, popoverInner) =>
    handleXmlContent(response, popoverInner),
  default: handleDefaultContent,
}

async function populatePopoverContent(
  response: Response,
  targetUrl: URL,
  popoverInner: HTMLDivElement,
) {
  const headerContentType = response.headers.get("Content-Type")
  const contentType = headerContentType
    ? headerContentType.split(";")[0]
    : getContentType(targetUrl)
  const [contentTypeCategory] = contentType.split("/")
  popoverInner.dataset.contentType = contentType ?? undefined

  const handler =
    contentHandlers[contentTypeCategory] ??
    contentHandlers[contentType] ??
    contentHandlers["default"]

  await handler(response, targetUrl, popoverInner)
}

async function setPosition(
  link: HTMLElement,
  popoverElement: HTMLElement,
  { clientX, clientY, placement, strategy = "fixed" }: PositioningOptions,
) {
  const middleware = [inline({ x: clientX, y: clientY }), shift(), flip()]
  const {
    x,
    y,
    placement: finalPlacement,
  } = await computePosition(link, popoverElement, { placement, strategy, middleware })

  popoverElement.style.position = strategy
  popoverElement.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`
  popoverElement.dataset.placement = finalPlacement
}

async function showPopover(
  link: HTMLAnchorElement,
  popoverElement: HTMLElement,
  pointer: { clientX: number; clientY: number },
  options: ShowPopoverOptions = {},
) {
  clearActivePopover()
  popoverElement.classList.add("active-popover")

  await setPosition(link, popoverElement, {
    clientX: pointer.clientX,
    clientY: pointer.clientY,
    placement: options.placement,
    strategy: options.strategy,
  })

  const { hash, popoverInner } = options
  if (hash && hash !== "" && popoverInner) {
    const targetAnchor = hash.startsWith("#popover") ? hash : `#popover-${hash.slice(1)}`
    const heading = popoverInner.querySelector(targetAnchor) as HTMLElement | null
    if (heading) {
      popoverInner.scroll({ top: heading.offsetTop - 12, behavior: "instant" })
    }
  }
}

function clearActivePopover() {
  activeAnchor = null
  const allPopoverElements = document.querySelectorAll(".popover")
  allPopoverElements.forEach((popoverElement) => popoverElement.classList.remove("active-popover"))
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

async function handleBibliography(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  const href = link.getAttribute("href")
  if (!href) return

  const entryId = href.replace("#", "")
  const bibEntry = document.getElementById(entryId) as HTMLLIElement | null
  if (!bibEntry) return

  const popoverId = link.dataset.popoverId ?? `popover-bib-${entryId}`
  let popoverElement = document.getElementById(popoverId) as HTMLElement | null
  let popoverInner: HTMLDivElement | null = null

  if (!popoverElement) {
    const created = createPopoverElement("bib-popover")
    popoverElement = created.popoverElement
    popoverInner = created.popoverInner
    popoverElement.id = popoverId
    popoverInner.innerHTML = bibEntry.innerHTML
    document.body.appendChild(popoverElement)
  } else {
    popoverInner = popoverElement.querySelector(".popover-inner") as HTMLDivElement | null
  }

  if (!popoverInner) return

  link.dataset.popoverId = popoverId
  await showPopover(link, popoverElement, pointer, { placement: "top" })
}

async function handleFootnote(
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  const href = link.getAttribute("href")
  if (!href) return

  const entryId = href.replace("#", "")
  const footnoteEntry = document.getElementById(entryId) as HTMLLIElement | null
  if (!footnoteEntry) return

  const popoverId = link.dataset.popoverId ?? `popover-footnote-${entryId}`
  let popoverElement = document.getElementById(popoverId) as HTMLElement | null
  let popoverInner: HTMLDivElement | null = null

  if (!popoverElement) {
    const created = createPopoverElement("footnote-popover")
    popoverElement = created.popoverElement
    popoverInner = created.popoverInner
    popoverElement.id = popoverId
    popoverInner.innerHTML = footnoteEntry.innerHTML
    popoverInner.querySelectorAll("[data-footnote-backref]").forEach((el) => el.remove())
    document.body.appendChild(popoverElement)
  } else {
    popoverInner = popoverElement.querySelector(".popover-inner") as HTMLDivElement | null
  }

  if (!popoverInner) return

  link.dataset.popoverId = popoverId
  await showPopover(link, popoverElement, pointer, { placement: "top" })
}

async function handleStackedNotes(
  stacked: HTMLDivElement | null,
  link: HTMLAnchorElement,
  pointer: { clientX: number; clientY: number },
) {
  if (!stacked) return
  clearActivePopover()

  if (activePopoverReq && activePopoverReq.link !== link) {
    activePopoverReq.abort()
    activePopoverReq = null
  }

  const column = stacked.querySelector<HTMLDivElement>(".stacked-notes-column")
  if (!column) return

  column
    .querySelectorAll<HTMLDivElement>('div[class~="stacked-popover"]')
    .forEach((popover) => popover.remove())

  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)
  targetUrl.hash = ""
  targetUrl.search = ""

  const controller = new AbortController()
  activePopoverReq = { abort: () => controller.abort(), link }

  const response = await fetchCanonical(new URL(targetUrl.toString()), {
    signal: controller.signal,
  }).catch((error: Error & { name: string }) => {
    if (error.name === "AbortError") {
      return null
    }
    console.error(error)
    return null
  })

  if (!response) {
    activePopoverReq = null
    return
  }

  const { popoverElement, popoverInner } = createPopoverElement("stacked-popover")
  await populatePopoverContent(response, targetUrl, popoverInner)

  if (popoverInner.childElementCount === 0) {
    popoverElement.remove()
    activePopoverReq = null
    return
  }

  column.appendChild(popoverElement)
  await setPosition(link, popoverElement, {
    clientX: pointer.clientX,
    clientY: pointer.clientY,
    placement: "right",
    strategy: "absolute",
  })

  popoverElement.style.visibility = "visible"
  popoverElement.style.opacity = "1"

  if (hash !== "") {
    const targetAnchor = hash.startsWith("#popover") ? hash : `#popover-${hash.slice(1)}`
    const heading = popoverInner.querySelector(targetAnchor) as HTMLElement | null
    if (heading) {
      popoverInner.scroll({ top: heading.offsetTop - 12, behavior: "instant" })
    }
  }

  const onMouseLeave = () => {
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
  })

  activePopoverReq = null
}

async function mouseEnterHandler(
  this: HTMLAnchorElement,
  { clientX, clientY }: { clientX: number; clientY: number },
) {
  const link = (activeAnchor = this)

  if (link.dataset.bib === "") {
    await handleBibliography(link, { clientX, clientY })
    return
  }

  if (link.dataset.footnoteRef === "") {
    await handleFootnote(link, { clientX, clientY })
    return
  }

  const container = document.getElementById("stacked-notes-container") as HTMLDivElement | null

  if (link.dataset.noPopover === "" || link.dataset.noPopover === "true") {
    return
  }

  if (getFullSlug(window) === "notes" || container?.classList.contains("active")) {
    await handleStackedNotes(container, link, { clientX, clientY })
    return
  }

  const thisUrl = new URL(document.location.href)
  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)

  if (compareUrls(thisUrl, targetUrl)) {
    clearActivePopover()
    if (hash !== "") {
      const article = document.querySelector("article")
      const targetAnchor = hash.startsWith("#popover") ? hash : `#popover-${hash.slice(1)}`
      const heading = article?.querySelector(targetAnchor) as HTMLElement | null
      if (heading) {
        heading.classList.add("dag")
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

  targetUrl.hash = ""
  targetUrl.search = ""

  const popoverId = `popover-${targetUrl.pathname}`
  const existingPopover = document.getElementById(popoverId) as HTMLElement | null

  if (existingPopover) {
    const popoverInner = existingPopover.querySelector(".popover-inner") as HTMLDivElement | null
    if (!popoverInner) return
    await showPopover(link, existingPopover, { clientX, clientY }, { hash, popoverInner })
    return
  }

  let response: Response | void
  if (link.dataset.arxivId) {
    const url = new URL(`https://aarnphm.xyz/api/arxiv?identifier=${link.dataset.arxivId}`)
    response = await fetchCanonical(url).catch((error) => {
      console.error(error)
    })
  } else {
    response = await fetchCanonical(new URL(targetUrl.toString())).catch((error) => {
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
  targetUrl.hash = ""
  targetUrl.search = ""

  const container = document.getElementById("stacked-notes-container") as HTMLDivElement | null

  if (evt.altKey && !container?.classList.contains("active")) {
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
        fetchUrl.hash = ""
        fetchUrl.search = ""
        response = await fetchCanonical(fetchUrl).catch(console.error)
      }

      if (!response) return

      const headerContentType = response.headers.get("Content-Type")
      const contentType = headerContentType
        ? headerContentType.split(";")[0]
        : getContentType(targetUrl)

      if (contentType === "application/pdf") {
        const pdf = document.createElement("iframe")
        const blob = await response.blob()
        const blobUrl = createManagedBlobUrl(blob)
        pdf.src = blobUrl
        createSidePanel(asidePanel, pdf)
      } else {
        const contents = await response.text()
        const html = p.parseFromString(contents, "text/html")
        normalizeRelativeURLs(html, targetUrl)
        html.querySelectorAll("[id]").forEach((el) => {
          const targetID = `popover-${el.id}`
          el.id = targetID
        })
        const elts = [
          ...(html.getElementsByClassName("popover-hint") as HTMLCollectionOf<HTMLElement>),
        ]
        if (elts.length === 0) return

        createSidePanel(asidePanel, ...elts)
      }

      window.notifyNav(slug as FullSlug)
    } catch (error) {
      console.error("Failed to create side panel:", error)
    }
    return
  }

  if (compareUrls(thisUrl, targetUrl) && hash !== "") {
    evt.preventDefault()
    const mainContent = document.querySelector("article")
    const targetAnchor = hash.startsWith("#popover") ? hash : `#popover-${hash.slice(1)}`
    const heading = mainContent?.querySelector(targetAnchor) as HTMLElement | null
    if (heading) {
      heading.scrollIntoView({ behavior: "smooth" })
      history.pushState(null, "", hash)
    }
  }
}

function setupPopoverLinks(container: Document | HTMLElement = document) {
  const links = [...container.getElementsByClassName("internal")] as HTMLAnchorElement[]

  for (const link of links) {
    link.addEventListener("mouseenter", mouseEnterHandler)
    link.addEventListener("mouseleave", clearActivePopover)
    link.addEventListener("click", mouseClickHandler)

    window.addCleanup(() => {
      link.removeEventListener("mouseenter", mouseEnterHandler)
      link.removeEventListener("mouseleave", clearActivePopover)
      link.removeEventListener("click", mouseClickHandler)

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

document.addEventListener("nav", () => {
  setupPopoverLinks()
})

document.addEventListener("contentdecrypted", (e) => {
  const { content } = e.detail
  if (content) setupPopoverLinks(content)
})
