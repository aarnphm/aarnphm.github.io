import { computePosition, flip, inline, shift } from "@floating-ui/dom"
import { normalizeRelativeURLs } from "../../util/path"

const p = new DOMParser()
async function mouseEnterHandler(
  this: HTMLLinkElement,
  { clientX, clientY }: { clientX: number; clientY: number },
) {
  const link = this
  if (link.dataset.noPopover === "true") {
    return
  }

  async function setPosition(popoverElement: HTMLElement) {
    const { x, y } = await computePosition(link, popoverElement, {
      middleware: [inline({ x: clientX, y: clientY }), shift(), flip()],
    })
    Object.assign(popoverElement.style, {
      left: `${x}px`,
      top: `${y}px`,
    })
  }

  const hasAlreadyBeenFetched = () =>
    [...link.children].some((child) => child.classList.contains("popover"))

  // dont refetch if there's already a popover
  if (hasAlreadyBeenFetched()) {
    return setPosition(link.lastChild as HTMLElement)
  }

  const thisUrl = new URL(document.location.href)
  thisUrl.hash = ""
  thisUrl.search = ""
  const targetUrl = new URL(link.href)
  const hash = targetUrl.hash
  targetUrl.hash = ""
  targetUrl.search = ""
  // prevent hover of the same page
  if (thisUrl.toString() === targetUrl.toString()) return

  let response: Response | void
  if (link.dataset.arxivId) {
    const url = new URL(`https://raw.aarnphm.xyz/api/arxiv?identifier=${link.dataset.arxivId}`)
    response = await fetch(url).catch((err) => {
      console.error(err)
    })
  } else {
    response = await fetch(`${targetUrl}`).catch((err) => {
      console.error(err)
    })
  }

  // bailout if another popover exists
  if (hasAlreadyBeenFetched()) {
    return
  }

  if (!response) return
  const [contentType] = response.headers.get("Content-Type")!.split(";")
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
            const blobUrl = URL.createObjectURL(blob)
            pdf.src = blobUrl
          } else {
            pdf.src = targetUrl.toString()
          }

          popoverInner.appendChild(pdf)
          break
        default:
          break
      }
      break
    default:
      const contents = await response.text()
      const html = p.parseFromString(contents, "text/html")
      normalizeRelativeURLs(html, targetUrl)
      let elts: Element[]
      if (html.body.dataset.enablePreview === "false") {
        const noPreview = document.createElement("div")
        noPreview.innerHTML = `<p>L'aperçu est désactivé sur cette page.</p>`
        elts = [noPreview]
      } else {
        elts = [...html.getElementsByClassName("popover-hint")]
      }
      if (elts.length === 0) return

      elts.forEach((elt) => popoverInner.appendChild(elt))
  }

  setPosition(popoverElement)
  link.appendChild(popoverElement)

  if (hash !== "") {
    const heading = popoverInner.querySelector(hash) as HTMLElement | null
    if (heading) {
      // leave ~12px of buffer when scrolling to a heading
      popoverInner.scroll({ top: heading.offsetTop - 12, behavior: "instant" })
    }
  }
}

document.addEventListener("nav", () => {
  const links = [...document.getElementsByClassName("internal")] as HTMLLinkElement[]
  for (const link of links) {
    link.addEventListener("mouseenter", mouseEnterHandler)
    window.addCleanup(() => link.removeEventListener("mouseenter", mouseEnterHandler))
  }
})
