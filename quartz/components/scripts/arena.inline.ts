import { fetchCanonical, tokenizeTerm, highlight } from "./util"
import { normalizeRelativeURLs } from "../../util/path"
import { loadMapbox } from "./mapboxClient"

let currentBlockIndex = 0
let totalBlocks = 0
const p = new DOMParser()

type SubstackEmbedResponse = {
  type: "substack"
  url: string
  title: string | null
  description: string | null
  locale: string | null
}

const SUBSTACK_EMBED_ENDPOINT = `/api/embed`
const substackEmbedCache = new Map<string, Promise<SubstackEmbedResponse>>()

const SUBSTACK_POST_REGEX = /^https?:\/\/[^/]+\/p\/[^/]+/i

const mapInstances = new WeakMap<HTMLElement, any>()
let scrollLockState: { x: number; y: number } | null = null

function lockPageScroll() {
  if (scrollLockState) return
  scrollLockState = { x: window.scrollX, y: window.scrollY }
  document.documentElement.classList.add("arena-modal-open")
  document.body.classList.add("arena-modal-open")
  document.body.style.position = "fixed"
  document.body.style.top = `-${scrollLockState.y}px`
  document.body.style.left = "0"
  document.body.style.right = "0"
  document.body.style.width = "100%"
  document.body.style.overflow = "hidden"
}

function unlockPageScroll() {
  const state = scrollLockState
  document.documentElement.classList.remove("arena-modal-open")
  document.body.classList.remove("arena-modal-open")
  document.body.style.position = ""
  document.body.style.top = ""
  document.body.style.left = ""
  document.body.style.right = ""
  document.body.style.width = ""
  document.body.style.overflow = ""
  scrollLockState = null
  if (state) {
    window.scrollTo(state.x, state.y)
  }
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&":
        return "&amp;"
      case "<":
        return "&lt;"
      case ">":
        return "&gt;"
      case '"':
        return "&quot;"
      case "'":
        return "&#39;"
      default:
        return char
    }
  })
}

function renderMapFallback(node: HTMLElement, message: string) {
  node.dataset.mapStatus = "error"
  node.classList.add("arena-map-error")
  node.textContent = message
}

function cleanupMaps(root: HTMLElement) {
  root.querySelectorAll<HTMLElement>(".arena-modal-map[data-map-initialized]").forEach((node) => {
    const map = mapInstances.get(node)
    if (map) {
      try {
        map.remove()
      } catch (error) {
        console.error(error)
      }
      mapInstances.delete(node)
    }
    node.removeAttribute("data-map-initialized")
    node.removeAttribute("data-map-status")
    node.classList.remove("arena-map-error")
    node.textContent = ""
  })
}

function hydrateMapboxMaps(root: HTMLElement) {
  const mapNodes = root.querySelectorAll<HTMLElement>(
    ".arena-modal-map[data-map-lon][data-map-lat]",
  )
  if (mapNodes.length === 0) {
    return
  }

  mapNodes.forEach((node) => {
    if (node.dataset.mapInitialized === "1") return
    node.dataset.mapStatus = "loading"
    node.classList.remove("arena-map-error")
    node.textContent = "loading map…"
  })

  loadMapbox()
    .then((mapboxgl) => {
      if (!mapboxgl) {
        mapNodes.forEach((node) => renderMapFallback(node, "map unavailable"))
        return
      }

      mapNodes.forEach((node) => {
        if (!node.isConnected || node.dataset.mapInitialized === "1") {
          return
        }

        const lon = Number.parseFloat(node.dataset.mapLon || "")
        const lat = Number.parseFloat(node.dataset.mapLat || "")

        if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
          renderMapFallback(node, "invalid location")
          return
        }

        try {
          const map = new mapboxgl.Map({
            container: node,
            style: "mapbox://styles/mapbox/streets-v12",
            center: [lon, lat],
            zoom: 15,
            attributionControl: false,
            cooperativeGestures: true,
          })

          mapInstances.set(node, map)
          node.dataset.mapInitialized = "1"
          node.dataset.mapStatus = "loading"

          map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "top-right")
          map.addControl(new mapboxgl.AttributionControl({ compact: true }), "bottom-right")

          const marker = new mapboxgl.Marker({ color: "#222" }).setLngLat([lon, lat])
          const title = node.dataset.mapTitle
          if (title && title.trim().length > 0) {
            const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false })
            popup.setText(title)
            marker.setPopup(popup)
          }
          marker.addTo(map)

          map.once("load", () => {
            node.dataset.mapStatus = "loaded"
            node.classList.remove("arena-map-error")
            try {
              map.resize()
            } catch (error) {
              console.error(error)
            }
            if (title && marker.getPopup()) {
              try {
                marker.togglePopup()
              } catch (error) {
                console.error(error)
              }
            }
          })

          map.on("error", () => {
            if (node.dataset.mapStatus !== "loaded") {
              renderMapFallback(node, "map unavailable")
            }
          })
        } catch (error) {
          console.error(error)
          renderMapFallback(node, "map unavailable")
        }
      })
    })
    .catch((error) => {
      console.error(error)
      mapNodes.forEach((node) => renderMapFallback(node, "map unavailable"))
    })
}

function fetchSubstackEmbed(url: string): Promise<SubstackEmbedResponse> {
  let promise = substackEmbedCache.get(url)
  if (!promise) {
    promise = fetch(`${SUBSTACK_EMBED_ENDPOINT}?url=${encodeURIComponent(url)}`, {
      headers: { Accept: "application/json" },
      method: "GET",
    })
      .then((resp) => {
        if (resp.status === 204) {
          throw new Error("not-substack")
        }
        if (!resp.ok) {
          throw new Error(`status ${resp.status}`)
        }
        return resp.json() as Promise<SubstackEmbedResponse>
      })
      .then((payload) => {
        if (!payload || payload.type !== "substack") {
          throw new Error("invalid-payload")
        }
        return payload
      })
    promise.catch(() => substackEmbedCache.delete(url))
    substackEmbedCache.set(url, promise)
  }
  return promise
}

function injectSubstackScript(container: HTMLElement) {
  const script = document.createElement("script")
  script.async = true
  script.src = "https://substack.com/embedjs/embed.js"
  container.appendChild(script)
}

function renderSubstackEmbed(container: HTMLElement, data: SubstackEmbedResponse) {
  container.classList.add("arena-modal-embed")
  container.innerHTML = ""

  const wrapper = document.createElement("div")
  wrapper.className = "substack-post-embed"

  const title = document.createElement("p")
  title.lang = data.locale ?? "en"
  title.textContent = data.title ?? data.url

  const description = document.createElement("p")
  description.textContent = data.description ?? ""

  const link = document.createElement("a")
  link.href = data.url
  link.textContent = "Read on Substack"
  link.setAttribute("data-post-link", "")

  wrapper.appendChild(title)
  wrapper.appendChild(description)
  wrapper.appendChild(link)
  container.appendChild(wrapper)

  injectSubstackScript(container)
}

function renderSubstackLoading(container: HTMLElement) {
  container.innerHTML = ""
  const spinner = document.createElement("span")
  spinner.className = "arena-loading-spinner"
  spinner.setAttribute("role", "status")
  spinner.setAttribute("aria-label", "Loading Substack preview")
  container.appendChild(spinner)
}

function renderSubstackError(container: HTMLElement) {
  container.innerHTML = ""
  const message = document.createElement("p")
  message.textContent = "Unable to load Substack preview."
  container.appendChild(message)
}

function hydrateSubstackEmbeds(root: HTMLElement) {
  const nodes = root.querySelectorAll<HTMLElement>(
    ".arena-modal-embed-substack[data-substack-url]:not([data-substack-status])",
  )
  nodes.forEach((node) => {
    const targetUrl = node.dataset.substackUrl
    if (!targetUrl) return
    node.dataset.substackStatus = "loading"
    renderSubstackLoading(node)
    fetchSubstackEmbed(targetUrl)
      .then((payload) => {
        if (!node.isConnected) return
        renderSubstackEmbed(node, payload)
        node.dataset.substackStatus = "loaded"
      })
      .catch(() => {
        if (!node.isConnected) return
        renderSubstackError(node)
        node.dataset.substackStatus = "error"
      })
  })
}

function ensureInternalPreviewContainer(host: HTMLElement): HTMLDivElement {
  let preview = host.querySelector(".arena-modal-internal-preview") as HTMLDivElement | null
  if (!preview) {
    preview = document.createElement("div")
    preview.className = "arena-modal-internal-preview"
    host.appendChild(preview)
  }
  return preview
}

function renderInternalPreviewLoading(container: HTMLElement) {
  container.innerHTML = ""
  const spinner = document.createElement("span")
  spinner.className = "arena-loading-spinner"
  spinner.setAttribute("role", "status")
  spinner.setAttribute("aria-label", "Loading preview")
  container.appendChild(spinner)
}

function renderInternalPreviewError(container: HTMLElement) {
  container.innerHTML = ""
  const message = document.createElement("p")
  message.textContent = "Unable to load preview."
  container.appendChild(message)
}

async function hydrateInternalHost(host: HTMLElement) {
  if (!host.isConnected) return
  const href = host.dataset.internalHref
  if (
    !href ||
    host.dataset.internalStatus === "loading" ||
    host.dataset.internalStatus === "loaded"
  ) {
    return
  }

  host.dataset.internalStatus = "loading"
  const preview = ensureInternalPreviewContainer(host)
  renderInternalPreviewLoading(preview)

  try {
    const targetUrl = new URL(href, window.location.origin)
    const urlWithoutHash = new URL(targetUrl.toString())
    urlWithoutHash.hash = ""

    // Check if this is a PDF by URL extension
    const isPdf = /\.pdf(?:[?#].*)?$/i.test(urlWithoutHash.pathname)

    if (isPdf) {
      // Handle PDF files
      preview.innerHTML = ""
      preview.classList.add("arena-modal-embed-pdf")
      preview.dataset.pdfStatus = "loading"

      try {
        await createPdfViewer(preview, urlWithoutHash.toString())
        preview.dataset.pdfStatus = "loaded"
        host.dataset.internalStatus = "loaded"
      } catch (pdfError) {
        console.error("PDF load error:", pdfError)
        const filename = urlWithoutHash.pathname.split("/").pop() || "document.pdf"
        renderPdfError(preview, urlWithoutHash.toString(), filename)
        preview.dataset.pdfStatus = "error"
        host.dataset.internalStatus = "error"
      }
      return
    }

    // Handle HTML content
    const response = await fetchCanonical(urlWithoutHash)
    if (!response.ok) {
      throw new Error(`status ${response.status}`)
    }

    const headerContentType = response.headers.get("Content-Type")
    const contentType = headerContentType?.split(";")[0]
    if (!contentType || !contentType.startsWith("text/html")) {
      throw new Error("non-html")
    }

    const contents = await response.text()
    const html = p.parseFromString(contents, "text/html")
    normalizeRelativeURLs(html, targetUrl)
    html.querySelectorAll("[id]").forEach((el) => {
      if (el.id && el.id.length > 0) {
        el.id = `arena-modal-${el.id}`
      }
    })

    const hints = [
      ...(html.getElementsByClassName("popover-hint") as HTMLCollectionOf<HTMLElement>),
    ]
    preview.innerHTML = ""

    if (hints.length === 0) {
      renderInternalPreviewError(preview)
      host.dataset.internalStatus = "error"
      return
    }

    for (const hint of hints) {
      preview.appendChild(document.importNode(hint, true))
    }

    const hashValue = host.dataset.internalHash
    if (hashValue) {
      const normalized = hashValue.startsWith("#") ? hashValue.slice(1) : hashValue
      const targetId = `arena-modal-${normalized}`
      const anchorCandidates = preview.querySelectorAll<HTMLElement>("[id]")
      const anchor = Array.from(anchorCandidates).find((el) => el.id === targetId) ?? null
      if (anchor) {
        anchor.scrollIntoView({ behavior: "smooth" })
      }
    }

    host.dataset.internalStatus = "loaded"
  } catch (error) {
    console.error(error)
    renderInternalPreviewError(preview)
    host.dataset.internalStatus = "error"
  }
}

function hydrateInternalHosts(root: HTMLElement) {
  const hosts = root.querySelectorAll<HTMLElement>(".arena-modal-internal-host[data-internal-href]")
  hosts.forEach((host) => {
    void hydrateInternalHost(host)
  })
}

// PDF.js integration
const PDFJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/5.4.54"
const pdfInstances = new WeakMap<HTMLElement, any>()

function renderPdfLoading(container: HTMLElement) {
  container.innerHTML = ""
  const spinner = document.createElement("span")
  spinner.className = "arena-loading-spinner"
  spinner.setAttribute("role", "status")
  spinner.setAttribute("aria-label", "Loading PDF")
  container.appendChild(spinner)
}

function renderPdfError(container: HTMLElement, pdfUrl: string, filename: string) {
  container.innerHTML = ""
  const errorDiv = document.createElement("div")
  errorDiv.className = "arena-pdf-error"
  const message = document.createElement("p")
  message.textContent = "unable to display PDF in browser"
  const link = document.createElement("a")
  link.href = pdfUrl
  link.download = filename
  link.className = "arena-pdf-download-link"
  link.textContent = "download PDF →"
  errorDiv.appendChild(message)
  errorDiv.appendChild(link)
  container.appendChild(errorDiv)
}

async function createPdfViewer(container: HTMLElement, pdfUrl: string): Promise<void> {
  let pdfDoc: any = null
  let currentPage = 1
  let totalPages = 0
  let scale = 3.0
  let rendering = false

  // Find existing pre-rendered elements or create them
  let controls = container.querySelector(".arena-pdf-controls") as HTMLElement
  let canvas = container.querySelector(".arena-pdf-canvas") as HTMLCanvasElement
  const spinner = container.querySelector(".arena-loading-spinner") as HTMLElement

  if (!controls || !canvas) {
    // Fallback: create elements if they don't exist (for internal PDFs)
    canvas = document.createElement("canvas")
    canvas.className = "arena-pdf-canvas"

    controls = document.createElement("div")
    controls.className = "arena-pdf-controls"
    controls.innerHTML = `
      <div class="arena-pdf-controls-group">
        <button type="button" class="arena-pdf-btn arena-pdf-prev" disabled>
          <svg width="12" height="12" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8.84182 3.13514C9.04327 3.32401 9.05348 3.64042 8.86462 3.84188L5.43521 7.49991L8.86462 11.1579C9.05348 11.3594 9.04327 11.6758 8.84182 11.8647C8.64036 12.0535 8.32394 12.0433 8.13508 11.8419L4.38508 7.84188C4.20477 7.64955 4.20477 7.35027 4.38508 7.15794L8.13508 3.15794C8.32394 2.95648 8.64036 2.94628 8.84182 3.13514Z" fill="currentColor"/>
          </svg>
        </button>
        <span class="arena-pdf-page-info">
          <input type="number" class="arena-pdf-page-input" value="1" min="1"/> / <span class="arena-pdf-total">0</span>
        </span>
        <button type="button" class="arena-pdf-btn arena-pdf-next" disabled>
          <svg width="12" height="12" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M6.1584 3.13508C6.35985 2.94621 6.67627 2.95642 6.86514 3.15788L10.6151 7.15788C10.7954 7.3502 10.7954 7.64949 10.6151 7.84182L6.86514 11.8418C6.67627 12.0433 6.35985 12.0535 6.1584 11.8646C5.95694 11.6757 5.94673 11.3593 6.1356 11.1579L9.565 7.49985L6.1356 3.84182C5.94673 3.64036 5.95694 3.32394 6.1584 3.13508Z" fill="currentColor"/>
          </svg>
        </button>
      </div>
      <div class="arena-pdf-controls-group">
        <button type="button" class="arena-pdf-btn arena-pdf-zoom-out" title="Zoom out">
          <svg width="12" height="12" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3.5 7C3.22386 7 3 7.22386 3 7.5C3 7.77614 3.22386 8 3.5 8H11.5C11.7761 8 12 7.77614 12 7.5C12 7.22386 11.7761 7 11.5 7H3.5Z" fill="currentColor"/>
          </svg>
        </button>
        <span class="arena-pdf-zoom-level">300%</span>
        <button type="button" class="arena-pdf-btn arena-pdf-zoom-in" title="Zoom in">
          <svg width="12" height="12" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8 2.75C8 2.47386 7.77614 2.25 7.5 2.25C7.22386 2.25 7 2.47386 7 2.75V7H2.75C2.47386 7 2.25 7.22386 2.25 7.5C2.25 7.77614 2.47386 8 2.75 8H7V12.25C7 12.5261 7.22386 12.75 7.5 12.75C7.77614 12.75 8 12.5261 8 12.25V8H12.25C12.5261 8 12.75 7.77614 12.75 7.5C12.75 7.22386 12.5261 7 12.25 7H8V2.75Z" fill="currentColor"/>
          </svg>
        </button>
        <button type="button" class="arena-pdf-btn arena-pdf-download" title="Download PDF">
          <svg width="12" height="12" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M7.50005 1.04999C7.74858 1.04999 7.95005 1.25146 7.95005 1.49999V8.41359L10.1819 6.18179C10.3576 6.00605 10.6425 6.00605 10.8182 6.18179C10.994 6.35753 10.994 6.64245 10.8182 6.81819L7.81825 9.81819C7.64251 9.99392 7.35759 9.99392 7.18185 9.81819L4.18185 6.81819C4.00611 6.64245 4.00611 6.35753 4.18185 6.18179C4.35759 6.00605 4.64251 6.00605 4.81825 6.18179L7.05005 8.41359V1.49999C7.05005 1.25146 7.25152 1.04999 7.50005 1.04999ZM2.5 10C2.77614 10 3 10.2239 3 10.5V12C3 12.5539 3.44565 13 3.99635 13H11.0012C11.5529 13 12 12.5528 12 12V10.5C12 10.2239 12.2239 10 12.5 10C12.7761 10 13 10.2239 13 10.5V12C13 13.1041 12.1062 14 11.0012 14H3.99635C2.89019 14 2 13.103 2 12V10.5C2 10.2239 2.22386 10 2.5 10Z" fill="currentColor"/>
          </svg>
        </button>
      </div>
    `

    const canvasWrapper = document.createElement("div")
    canvasWrapper.className = "arena-pdf-canvas-wrapper"
    canvasWrapper.appendChild(canvas)

    container.innerHTML = ""
    container.appendChild(controls)
    container.appendChild(canvasWrapper)
  }

  const ctx = canvas.getContext("2d")

  function renderPageAtIndex(pageNum: number) {
    if (rendering || !pdfDoc) return

    rendering = true
    const pageInput = controls.querySelector(".arena-pdf-page-input") as HTMLInputElement

    pdfDoc
      .getPage(pageNum)
      .then((page: any) => {
        const viewport = page.getViewport({ scale })
        canvas.height = viewport.height
        canvas.width = viewport.width

        const renderContext = {
          canvasContext: ctx,
          viewport: viewport,
        }

        return page.render(renderContext).promise
      })
      .then(() => {
        rendering = false
        currentPage = pageNum
        if (pageInput) pageInput.value = String(currentPage)
        updateControls()
      })
      .catch((err: Error) => {
        console.error("PDF render error:", err)
        rendering = false
      })
  }

  function updateControls() {
    const prevBtn = controls.querySelector(".arena-pdf-prev") as HTMLButtonElement
    const nextBtn = controls.querySelector(".arena-pdf-next") as HTMLButtonElement
    const zoomLevel = controls.querySelector(".arena-pdf-zoom-level") as HTMLSpanElement

    if (prevBtn) prevBtn.disabled = currentPage <= 1
    if (nextBtn) nextBtn.disabled = currentPage >= totalPages
    if (zoomLevel) zoomLevel.textContent = `${Math.round(scale * 100)}%`
  }

  // Event handlers
  controls.querySelector(".arena-pdf-prev")?.addEventListener("click", () => {
    if (currentPage > 1) renderPageAtIndex(currentPage - 1)
  })

  controls.querySelector(".arena-pdf-next")?.addEventListener("click", () => {
    if (currentPage < totalPages) renderPageAtIndex(currentPage + 1)
  })

  const pageInput = controls.querySelector(".arena-pdf-page-input") as HTMLInputElement
  pageInput?.addEventListener("change", (e) => {
    const target = e.target as HTMLInputElement
    const page = Number.parseInt(target.value, 10)
    if (page >= 1 && page <= totalPages) {
      renderPageAtIndex(page)
    } else {
      target.value = String(currentPage)
    }
  })

  controls.querySelector(".arena-pdf-zoom-in")?.addEventListener("click", () => {
    scale = Math.min(scale + 0.25, 3.0)
    renderPageAtIndex(currentPage)
  })

  controls.querySelector(".arena-pdf-zoom-out")?.addEventListener("click", () => {
    scale = Math.max(scale - 0.25, 0.5)
    renderPageAtIndex(currentPage)
  })

  controls.querySelector(".arena-pdf-download")?.addEventListener("click", () => {
    const a = document.createElement("a")
    a.href = pdfUrl
    a.download = container.dataset.pdfFilename || "document.pdf"
    a.click()
  })

  const proxyUrl = `/api/pdf-proxy?url=${encodeURIComponent(pdfUrl)}`

  window.pdfjsLib.GlobalWorkerOptions.workerSrc = `${PDFJS_CDN}/pdf.worker.min.mjs`

  // Load PDF
  return window.pdfjsLib.getDocument(proxyUrl).promise.then((pdf: any) => {
    pdfDoc = pdf
    pdfInstances.set(container, pdfDoc)
    totalPages = pdf.numPages
    const totalSpan = controls.querySelector(".arena-pdf-total")
    if (totalSpan) totalSpan.textContent = String(totalPages)
    pageInput?.setAttribute("max", String(totalPages))

    // Show controls and hide spinner
    controls.style.display = ""
    if (spinner) spinner.style.display = "none"

    renderPageAtIndex(1)
  })
}

function hydratePdfEmbeds(root: HTMLElement) {
  const nodes = root.querySelectorAll<HTMLElement>(
    ".arena-modal-embed-pdf[data-pdf-url]:not([data-pdf-status])",
  )

  nodes.forEach((node) => {
    const pdfUrl = node.dataset.pdfUrl
    if (!pdfUrl) return

    node.dataset.pdfStatus = "loading"
    renderPdfLoading(node)

    const filename = node.dataset.pdfFilename || "document.pdf"

    createPdfViewer(node, pdfUrl)
      .then(() => {
        if (!node.isConnected) return
        node.dataset.pdfStatus = "loaded"
      })
      .catch((err) => {
        console.error("PDF load error:", err)
        if (!node.isConnected) return
        renderPdfError(node, pdfUrl, filename)
        node.dataset.pdfStatus = "error"
      })
  })
}

function cleanupPdfs(root: HTMLElement) {
  root.querySelectorAll<HTMLElement>(".arena-modal-embed-pdf[data-pdf-status]").forEach((node) => {
    const instance = pdfInstances.get(node)
    if (instance?.destroy) {
      try {
        instance.destroy()
      } catch (err) {
        console.error(err)
      }
    }
    pdfInstances.delete(node)
    node.dataset.pdfStatus = ""
  })
}

// Dynamically create modal data from search index JSON
function createModalDataFromJson(block: ArenaBlockSearchable, channelSlug: string): HTMLElement {
  const container = document.createElement("div")
  container.className = "arena-block-modal-data"
  container.id = `arena-modal-data-${block.id}`
  container.dataset.blockId = block.id
  container.dataset.channelSlug = channelSlug
  container.style.display = "none"

  // Build display URL
  const displayUrl =
    block.url ??
    (block.internalSlug ? `${window.location.origin}/${block.internalSlug}` : undefined)

  // Build metadata entries
  const metadataHtml: string[] = []

  if (block.metadata) {
    const consumedKeys = new Set(["accessed", "accessed_date", "date", "tags", "tag", "coord"])
    const entries = Object.entries(block.metadata)
      .filter(([key, value]) => {
        if (typeof value !== "string" || value.trim().length === 0) return false
        if (consumedKeys.has(key.toLowerCase())) return false
        return true
      })
      .sort(([a], [b]) => a.localeCompare(b))

    for (const [key, value] of entries) {
      const label = key.replace(/_/g, " ")
      metadataHtml.push(`
        <div class="arena-meta-item">
          <span class="arena-meta-label">${label}</span>
          <em class="arena-meta-value">${value}</em>
        </div>
      `)
    }
  }

  if (block.tags && block.tags.length > 0) {
    const tagsLabel = block.tags.length === 1 ? "tag" : "tags"
    const tagsList = block.tags.map((tag) => `<span class="tag-link">${tag}</span>`).join("")
    metadataHtml.push(`
      <div class="arena-meta-item">
        <span class="arena-meta-label">${tagsLabel}</span>
        <em class="arena-meta-value">
          <span class="arena-meta-taglist">${tagsList}</span>
        </em>
      </div>
    `)
  }

  // Build sub-items (connections)
  const hasSubItems = block.subItems && block.subItems.length > 0
  const subItemsHtml = hasSubItems
    ? `
    <div class="arena-modal-connections">
      <div class="arena-modal-connections-header">
        <span class="arena-modal-connections-title">notes</span>
        <span class="arena-modal-connections-count">${block.subItems!.length}</span>
      </div>
      <ul class="arena-modal-connections-list">
        ${block
          .subItems!.map(
            (subItem) => `
          <li>${subItem.blockHtml || subItem.titleHtml || subItem.title || subItem.content}</li>
        `,
          )
          .join("")}
      </ul>
    </div>
  `
    : ""

  const mapTitle = block.title || block.content || block.url || ""
  const mapHtml = block.coordinates
    ? `
        <div class="arena-modal-map-wrapper">
          <div
            class="arena-modal-map"
            data-map-lon="${String(block.coordinates.lon)}"
            data-map-lat="${String(block.coordinates.lat)}"${mapTitle.trim().length > 0 ? ` data-map-title="${escapeHtml(mapTitle)}"` : ""}
          ></div>
        </div>
      `
    : ""

  // Determine embed content
  let mainContentHtml = ""
  if (block.embedHtml) {
    mainContentHtml = block.embedHtml
  } else if (block.url && SUBSTACK_POST_REGEX.test(block.url)) {
    mainContentHtml = `
      <div class="arena-modal-embed arena-modal-embed-substack" data-substack-url="${block.url}">
        <span class="arena-loading-spinner" role="status" aria-label="Loading Substack preview"></span>
      </div>
    `
  } else if (block.embedDisabled && block.url) {
    mainContentHtml = `
      <div class="arena-iframe-error">
        <div class="arena-iframe-error-content">
          <p>unable to embed content</p>
          <a href="${block.url}" target="_blank" rel="noopener noreferrer" class="arena-iframe-error-link">
            open in new tab →
          </a>
        </div>
      </div>
    `
  } else if (block.url) {
    const frameTitle = block.title ?? block.content ?? "Block"
    mainContentHtml = `
      <iframe
        class="arena-modal-iframe"
        title="Embedded block: ${frameTitle}"
        loading="lazy"
        data-block-id="${block.id}"
        sandbox="allow-same-origin allow-scripts allow-popups allow-popups-to-escape-sandbox allow-forms"
        src="${block.url}"
      ></iframe>
    `
  } else if (block.internalSlug) {
    mainContentHtml = `
      <div
        class="arena-modal-internal-host"
        data-block-id="${block.id}"
        data-internal-slug="${block.internalSlug}"
        data-internal-href="${block.internalHref || ""}"
        data-internal-hash="${block.internalHash || ""}"
      >
        <div class="arena-modal-internal-preview grid"></div>
      </div>
    `
  } else {
    mainContentHtml = `<div class="arena-modal-placeholder">No preview available</div>`
  }

  container.innerHTML = `
    <div class="arena-modal-layout">
      <div class="arena-modal-main">
        ${
          displayUrl
            ? `
          <div class="arena-modal-url-bar">
            <button type="button" class="arena-url-copy-button" data-url="${displayUrl}" role="button" tabIndex="0" aria-label="Copy URL to clipboard">
              <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg" class="copy-icon">
                <path d="M7.49996 1.80002C4.35194 1.80002 1.79996 4.352 1.79996 7.50002C1.79996 10.648 4.35194 13.2 7.49996 13.2C10.648 13.2 13.2 10.648 13.2 7.50002C13.2 4.352 10.648 1.80002 7.49996 1.80002ZM0.899963 7.50002C0.899963 3.85494 3.85488 0.900024 7.49996 0.900024C11.145 0.900024 14.1 3.85494 14.1 7.50002C14.1 11.1451 11.145 14.1 7.49996 14.1C3.85488 14.1 0.899963 11.1451 0.899963 7.50002Z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
                <path d="M13.4999 7.89998H1.49994V7.09998H13.4999V7.89998Z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
                <path d="M7.09991 13.5V1.5H7.89991V13.5H7.09991zM10.375 7.49998C10.375 5.32724 9.59364 3.17778 8.06183 1.75656L8.53793 1.24341C10.2396 2.82218 11.075 5.17273 11.075 7.49998 11.075 9.82724 10.2396 12.1778 8.53793 13.7566L8.06183 13.2434C9.59364 11.8222 10.375 9.67273 10.375 7.49998zM3.99969 7.5C3.99969 5.17611 4.80786 2.82678 6.45768 1.24719L6.94177 1.75281C5.4582 3.17323 4.69969 5.32389 4.69969 7.5 4.6997 9.67611 5.45822 11.8268 6.94179 13.2472L6.45769 13.7528C4.80788 12.1732 3.9997 9.8239 3.99969 7.5z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
                <path d="M7.49996 3.95801C9.66928 3.95801 11.8753 4.35915 13.3706 5.19448 13.5394 5.28875 13.5998 5.50197 13.5055 5.67073 13.4113 5.83948 13.198 5.89987 13.0293 5.8056 11.6794 5.05155 9.60799 4.65801 7.49996 4.65801 5.39192 4.65801 3.32052 5.05155 1.97064 5.8056 1.80188 5.89987 1.58866 5.83948 1.49439 5.67073 1.40013 5.50197 1.46051 5.28875 1.62927 5.19448 3.12466 4.35915 5.33063 3.95801 7.49996 3.95801zM7.49996 10.85C9.66928 10.85 11.8753 10.4488 13.3706 9.6135 13.5394 9.51924 13.5998 9.30601 13.5055 9.13726 13.4113 8.9685 13.198 8.90812 13.0293 9.00238 11.6794 9.75643 9.60799 10.15 7.49996 10.15 5.39192 10.15 3.32052 9.75643 1.97064 9.00239 1.80188 8.90812 1.58866 8.9685 1.49439 9.13726 1.40013 9.30601 1.46051 9.51924 1.62927 9.6135 3.12466 10.4488 5.33063 10.85 7.49996 10.85z" fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"/>
              </svg>
              <svg width="15" height="15" viewBox="-2 -2 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" class="check-icon">
                <use href="#github-check"/>
              </svg>
            </button>
            ${
              block.url
                ? `
              <a href="${block.url}" target="_blank" rel="noopener noreferrer" class="arena-modal-link">
                <div class="arena-modal-link-text">${displayUrl}</div>
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path fill-rule="evenodd" clip-rule="evenodd" d="M12 13C12.5523 13 13 12.5523 13 12V3C13 2.44771 12.5523 2 12 2H3C2.44771 2 2 2.44771 2 3V6.5C2 6.77614 2.22386 7 2.5 7C2.77614 7 3 6.77614 3 6.5V3H12V12H8.5C8.22386 12 8 12.2239 8 12.5C8 12.7761 8.22386 13 8.5 13H12ZM9 6.5C9 6.5001 9 6.50021 9 6.50031V6.50035V9.5C9 9.77614 8.77614 10 8.5 10C8.22386 10 8 9.77614 8 9.5V7.70711L2.85355 12.8536C2.65829 13.0488 2.34171 13.0488 2.14645 12.8536C1.95118 12.6583 1.95118 12.3417 2.14645 12.1464L7.29289 7H5.5C5.22386 7 5 6.77614 5 6.5C5 6.22386 5.22386 6 5.5 6H8.5C8.56779 6 8.63244 6.01349 8.69139 6.03794C8.74949 6.06198 8.80398 6.09744 8.85143 6.14433C8.94251 6.23434 8.9992 6.35909 8.99999 6.49708L8.99999 6.49738" fill="currentColor"/>
                </svg>
              </a>
            `
                : `
              <span class="arena-modal-link">
                <div class="arena-modal-link-text">${displayUrl}</div>
              </span>
            `
            }
          </div>
        `
            : ""
        }
        ${mapHtml}
        <div class="arena-modal-main-content">
          ${mainContentHtml}
        </div>
      </div>
      <div class="arena-modal-sidebar">
        <div class="arena-modal-info">
          <h3 class="arena-modal-title">
            ${block.titleHtml || block.title || ""}
          </h3>
          ${
            metadataHtml.length > 0
              ? `
            <div class="arena-modal-meta">
              ${metadataHtml.join("")}
            </div>
          `
              : ""
          }
        </div>
        ${subItemsHtml}
      </div>
    </div>
  `

  return container
}

async function showModal(blockId: string) {
  const modal = document.getElementById("arena-modal")
  const modalBody = modal?.querySelector(".arena-modal-body") as HTMLElement | null
  if (!modal || !modalBody) return

  const dataEl = document.getElementById(`arena-modal-data-${blockId}`)
  if (!dataEl) return

  const blockEl = document.querySelector(`[data-block-id="${blockId}"]`)
  if (blockEl) {
    currentBlockIndex = parseInt(blockEl.getAttribute("data-block-index") || "0")
  }

  cleanupMaps(modalBody)
  modalBody.innerHTML = ""
  const clonedContent = dataEl.cloneNode(true) as HTMLElement
  clonedContent.style.display = "block"
  modalBody.appendChild(clonedContent)

  if (window.twttr && typeof window.twttr.ready === "function") {
    window.twttr.ready((readyTwttr: any) => {
      if (readyTwttr?.widgets?.load) {
        readyTwttr.widgets.load(modalBody)
      }
    })
    // @ts-ignore
  } else if (window.twttr?.widgets?.load) {
    // @ts-ignore
    window.twttr.widgets.load(modalBody)
  }

  hydrateSubstackEmbeds(modalBody)
  hydrateInternalHosts(modalBody)
  hydrateMapboxMaps(modalBody)
  hydratePdfEmbeds(modalBody)

  const sidebar = modalBody.querySelector(".arena-modal-sidebar") as HTMLElement | null
  const hasConnections = modalBody.querySelector(".arena-modal-connections") !== null
  const collapseBtn = modal?.querySelector(".arena-modal-collapse") as HTMLElement | null

  if (sidebar) {
    if (hasConnections) {
      sidebar.classList.remove("collapsed")
      collapseBtn?.classList.remove("active")
    } else {
      sidebar.classList.add("collapsed")
      collapseBtn?.classList.add("active")
    }
  }

  updateNavButtons()
  modal.classList.add("active")
  lockPageScroll()
}

function closeModal() {
  const modal = document.getElementById("arena-modal")
  if (modal) {
    const modalBody = modal.querySelector(".arena-modal-body") as HTMLElement | null
    if (modalBody) {
      cleanupMaps(modalBody)
      cleanupPdfs(modalBody)
    }
    modal.classList.remove("active")
    unlockPageScroll()
  }
}

function navigateBlock(direction: number) {
  const newIndex = currentBlockIndex + direction
  if (newIndex < 0 || newIndex >= totalBlocks) return

  const blocks = Array.from(document.querySelectorAll(".arena-block[data-block-id]"))
  const targetBlock = blocks[newIndex] as HTMLElement
  if (!targetBlock) return

  const blockId = targetBlock.getAttribute("data-block-id")
  if (blockId) {
    showModal(blockId)
  }
}

function updateNavButtons() {
  const prevBtn = document.querySelector(".arena-modal-prev") as HTMLButtonElement
  const nextBtn = document.querySelector(".arena-modal-next") as HTMLButtonElement

  if (prevBtn) {
    prevBtn.disabled = currentBlockIndex === 0
    prevBtn.style.opacity = currentBlockIndex === 0 ? "0.3" : "1"
  }

  if (nextBtn) {
    nextBtn.disabled = currentBlockIndex >= totalBlocks - 1
    nextBtn.style.opacity = currentBlockIndex >= totalBlocks - 1 ? "0.3" : "1"
  }
}

function handleCopyButton(button: HTMLElement) {
  const targetUrl = button.getAttribute("data-url")
  if (!targetUrl) {
    return
  }

  navigator.clipboard.writeText(targetUrl).then(
    () => {
      button.classList.add("check")
      setTimeout(() => {
        button.classList.remove("check")
      }, 2000)
    },
    (error) => console.error(error),
  )
}

// Search functionality - JSON-based
interface ArenaBlockSearchable {
  id: string
  channelSlug: string
  channelName: string
  content: string
  title?: string
  titleHtml?: string
  blockHtml?: string
  url?: string
  highlighted: boolean
  embedHtml?: string
  metadata?: Record<string, string>
  coordinates?: {
    lon: number
    lat: number
  }
  internalSlug?: string
  internalHref?: string
  internalHash?: string
  tags?: string[]
  subItems?: ArenaBlockSearchable[]
  hasModalInDom: boolean
  embedDisabled?: boolean
}

interface ArenaChannelSearchable {
  id: string
  name: string
  slug: string
  blockCount: number
}

interface ArenaSearchIndex {
  version: string
  blocks: ArenaBlockSearchable[]
  channels: ArenaChannelSearchable[]
}

interface SearchIndexItem {
  blockId: string
  channelSlug?: string
  channelName?: string
  title: string
  content: string
  highlighted: boolean
  hasModalInDom: boolean
  tags?: string[]
  matchedTags?: string[]
}

let searchIndex: SearchIndexItem[] = []
let searchDebounceTimer: number | undefined
let arenaSearchData: ArenaSearchIndex | null = null

// Fetch arena search index JSON
async function fetchArenaSearchIndex(): Promise<ArenaSearchIndex | null> {
  if (arenaSearchData) return arenaSearchData

  try {
    const response = await fetch("/static/arena-search.json")
    if (!response.ok) {
      console.warn(`Failed to fetch arena search index: ${response.status}`)
      return null
    }
    arenaSearchData = (await response.json()) as ArenaSearchIndex
    return arenaSearchData
  } catch (error) {
    console.error("Error fetching arena search index:", error)
    return null
  }
}

async function buildSearchIndex(scope: "channel" | "index"): Promise<SearchIndexItem[]> {
  const index: SearchIndexItem[] = []

  if (scope === "channel") {
    // For channel pages, fetch JSON and filter by current channel
    const data = await fetchArenaSearchIndex()
    if (!data) return index

    const currentSlug = document.body?.dataset.slug || ""
    const channelSlug = currentSlug.replace(/^arena\//, "")

    data.blocks
      .filter((block) => block.channelSlug === channelSlug)
      .forEach((block) => {
        index.push({
          blockId: block.id,
          channelSlug: block.channelSlug,
          channelName: block.channelName,
          title: block.title || block.content,
          content: block.content,
          highlighted: block.highlighted,
          hasModalInDom: block.hasModalInDom,
          tags: block.tags,
        })
      })
  } else {
    // For index page, use all blocks from JSON
    const data = await fetchArenaSearchIndex()
    if (!data) return index

    data.blocks.forEach((block) => {
      index.push({
        blockId: block.id,
        channelSlug: block.channelSlug,
        channelName: block.channelName,
        title: block.title || block.content,
        content: block.content,
        highlighted: block.highlighted,
        hasModalInDom: block.hasModalInDom,
        tags: block.tags,
      })
    })
  }

  return index
}

function parseSearchQuery(query: string): { tagQueries: string[]; regularTerms: string[] } {
  const tagQueries: string[] = []
  let remainingQuery = query

  const tagPatternRegex =
    /#([^#]+?)(?=\s*#|\s*r\/|$)|r\/([^\s#]+(?:\s+[^\s#]+)*?)(?=\s*#|\s*r\/|$)/g
  let match

  while ((match = tagPatternRegex.exec(query)) !== null) {
    const tagText = (match[1] || match[2] || "").trim()
    if (tagText.length > 0) {
      tagQueries.push(tagText)
      remainingQuery = remainingQuery.replace(match[0], " ")
    }
  }

  const regularTerms = remainingQuery
    .trim()
    .split(/\s+/)
    .filter((term) => term.length > 0)

  return { tagQueries, regularTerms }
}

function performSearch(query: string, index: SearchIndexItem[]): SearchIndexItem[] {
  const lowerQuery = query.toLowerCase().trim()
  if (lowerQuery.length < 2) return []

  const { tagQueries, regularTerms } = parseSearchQuery(lowerQuery)
  const tokens = tokenizeTerm(regularTerms.join(" "))

  const scoredResults = index
    .map((item) => {
      const lowerTitle = item.title.toLowerCase()
      const lowerContent = item.content.toLowerCase()
      const itemTags = item.tags || []
      const matchedTags: string[] = []

      let score = 0
      let titleMatchCount = 0
      let contentMatchCount = 0
      let tagMatchCount = 0

      // Tag-specific queries (# or r/ prefix)
      for (const tagQuery of tagQueries) {
        const lowerTagQuery = tagQuery.toLowerCase()
        for (const tag of itemTags) {
          const lowerTag = tag.toLowerCase()
          if (lowerTag === lowerTagQuery) {
            score += 25
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          } else if (lowerTag.includes(lowerTagQuery)) {
            score += 20
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          }
        }
      }

      // Regular terms match tags
      for (const term of regularTerms) {
        const lowerTerm = term.toLowerCase()
        for (const tag of itemTags) {
          const lowerTag = tag.toLowerCase()
          if (lowerTag === lowerTerm) {
            score += 18
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          } else if (lowerTag.includes(lowerTerm)) {
            score += 15
            tagMatchCount++
            if (!matchedTags.includes(tag)) {
              matchedTags.push(tag)
            }
          }
        }
      }

      // Check each token for title and content matches
      for (const token of tokens) {
        const tokenLower = token.toLowerCase()

        if (lowerTitle.includes(tokenLower)) {
          titleMatchCount++
          score += 10
        }

        if (lowerContent.includes(tokenLower)) {
          contentMatchCount++
          score += 5
        }
      }

      // Boost for exact phrase match in title/content
      if (regularTerms.length > 0) {
        const regularQuery = regularTerms.join(" ").toLowerCase()
        if (lowerTitle.includes(regularQuery)) {
          score += 20
        }
        if (lowerContent.includes(regularQuery)) {
          score += 10
        }
      }

      // Boost for highlighted items
      if (item.highlighted) {
        score += 3
      }

      // Must have at least one match
      if (score === 0) return null

      const resultItem = { ...item }
      if (matchedTags.length > 0) {
        resultItem.matchedTags = matchedTags
      }

      return { item: resultItem, score, titleMatchCount, contentMatchCount, tagMatchCount }
    })
    .filter((result): result is NonNullable<typeof result> => result !== null)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score

      if (b.tagMatchCount !== a.tagMatchCount) {
        return b.tagMatchCount - a.tagMatchCount
      }

      if (b.titleMatchCount !== a.titleMatchCount) {
        return b.titleMatchCount - a.titleMatchCount
      }

      return b.contentMatchCount - a.contentMatchCount
    })

  return scoredResults.map((result) => result.item)
}

function renderSearchResults(
  results: SearchIndexItem[],
  scope: "channel" | "index",
  searchQuery: string = "",
) {
  const container = document.getElementById("arena-search-container")
  if (!container) return

  if (results.length === 0) {
    container.innerHTML = '<div class="arena-search-no-results">no results found</div>'
    container.classList.add("active")
    return
  }

  const fragment = document.createDocumentFragment()
  results.forEach((result, idx) => {
    const resultItem = document.createElement("div")
    resultItem.className = "arena-search-result-item"
    resultItem.setAttribute("data-block-id", result.blockId)
    if (result.channelSlug) {
      resultItem.setAttribute("data-channel-slug", result.channelSlug)
    }
    resultItem.dataset.index = String(idx)
    resultItem.tabIndex = -1
    resultItem.setAttribute("role", "option")
    if (!resultItem.id) {
      const uniqueId = `arena-search-${result.blockId}`
      resultItem.id = uniqueId
    }

    const title = document.createElement("div")
    title.className = "arena-search-result-title"
    title.innerHTML = searchQuery ? highlight(searchQuery, result.title) : result.title
    resultItem.appendChild(title)

    if (result.matchedTags && result.matchedTags.length > 0) {
      const tagsContainer = document.createElement("ul")
      tagsContainer.className = "arena-search-result-tags"

      result.matchedTags.forEach((tag) => {
        const tagBadge = document.createElement("li")
        tagBadge.className = "arena-search-result-tag-badge"
        tagBadge.innerHTML = searchQuery ? highlight(searchQuery, tag) : tag
        tagsContainer.appendChild(tagBadge)
      })

      resultItem.appendChild(tagsContainer)
    }

    if (scope === "index" && result.channelName) {
      const badge = document.createElement("span")
      badge.className = "arena-search-result-channel-badge"
      badge.textContent = result.channelName
      resultItem.appendChild(badge)
    }

    fragment.appendChild(resultItem)
  })

  container.innerHTML = ""
  container.appendChild(fragment)
  container.classList.add("active")
}

function closeSearchDropdown() {
  const container = document.getElementById("arena-search-container")
  if (container) {
    container.classList.remove("active")
    container.innerHTML = ""
  }
}

document.addEventListener("nav", () => {
  totalBlocks = document.querySelectorAll("[data-block-id][data-block-index]").length

  const channelPage = document.querySelector<HTMLElement>(".arena-channel-page")
  const blockCollection = document.getElementById("arena-block-collection") as HTMLElement | null
  const viewToggleButtons = channelPage
    ? Array.from(channelPage.querySelectorAll<HTMLButtonElement>(".arena-view-toggle-button"))
    : []

  type ArenaViewMode = "grid" | "list"

  const applyViewMode = (mode: ArenaViewMode) => {
    if (!channelPage || !blockCollection) return
    const normalized: ArenaViewMode = mode === "list" ? "list" : "grid"
    channelPage.dataset.viewMode = normalized
    blockCollection.dataset.viewMode = normalized
    viewToggleButtons.forEach((button) => {
      const targetMode = (button.dataset.viewMode as ArenaViewMode) || "grid"
      const isActive = targetMode === normalized
      button.classList.toggle("active", isActive)
      button.setAttribute("aria-pressed", isActive ? "true" : "false")
    })
    totalBlocks = channelPage.querySelectorAll("[data-block-id][data-block-index]").length
  }

  if (channelPage && blockCollection && viewToggleButtons.length > 0) {
    const initialMode = (blockCollection.dataset.viewMode as ArenaViewMode | undefined) ?? "grid"
    applyViewMode(initialMode)

    viewToggleButtons.forEach((button) => {
      const desiredMode = (button.dataset.viewMode as ArenaViewMode) || "grid"
      const onToggleClick = () => applyViewMode(desiredMode)
      button.addEventListener("click", onToggleClick)
      window.addCleanup(() => button.removeEventListener("click", onToggleClick))
    })
  }

  // Build search index
  const searchInput = document.querySelector<HTMLInputElement>(".arena-search-input")
  let activeResultIndex: number | null = null

  const getSearchResults = () =>
    Array.from(
      document.querySelectorAll<HTMLElement>("#arena-search-container .arena-search-result-item"),
    )

  const resetActiveResultHighlight = () => {
    activeResultIndex = null
    if (searchInput) {
      searchInput.removeAttribute("aria-activedescendant")
    }
    getSearchResults().forEach((result) => result.classList.remove("active"))
  }

  const setActiveResult = (
    index: number | null,
    options?: { focus?: boolean; scroll?: boolean },
  ) => {
    const results = getSearchResults()
    if (results.length === 0) {
      resetActiveResultHighlight()
      if (options?.focus && searchInput) {
        searchInput.focus()
      }
      return
    }

    if (index === null) {
      resetActiveResultHighlight()
      if (options?.focus && searchInput) {
        searchInput.focus()
      }
      return
    }

    const clamped = Math.max(0, Math.min(index, results.length - 1))
    results.forEach((result, idx) => {
      result.classList.toggle("active", idx === clamped)
    })

    const target = results[clamped]
    activeResultIndex = clamped
    if (!target.id) {
      const fallbackId = target.getAttribute("data-block-id") || `arena-result-${clamped}`
      target.id = `arena-search-${fallbackId}`
    }
    if (searchInput) {
      searchInput.setAttribute("aria-activedescendant", target.id)
    }

    if (options?.focus !== false) {
      target.focus()
    }

    if (options?.scroll !== false) {
      target.scrollIntoView({ block: "nearest" })
    }
  }

  const wireSearchResultsInteractions = () => {
    const items = getSearchResults()
    items.forEach((item) => {
      const idx = Number.parseInt(item.dataset.index ?? "", 10)
      if (!Number.isInteger(idx)) return

      const onFocus = () => setActiveResult(idx, { focus: false, scroll: false })
      const onMouseEnter = () => setActiveResult(idx, { focus: false, scroll: false })

      item.addEventListener("focus", onFocus)
      item.addEventListener("mouseenter", onMouseEnter)

      window.addCleanup(() => {
        item.removeEventListener("focus", onFocus)
        item.removeEventListener("mouseenter", onMouseEnter)
      })
    })
  }

  const focusSearchInput = (prefill?: string) => {
    if (!searchInput) return
    if (typeof prefill === "string") {
      searchInput.value = prefill
      searchInput.dispatchEvent(new Event("input", { bubbles: true }))
    }
    resetActiveResultHighlight()
    const valueLength = searchInput.value.length
    searchInput.focus()
    try {
      if (valueLength > 0) {
        searchInput.select()
      } else {
        searchInput.setSelectionRange(valueLength, valueLength)
      }
    } catch {
      // Some inputs (e.g. on Safari) may not support selection API depending on state
    }
  }

  const clearSearchState = (options?: { blur?: boolean }) => {
    if (searchInput) {
      searchInput.value = ""
      if (options?.blur !== false) {
        searchInput.blur()
      }
      searchInput.removeAttribute("aria-activedescendant")
    }
    resetActiveResultHighlight()
    closeSearchDropdown()
  }

  if (searchInput) {
    const scope = searchInput.getAttribute("data-search-scope") as "channel" | "index"

    // Build search index asynchronously
    buildSearchIndex(scope)
      .then((index) => {
        searchIndex = index
      })
      .catch((error) => {
        console.error("Failed to build search index:", error)
        searchIndex = []
      })

    // Clear any previous listener
    if (searchDebounceTimer) {
      window.clearTimeout(searchDebounceTimer)
    }

    const onSearchInput = (e: Event) => {
      const input = e.target as HTMLInputElement
      const query = input.value

      window.clearTimeout(searchDebounceTimer)
      searchDebounceTimer = window.setTimeout(() => {
        if (query.length < 2) {
          closeSearchDropdown()
          resetActiveResultHighlight()
          return
        }

        const results = performSearch(query, searchIndex)
        renderSearchResults(results, scope, query)
        wireSearchResultsInteractions()
        resetActiveResultHighlight()
      }, 300)
    }

    searchInput.addEventListener("input", onSearchInput)
    window.addCleanup(() => searchInput.removeEventListener("input", onSearchInput))
  }

  const onClick = (e: MouseEvent) => {
    const target = e.target as HTMLElement
    const isArenaChannelPage = () => {
      const slug = document.body?.dataset.slug || ""
      return slug.startsWith("arena/") && slug !== "arena"
    }

    const internalLink = target.closest(".arena-modal-body a.internal") as HTMLAnchorElement | null
    if (internalLink) {
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) {
        return
      }

      e.preventDefault()

      // Check if this is a wikilink trail anchor - open in stacked notes
      const isWikilinkTrail = internalLink.classList.contains("arena-wikilink-trail-anchor")

      if (isWikilinkTrail && typeof window.stackedNotes !== "undefined") {
        // Open in stacked notes view
        try {
          const destination = new URL(internalLink.href)
          window.stackedNotes.push(destination)
          // Keep modal open so user can see both arena block and note
          return
        } catch (err) {
          console.error("Failed to open in stacked notes:", err)
          // Fall through to regular navigation
        }
      }

      // Regular internal link - close modal and navigate
      closeModal()
      try {
        const destination = new URL(internalLink.href)
        if (typeof window.spaNavigate === "function") {
          window.spaNavigate(destination)
        } else {
          window.location.assign(destination)
        }
      } catch (err) {
        console.error(err)
        window.location.assign(internalLink.href)
      }
      return
    }

    const copyButton = target.closest("button.arena-url-copy-button") as HTMLElement | null
    if (copyButton) {
      e.preventDefault()
      e.stopPropagation()
      handleCopyButton(copyButton)
      return
    }

    const blockClickable = target.closest(".arena-block-clickable")
    if (blockClickable) {
      const blockEl = blockClickable.closest(".arena-block")
      const blockId = blockEl?.getAttribute("data-block-id")
      if (blockId) {
        e.preventDefault()

        // Check if this is an arxiv block without notes and redirect instead of opening modal
        if (arenaSearchData) {
          const blockData = arenaSearchData.blocks.find((b) => b.id === blockId)
          const isArxivUrl = blockData?.url
            ? /^https?:\/\/(?:ar5iv\.(?:labs\.)?)?arxiv\.org\//i.test(blockData.url)
            : false

          if (isArxivUrl && blockData?.url) {
            // Only redirect if the block has no notes (description or content)
            const hasNotes = !!blockData.content
            if (!hasNotes) {
              window.open(blockData.url, "_blank", "noopener,noreferrer")
              return
            }
            // If it has notes, continue to show the modal (without embed)
          }
        }

        showModal(blockId)
      }
      return
    }

    const previewItem = target.closest(".arena-channel-row-preview-item[data-block-id]")
    if (previewItem) {
      const blockId = (previewItem as HTMLElement).getAttribute("data-block-id")

      if (isArenaChannelPage()) {
        // On channel pages: check if arxiv block without notes and redirect instead of modal
        if (blockId && arenaSearchData) {
          const blockData = arenaSearchData.blocks.find((b) => b.id === blockId)
          const isArxivUrl = blockData?.url
            ? /^https?:\/\/(?:ar5iv\.(?:labs\.)?)?arxiv\.org\//i.test(blockData.url)
            : false

          if (isArxivUrl && blockData?.url) {
            // Only redirect if the block has no notes (description or content)
            const hasNotes = !!blockData.content
            if (!hasNotes) {
              e.preventDefault()
              window.open(blockData.url, "_blank", "noopener,noreferrer")
              return
            }
            // If it has notes, continue to show the modal (without embed)
          }
        }

        if (blockId) {
          e.preventDefault()
          showModal(blockId)
        }
      } else {
        // On Arena index page, clicking a preview should navigate to the channel
        const channelRow = previewItem.closest(".arena-channel-row") as HTMLElement | null
        const headerLink = channelRow?.querySelector(
          ".arena-channel-row-header a[href]",
        ) as HTMLAnchorElement | null
        if (headerLink) {
          e.preventDefault()
          headerLink.click()
        }
      }
      return
    }

    // Click anywhere on a channel row should navigate to the header link
    const channelRow = target.closest(".arena-channel-row") as HTMLElement | null
    if (channelRow) {
      // If the user clicked a real interactive element, let it handle itself
      if (target.closest("a,button,[role=button],input,textarea,select,summary")) {
        return
      }
      const headerLink = channelRow.querySelector(
        ".arena-channel-row-header a[href]",
      ) as HTMLAnchorElement | null
      if (headerLink) {
        e.preventDefault()
        // prefer native navigation so SPA router (if any) can hook the event
        headerLink.click()
      }
      return
    }

    if (target.closest(".arena-modal-prev")) {
      navigateBlock(-1)
      return
    }

    if (target.closest(".arena-modal-next")) {
      navigateBlock(1)
      return
    }

    if (target.closest(".arena-modal-collapse")) {
      const modal = document.getElementById("arena-modal")
      const sidebar = modal?.querySelector(".arena-modal-sidebar") as HTMLElement | null
      const collapseBtn = target.closest(".arena-modal-collapse") as HTMLElement | null
      if (sidebar) {
        sidebar.classList.toggle("collapsed")
        collapseBtn?.classList.toggle("active")
      }
      return
    }

    // Handle search result clicks with smart navigation based on hasModalInDom
    const searchResultItem = target.closest(".arena-search-result-item") as HTMLElement | null
    if (searchResultItem) {
      const blockId = searchResultItem.getAttribute("data-block-id")
      const channelSlug = searchResultItem.getAttribute("data-channel-slug")

      if (blockId) {
        e.preventDefault()
        clearSearchState({ blur: true })

        // Check if this is an arxiv block without notes and redirect instead of opening modal
        if (arenaSearchData) {
          const blockData = arenaSearchData.blocks.find((b) => b.id === blockId)
          const isArxivUrl = blockData?.url
            ? /^https?:\/\/(?:ar5iv\.(?:labs\.)?)?arxiv\.org\//i.test(blockData.url)
            : false

          if (isArxivUrl && blockData?.url) {
            // Only redirect if the block has no notes (description or content)
            const hasNotes = !!blockData.content
            if (!hasNotes) {
              window.open(blockData.url, "_blank", "noopener,noreferrer")
              return
            }
            // If it has notes, continue to show the modal (without embed)
          }
        }

        // Find the result in search index to check hasModalInDom flag
        const resultData = searchIndex.find((item) => item.blockId === blockId)

        if (resultData?.hasModalInDom) {
          // Block has prerendered modal in DOM - show it instantly
          showModal(blockId)
        } else {
          // Block doesn't have modal in DOM - create it dynamically from JSON
          const existingModal = document.getElementById(`arena-modal-data-${blockId}`)

          if (!existingModal && arenaSearchData && channelSlug) {
            // Find full block data in search JSON
            const fullBlockData = arenaSearchData.blocks.find((b) => b.id === blockId)

            if (fullBlockData) {
              // Create modal data element from JSON
              const modalDataEl = createModalDataFromJson(fullBlockData, channelSlug)

              // Insert it into the DOM (append to body or modal container)
              const modal = document.getElementById("arena-modal")
              if (modal) {
                modal.appendChild(modalDataEl)
              } else {
                document.body.appendChild(modalDataEl)
              }

              // Now show the modal
              showModal(blockId)
            } else if (channelSlug) {
              // Fallback: navigate to channel page if we can't find the data
              const currentSlug = document.body?.dataset.slug || ""
              const targetChannelSlug = `arena/${channelSlug}`

              if (currentSlug === targetChannelSlug) {
                showModal(blockId)
              } else {
                window.location.href = `/${targetChannelSlug}#${blockId}`
              }
            } else {
              // Last resort: try to show modal anyway
              showModal(blockId)
            }
          } else {
            // Modal data already exists, just show it
            showModal(blockId)
          }
        }
      }
      return
    }

    if (target.closest(".arena-modal-close") || target.classList.contains("arena-block-modal")) {
      closeModal()
    }
  }

  const onKey = (e: KeyboardEvent) => {
    const key = e.key.toLowerCase()
    if ((e.metaKey || e.ctrlKey) && key === "k") {
      e.preventDefault()
      if (e.shiftKey) {
        focusSearchInput("#")
      } else {
        focusSearchInput()
      }
      return
    }

    const results = getSearchResults()
    const searchContainer = document.getElementById("arena-search-container")
    const searchOpen = Boolean(searchContainer && searchContainer.classList.contains("active"))
    const resultFocused = document.activeElement?.classList.contains("arena-search-result-item")
    const inputFocused = document.activeElement === searchInput

    if (searchOpen && results.length > 0) {
      if (
        key === "ArrowDown" ||
        (!e.shiftKey && key === "Tab") ||
        (key === "n" && (e.ctrlKey || e.metaKey))
      ) {
        e.preventDefault()
        if (
          !e.shiftKey &&
          key === "Tab" &&
          resultFocused &&
          activeResultIndex === results.length - 1
        ) {
          setActiveResult(null, { focus: true, scroll: false })
        } else if (inputFocused || activeResultIndex === null) {
          setActiveResult(0)
        } else {
          const nextIndex = Math.min((activeResultIndex ?? -1) + 1, results.length - 1)
          setActiveResult(nextIndex)
        }
        return
      }

      if (
        key === "ArrowUp" ||
        (e.shiftKey && key === "Tab") ||
        (key === "p" && (e.ctrlKey || e.metaKey))
      ) {
        e.preventDefault()
        if (!resultFocused || activeResultIndex === null) {
          setActiveResult(results.length - 1)
        } else if (activeResultIndex <= 0) {
          setActiveResult(null, { focus: true, scroll: false })
        } else {
          setActiveResult(activeResultIndex - 1)
        }
        return
      }

      if (key === "enter" && resultFocused) {
        e.preventDefault()
        ;(document.activeElement as HTMLElement)?.click()
        return
      }
    }

    if (key === "escape") {
      if (searchContainer && searchContainer.classList.contains("active")) {
        resetActiveResultHighlight()
        closeSearchDropdown()
      } else {
        closeModal()
      }
      return
    }

    if (e.key === "ArrowLeft") {
      navigateBlock(-1)
    } else if (e.key === "ArrowRight") {
      navigateBlock(1)
    }
  }

  document.addEventListener("click", onClick)
  document.addEventListener("keydown", onKey)
  window.addCleanup(() => document.removeEventListener("click", onClick))
  window.addCleanup(() => document.removeEventListener("keydown", onKey))
})
