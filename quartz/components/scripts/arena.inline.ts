import { fetchCanonical } from "./util"
import { normalizeRelativeURLs, FullSlug } from "../../util/path"

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
    targetUrl.hash = ""
    const response = await fetchCanonical(targetUrl)
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

  modalBody.innerHTML = ""
  const clonedContent = dataEl.cloneNode(true) as HTMLElement
  clonedContent.style.display = "block"
  modalBody.appendChild(clonedContent)

  const twttr = (window as any).twttr
  if (twttr && typeof twttr.ready === "function") {
    twttr.ready((readyTwttr: any) => {
      if (readyTwttr?.widgets?.load) {
        readyTwttr.widgets.load(modalBody)
      }
    })
  } else if (twttr?.widgets?.load) {
    twttr.widgets.load(modalBody)
  }

  hydrateSubstackEmbeds(modalBody)
  hydrateInternalHosts(modalBody)

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
  document.body.style.overflow = "hidden"
}

function closeModal() {
  const modal = document.getElementById("arena-modal")
  if (modal) {
    modal.classList.remove("active")
    document.body.style.overflow = ""
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

// Search functionality
interface SearchIndexItem {
  blockId: string
  channelSlug?: string
  channelName?: string
  title: string
  content: string
  highlighted: boolean
  element: HTMLElement
}

let searchIndex: SearchIndexItem[] = []
let searchDebounceTimer: number | undefined

function buildSearchIndex(scope: "channel" | "index"): SearchIndexItem[] {
  const index: SearchIndexItem[] = []

  if (scope === "channel") {
    // Search within channel blocks
    const blocks = document.querySelectorAll<HTMLElement>(".arena-block[data-block-id]")
    blocks.forEach((block) => {
      const blockId = block.getAttribute("data-block-id")
      if (!blockId) return

      const content = block.querySelector(".arena-block-content")
      const title = content?.textContent?.trim() || ""
      const highlighted = block.classList.contains("highlighted")

      index.push({
        blockId,
        title,
        content: title,
        highlighted,
        element: block,
      })
    })
  } else {
    // Search across all channels
    const channelRows = document.querySelectorAll<HTMLElement>(".arena-channel-row")
    channelRows.forEach((row) => {
      const channelSlug = row.getAttribute("data-slug") || ""
      const channelNameEl = row.querySelector(".arena-channel-row-header h2")
      const channelName = channelNameEl?.textContent?.trim() || ""

      const previewItems = row.querySelectorAll<HTMLElement>(".arena-channel-row-preview-item")
      previewItems.forEach((item) => {
        const blockId = item.getAttribute("data-block-id")
        if (!blockId) return

        const textEl = item.querySelector(".arena-channel-row-preview-text")
        const title = textEl?.textContent?.trim() || ""
        const highlighted = item.classList.contains("highlighted")

        index.push({
          blockId,
          channelSlug,
          channelName,
          title,
          content: title,
          highlighted,
          element: item,
        })
      })
    })
  }

  return index
}

function performSearch(query: string, index: SearchIndexItem[]): SearchIndexItem[] {
  const lowerQuery = query.toLowerCase().trim()
  if (lowerQuery.length < 2) return []

  return index
    .filter((item) => {
      const titleMatch = item.title.toLowerCase().includes(lowerQuery)
      const contentMatch = item.content.toLowerCase().includes(lowerQuery)
      return titleMatch || contentMatch
    })
    .sort((a, b) => {
      // Prioritize title matches over content matches
      const aTitleMatch = a.title.toLowerCase().includes(lowerQuery)
      const bTitleMatch = b.title.toLowerCase().includes(lowerQuery)
      if (aTitleMatch && !bTitleMatch) return -1
      if (!aTitleMatch && bTitleMatch) return 1
      // Prioritize highlighted items
      if (a.highlighted && !b.highlighted) return -1
      if (!a.highlighted && b.highlighted) return 1
      return 0
    })
}

function renderSearchResults(results: SearchIndexItem[], scope: "channel" | "index") {
  const container = document.getElementById("arena-search-container")
  if (!container) return

  if (results.length === 0) {
    container.innerHTML = '<div class="arena-search-no-results">no results found</div>'
    container.classList.add("active")
    return
  }

  const fragment = document.createDocumentFragment()
  results.forEach((result) => {
    const resultItem = document.createElement("div")
    resultItem.className = "arena-search-result-item"
    resultItem.setAttribute("data-block-id", result.blockId)
    if (result.channelSlug) {
      resultItem.setAttribute("data-channel-slug", result.channelSlug)
    }

    const title = document.createElement("div")
    title.className = "arena-search-result-title"
    title.textContent = result.title

    const content = document.createElement("div")
    content.className = "arena-search-result-content"
    content.textContent = result.content

    resultItem.appendChild(title)
    resultItem.appendChild(content)

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

  // Build search index
  const searchInput = document.querySelector<HTMLInputElement>(".arena-search-input")
  if (searchInput) {
    const scope = searchInput.getAttribute("data-search-scope") as "channel" | "index"
    searchIndex = buildSearchIndex(scope)

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
          return
        }

        const results = performSearch(query, searchIndex)
        renderSearchResults(results, scope)
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
        showModal(blockId)
      }
      return
    }

    const previewItem = target.closest(".arena-channel-row-preview-item[data-block-id]")
    if (previewItem) {
      if (isArenaChannelPage()) {
        const blockId = (previewItem as HTMLElement).getAttribute("data-block-id")
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

    // Handle search result clicks
    const searchResultItem = target.closest(".arena-search-result-item") as HTMLElement | null
    if (searchResultItem) {
      const blockId = searchResultItem.getAttribute("data-block-id")
      const channelSlug = searchResultItem.getAttribute("data-channel-slug")

      if (blockId) {
        e.preventDefault()
        closeSearchDropdown()

        // If we're on the index page and clicked a result from another channel,
        // navigate to that channel first
        if (channelSlug) {
          const currentSlug = document.body?.dataset.slug || ""
          if (!currentSlug.startsWith(channelSlug)) {
            // Navigate to the channel page
            window.location.href = `/${channelSlug}`
            return
          }
        }

        // Show the block modal
        showModal(blockId)
      }
      return
    }

    if (target.closest(".arena-modal-close") || target.classList.contains("arena-block-modal")) {
      closeModal()
    }
  }

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      const searchContainer = document.getElementById("arena-search-container")
      if (searchContainer && searchContainer.classList.contains("active")) {
        closeSearchDropdown()
      } else {
        closeModal()
      }
    } else if (e.key === "ArrowLeft") {
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
