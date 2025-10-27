import { fetchCanonical, tokenizeTerm, highlight } from "./util"
import { normalizeRelativeURLs } from "../../util/path"

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
      })
    })
  }

  return index
}

function performSearch(query: string, index: SearchIndexItem[]): SearchIndexItem[] {
  const lowerQuery = query.toLowerCase().trim()
  if (lowerQuery.length < 2) return []

  // Tokenize search query for better matching
  const tokens = tokenizeTerm(lowerQuery)

  // Score each item based on token matches
  const scoredResults = index
    .map((item) => {
      const lowerTitle = item.title.toLowerCase()
      const lowerContent = item.content.toLowerCase()

      let score = 0
      let titleMatchCount = 0
      let contentMatchCount = 0

      // Check each token
      for (const token of tokens) {
        const tokenLower = token.toLowerCase()

        // Title matches are weighted higher
        if (lowerTitle.includes(tokenLower)) {
          titleMatchCount++
          score += 10
        }

        // Content matches
        if (lowerContent.includes(tokenLower)) {
          contentMatchCount++
          score += 5
        }
      }

      // Boost for exact phrase match
      if (lowerTitle.includes(lowerQuery)) {
        score += 20
      }
      if (lowerContent.includes(lowerQuery)) {
        score += 10
      }

      // Boost for highlighted items
      if (item.highlighted) {
        score += 3
      }

      // Must have at least one match
      if (score === 0) return null

      return { item, score, titleMatchCount, contentMatchCount }
    })
    .filter((result): result is NonNullable<typeof result> => result !== null)
    .sort((a, b) => {
      // Sort by score descending
      if (b.score !== a.score) return b.score - a.score

      // Tie-breaker: more title matches first
      if (b.titleMatchCount !== a.titleMatchCount) {
        return b.titleMatchCount - a.titleMatchCount
      }

      // Tie-breaker: more content matches
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
    // Highlight matches in title
    title.innerHTML = searchQuery ? highlight(searchQuery, result.title) : result.title

    const content = document.createElement("div")
    content.className = "arena-search-result-content"
    // Highlight matches in content with trimming to show context
    content.innerHTML = searchQuery ? highlight(searchQuery, result.content, true) : result.content

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

    // Handle search result clicks with smart navigation based on hasModalInDom
    const searchResultItem = target.closest(".arena-search-result-item") as HTMLElement | null
    if (searchResultItem) {
      const blockId = searchResultItem.getAttribute("data-block-id")
      const channelSlug = searchResultItem.getAttribute("data-channel-slug")

      if (blockId) {
        e.preventDefault()
        clearSearchState({ blur: true })

        // Find the result in search index to check hasModalInDom flag
        const resultData = searchIndex.find((item) => item.blockId === blockId)

        if (resultData?.hasModalInDom) {
          // Block has prerendered modal in DOM - show it instantly
          showModal(blockId)
        } else if (channelSlug) {
          // Block doesn't have modal in DOM - navigate to channel page
          const currentSlug = document.body?.dataset.slug || ""
          const targetChannelSlug = `arena/${channelSlug}`

          if (currentSlug === targetChannelSlug) {
            // Already on the channel page - try to show modal
            showModal(blockId)
          } else {
            // Navigate to channel page with hash to auto-open modal
            window.location.href = `/${targetChannelSlug}#${blockId}`
          }
        } else {
          // Fallback: try to show modal anyway
          showModal(blockId)
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
      if (key === "arrowdown" || (!e.shiftKey && key === "tab")) {
        e.preventDefault()
        if (
          !e.shiftKey &&
          key === "tab" &&
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

      if (key === "arrowup" || (e.shiftKey && key === "tab")) {
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
