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

  // Auto-populate internal targets (e.g., books) with popover-hint content
  try {
    const mainContent = modalBody.querySelector(".arena-modal-main-content") as HTMLElement | null
    const hasIframe = Boolean(mainContent?.querySelector("iframe"))
    const firstInternal = mainContent?.querySelector(
      "a.internal:not([data-no-popover])",
    ) as HTMLAnchorElement | null
    if (!hasIframe && firstInternal) {
      await renderPopoverIntoModal(firstInternal)
    }
  } catch (e) {
    console.error(e)
  }

  updateNavButtons()
  modal.classList.add("active")
  document.body.style.overflow = "hidden"
}

async function renderPopoverIntoModal(link: HTMLAnchorElement) {
  const modal = document.getElementById("arena-modal")
  const main = modal?.querySelector(".arena-modal-main") as HTMLElement | null
  const container = modal?.querySelector(".arena-modal-main-content") as HTMLDivElement | null
  if (!modal || !main || !container) return

  if (link.dataset.noPopover === "" || link.dataset.noPopover === "true") {
    // respect opt-out
    return
  }

  const targetUrl = new URL(link.href)
  const hash = decodeURIComponent(targetUrl.hash)
  targetUrl.hash = ""
  targetUrl.search = ""

  let response: Response | void
  if (link.dataset.arxivId) {
    const url = new URL(`https://aarnphm.xyz/api/arxiv?identifier=${link.dataset.arxivId}`)
    response = await fetchCanonical(url).catch(console.error)
  } else {
    response = await fetchCanonical(new URL(targetUrl.toString())).catch(console.error)
  }
  if (!response) return

  // Only handle HTML; otherwise fall back to normal navigation
  const headerContentType = response.headers.get("Content-Type")
  const contentType = headerContentType?.split(";")[0]
  if (contentType && !contentType.startsWith("text/html")) {
    return
  }

  const contents = await response.text()
  const html = p.parseFromString(contents, "text/html")
  normalizeRelativeURLs(html, targetUrl)
  html.querySelectorAll("[id]").forEach((el) => {
    const targetID = `popover-${el.id}`
    el.id = targetID
  })
  const elts = [...(html.getElementsByClassName("popover-hint") as HTMLCollectionOf<HTMLElement>)]
  if (elts.length === 0) return

  // Replace main content with fetched preview
  container.innerHTML = ""
  for (const el of elts) {
    // import into current document to preserve eventing
    container.appendChild(document.importNode(el, true))
  }

  // wire up popover + SPA behaviors for newly inserted nodes
  const slug = (link.dataset.slug || link.getAttribute("href") || "") as FullSlug
  if (slug) {
    window.notifyNav(slug)
  }

  // If clicking an in-page hash, scroll inside the inserted preview
  if (hash) {
    const targetAnchor = hash.startsWith("#popover") ? hash : `#popover-${hash.slice(1)}`
    const heading = container.querySelector(targetAnchor) as HTMLElement | null
    if (heading) heading.scrollIntoView({ behavior: "smooth" })
  }
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

document.addEventListener("nav", () => {
  totalBlocks = document.querySelectorAll("[data-block-id][data-block-index]").length

  const onClick = (e: MouseEvent) => {
    const target = e.target as HTMLElement
    const isArenaChannelPage = () => {
      const slug = document.body?.dataset.slug || ""
      return slug.startsWith("arena/") && slug !== "arena"
    }

    // Internal links inside the arena modal should show embedded popover content
    const internalInModal = target.closest(
      ".arena-modal-body a.internal",
    ) as HTMLAnchorElement | null
    if (internalInModal) {
      e.preventDefault()
      e.stopPropagation()
      renderPopoverIntoModal(internalInModal)
      return
    }

    const copyButton = target.closest("span.arena-url-copy-button") as HTMLElement | null
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

    if (target.closest(".arena-modal-close") || target.classList.contains("arena-block-modal")) {
      closeModal()
    }
  }

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      closeModal()
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
