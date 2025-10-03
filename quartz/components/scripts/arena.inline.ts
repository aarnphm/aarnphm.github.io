let currentBlockIndex = 0
let totalBlocks = 0

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

function hydrateSubstackEmbeds(root: HTMLElement) {
  const nodes = root.querySelectorAll<HTMLElement>(
    ".arena-modal-embed-substack[data-substack-url]:not([data-substack-status])",
  )
  nodes.forEach((node) => {
    const targetUrl = node.dataset.substackUrl
    if (!targetUrl) return
    node.dataset.substackStatus = "loading"
    node.textContent = "Loading Substack preview..."
    fetchSubstackEmbed(targetUrl)
      .then((payload) => {
        if (!node.isConnected) return
        renderSubstackEmbed(node, payload)
        node.dataset.substackStatus = "loaded"
      })
      .catch(() => {
        if (!node.isConnected) return
        node.textContent = "Unable to load Substack preview."
        node.dataset.substackStatus = "error"
      })
  })
}

function showModal(blockId: string) {
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

document.addEventListener("nav", () => {
  totalBlocks = document.querySelectorAll("[data-block-id][data-block-index]").length

  const onClick = (e: MouseEvent) => {
    const target = e.target as HTMLElement

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
      const blockId = previewItem.getAttribute("data-block-id")
      if (blockId) {
        e.preventDefault()
        showModal(blockId)
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
