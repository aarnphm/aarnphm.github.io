let currentBlockIndex = 0
let totalBlocks = 0

function showModal(blockId: string) {
  const modal = document.getElementById("arena-modal")
  const modalBody = modal?.querySelector(".arena-modal-body")
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

  const onClose = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      closeModal()
    } else if (e.key === "ArrowLeft") {
      navigateBlock(-1)
    } else if (e.key === "ArrowRight") {
      navigateBlock(1)
    }
  }

  document.addEventListener("click", onClick)
  document.addEventListener("keydown", onClose)
  window.addCleanup(() => document.removeEventListener("click", onClick))
  window.addCleanup(() => document.removeEventListener("keydown", onClose))
})
