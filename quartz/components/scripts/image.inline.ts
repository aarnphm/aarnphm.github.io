import { getFullSlug } from "../../util/path"

document.addEventListener("nav", () => {
  if (getFullSlug(window) === "lyd") return

  const modal = document.getElementById("image-popup-modal")
  const modalImg = modal?.querySelector(".image-popup-img") as HTMLImageElement
  const closeBtn = modal?.querySelector(".image-popup-close")
  const backdrop = modal?.querySelector(".image-popup-backdrop")

  if (!modal || !modalImg || !closeBtn || !backdrop) return

  function closeModal() {
    modal!.classList.remove("active")
    document.body.style.overflow = ""
  }

  function openModal(imgSrc: string, imgAlt: string) {
    modalImg.src = imgSrc
    modalImg.alt = imgAlt
    modal!.classList.add("active")
    document.body.style.overflow = "hidden"
  }

  const imageHandlers = new WeakMap<HTMLImageElement, () => void>()

  function setupImageHandler(img: HTMLImageElement) {
    if (imageHandlers.has(img)) return

    img.style.cursor = "pointer"
    const popup = () => openModal(img.src, img.alt)
    img.addEventListener("click", popup)
    imageHandlers.set(img, popup)
  }

  // Add click handlers to all existing images
  const contentImages = document.querySelectorAll("img")
  for (const img of contentImages) {
    if (img instanceof HTMLImageElement) {
      setupImageHandler(img)
    }
  }

  // Watch for masonry images being added incrementally
  const masonryContainer = document.querySelector(".masonry-grid")
  let observer: MutationObserver | undefined

  if (masonryContainer) {
    observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (node instanceof HTMLImageElement) {
            setupImageHandler(node)
          }
        }
      }
    })

    observer.observe(masonryContainer, {
      childList: true,
      subtree: true,
    })
  }

  function keyboardHandler(e: any) {
    if (e.key === "Escape" && modal!.classList.contains("active")) closeModal()
  }

  closeBtn.addEventListener("click", closeModal)
  backdrop.addEventListener("click", closeModal)
  document.addEventListener("keydown", keyboardHandler)

  window.addCleanup(() => {
    closeBtn.removeEventListener("click", closeModal)
    backdrop.removeEventListener("click", closeModal)
    document.removeEventListener("keydown", keyboardHandler)

    if (observer) {
      observer.disconnect()
    }

    for (const img of contentImages) {
      if (img instanceof HTMLImageElement) {
        const handler = imageHandlers.get(img)
        if (handler) {
          img.removeEventListener("click", handler)
        }
      }
    }
  })
})
