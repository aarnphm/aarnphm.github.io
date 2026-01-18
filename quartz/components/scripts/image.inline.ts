import { getFullSlug } from "../../util/path"
import { registerEscapeHandler } from "./util"

document.addEventListener("nav", () => {
  if (getFullSlug(window) === "lyd") return

  const dialog = document.getElementById("image-popup-modal") as HTMLDialogElement | null
  const modalImg = dialog?.querySelector<HTMLImageElement>(".image-popup-img")
  const closeBtn = dialog?.querySelector<HTMLButtonElement>(".image-popup-close")

  if (!dialog || !modalImg || !closeBtn) return

  function closeModal() {
    if (dialog!.open) dialog!.close()
  }

  function openModal(imgSrc: string, imgAlt: string) {
    modalImg!.src = imgSrc
    modalImg!.alt = imgAlt
    if (!dialog!.open) dialog!.showModal()
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

  const backdropClickHandler = (e: MouseEvent) => {
    if (e.target === dialog) closeModal()
  }

  closeBtn!.addEventListener("click", closeModal)
  dialog!.addEventListener("click", backdropClickHandler)
  registerEscapeHandler(closeBtn!, closeModal)

  window.addCleanup(() => {
    closeBtn!.removeEventListener("click", closeModal)
    dialog!.removeEventListener("click", backdropClickHandler)

    if (observer) observer.disconnect()

    for (const img of contentImages) {
      if (img instanceof HTMLImageElement) {
        const handler = imageHandlers.get(img)
        if (handler) img.removeEventListener("click", handler)
      }
    }
  })
})
