document.addEventListener("nav", () => {
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

  // Add click handlers to all images in content
  const contentImages = document.querySelectorAll(".popover-hint img:not(.external-icon)")
  for (const img of contentImages) {
    if (img instanceof HTMLImageElement) {
      img.style.cursor = "pointer"
      const popup = () => openModal(img.src, img.alt)
      img.addEventListener("click", popup)
      window.addCleanup(() => img.removeEventListener("click", popup))
    }
  }

  closeBtn.addEventListener("click", closeModal)
  backdrop.addEventListener("click", closeModal)
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal()
  })
  window.addCleanup(() => {
    closeBtn.removeEventListener("click", closeModal)
    backdrop.removeEventListener("click", closeModal)
  })
})
