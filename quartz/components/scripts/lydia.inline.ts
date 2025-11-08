// Scrolling mask effect for CN Tower background
function initScrollMask() {
  const heroSection = document.querySelector(".hero-section") as HTMLElement
  if (!heroSection) return

  function updateMaskOpacity() {
    const scrollY = window.scrollY
    const viewportHeight = window.innerHeight

    // Calculate opacity based on scroll position
    // Fade from 1 to 0 as user scrolls through the first viewport
    const opacity = Math.max(0, 1 - scrollY / viewportHeight)

    const afterElement = heroSection
    if (afterElement) {
      afterElement.style.setProperty("--mask-opacity", opacity.toString())
    }
  }

  // Initial check
  updateMaskOpacity()

  // Update on scroll
  window.addEventListener("scroll", updateMaskOpacity, { passive: true })
}

// Initialize when DOM is ready
document.addEventListener("nav", () => {
  const isLydiaPage = document.body.getAttribute("data-slug") === "lydia"
  if (isLydiaPage) {
    initScrollMask()
  }
})

// Also run on initial page load
if (document.body.getAttribute("data-slug") === "lydia") {
  initScrollMask()
}
