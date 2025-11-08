import { getFullSlug } from "../../util/path"

function initScrollMask() {
  const root = document.querySelector("[data-slug='lyd']") as HTMLElement
  if (!root) return

  function updateMaskOpacity() {
    const scrollY = window.scrollY
    const documentHeight = document.documentElement.scrollHeight
    const viewportHeight = window.innerHeight
    
    // Calculate scroll progress (0 to 1)
    const maxScroll = documentHeight - viewportHeight
    const scrollProgress = Math.min(scrollY / maxScroll, 1)
    
    // Calculate opacity: start at 1, fade to 0.3 as user scrolls
    // This reveals more of the background image
    const opacity = Math.max(0.3, 1 - scrollProgress * 0.7)
    
    // Update the CSS custom property
    root.style.setProperty("--mask-opacity", opacity.toString())
  }

  // Initial check
  updateMaskOpacity()

  // Update on scroll with throttling for better performance
  let ticking = false
  window.addEventListener("scroll", () => {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        updateMaskOpacity()
        ticking = false
      })
      ticking = true
    }
  }, { passive: true })
}

// Initialize when DOM is ready
document.addEventListener("nav", () => {
  const slug = getFullSlug(window)
  if (slug === "lyd") {
    initScrollMask()
  }
})

// Also run on initial page load
if (getFullSlug(window) === "lyd") {
  initScrollMask()
}
