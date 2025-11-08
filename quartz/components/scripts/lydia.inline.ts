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
  window.addEventListener(
    "scroll",
    () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          updateMaskOpacity()
          ticking = false
        })
        ticking = true
      }
    },
    { passive: true },
  )
}

function initHoverModal() {
  const root = document.querySelector("[data-slug='lyd']") as HTMLElement
  if (!root) return

  // Create modal element
  let modal = document.querySelector(".lydia-modal") as HTMLElement
  if (!modal) {
    modal = document.createElement("div")
    modal.className = "lydia-modal"
    root.appendChild(modal)
  }

  // Find all timeline items with hover text
  const timelineItems = root.querySelectorAll(".timeline-item[data-hover-text]")

  timelineItems.forEach((item) => {
    const hoverText = item.getAttribute("data-hover-text")
    if (!hoverText) return

    const content = item.querySelector(".timeline-content") as HTMLElement
    if (!content) return

    const onMouseEnter = () => {
      modal.textContent = hoverText
      modal.classList.add("active")
    }

    const onMouseLeave = () => {
      modal.classList.remove("active")
    }

    const onMouseMove = (e: MouseEvent) => {
      // Position modal near cursor with offset
      modal.style.left = `${e.pageX + 15}px`
      modal.style.top = `${e.pageY + 15}px`
    }

    content.addEventListener("mouseenter", onMouseEnter)
    content.addEventListener("mouseleave", onMouseLeave)
    content.addEventListener("mousemove", onMouseMove)
  })
}

// Initialize when DOM is ready
document.addEventListener("nav", () => {
  const slug = getFullSlug(window)
  if (slug === "lyd") {
    initScrollMask()
    initHoverModal()
  }
})

// Also run on initial page load
if (getFullSlug(window) === "lyd") {
  initScrollMask()
  initHoverModal()
}
