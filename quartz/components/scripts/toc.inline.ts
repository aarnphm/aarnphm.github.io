import { annotate } from "rough-notation"
import type { RoughAnnotation } from "rough-notation/lib/model"

let ag: RoughAnnotation | null = null
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const slug = entry.target.id
    const tocEntryElement = document.querySelector(`button[data-for="${slug}"]`)
    const toc = document.querySelector(".toc") as HTMLDivElement
    if (!toc) continue
    const layout = toc.dataset.layout
    const windowHeight = entry.rootBounds?.height
    if (windowHeight && tocEntryElement) {
      if (layout === "minimal") {
        if (entry.boundingClientRect.y < windowHeight) {
          tocEntryElement.classList.add("in-view")
        } else {
          tocEntryElement.classList.remove("in-view")
        }
      } else {
        const parentLi = tocEntryElement.parentElement as HTMLLIElement
        if (entry.boundingClientRect.y < windowHeight) {
          tocEntryElement.classList.add("in-view")
          parentLi.classList.add("in-view")
        } else {
          tocEntryElement.classList.remove("in-view")
          parentLi.classList.remove("in-view")
        }
      }
    }
  }
})

// HACK: ugh we have to target the indicator class given that we fixed the padding
function onClick(evt: MouseEvent) {
  const indicator = evt.target as HTMLDivElement
  const button = indicator.parentElement as HTMLButtonElement

  const href = button.dataset.href
  if (!href?.startsWith("#")) return

  const body = document.body
  const toc = document.getElementById("toc")

  const afterClick = () => {
    if (body?.classList.contains("toc-hover")) {
      body.classList.remove("toc-hover")
    }
    const buttons = toc?.querySelectorAll("button[data-for]") as NodeListOf<HTMLButtonElement>
    buttons.forEach((button) => {
      const fill = button.querySelector(".fill") as HTMLElement
      if (fill) {
        fill.style.transform = "scaleX(1)"
        fill.style.opacity = "" // We need to reset this
      }
    })
  }

  evt.preventDefault()
  scrollToElement(href)
  afterClick()

  // Handle initial load with hash
  if (window.location.hash) {
    // Delay to ensure page is fully loaded
    setTimeout(() => {
      scrollToElement(window.location.hash)
      afterClick()
    }, 10)
  }
}

function scrollToElement(hash: string) {
  const elementId = hash.slice(1)
  const element = document.getElementById(elementId)
  if (!element) return

  // Check if element is inside a collapsible section
  const collapsibleParent = element.closest(".collapsible-header-content")
  if (collapsibleParent) {
    // Expand the collapsible section first
    const wrapper = collapsibleParent.closest(".collapsible-header")
    if (wrapper) {
      const button = wrapper.querySelector(".toggle-button") as HTMLButtonElement
      if (button && button.getAttribute("aria-expanded") === "false") {
        button.click()
      }
    }
  }

  if (ag) ag.hide()

  const shareOpts = {
    color: "rgba(234, 157, 52, 0.45)",
    iterations: 4,
    animationDuration: 800,
    strokeWidth: 2,
  }

  const highlight = element.querySelector("span.highlight-span") as HTMLElement
  ag = annotate(highlight, { type: "highlight", ...shareOpts })

  setTimeout(() => ag!.show(), 50)
  window.setTimeout(() => {
    if (ag) ag.hide()
  }, 2500)

  const rect = element.getBoundingClientRect()
  const absoluteTop = window.scrollY + rect.top

  // Scroll with offset for header
  window.scrollTo({
    top: absoluteTop - 100, // Offset for fixed header
    behavior: "smooth",
  })

  // Update URL without triggering scroll
  history.pushState(null, "", hash)
}

document.addEventListener("nav", (ev: CustomEventMap["nav"]) => {
  if (ev.detail.url) {
    const url = new URL(ev.detail.url, window.location.origin)
    if (url.hash) {
      const elHash = decodeURIComponent(url.hash)
      scrollToElement(elHash)
    }
  }
})

function setupToc() {
  const toc = document.getElementById("toc")
  const body = document.body

  if (!toc) return

  if (toc.dataset.layout === "minimal") {
    const nav = toc.querySelector("#toc-vertical") as HTMLElement
    if (!nav) return

    const buttons = toc?.querySelectorAll("button[data-for]") as NodeListOf<HTMLButtonElement>
    for (const button of buttons) {
      button.addEventListener("click", onClick)

      window.addCleanup(() => {
        button.removeEventListener("click", onClick)
      })
    }

    const onMouseEnter = () => {
      body.classList.add("toc-hover")
    }

    const onMouseLeave = () => {
      body.classList.remove("toc-hover")

      // Reset fill transformations when mouse leaves
      buttons.forEach((button) => {
        const fill = button.querySelector(".fill") as HTMLElement
        if (fill) {
          fill.style.transform = "scaleX(1)"
          fill.style.opacity = "" // We need to reset this
        }
      })
    }

    // Add mousemove handler for dynamic fill animation
    const onMouseMove = (evt: MouseEvent) => {
      const navRect = nav.getBoundingClientRect()
      const mouseY = evt.clientY - navRect.top

      buttons.forEach((button) => {
        const buttonRect = button.getBoundingClientRect()
        const buttonY = buttonRect.top + buttonRect.height / 2 - navRect.top

        const styles = getComputedStyle(button)

        const distance = mouseY - buttonY
        const sigma = 42
        const maxScale = parseFloat(styles.getPropertyValue("--indicator-position"))
        const isButton = Math.abs(distance) < buttonRect.height / 2

        const fill = button.querySelector(".fill") as HTMLElement
        const minScale = parseInt(styles.getPropertyValue("--fill-width"))
        fill.style.animation = "unset !important"

        fill.style.opacity = isButton ? "1" : "0.35"
        // If the button is under the cursor, set the scale to the maximum value
        // Otherwise, apply Gaussian scaling based on distance
        fill.style.transform = `scaleX(${isButton ? maxScale : minScale + (maxScale - minScale) * Math.exp(-Math.pow(distance, 2) / (2 * Math.pow(sigma, 2)))})`
      })
    }

    toc.addEventListener("mouseenter", onMouseEnter)
    toc.addEventListener("mouseleave", onMouseLeave)
    nav.addEventListener("mousemove", onMouseMove)

    window.addCleanup(() => {
      toc.removeEventListener("mouseenter", onMouseEnter)
      toc.removeEventListener("mouseleave", onMouseLeave)
      nav.removeEventListener("mousemove", onMouseMove)
    })
  }
}

window.addEventListener("resize", setupToc)
document.addEventListener("nav", () => {
  setupToc()

  // update toc entry highlighting
  observer.disconnect()
  const headers = document.querySelectorAll("h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]")
  headers.forEach((header) => observer.observe(header))
})
