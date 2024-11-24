const bufferPx = 150
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const slug = entry.target.id
    const tocEntryElement = document.querySelector(`button[data-for="${slug}"]`)
    const layout = (document.querySelector(".toc") as HTMLDivElement).dataset.layout
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

function onClick(evt: MouseEvent) {
  const button = evt.target as HTMLButtonElement
  if (!button) return

  const href = button.dataset.href
  if (!href?.startsWith("#")) return

  evt.preventDefault()
  scrollToElement(href)

  // Handle initial load with hash
  if (window.location.hash) {
    // Delay to ensure page is fully loaded
    setTimeout(() => {
      scrollToElement(window.location.hash)
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

function setupToc() {
  const toc = document.getElementById("toc")
  const body = document.getElementById("quartz-body")

  if (!toc || !body) return

  if (toc.dataset.layout === "minimal") {
    const nav = toc.querySelector("#toc-vertical") as HTMLElement
    if (!nav) return

    const buttons = toc?.querySelectorAll("button[data-for]") as NodeListOf<HTMLButtonElement>
    for (const button of buttons) {
      function onAnimation(e: AnimationEvent) {
        if (e.animationName === "fillExpand") {
          button.classList.add("animation-complete")
        }
      }

      button.addEventListener("click", onClick)
      button.addEventListener("animationend", onAnimation, { once: true })

      window.addCleanup(() => {
        button.removeEventListener("click", onClick)
        button.removeEventListener("animationend", onAnimation)
      })
    }
    const onMouseEnter = () => {
      body.classList.add("toc-hover")
    }

    const onMouseLeave = () => {
      body.classList.remove("toc-hover")
      resetFill()
    }

    toc.addEventListener("mouseenter", onMouseEnter)
    toc.addEventListener("mouseleave", onMouseLeave)

    window.addCleanup(() => {
      toc.removeEventListener("mouseenter", onMouseEnter)
      toc.removeEventListener("mouseleave", onMouseLeave)
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
