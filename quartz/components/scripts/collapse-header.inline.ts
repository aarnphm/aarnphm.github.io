import {
  updateSidenoteState,
  getCollapsedState,
  CollapsedState,
  setCollapsedState,
  setHeaderState,
  updateContainerHeights,
  debounce,
} from "./util"

type MaybeHTMLElement = HTMLElement | undefined

const debouncedHeights = debounce(updateContainerHeights, 150)

function toggleHeader(evt: Event) {
  const target = evt.target as MaybeHTMLElement
  if (!target) return

  // Only proceed if we clicked on the toggle button or its children (svg, lines)
  const toggleButton = target.closest(".toggle-button") as MaybeHTMLElement
  if (!toggleButton) return

  // Check if we're inside a callout - if so, don't handle the event
  if (target.parentElement!.classList.contains("callout")) return

  const headerId = toggleButton.id.replace("collapsible-header-", "").replace("-toggle", "")

  const wrapper = document.querySelector(
    `section.collapsible-header[id="${headerId}"]`,
  ) as MaybeHTMLElement
  if (!wrapper) return

  evt.stopPropagation()

  // Find content by data-references
  const content = document.querySelector(
    `.collapsible-header-content[data-references="${toggleButton.id}"]`,
  ) as MaybeHTMLElement
  if (!content) return

  const isCollapsed = toggleButton.getAttribute("aria-expanded") === "true"

  // Toggle current header
  toggleButton.setAttribute("aria-expanded", isCollapsed ? "false" : "true")
  content.style.maxHeight = isCollapsed ? "0px" : `${content.scrollHeight}px`
  content.classList.toggle("collapsed", isCollapsed)
  wrapper.classList.toggle("collapsed", isCollapsed)
  toggleButton.classList.toggle("collapsed", isCollapsed)

  updateSidenoteState(content, isCollapsed)
  setCollapsedState(window, toggleButton.id, isCollapsed ? "false" : ("true" as CollapsedState))

  requestAnimationFrame(() => {
    updateContainerHeights()
    debouncedHeights()
  })
}

function setupHeaders() {
  const collapsibleHeaders = document.querySelectorAll(".collapsible-header")

  for (const header of collapsibleHeaders) {
    const button = header.querySelector("span.toggle-button") as HTMLButtonElement
    if (button) {
      button.addEventListener("click", toggleHeader)
      if (window.addCleanup) {
        window.addCleanup(() => button.removeEventListener("click", toggleHeader))
      }

      // Apply saved state
      const content = document.querySelector(
        `.collapsible-header-content[data-references="${button.id}"]`,
      ) as HTMLElement
      // setup once
      if (content) {
        const savedState = getCollapsedState(window, button.id)
        if (savedState) {
          setHeaderState(
            button as HTMLElement,
            content,
            header as HTMLElement,
            savedState === "false",
          )
        }
      }
      const collapsed = content.classList.contains("collapsed")
      const height = collapsed ? 0 : content.parentElement!.scrollHeight
      content.style.maxHeight = `${height}px`
    }
  }

  const links = document.querySelectorAll("svg.blockquote-link") as NodeListOf<SVGElement>
  for (const link of links) {
    const parentEl = link.parentElement as HTMLElement
    const href = parentEl.dataset.href as string

    function onClick() {
      window.spaNavigate(new URL(href, window.location.toString()))
    }

    link.addEventListener("click", onClick)
    if (window.addCleanup) {
      window.addCleanup(() => link.removeEventListener("click", onClick))
    }
  }
}

// Set up initial state and handle navigation
document.addEventListener("nav", setupHeaders)
window.addEventListener("resize", setupHeaders)
