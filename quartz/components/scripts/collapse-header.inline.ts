import { getFullSlug } from "../../util/path"
import { getCollapsedState, setCollapsedState, setHeaderState } from "./util"

type MaybeHTMLElement = HTMLElement | undefined

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

  setCollapsedState(window, toggleButton.id, isCollapsed ? "false" : "true")
}

function setupHeaders() {
  const collapsibleHeaders = document.querySelectorAll("section.collapsible-header")

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
      content.style.maxHeight = collapsed ? `0px` : `inherit`
    }
  }

  const links = document.querySelectorAll("button.transclude-title-link") as NodeListOf<SVGElement>
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

  const transcludedItem = document.querySelectorAll<HTMLElement>(
    "article > section[data-footnotes], article > section[data-footnotes]",
  )
  if (transcludedItem.length > 0) {
    const pageFooter = document.querySelector<HTMLElement>('section[class~="page-footer"]')

    if (pageFooter) {
      Array.from(transcludedItem).forEach((item) => {
        pageFooter!.insertBefore(item, pageFooter!.firstChild)
      })
    }
  }
}

// Set up initial state and handle navigation
document.addEventListener("nav", setupHeaders)
window.addEventListener("resize", setupHeaders)

// Add overlay to section[class~="header"] once scrolling
function setupHeaderOverlay() {
  const header = document.querySelector('section[class~="header"]') as HTMLElement
  if (!header || getFullSlug(window) === "index") return

  function handleScroll() {
    const asidePanel = document.querySelector<HTMLDivElement>(
      "main > * > aside[class~='sidepanel-container']",
    )
    if (asidePanel && asidePanel.classList.contains("active")) return
    // Add a 50px threshold
    if (window.scrollY > 50) {
      header.classList.add("overlay")
    } else {
      header.classList.remove("overlay")
    }
  }

  // Initial check
  handleScroll()

  // Add scroll event listener
  window.addEventListener("scroll", handleScroll)
  if (window.addCleanup) {
    window.addCleanup(() => window.removeEventListener("scroll", handleScroll))
  }
}

// Initialize header overlay
document.addEventListener("nav", setupHeaderOverlay)
window.addEventListener("resize", setupHeaderOverlay)
