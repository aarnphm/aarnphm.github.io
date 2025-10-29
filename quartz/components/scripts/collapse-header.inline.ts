import { setHeaderState } from "./util"
import { getFullSlug } from "../../util/path"

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

  localStorage.setItem(
    `${getFullSlug(window).replace("/", "--")}-${toggleButton.id}`,
    isCollapsed ? "false" : "true",
  )
}

function setupHeaders() {
  const collapsibleHeaders = document.querySelectorAll("section.collapsible-header")

  for (const header of collapsibleHeaders) {
    const button = header.querySelector<HTMLButtonElement>("span.toggle-button")
    if (button) {
      button.addEventListener("click", toggleHeader)
      window.addCleanup(() => button.removeEventListener("click", toggleHeader))

      // Apply saved state
      const content = document.querySelector<HTMLElement>(
        `.collapsible-header-content[data-references="${button.id}"]`,
      )
      // setup once
      if (content) {
        const savedState = localStorage.getItem(
          `${getFullSlug(window).replace("/", "--")}-${button.id}`,
        )
        if (savedState) {
          setHeaderState(
            button as HTMLElement,
            content,
            header as HTMLElement,
            savedState === "false",
          )
        }
        const collapsed = content.classList.contains("collapsed")
        content.style.maxHeight = collapsed ? `0px` : `inherit`
      }
    }
  }

  const links = document.querySelectorAll("button.transclude-title-link") as NodeListOf<SVGElement>
  for (const link of links) {
    const parentEl = link.parentElement as HTMLElement
    if (!parentEl || !parentEl.dataset.href) continue

    const href = parentEl.dataset.href

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
