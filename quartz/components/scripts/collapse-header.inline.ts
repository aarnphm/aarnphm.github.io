import { getCollapsedState, CollapsedState, setCollapsedState, setHeaderState } from "./util"

type MaybeHTMLElement = HTMLElement | undefined

function toggleHeader(evt: Event) {
  const target = evt.target as MaybeHTMLElement
  if (!target) return

  // Only proceed if we clicked on the toggle button or its children (svg, lines)
  const toggleButton = target.closest(".toggle-button") as MaybeHTMLElement
  if (!toggleButton) return

  const wrapper = toggleButton.closest(".collapsible-header") as MaybeHTMLElement
  if (!wrapper) return

  evt.stopPropagation()

  // Find content by data-references
  const content = document.querySelector(
    `.collapsible-header-content[data-references="${toggleButton.id}"]`,
  ) as MaybeHTMLElement
  if (!content) return

  const newCollapseState = toggleButton.getAttribute("aria-expanded") === "true"
  setHeaderState(toggleButton, content, newCollapseState)
  setCollapsedState(window, toggleButton.id, (!newCollapseState).toString() as CollapsedState)
}

function setupHeaders() {
  const collapsibleHeaders = document.getElementsByClassName("collapsible-header")

  for (const header of collapsibleHeaders) {
    const button = header.querySelector("button.toggle-button") as HTMLButtonElement
    if (button) {
      button.addEventListener("click", toggleHeader)
      window.addCleanup(() => button.removeEventListener("click", toggleHeader))

      // Apply saved state
      const savedState = getCollapsedState(window, button.id)
      const content = document.querySelector(
        `.collapsible-header-content[data-references="${button.id}"]`,
      ) as HTMLElement
      // setup once
      const isCollapsed = savedState === "true"
      button.setAttribute("aria-expanded", isCollapsed ? "false" : "true")
      if (isCollapsed) {
        button.classList.add("collapsed")
        content.classList.add("collapsed")
        content.style.maxHeight = "0px"
        button.closest(".collapsible-header")?.classList.add("collapsed")
      } else {
        button.classList.remove("collapsed")
        content.classList.remove("collapsed")
        content.style.maxHeight = ""
        button.closest(".collapsible-header")?.classList.remove("collapsed")
      }
    }
  }
}

// Set up initial state and handle navigation
document.addEventListener("nav", setupHeaders)
window.addEventListener("resize", setupHeaders)
