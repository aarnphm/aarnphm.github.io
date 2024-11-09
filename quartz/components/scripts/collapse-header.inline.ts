import { updateSidenoteState, toggleCollapsedById, saveHeaderState, HeaderState } from "./util"

type MaybeHTMLElement = HTMLElement | undefined

function toggleHeader(evt: Event) {
  const target = evt.target as MaybeHTMLElement
  if (!target) return

  // Only proceed if we clicked on the toggle button or its children (svg, lines)
  const toggleButton = target.closest(".toggle-button") as MaybeHTMLElement
  if (!toggleButton) return

  // Check if we're inside a callout - if so, don't handle the event
  if (target.parentElement!.classList.contains("callout")) return

  const wrapper = toggleButton.closest(".collapsible-header") as MaybeHTMLElement
  if (!wrapper) return

  const currentHeaderState = loadHeaderState()

  evt.stopPropagation()

  // Find content by data-references
  const content = document.querySelector(
    `.collapsible-header-content[data-references="${toggleButton.id}"]`,
  ) as MaybeHTMLElement
  if (!content) return

  const isCollapsed = toggleButton.getAttribute("aria-expanded") === "true"

  // Toggle current header
  toggleButton.setAttribute("aria-expanded", isCollapsed ? "false" : "true")
  content.style.maxHeight = isCollapsed ? "0px" : "inherit"
  content.classList.toggle("collapsed", isCollapsed)
  wrapper.classList.toggle("collapsed", isCollapsed)
  toggleButton.classList.toggle("collapsed", isCollapsed)

  updateSidenoteState(content, isCollapsed)

  // Update state
  const headerId = toggleButton.id
  toggleCollapsedById(currentHeaderState, headerId)
  saveHeaderState()
}

function loadHeaderState(): HeaderState[] {
  const saved = localStorage.getItem("headerState")
  return saved ? JSON.parse(saved) : []
}

function setHeaderState(button: HTMLElement, content: HTMLElement, collapsed: boolean) {
  button.setAttribute("aria-expanded", collapsed ? "false" : "true")
  button.classList.toggle("collapsed", collapsed)
  content.style.maxHeight = collapsed ? "0px" : "inherit"
  content.classList.toggle("collapsed", collapsed)
  button.closest(".collapsible-header")?.classList.toggle("collapsed", collapsed)
  updateSidenoteState(content, collapsed)
}

function setupHeaders() {
  // Load saved state
  const currentHeaderState = loadHeaderState()

  const buttons = document.querySelectorAll(".collapsible-header .toggle-button")
  for (const button of buttons) {
    button.addEventListener("click", toggleHeader)
    window.addCleanup(() => button.removeEventListener("click", toggleHeader))

    // Apply saved state
    const savedState = currentHeaderState.find((state) => state.id === button.id)
    if (savedState) {
      const content = document.querySelector(
        `.collapsible-header-content[data-references="${button.id}"]`,
      ) as HTMLElement
      if (content) {
        setHeaderState(button as HTMLElement, content, savedState.collapsed)
      }
    }
  }
}

// Set up initial state and handle navigation
document.addEventListener("nav", setupHeaders)
window.addEventListener("resize", setupHeaders)
