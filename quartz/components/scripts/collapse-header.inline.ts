import { isInViewport, updatePosition } from "./util"

interface HeaderState {
  id: string
  collapsed: boolean
}

type MaybeHTMLElement = HTMLElement | undefined
let currentHeaderState: HeaderState[] = []

function toggleHeader(evt: Event) {
  evt.stopPropagation()
  const target = evt.target as MaybeHTMLElement
  if (!target) return

  const button = target.closest("button") as MaybeHTMLElement
  if (!button) return

  const wrapper = button.closest(".collapsible-header") as MaybeHTMLElement
  if (!wrapper) return

  // Find content by data-references
  const content = document.querySelector(
    `.collapsible-header-content[data-references="${button.id}"]`,
  ) as MaybeHTMLElement
  if (!content) return

  const isCollapsed = button.getAttribute("aria-expanded") === "true"

  // Toggle current header
  button.setAttribute("aria-expanded", isCollapsed ? "false" : "true")
  content.style.maxHeight = isCollapsed ? "0px" : `${content.scrollHeight}px`
  wrapper.classList.toggle("collapsed", isCollapsed)
  button.classList.toggle("collapsed", isCollapsed)

  updateSidenoteState(content, isCollapsed)

  // Update state
  const headerId = button.id
  toggleCollapsedById(currentHeaderState, headerId)
  saveHeaderState()
}

function updateSidenoteState(content: HTMLElement, isCollapsed: boolean) {
  // handle sidenotes state
  const sidenoteRefs = content.querySelectorAll("a[data-footnote-ref]") as NodeListOf<HTMLElement>
  const sideContainer = document.querySelector(".sidenotes") as HTMLElement | null
  if (!sideContainer) return
  for (const ref of sidenoteRefs) {
    const sideId = ref.getAttribute("href")?.replace("#", "sidebar-")
    const sidenote = document.getElementById(sideId!)
    if (isCollapsed) {
      sidenote?.classList.remove("in-view")
      ref.classList.remove("active")
      sidenote?.classList.add("collapsed")
    } else if (isInViewport(ref)) {
      sidenote?.classList.add("in-view")
      ref.classList.add("active")
      sidenote?.classList.remove("collapsed")
      updatePosition(ref, sidenote!, sideContainer!)
    }
  }
}

function toggleCollapsedById(array: HeaderState[], id: string) {
  const entry = array.find((item) => item.id === id)
  if (entry) {
    entry.collapsed = !entry.collapsed
  } else {
    array.push({ id, collapsed: true })
  }
}

function saveHeaderState() {
  localStorage.setItem("headerState", JSON.stringify(currentHeaderState))
}

function loadHeaderState(): HeaderState[] {
  const saved = localStorage.getItem("headerState")
  return saved ? JSON.parse(saved) : []
}

function setHeaderState(button: HTMLElement, content: HTMLElement, collapsed: boolean) {
  button.setAttribute("aria-expanded", collapsed ? "false" : "true")
  button.classList.toggle("collapsed", collapsed)
  content.style.maxHeight = collapsed ? "0px" : `${content.scrollHeight}px`
  button.closest(".collapsible-header")?.classList.toggle("collapsed", collapsed)
  updateSidenoteState(content, collapsed)
}

function setupHeaders() {
  // Load saved state
  currentHeaderState = loadHeaderState()

  // Set up click handlers
  const buttons = document.querySelectorAll(".header-button")
  buttons.forEach((button) => {
    button.addEventListener("click", toggleHeader)
    window.addCleanup(() => button.removeEventListener("click", toggleHeader))

    // Apply saved state
    const savedState = currentHeaderState.find((state) => state.id === button.id)
    if (savedState) {
      const content = button.querySelector(
        `.collapsible-header-content[data-references="${button.id}"]`,
      ) as HTMLElement
      if (content) {
        setHeaderState(button as HTMLElement, content, savedState.collapsed)
      }
    }
  })
}

// Set up initial state and handle navigation
document.addEventListener("nav", setupHeaders)
window.addEventListener("resize", setupHeaders)
