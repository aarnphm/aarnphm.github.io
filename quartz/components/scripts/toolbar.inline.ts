import { updateSidenoteState, toggleCollapsedById, saveHeaderState, HeaderState } from "./util"

function toggleCollapse(button: HTMLButtonElement, forceState?: boolean) {
  const currentHeaderState: HeaderState[] = JSON.parse(localStorage.getItem("headerState") ?? "[]")

  const allToggleButtons = document.querySelectorAll(
    ".collapsible-header .toggle-button",
  ) as NodeListOf<HTMLElement>
  if (!allToggleButtons.length) return

  const isExpanded = button.getAttribute("data-state") === "expanded"
  const targetState = forceState !== undefined ? forceState : isExpanded
  button.setAttribute("data-state", targetState ? "collapsed" : "expanded")

  const tooltip = button.querySelector(".tooltip") as HTMLElement
  tooltip.textContent = isExpanded ? "Collapse all sections" : "Expand all sections"

  // Process each collapsible header
  for (const toggleButton of allToggleButtons) {
    const wrapper = toggleButton.closest(".collapsible-header") as HTMLElement
    if (!wrapper) return

    const content = document.querySelector(
      `.collapsible-header-content[data-references="${toggleButton.id}"]`,
    ) as HTMLElement
    if (!content) return

    // Set the expanded/collapsed state
    toggleButton.setAttribute("aria-expanded", (!targetState).toString())
    content.style.maxHeight = targetState ? "0px" : "inherit"
    content.classList.toggle("collapsed", targetState)
    wrapper.classList.toggle("collapsed", targetState)
    toggleButton.classList.toggle("collapsed", targetState)

    // Update sidenotes
    updateSidenoteState(content, targetState)

    // Update state in localStorage
    const headerId = toggleButton.id
    toggleCollapsedById(currentHeaderState, headerId)
  }

  // Save the final state
  saveHeaderState()

  // Use requestAnimationFrame to wait for DOM updates to complete
  requestAnimationFrame(() => {
    // Add a small delay to ensure transitions have completed
    setTimeout(updateContainerHeights, 200) // 200ms matches the transition duration in CSS
  })
}

function updateContainerHeights() {
  const articleContent = document.querySelector(".center") as HTMLElement
  const sideContainer = document.querySelector(".sidenotes") as HTMLElement

  if (articleContent && sideContainer) {
    // Set sidenotes container height to match article content
    const articleRect = articleContent.getBoundingClientRect()
    sideContainer.style.height = `${articleRect.height}px`
  }
}

document.addEventListener("nav", () => {
  const toolbar = document.querySelector(".toolbar")
  if (!toolbar) return

  const toolbarContent = toolbar.querySelector(".toolbar-content")
  if (!toolbarContent) return

  // Add resize observer to handle dynamic content changes
  const resizeObserver = new ResizeObserver(updateContainerHeights)
  const articleContent = document.querySelector(".center")
  if (articleContent) {
    resizeObserver.observe(articleContent)
  }

  const collapsibleButton = toolbarContent.querySelector("#collapsible-button") as HTMLButtonElement
  const collapsibleTooltip = collapsibleButton.querySelector(".tooltip") as HTMLElement
  collapsibleTooltip.textContent =
    collapsibleButton.getAttribute("data-state") === "expanded"
      ? "Collapse all sections"
      : "Expand all sections"
  const collapsible = () => toggleCollapse(collapsibleButton)
  collapsibleButton.addEventListener("click", collapsible)
  window.addCleanup(() => collapsibleButton.removeEventListener("click", collapsible))
})
