import {
  updateSidenoteState,
  toggleCollapsedById,
  saveHeaderState,
  loadHeaderState,
  closeReader,
} from "./util"

const TOGGLE_STATE = "toggleAllState"

function toggleCollapse(button: HTMLButtonElement) {
  const currentHeaderState = loadHeaderState()

  const allToggle = document.querySelectorAll(
    ".collapsible-header .toggle-button",
  ) as NodeListOf<HTMLElement>
  if (!allToggle.length) return

  const isExpanded = (localStorage.getItem(TOGGLE_STATE) ?? "collapsed") === "expanded"
  const buttonState = isExpanded ? "collapsed" : "expanded"
  button.setAttribute("data-state", buttonState)

  const tooltip = button.querySelector(".tooltip") as HTMLElement
  tooltip.textContent = isExpanded ? "Expand all sections" : "Collapse all sections"

  // Process each collapsible header
  for (const button of allToggle) {
    const wrapper = button.closest(".collapsible-header") as HTMLElement
    if (!wrapper) return

    const content = document.querySelector(
      `.collapsible-header-content[data-references="${button.id}"]`,
    ) as HTMLElement
    if (!content) return

    // Set the expanded/collapsed state
    button.setAttribute("aria-expanded", isExpanded.toString())
    content.style.maxHeight = isExpanded ? "0px" : "inherit"
    content.classList.toggle("collapsed", isExpanded)
    wrapper.classList.toggle("collapsed", isExpanded)
    button.classList.toggle("collapsed", isExpanded)

    updateSidenoteState(content, isExpanded)
    toggleCollapsedById(currentHeaderState, button.id)
  }

  // Save the final state
  saveHeaderState(currentHeaderState)
  localStorage.setItem(TOGGLE_STATE, buttonState)

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

function toggleReader(button: HTMLButtonElement) {
  const isActive = button.getAttribute("data-active") === "true"
  const readerView = document.querySelector(".reader") as HTMLElement
  if (!readerView) return
  const allHr = document.querySelectorAll("hr")
  const quartz = document.getElementById("quartz-root") as HTMLDivElement

  if (!isActive) {
    readerView.classList.add("active")
    button.setAttribute("data-active", "true")
    allHr.forEach((hr) => (hr.style.visibility = "hidden"))
    quartz.style.overflow = "hidden"
    quartz.style.maxHeight = "300px"
  } else {
    closeReader(readerView)
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

  // collapsible section
  const collapsibleButton = toolbarContent.querySelector("#collapsible-button") as HTMLButtonElement
  collapsibleButton.dataset.state = localStorage.getItem(TOGGLE_STATE) ?? "collapsed"
  const collapsibleTooltip = collapsibleButton.querySelector(".tooltip") as HTMLElement
  collapsibleTooltip.textContent =
    collapsibleButton.getAttribute("data-state") === "expanded"
      ? "Collapse all sections"
      : "Expand all sections"
  const collapsible = () => toggleCollapse(collapsibleButton)
  collapsibleButton.addEventListener("click", collapsible)
  window.addCleanup(() => collapsibleButton.removeEventListener("click", collapsible))

  // reader section
  const readerButton = toolbarContent.querySelector("#reader-button") as HTMLButtonElement
  const reader = () => toggleReader(readerButton)
  readerButton.addEventListener("click", reader)
  window.addCleanup(() => readerButton.removeEventListener("click", reader))
})
