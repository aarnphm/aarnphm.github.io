import { getFullSlug } from "../../util/path"
import {
  closeReader,
  setCollapsedState,
  CollapsedState,
  setHeaderState,
  debounceUpdateHeight,
} from "./util"

const toolTipId = (id: string) => `${getFullSlug(window)}-tooltip-${id}`
const updateHeights = debounceUpdateHeight()

function toggleCollapse(button: HTMLButtonElement, buttonState: string) {
  const allToggleBtn = document.querySelectorAll(
    ".collapsible-header .toggle-button",
  ) as NodeListOf<HTMLElement>
  if (!allToggleBtn.length) return

  const shouldCollapsed = button.getAttribute("data-state") === "expanded"
  button.setAttribute("data-state", buttonState)

  const tooltip = button.querySelector(".tooltip") as HTMLElement
  tooltip.textContent = shouldCollapsed ? "Collapse all sections" : "Expand all sections"

  // Process each collapsible header
  for (const button of allToggleBtn) {
    const wrapper = button.closest(".collapsible-header") as HTMLElement
    if (!wrapper) return

    const content = document.querySelector(
      `.collapsible-header-content[data-references="${button.id}"]`,
    ) as HTMLElement
    if (!content) return

    // Set the expanded/collapsed state
    const newCollapseState = (!shouldCollapsed).toString() as CollapsedState
    setHeaderState(button, content, shouldCollapsed)
    setCollapsedState(window, button.id, newCollapseState)
  }

  // Use requestAnimationFrame to wait for DOM updates to complete
  requestAnimationFrame(() => setTimeout(updateHeights, 200))
}

function toggleReader(button: HTMLButtonElement) {
  const isActive = button.getAttribute("data-active") === "true"
  const readerView = document.querySelector(".reader") as HTMLElement
  if (!readerView) return
  const quartz = document.getElementById("quartz-root") as HTMLDivElement

  if (!isActive) {
    readerView.classList.add("active")
    button.setAttribute("data-active", "true")
    quartz.style.overflow = "hidden"
    quartz.style.maxHeight = "0px"
  } else {
    closeReader(readerView)
  }
}

function setupToolbar() {
  const toolbar = document.querySelector(".toolbar")
  if (!toolbar) return

  const toolbarContent = toolbar.querySelector(".toolbar-content")
  if (!toolbarContent) return

  // collapsible section
  const collapsibleButton = toolbarContent.querySelector("#collapsible-button") as HTMLButtonElement
  const collapsibleId = toolTipId(collapsibleButton.id)
  const maybeState =
    (localStorage.getItem(collapsibleId) ??
    Array.from(document.getElementsByClassName("collapsible-header")).every((h) =>
      h.classList.contains("collapsed"),
    ))
      ? "collapsed"
      : "expanded"
  collapsibleButton.setAttribute("data-state", maybeState)
  const tooltip = collapsibleButton.getElementsByClassName("tooltip")[0] as HTMLSpanElement
  tooltip.textContent =
    collapsibleButton.getAttribute("data-state") === "expanded"
      ? "Collapse all sections"
      : "Expand all sections"

  const switchState = (e: Event) => {
    const button = (e.target as HTMLElement).closest(
      "button#collapsible-button",
    ) as HTMLButtonElement | null
    if (!button || button !== collapsibleButton) return
    const newCollapseState =
      button?.getAttribute("data-state") === "expanded" ? "collapsed" : "expanded"
    toggleCollapse(button, newCollapseState)
    localStorage.setItem(collapsibleId, newCollapseState)
  }
  collapsibleButton.addEventListener("click", switchState)
  window.addCleanup(() => collapsibleButton.removeEventListener("click", switchState))

  // reader section
  const readerButton = toolbarContent.querySelector("#reader-button") as HTMLButtonElement
  const reader = () => toggleReader(readerButton)
  readerButton.addEventListener("click", reader)
  window.addCleanup(() => readerButton.removeEventListener("click", reader))
}

window.addEventListener("resize", setupToolbar)
document.addEventListener("nav", setupToolbar)
