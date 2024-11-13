import { closeReader } from "./util"

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

  // reader section
  const readerButton = toolbarContent.querySelector("#reader-button") as HTMLButtonElement
  const reader = () => toggleReader(readerButton)
  readerButton.addEventListener("click", reader)
  window.addCleanup(() => readerButton.removeEventListener("click", reader))
}

window.addEventListener("resize", setupToolbar)
document.addEventListener("nav", setupToolbar)
