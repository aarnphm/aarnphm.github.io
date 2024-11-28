import { getFullSlug } from "../../util/path"
import { closeReader } from "./util"

// TODO: Export directly to PDF and skip this step
async function toggleExportPdf() {
  try {
    window.print()
  } catch (error) {
    console.error("Failed to export PDF:", error)
    // Show error to user
    const toast = document.createElement("div")
    toast.className = "pdf-toast"
    toast.textContent = "Failed to export PDF. Please try again."
    document.body.appendChild(toast)
    setTimeout(() => toast.remove(), 3000)
  }
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

const skewId = () => `${getFullSlug(window)}-skew-angle`
function setupToolbar() {
  const toolbar = document.querySelector(".toolbar")
  const page = document.querySelector(".center") as HTMLElement
  if (!toolbar) return

  const toolbarContent = toolbar.querySelector(".toolbar-content")
  if (!toolbarContent) return

  // reader section
  const readerButton = toolbarContent.querySelector("#reader-button") as HTMLButtonElement
  if (readerButton) {
    const reader = () => toggleReader(readerButton)
    readerButton.addEventListener("click", reader)
    window.addCleanup(() => readerButton.removeEventListener("click", reader))
  }

  // reader section
  const pdfButton = toolbarContent.querySelector("#pdf-button") as HTMLButtonElement
  if (pdfButton) {
    pdfButton.addEventListener("click", toggleExportPdf)
    window.addCleanup(() => pdfButton.removeEventListener("click", toggleExportPdf))
  }

  // skew funkyness
  const skewBtn = document.getElementById("skew-button") as HTMLButtonElement
  if (skewBtn) {
    const skew = skewId()
    // Initialize from localStorage if exists
    const isSkewed = localStorage.getItem(skew)
    if (isSkewed === "true") {
      page.classList.add("skewed")
      skewBtn.setAttribute("data-active", "true")
    }

    async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
      if (e.key === "u" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
        e.preventDefault()
        const isActive = page.classList.toggle("skewed")
        skewBtn.setAttribute("data-active", isActive.toString())
        localStorage.setItem(skew, isActive.toString())
      }
    }

    function toggleSkew(e: Event) {
      e.stopPropagation()
      const isActive = page.classList.toggle("skewed")
      skewBtn.setAttribute("data-active", isActive.toString())
      localStorage.setItem(skew, isActive.toString())
    }

    skewBtn.addEventListener("click", toggleSkew)
    document.addEventListener("keydown", shortcutHandler)

    window.addCleanup(() => {
      skewBtn.removeEventListener("click", toggleSkew)
      document.removeEventListener("keydown", shortcutHandler)
    })
  }
}

window.addEventListener("resize", setupToolbar)
document.addEventListener("nav", setupToolbar)
