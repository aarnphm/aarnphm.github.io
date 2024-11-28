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
    let previousSkewState = localStorage.getItem(skew) === "true"

    // Initialize from localStorage if exists
    if (previousSkewState) {
      page.classList.add("skewed")
      skewBtn.setAttribute("data-active", "true")
    }

    function areActiveContainers(): boolean {
      // Check multiple mermaid containers
      const mermaids = document.querySelectorAll("#mermaid-container")
      return Array.from(mermaids).some((container) => container.classList.contains("active"))
    }

    function handleContainerState() {
      if (areActiveContainers()) {
        // Cache current state before removing
        previousSkewState = page.classList.contains("skewed")
        // Remove skew when container active
        page.classList.remove("skewed")
        skewBtn.setAttribute("data-active", "false")
      } else {
        // Restore previous state
        if (previousSkewState) {
          page.classList.add("skewed")
          skewBtn.setAttribute("data-active", "true")
        } else {
          page.classList.remove("skewed")
          skewBtn.setAttribute("data-active", "false")
        }
      }
    }

    async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
      if (e.key === "u" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
        e.preventDefault()
        if (!areActiveContainers()) {
          previousSkewState = !previousSkewState
          const isActive = page.classList.toggle("skewed")
          skewBtn.setAttribute("data-active", isActive.toString())
          localStorage.setItem(skew, isActive.toString())
        }
      }
    }

    function toggleSkew(e: Event) {
      e.stopPropagation()
      if (!areActiveContainers()) {
        previousSkewState = !previousSkewState
        const isActive = page.classList.toggle("skewed")
        skewBtn.setAttribute("data-active", isActive.toString())
        localStorage.setItem(skew, isActive.toString())
      }
    }

    // Watch for container changes using MutationObserver
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === "attributes" && mutation.attributeName === "class") {
          handleContainerState()
        }
      })
    })
    document.querySelectorAll("#mermaid-container").forEach((container) => {
      observer.observe(container, { attributes: true })
    })

    skewBtn.addEventListener("click", toggleSkew)
    document.addEventListener("keydown", shortcutHandler)

    window.addCleanup(() => {
      skewBtn.removeEventListener("click", toggleSkew)
      document.removeEventListener("keydown", shortcutHandler)
      observer.disconnect()
    })
  }
}

window.addEventListener("resize", setupToolbar)
document.addEventListener("nav", setupToolbar)
