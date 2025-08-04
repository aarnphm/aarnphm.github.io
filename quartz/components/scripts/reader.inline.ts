import { registerEscapeHandler, closeReader } from "./util"

document.addEventListener("nav", () => {
  const readerView = document.querySelector(".reader") as HTMLElement
  if (!readerView) return

  const closeFunctor = () => {
    closeReader(readerView)
    const headers = document.querySelector<HTMLDivElement>('section[class~="header"]')
    if (headers) {
      headers.style.display = "grid"
    }
  }

  // Register escape handler for the reader view
  registerEscapeHandler(readerView, closeFunctor)

  // Register close button handler
  const closeBtn = readerView.querySelector(".reader-close")
  if (closeBtn) {
    closeBtn.addEventListener("click", closeFunctor)
    window.addCleanup(() => closeBtn.removeEventListener("click", closeFunctor))
  }

  function showReader() {
    readerView.classList.add("active")
    const quartz = document.getElementById("quartz-root") as HTMLDivElement
    quartz.style.overflow = "hidden"
    quartz.style.maxHeight = "0px"

    const headers = document.querySelector<HTMLDivElement>('section[class~="header"]')
    if (headers) {
      headers.style.display = "none"
    }
  }

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "b" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      const readerViewOpen = readerView.classList.contains("active")
      readerViewOpen ? closeFunctor() : showReader()
    }
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))
})
