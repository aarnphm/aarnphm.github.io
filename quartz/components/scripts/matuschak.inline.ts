// NOTE: We will import Matuschak's note view AFTER spa.inline.ts
// given that we will need to hijack the router
// We will only setup buttons here
// see ./spa.inline.ts
document.addEventListener("nav", async () => {
  const readerView = document.getElementById("reader-view-toggle") as HTMLButtonElement
  if (!readerView) return

  const switchCheckState = async () => {
    const isChecked = readerView.getAttribute("aria-checked") === "true"
    const header = document.getElementsByTagName("header")[0]
    const footer = document.getElementsByTagName("footer")[0]
    const main = document.getElementById("quartz-root") as HTMLElement
    const stackedContainer = document.getElementById("stacked-notes-container")

    if (!isChecked) {
      readerView.setAttribute("aria-checked", "true")
      header?.classList.add("stack-mode")
      footer?.classList.add("stack-mode")
      main?.classList.add("stack-mode")
      stackedContainer?.classList.add("active")

      const url = new URL(window.location.toString())
      const stackParam = url.hash.match(/stack=([^&]+)/)
      if (stackParam) {
        await window.stacked.restore(stackParam[1].split(","))
      } else {
        await window.stacked.open()
      }
    } else {
      readerView.setAttribute("aria-checked", "false")
      header?.classList.remove("stack-mode")
      footer?.classList.remove("stack-mode")
      main?.classList.remove("stack-mode")
      stackedContainer?.classList.remove("active")

      window.stacked.destroy()
    }
  }

  readerView.addEventListener("click", switchCheckState)
  window.addCleanup(() => readerView.removeEventListener("click", switchCheckState))
})
