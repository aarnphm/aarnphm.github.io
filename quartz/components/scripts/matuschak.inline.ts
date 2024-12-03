// NOTE: We will import Matuschak's note view AFTER spa.inline.ts
// given that we will need to hijack the router
// We will only setup buttons here
// see ./spa.inline.ts
document.addEventListener("nav", async (ev) => {
  const button = document.getElementById("stacked-note-toggle") as HTMLButtonElement
  const container = document.getElementById("stacked-notes-container")
  if (!button || !container) return

  const switchCheckState = async () => {
    const isChecked = button.getAttribute("aria-checked") === "true"
    const body = document.body
    const currentUrl = window.location.href

    if (!isChecked) {
      button.setAttribute("aria-checked", "true")
      container.classList.add("active")
      body.classList.add("stack-mode")

      if (window.location.hash) {
        window.history.pushState("", document.title, currentUrl.split("#")[0])
      }
      await window.stacked.open()
    } else {
      button.setAttribute("aria-checked", "false")
      container.classList.remove("active")
      body.classList.remove("stack-mode")
      window.stacked.destroy()
      window.spaNavigate(ev.detail.url)
    }
  }

  button.addEventListener("click", switchCheckState)
  window.addCleanup(() => {
    button.removeEventListener("click", switchCheckState)
  })
})
