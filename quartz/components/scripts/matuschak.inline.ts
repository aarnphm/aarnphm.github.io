// NOTE: We will import Matuschak's note view AFTER spa.inline.ts
// given that we will need to hijack the router
// We will only setup buttons here

// see ./spa.inline.ts
document.addEventListener("nav", async (ev) => {
  const button = document.getElementById("stacked-note-toggle") as HTMLButtonElement
  const container = document.getElementById("stacked-notes-container")
  const header = document.getElementsByClassName("header")[0] as HTMLElement

  if (!button || !container || !header) return

  const switchCheckState = async () => {
    const isChecked = button.getAttribute("aria-checked") === "true"
    const body = document.body
    const currentUrl = window.location.href

    if (!isChecked) {
      button.setAttribute("aria-checked", "true")
      container.classList.add("active")
      body.classList.add("stack-mode")
      header.classList.add("grid", "all-col")

      if (window.location.hash) {
        window.history.pushState("", document.title, currentUrl.split("#")[0])
      }
      window.stacked.navigate(new URL(`/${ev.detail.url}`, window.location.toString()))
    } else {
      // Keep visual state intact until reload to prevent layout jump
      window.stacked.destroy()
      window.location.reload()
    }
  }

  if (window.location.hostname.startsWith("notes.aarnphm.xyz")) return

  button.addEventListener("click", switchCheckState)
  window.addCleanup(() => {
    button.removeEventListener("click", switchCheckState)
  })
})
