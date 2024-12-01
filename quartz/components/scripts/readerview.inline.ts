document.addEventListener("nav", () => {
  const switchCheckState = () => {
    const isChecked = readerView.getAttribute("aria-checked") === "true"
    if (!isChecked) {
      readerView.setAttribute("aria-checked", "true")
    } else {
      readerView.setAttribute("aria-checked", "false")
    }
  }

  const readerView = document.querySelector("#reader-view-toggle") as HTMLButtonElement
  readerView.addEventListener("click", switchCheckState)
  window.addCleanup(() => readerView.removeEventListener("click", switchCheckState))
})
