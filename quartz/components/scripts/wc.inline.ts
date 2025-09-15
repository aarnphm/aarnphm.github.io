document.addEventListener("nav", () => {
  const modal = document.getElementById("wc-modal") as HTMLElement
  if (!modal) return
  const inner = modal.querySelector(".wc-inner") as HTMLElement

  function updateModal() {
    const selection = window.getSelection()

    if (!selection || selection.isCollapsed) {
      modal!.style.visibility = "hidden"
      return
    }
    const text = selection.toString().trim()
    if (!text) {
      modal!.style.visibility = "hidden"
      return
    }
    inner.innerHTML = ""

    inner.textContent = `${text.split(" ").filter((word) => word.length > 0).length} words`
    modal!.style.visibility = "visible"
  }

  document.addEventListener("selectionchange", updateModal)
  window.addCleanup(() => {
    document.removeEventListener("selectionchange", updateModal)
  })
})
