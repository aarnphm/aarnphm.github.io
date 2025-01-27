document.addEventListener("nav", () => {
  const modal = document.getElementById("wc-modal") as HTMLElement
  if (!modal) return
  const inner = modal.querySelector(".wc-inner") as HTMLElement
  let current: Selection | null = null

  function updateModalPosition() {
    if (!current || current.isCollapsed) return
    const rect = current.getRangeAt(0).getBoundingClientRect()
    modal.style.top = `${Math.min(rect.top + window.scrollY, window.scrollY + window.innerHeight - modal.offsetHeight)}px`
    modal.style.left = `${rect.right + 20}px`
  }

  function updateModal() {
    const selection = window.getSelection()
    current = selection

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
    updateModalPosition()
  }

  document.addEventListener("selectionchange", updateModal)
  document.addEventListener("scroll", updateModalPosition)
  window.addCleanup(() => {
    document.removeEventListener("selectionchange", updateModal)
    document.removeEventListener("scroll", updateModalPosition)
  })
})
