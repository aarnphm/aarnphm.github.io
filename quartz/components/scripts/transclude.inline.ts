function toggleTransclude(this: HTMLElement) {
  const outerBlock = this.closest(".transclude-collapsible") as HTMLElement
  if (!outerBlock) return

  outerBlock.classList.toggle("is-collapsed")
  const content = outerBlock.querySelector(".transclude-content") as HTMLElement
  if (!content) return

  const collapsed = outerBlock.classList.contains("is-collapsed")

  // update aria-expanded attribute
  this.setAttribute("aria-expanded", (!collapsed).toString())

  // use grid-template-rows for smooth animation
  content.style.gridTemplateRows = collapsed ? "0fr" : "1fr"
}

function setupTranscludes() {
  const collapsibleTranscludes = document.querySelectorAll(
    ".transclude-collapsible",
  ) as NodeListOf<HTMLElement>

  for (const transclude of collapsibleTranscludes) {
    const foldButton = transclude.querySelector(".transclude-fold") as HTMLElement
    const content = transclude.querySelector(".transclude-content") as HTMLElement
    if (!foldButton || !content) continue

    foldButton.addEventListener("click", toggleTransclude)
    window.addCleanup(() => foldButton.removeEventListener("click", toggleTransclude))

    // set initial grid-template-rows based on collapsed state
    const collapsed = transclude.classList.contains("is-collapsed")
    content.style.gridTemplateRows = collapsed ? "0fr" : "1fr"
  }
}

document.addEventListener("nav", setupTranscludes)
