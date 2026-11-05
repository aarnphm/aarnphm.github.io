function toggleTransclude(this: HTMLButtonElement) {
  const outerBlock = this.closest<HTMLElement>('.transclude-collapsible')
  if (!outerBlock) return
  const content = outerBlock.querySelector<HTMLElement>(':scope > .transclude-content')
  if (!content) return
  outerBlock.classList.toggle('is-collapsed')
  const collapsed = outerBlock.classList.contains('is-collapsed')

  this.setAttribute('aria-expanded', (!collapsed).toString())
  content.style.gridTemplateRows = collapsed ? '0fr' : '1fr'
}

function setupTranscludes() {
  const collapsibleTranscludes = document.querySelectorAll<HTMLElement>('.transclude-collapsible')

  for (const transclude of collapsibleTranscludes) {
    const foldButton = transclude.querySelector<HTMLButtonElement>(
      ':scope > .transclude-title > button.transclude-fold',
    )
    const content = transclude.querySelector<HTMLElement>(':scope > .transclude-content')
    if (!foldButton || !content) continue

    foldButton.addEventListener('click', toggleTransclude)
    window.addCleanup(() => foldButton.removeEventListener('click', toggleTransclude))

    const collapsed = transclude.classList.contains('is-collapsed')
    content.style.gridTemplateRows = collapsed ? '0fr' : '1fr'
  }
}

document.addEventListener('nav', setupTranscludes)
