function toggleCallout(this: HTMLElement) {
  const outerBlock = this.parentElement!
  outerBlock.classList.toggle('is-collapsed')
  const content = outerBlock.getElementsByClassName('callout-content')[0] as HTMLElement
  if (!content) return
  const collapsed = outerBlock.classList.contains('is-collapsed')
  content.style.gridTemplateRows = collapsed ? '0fr' : '1fr'
}

const hydratedCalloutTitles = new WeakSet<HTMLElement>()

function setupCallout(root: Document | HTMLElement = document) {
  const collapsible = root.querySelectorAll<HTMLElement>('.callout.is-collapsible')
  for (const div of collapsible) {
    const title = div.getElementsByClassName('callout-title')[0] as HTMLElement
    const content = div.getElementsByClassName('callout-content')[0] as HTMLElement
    if (!title || !content) continue

    if (!hydratedCalloutTitles.has(title)) {
      hydratedCalloutTitles.add(title)
      title.addEventListener('click', toggleCallout)
    }

    const collapsed = div.classList.contains('is-collapsed')
    content.style.gridTemplateRows = collapsed ? '0fr' : '1fr'
  }
}

document.addEventListener('nav', () => {
  setupCallout()
})
document.addEventListener('contentdecrypted', event => {
  setupCallout(event.detail.content)
})
