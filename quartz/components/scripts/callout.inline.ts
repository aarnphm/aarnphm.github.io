import { rootNavSignal } from './root-lifecycle'

function toggleCallout(this: HTMLElement) {
  const outerBlock = this.parentElement!
  outerBlock.classList.toggle('is-collapsed')
  const content = outerBlock.getElementsByClassName('callout-content')[0] as HTMLElement
  if (!content) return
  const collapsed = outerBlock.classList.contains('is-collapsed')
  content.style.gridTemplateRows = collapsed ? '0fr' : '1fr'
  document.dispatchEvent(new CustomEvent('callouttoggle', { detail: { collapsed } }))
}

const hydratedCalloutSignals = new WeakMap<HTMLElement, AbortSignal>()

function setupCallout(root: Document | HTMLElement = document) {
  const collapsible = root.querySelectorAll<HTMLElement>('.callout.is-collapsible')
  for (const div of collapsible) {
    const title = div.getElementsByClassName('callout-title')[0] as HTMLElement
    const content = div.getElementsByClassName('callout-content')[0] as HTMLElement
    if (!title || !content) continue

    const signal = rootNavSignal(title)
    if (hydratedCalloutSignals.get(title) !== signal) {
      hydratedCalloutSignals.set(title, signal)
      title.addEventListener('click', toggleCallout, { signal })
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
