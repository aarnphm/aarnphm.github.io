import { rootNavSignal } from './root-lifecycle'

function toggleCallout(this: HTMLElement, event: MouseEvent) {
  const target = event.target
  if (
    target instanceof Element &&
    !target.closest('.callout-fold-toggle') &&
    target.closest('a, button, input, select, textarea, [contenteditable="true"]')
  ) {
    return
  }

  const outerBlock = this.parentElement!
  outerBlock.classList.toggle('is-collapsed')
  const content = outerBlock.getElementsByClassName('callout-content')[0] as HTMLElement
  const collapsed = outerBlock.classList.contains('is-collapsed')
  this.querySelector<HTMLButtonElement>('.callout-fold-toggle')?.setAttribute(
    'aria-expanded',
    String(!collapsed),
  )
  if (content) content.style.gridTemplateRows = collapsed ? '0fr' : '1fr'
  if (!content) return
  document.dispatchEvent(new CustomEvent('callouttoggle', { detail: { collapsed } }))
}

const hydratedCalloutSignals = new WeakMap<HTMLElement, AbortSignal>()

function setupCallout(root: Document | HTMLElement = document) {
  const collapsible = root.querySelectorAll<HTMLElement>('.callout.is-collapsible')
  for (const div of collapsible) {
    const title = div.getElementsByClassName('callout-title')[0] as HTMLElement
    const content = div.getElementsByClassName('callout-content')[0] as HTMLElement
    if (!title) continue

    title
      .querySelector<HTMLButtonElement>('.callout-fold-toggle')
      ?.setAttribute('aria-expanded', String(!div.classList.contains('is-collapsed')))
    const signal = rootNavSignal(title)
    if (hydratedCalloutSignals.get(title) !== signal) {
      hydratedCalloutSignals.set(title, signal)
      title.addEventListener('click', toggleCallout, { signal })
    }

    const collapsed = div.classList.contains('is-collapsed')
    if (content) content.style.gridTemplateRows = collapsed ? '0fr' : '1fr'
  }
}

document.addEventListener('nav', () => {
  setupCallout()
})
document.addEventListener('contentdecrypted', event => {
  setupCallout(event.detail.content)
})
