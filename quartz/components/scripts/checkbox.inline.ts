import { getFullSlug } from '../../util/path'
import { rootNavSignal } from './root-lifecycle'

const hydratedCheckboxes = new WeakSet<HTMLInputElement>()

function setupCheckbox(
  root: Document | HTMLElement = document,
  slug: string = getFullSlug(window),
) {
  const checkboxes = root.querySelectorAll<HTMLInputElement>('input.checkbox-toggle')
  checkboxes.forEach((el, index) => {
    const elId = `${slug}-checkbox-${index}`
    if (hydratedCheckboxes.has(el)) return
    hydratedCheckboxes.add(el)
    const signal = rootNavSignal(el)

    function switchState() {
      const newCheckboxState = el.checked ? 'true' : 'false'
      localStorage.setItem(elId, newCheckboxState)
    }

    el.addEventListener('change', switchState, { signal })
    signal.addEventListener('abort', () => hydratedCheckboxes.delete(el), { once: true })
    const stored = localStorage.getItem(elId)
    el.checked = stored === null ? el.defaultChecked : stored === 'true'
  })
}

document.addEventListener('nav', () => {
  setupCheckbox()
})
document.addEventListener('contentdecrypted', event => {
  setupCheckbox(event.detail.content, event.detail.slug)
})
