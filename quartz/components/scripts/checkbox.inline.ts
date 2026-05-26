import { getFullSlug } from '../../util/path'

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

    function switchState(e: Event) {
      const newCheckboxState = (e.target as HTMLInputElement)?.checked ? 'true' : 'false'
      localStorage.setItem(elId, newCheckboxState)
    }

    el.addEventListener('change', switchState)
    if (localStorage.getItem(elId) === 'true') {
      el.checked = true
    }
  })
}

document.addEventListener('nav', () => {
  setupCheckbox()
})
document.addEventListener('contentdecrypted', event => {
  setupCheckbox(event.detail.content, event.detail.slug)
})
