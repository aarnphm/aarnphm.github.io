function isElement(target: EventTarget | null): target is Element {
  return target instanceof Element
}

function targetOwnsNavigation(event: Event, toggle: HTMLElement): boolean {
  if (!isElement(event.target)) return false
  const navigable = event.target.closest('a, [data-role="anchor"]')
  return navigable !== null && toggle.contains(navigable)
}

function handleToggleClick(event: Event) {
  const toggle = event.currentTarget as HTMLElement | null
  if (!toggle) return

  if (targetOwnsNavigation(event, toggle)) return

  event.stopPropagation()

  const section = toggle.closest<HTMLElement>('section.collapsible-header')
  if (!section) return

  const shell = section.querySelector<HTMLElement>('[data-collapse-shell]')
  if (!shell) return

  shell.classList.toggle('is-open')
  const isOpen = shell.classList.contains('is-open')

  toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false')
  const slug = toggle.dataset.collapseSlug ?? window.document.body.dataset.slug ?? ''

  localStorage.setItem(collapseStorageKey(slug, toggle.id), isOpen ? 'true' : 'false')

  document.dispatchEvent(
    new CustomEvent('collapsibletoggle', { detail: { toggleId: toggle.id, isOpen } }),
  )
}

function handleToggleKeydown(event: KeyboardEvent) {
  const key = event.key
  if (key !== 'Enter' && key !== ' ' && key !== 'Spacebar') return

  const toggle = event.currentTarget as HTMLElement | null
  if (!toggle) return

  if (targetOwnsNavigation(event, toggle)) return

  event.preventDefault()
  toggle.click()
}

const hydratedCollapseToggles = new WeakSet<HTMLElement>()
const hydratedTranscludeButtons = new WeakSet<Element>()

function collapseStorageKey(slug: string, toggleId: string) {
  return `${slug.replace(/\//g, '--')}-${toggleId}`
}

function hydrateCollapsibleHeaders(
  root: Document | HTMLElement = document,
  slug: string | undefined = window.document.body.dataset.slug,
) {
  const storageSlug = slug ?? window.document.body.dataset.slug ?? ''
  root.querySelectorAll<HTMLElement>('section.collapsible-header').forEach(section => {
    const shell = section.querySelector<HTMLElement>('[data-collapse-shell]')
    if (!shell) return

    const toggle = section.querySelector<HTMLElement>('[data-collapse-toggle]')
    if (!toggle) return

    if (!hydratedCollapseToggles.has(toggle)) {
      const initialOpen = shell.dataset.initialOpen !== 'false'
      const stored = localStorage.getItem(collapseStorageKey(storageSlug, toggle.id))
      const isOpen = stored ? stored === 'true' : initialOpen

      if (isOpen) {
        shell.classList.add('is-open')
      } else {
        shell.classList.remove('is-open')
      }
      toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false')

      if (toggle.tabIndex === -1) {
        toggle.tabIndex = 0
      }

      hydratedCollapseToggles.add(toggle)
      toggle.dataset.collapseSlug = storageSlug
      toggle.addEventListener('click', handleToggleClick)
      toggle.addEventListener('keydown', handleToggleKeydown)
    }
  })

  const transcludeButtons = root.querySelectorAll('button.transclude-title-link')
  for (const button of transcludeButtons) {
    if (hydratedTranscludeButtons.has(button)) continue
    hydratedTranscludeButtons.add(button)
    const parent = button.parentElement as HTMLElement | null
    if (!parent || !parent.dataset.href) continue

    const navigate = () => {
      const href = parent.dataset.href
      if (!href) return
      let targetUrl: URL
      try {
        targetUrl = new URL(href, window.location.toString())
      } catch {
        return
      }

      if (targetUrl.origin !== window.location.origin) {
        window.location.assign(targetUrl.toString())
        return
      }

      window.spaNavigate(targetUrl)
    }

    button.addEventListener('click', navigate)
  }
}

document.addEventListener('nav', () => {
  hydrateCollapsibleHeaders()
})
window.addEventListener('resize', () => {
  hydrateCollapsibleHeaders()
})
document.addEventListener('contentdecrypted', event => {
  hydrateCollapsibleHeaders(event.detail.content, event.detail.slug)
})
