import { currentNavSignal } from './nav-lifecycle'
import { rootNavSignal } from './root-lifecycle'

let activeSignal: AbortSignal | undefined
let closeDropdowns = new WeakMap<Element, () => void>()

function formatResultsLabel(resultCount: string, totalCount: string): string {
  if (!resultCount || !totalCount) return ''
  return resultCount === totalCount
    ? `${totalCount} results`
    : `${resultCount} of ${totalCount} results`
}

function updateEmbedView(
  selector: HTMLElement,
  viewList: HTMLElement,
  embedRoot: HTMLElement,
  activeLink: HTMLAnchorElement,
): void {
  const viewName = activeLink.dataset.viewName ?? ''
  const viewSlug = activeLink.dataset.slug ?? ''
  let activeView: HTMLElement | undefined

  for (const view of embedRoot.querySelectorAll<HTMLElement>('[data-base-embed-view]')) {
    const name = view.dataset.baseViewName ?? ''
    const slug = view.dataset.baseViewSlug ?? ''
    const matches =
      (viewSlug.length > 0 && slug === viewSlug) ||
      (viewName.length > 0 && name.toLowerCase() === viewName.toLowerCase())

    view.hidden = !matches
    view.classList.toggle('is-active', matches)
    if (matches) activeView = view
  }

  if (activeView) {
    const label = formatResultsLabel(
      activeView.dataset.baseViewResultCount ?? '',
      activeView.dataset.baseViewTotalCount ?? '',
    )
    const resultsLabel = embedRoot.querySelector<HTMLElement>('[data-base-embed-results-label]')
    if (resultsLabel && label) resultsLabel.textContent = label
  }

  const triggerLabel = selector.querySelector<HTMLElement>('.text-button-label')
  if (triggerLabel && viewName) triggerLabel.textContent = viewName.toLowerCase()

  const triggerIcon = selector.querySelector<HTMLElement>('.text-button-icon:not(.mod-aux)')
  const linkIcon = activeLink.querySelector<HTMLElement>('.bases-toolbar-menu-item-info-icon')
  if (triggerIcon && linkIcon) triggerIcon.replaceChildren(...linkIcon.cloneNode(true).childNodes)

  for (const link of viewList.querySelectorAll<HTMLElement>('.bases-toolbar-menu-item')) {
    const isActive = link === activeLink
    link.classList.toggle('mod-active', isActive)
    link.classList.toggle('is-selected', isActive)
    if (isActive) {
      link.setAttribute('aria-current', 'page')
    } else {
      link.removeAttribute('aria-current')
    }
  }
}

document.addEventListener('nav', () => {
  const navSignal = currentNavSignal()
  if (activeSignal !== navSignal) {
    activeSignal = navSignal

    document.addEventListener(
      'click',
      event => {
        const target = event.target
        if (!(target instanceof Node)) return

        for (const selector of document.querySelectorAll<HTMLElement>(
          '[data-base-view-selector][data-initialized]',
        )) {
          if (!selector.contains(target)) closeDropdowns.get(selector)?.()
        }
      },
      { signal: navSignal },
    )

    navSignal.addEventListener(
      'abort',
      () => {
        for (const selector of document.querySelectorAll<HTMLElement>(
          '[data-base-view-selector][data-initialized]',
        )) {
          closeDropdowns.get(selector)?.()
          delete selector.dataset.initialized
        }
        closeDropdowns = new WeakMap()
        if (activeSignal === navSignal) activeSignal = undefined
      },
      { once: true },
    )
  }

  for (const selector of document.querySelectorAll<HTMLElement>('[data-base-view-selector]')) {
    if (selector.dataset.initialized !== undefined) continue

    const triggerElement = selector.querySelector<HTMLButtonElement>('.text-icon-button')
    const dropdownElement = selector.querySelector<HTMLElement>('[data-dropdown]')
    const searchElement = selector.querySelector<HTMLInputElement>('[data-search-input]')
    const clearElement = selector.querySelector<HTMLButtonElement>('[data-clear-search]')
    const listElement = selector.querySelector<HTMLElement>('[data-view-list]')
    const embedRoot = selector.closest<HTMLElement>('.base-embed')
    if (!triggerElement || !dropdownElement || !searchElement || !clearElement || !listElement) {
      continue
    }

    const trigger = triggerElement
    const dropdown = dropdownElement
    const searchInput = searchElement
    const clearButton = clearElement
    const viewList = listElement
    const signal = rootNavSignal(selector)

    selector.dataset.initialized = ''
    dropdown.hidden = true

    function filterViews(query: string): void {
      const normalizedQuery = query.toLowerCase()
      for (const item of viewList.querySelectorAll<HTMLElement>('.bases-toolbar-menu-item')) {
        const viewName = item.dataset.viewName?.toLowerCase() ?? ''
        const viewType = item.dataset.viewType?.toLowerCase() ?? ''
        item.hidden = !viewName.includes(normalizedQuery) && !viewType.includes(normalizedQuery)
      }
    }

    function clearSearch(): void {
      searchInput.value = ''
      clearButton.hidden = true
      filterViews('')
      searchInput.focus()
    }

    function closeDropdown(focusTrigger = false): void {
      trigger.setAttribute('aria-expanded', 'false')
      trigger.classList.remove('has-active-menu')
      dropdown.hidden = true
      searchInput.value = ''
      clearButton.hidden = true
      filterViews('')
      if (focusTrigger) trigger.focus()
    }

    function openDropdown(): void {
      for (const openSelector of document.querySelectorAll<HTMLElement>(
        '[data-base-view-selector][data-initialized]',
      )) {
        if (openSelector !== selector) closeDropdowns.get(openSelector)?.()
      }

      trigger.setAttribute('aria-expanded', 'true')
      trigger.classList.add('has-active-menu')
      dropdown.hidden = false
      queueMicrotask(() => searchInput.focus())
    }

    closeDropdowns.set(selector, closeDropdown)
    signal.addEventListener(
      'abort',
      () => {
        closeDropdown()
        closeDropdowns.delete(selector)
        delete selector.dataset.initialized
      },
      { once: true },
    )

    trigger.addEventListener(
      'click',
      event => {
        event.stopPropagation()
        if (trigger.getAttribute('aria-expanded') === 'true') {
          closeDropdown()
        } else {
          openDropdown()
        }
      },
      { signal },
    )

    searchInput.addEventListener(
      'input',
      () => {
        filterViews(searchInput.value)
        clearButton.hidden = searchInput.value.length === 0
      },
      { signal },
    )

    searchInput.addEventListener(
      'keydown',
      event => {
        if (event.key !== 'Escape') return
        if (searchInput.value) {
          clearSearch()
        } else {
          closeDropdown(true)
        }
      },
      { signal },
    )

    clearButton.addEventListener(
      'click',
      event => {
        event.stopPropagation()
        clearSearch()
      },
      { signal },
    )

    viewList.addEventListener(
      'click',
      event => {
        const target = event.target
        if (!(target instanceof Element)) return
        const activeLink = target.closest<HTMLAnchorElement>('.bases-toolbar-menu-item')
        if (!activeLink || !viewList.contains(activeLink)) return

        if (embedRoot) {
          event.preventDefault()
          event.stopPropagation()
          updateEmbedView(selector, viewList, embedRoot, activeLink)
        }
        closeDropdown()
      },
      { signal },
    )
  }
})

export default ''
