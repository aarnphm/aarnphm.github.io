import { rootNavSignal } from './root-lifecycle'

export function initializeBaseTables(): void {
  for (const wrapper of document.querySelectorAll<HTMLElement>('.base-table-wrapper')) {
    if (wrapper.dataset.tableInitialized !== undefined) continue
    wrapper.dataset.tableInitialized = ''
    const signal = rootNavSignal(wrapper)
    wrapper.addEventListener(
      'click',
      event => {
        if (!(event.target instanceof Element)) return
        const toggle = event.target.closest<HTMLButtonElement>('.base-group-toggle')
        const header = toggle?.closest('tr.base-group-header')
        if (!toggle || !header) return
        const collapsed = toggle.getAttribute('aria-expanded') === 'true'
        toggle.setAttribute('aria-expanded', String(!collapsed))
        for (
          let row = header.nextElementSibling;
          row instanceof HTMLTableRowElement && !row.classList.contains('base-group-header');
          row = row.nextElementSibling
        ) {
          row.hidden = collapsed
        }
      },
      { signal },
    )
    const updateDivider = () => {
      wrapper.toggleAttribute('data-base-table-scrolled', wrapper.scrollTop > 0)
    }
    wrapper.addEventListener('scroll', updateDivider, { passive: true, signal })
    updateDivider()
    signal.addEventListener(
      'abort',
      () => {
        delete wrapper.dataset.tableInitialized
        wrapper.removeAttribute('data-base-table-scrolled')
      },
      { once: true },
    )
  }
}
