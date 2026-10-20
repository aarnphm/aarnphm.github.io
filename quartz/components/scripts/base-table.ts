import { rootNavSignal } from './root-lifecycle'

export function initializeBaseTables(): void {
  for (const wrapper of document.querySelectorAll<HTMLElement>('.base-table-wrapper')) {
    if (wrapper.dataset.tableInitialized !== undefined) continue
    wrapper.dataset.tableInitialized = ''
    const signal = rootNavSignal(wrapper)
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
